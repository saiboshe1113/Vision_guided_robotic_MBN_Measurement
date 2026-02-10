import os
import random
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A

# =========================
# 路径与超参
# =========================
DATASET_DIR = r".\dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images1")
MSK_DIR = os.path.join(DATASET_DIR, "masks1")

IMG_SIZE     = 512
BATCH_SIZE   = 2
EPOCHS       = 100
LR           = 1e-4
VAL_SPLIT    = 0.15   # 验证少一点
SEED         = 42
NUM_WORKERS  = 4
SAVE_DIR     = r".\runs_deeplab"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_CLASSES  = 4  # 背景+3类

# 固定随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# =========================
# 数据集
# =========================
def list_pairs(img_dir, msk_dir, min_fg_pixels: int = 200):
    """从目录里成对读图，并过滤掉前景像素太少的"""
    imgs = sorted([
        p for p in glob(os.path.join(img_dir, "*"))
        if os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])
    pairs = []
    for ip in imgs:
        name = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(msk_dir, f"{name}.png")
        if not os.path.exists(mp):
            continue

        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        m = (m / 80).round().astype(np.uint8)  # 0/80/160/240 -> 0/1/2/3
        if (m > 0).sum() < min_fg_pixels:
            continue

        pairs.append((ip, mp))
    return pairs


class SegDataset(Dataset):
    def __init__(self, pairs, img_size=512, aug=False):
        self.pairs = pairs
        self.img_size = img_size
        self.aug = aug

        # 训练：只做尺寸对齐 + 很轻的扰动
        self.tf_train = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                img_size, img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),
            A.RandomCrop(img_size, img_size, p=0.3),
            A.RandomBrightnessContrast(
                p=0.2,
                brightness_limit=0.1,
                contrast_limit=0.1
            ),
        ])

        # 验证：固定
        self.tf_val = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                img_size, img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),
            A.CenterCrop(img_size, img_size),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = cv2.imread(ip)[:, :, ::-1]
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)

        # 0/80/160/240 -> 0/1/2/3
        msk = (msk / 80).round().astype(np.uint8)
        msk = np.clip(msk, 0, NUM_CLASSES - 1)

        data = (self.aug and self.tf_train or self.tf_val)(image=img, mask=msk)
        img, msk = data["image"], data["mask"]

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        msk = msk.astype(np.int64)

        return torch.from_numpy(img), torch.from_numpy(msk)


# =========================
# 指标
# =========================
@torch.no_grad()
def mean_dice_iou_multiclass(pred, target,
                             num_classes=NUM_CLASSES,
                             ignore_index=0,
                             eps=1e-6):
    dices, ious = [], []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_c = (pred == cls).float()
        targ_c = (target == cls).float()
        inter = (pred_c * targ_c).sum(dim=(1, 2))
        pred_sum = pred_c.sum(dim=(1, 2))
        targ_sum = targ_c.sum(dim=(1, 2))
        union = pred_sum + targ_sum - inter

        dice = (2 * inter + eps) / (pred_sum + targ_sum + eps)
        iou = (inter + eps) / (union + eps)
        dices.append(dice.mean().item())
        ious.append(iou.mean().item())

    if not dices:
        return 0.0, 0.0
    return float(np.mean(dices)), float(np.mean(ious))


# =========================
# Dice Loss
# =========================
def dice_loss_multiclass(logits, targets, eps=1e-6):
    """
    logits: [B, C, H, W]
    targets: [B, H, W]
    """
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(
        targets, num_classes=num_classes
    ).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    inter = torch.sum(probs * one_hot, dims)       # [C]
    den = torch.sum(probs + one_hot, dims)         # [C]
    dice_per_class = (2 * inter + eps) / (den + eps)

    # 去掉背景
    if dice_per_class.numel() > 1:
        dice_fg = dice_per_class[1:]
    else:
        dice_fg = dice_per_class

    return 1 - dice_fg.mean()


# =========================
# Focal Loss
# =========================
def focal_loss(logits, targets, alpha=None, gamma=2.0):
    """
    logits: [B,C,H,W]
    targets: [B,H,W]
    alpha: 可传每类的权重 [C]，比如你的 class_weights
    """
    ce = F.cross_entropy(logits, targets, reduction="none")  # [B,H,W]
    pt = torch.exp(-ce)                                      # [B,H,W]
    focal = (1 - pt) ** gamma * ce                           # [B,H,W]

    if alpha is not None:
        # alpha: [C] → [B,H,W]
        focal = alpha[targets] * focal

    return focal.mean()


# =========================
# 模型
# =========================
def create_model(num_classes=NUM_CLASSES):
    # 优先用 resnet50 预训练
    try:
        from torchvision.models.segmentation import (
            deeplabv3_resnet50,
            DeepLabV3_ResNet50_Weights,
        )
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        model = deeplabv3_resnet50(weights=weights)
    except Exception:
        from torchvision.models.segmentation import deeplabv3_resnet101
        model = deeplabv3_resnet101(weights=None)

    in_ch = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    if getattr(model, "aux_classifier", None) is not None:
        aux_in = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_in, num_classes, kernel_size=1)

    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    pairs = list_pairs(IMG_DIR, MSK_DIR)
    assert len(pairs) > 0, f"No pairs found in {IMG_DIR} and {MSK_DIR}"
    print(f"Total pairs (after filter): {len(pairs)}")

    random.shuffle(pairs)
    split = int(len(pairs) * (1 - VAL_SPLIT))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    train_ds = SegDataset(train_pairs, IMG_SIZE, aug=True)
    val_ds = SegDataset(val_pairs, IMG_SIZE, aug=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = create_model(num_classes=NUM_CLASSES).to(device)

    # 你之前的类权重
    class_weights = torch.tensor(
        [0.011286975091191085, 0.4371271256108732, 2.165511103918138, 1.3860747953797974],
        device=device, dtype=torch.float32
    )
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 先预热 5 epoch，再余弦
    def get_scheduler(optimizer, num_warmup=5, num_epochs=EPOCHS):
        def lr_lambda(current_epoch):
            if current_epoch < num_warmup:
                return float(current_epoch + 1) / float(num_warmup)
            progress = (current_epoch - num_warmup) / float(max(1, num_epochs - num_warmup))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = get_scheduler(optimizer)

    # AMP
    try:
        scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
        use_new_autocast = True
    except Exception:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=(device == "cuda"))
        use_new_autocast = False

    best_iou, best_path = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_meter = 0.0
        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}")

        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_new_autocast:
                with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                    out = model(imgs)
                    logits = out["out"]

                    ce = ce_criterion(logits, masks)
                    fl = focal_loss(logits, masks, alpha=class_weights, gamma=2.0)
                    dl = dice_loss_multiclass(logits, masks)

                    loss = 0.4 * ce + 0.3 * fl + 0.3 * dl
            else:
                from torch.cuda.amp import autocast
                with autocast(enabled=(device == "cuda")):
                    out = model(imgs)
                    logits = out["out"]

                    ce = ce_criterion(logits, masks)
                    fl = focal_loss(logits, masks, alpha=class_weights, gamma=2.0)
                    dl = dice_loss_multiclass(logits, masks)

                    loss = 0.4 * ce + 0.3 * fl + 0.3 * dl

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = loss_meter / len(train_ds)

        # ===== 验证 =====
        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc="[Val]"):
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)["out"]
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

                d, i = mean_dice_iou_multiclass(
                    preds, masks,
                    num_classes=NUM_CLASSES,
                    ignore_index=0
                )
                dices.append(d)
                ious.append(i)

        md, mi = float(np.mean(dices)), float(np.mean(ious))
        print(f"Epoch {epoch}: loss={avg_loss:.4f}  mDice={md:.4f}  mIoU={mi:.4f}")

        scheduler.step()

        if mi > best_iou:
            best_iou = mi
            best_path = os.path.join(
                SAVE_DIR,
                f"best_deeplab_resnet50_mIoU{best_iou:.4f}.pth"
            )
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best to: {best_path}")

    print("Training finished.")
    if best_path:
        print("Best model:", best_path)


if __name__ == "__main__":
    main()
