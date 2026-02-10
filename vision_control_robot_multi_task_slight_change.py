# -*- coding: utf-8 -*-
import time
import math
import cv2
import numpy as np
import pyorbbecsdk as ob
import torch
import torch.nn as nn
import torchvision
import msvcrt
from collections import deque
import os  # 🔹 用于删除 txt

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# =============== 基本配置 ===============

MODEL_PATH = r"./runs_deeplab/best_deeplab_resnet50_mIoU0.9079.pth"
ROBOT_IP = "172.168.1.100"

NUM_CLASSES = 4
INPUT_SIZE = 512

# 分类别阈值
CLS_THRESH = {1: 0.85, 2: 0.80, 3: 0.80}
CLS_AREA = {1: 1800, 2: 1300, 3: 600}

# 相机/工具/传感器几何关系
SAFE_OFFSET = -0.01  # 识别后，沿工具Z再下 1cm（安全）
CAM_Z_DOWN = -0.02  # 相机比 TCP 低 1cm
CAM_ORI_ID = 0  # 相机与工具姿态关系（目前设成一致）

ACC = 0.1
VEL = 0.1

# 🔹 Z 轴粗/细调用更慢的速度（大范围移动仍用上面的 VEL/ACC）
VEL_Z_COARSE = 0.005  # 粗调：3 cm/s
VEL_Z_FINE = 0.001  # 细调：1 cm/s
ACC_Z = 0.02  # Z 向加速度小一点，更柔和

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class_names = ["BG", "sample1", "sample2", "sample3"]
class_colors = [(0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

# 类间优先级（用于 2 秒识别阶段的互斥）
CLASS_PRIORITY = {1: 3, 2: 2, 3: 1}

# 时序多数投票（抑制瞬时误检）
TEMPORAL_BUF = {1: {}, 2: {}, 3: {}}  # {cls: {track_key: (cx,cy,deque)}}
MAX_TRACKS_PER_CLASS = 8
HISTORY = 5
VOTE_K = 3

# 传感器相对工具 Y 方向偏移 55mm
SENSOR_OFFSET_y = -0.0875  # 方向反了就改成 +0.055
SENSOR_OFFSET_X = -0.0075

# 每个类别最多测几个目标
MAX_INST_PER_CLASS = 4  # 想测完可设大一点，例如 99

# =============== 粗/细调 Z（基于 txt） ===============

# 传感器输出 txt（每次追加一行，最后一行是最新信号）
TXT_PATH = r"C:/Users/Saber/Desktop/test2/test2/mbn_rms_log.txt"

# 读数间隔：粗调 + 微调约 6 步 → 大约 5–6 s
READ_INTERVAL = 1.5

THRESHOLD = 12.0  # 粗/细调信号阈值

# 步长：从 3–5 mm 走到表面，一般 6 步左右
STEP_COARSE = -0.0005  # 粗调 0.8 mm 每步
STEP_FINE = -0.0001  # 细调 0.2 mm 每步

DELTA_LIMIT = 0.8
N_STABLE = 3  # 连续 N 次 Δsignal 很小 → 到达表面

FINAL_HOLD_SECONDS = 5.0  # 到达表面后保持位置多少秒采集最终信号


# =============== 工具函数 ===============

def convert_color_frame_to_bgr(color_frame):
    fmt = color_frame.get_format()
    w = color_frame.get_width()
    h = color_frame.get_height()
    data = color_frame.get_data()
    if fmt == ob.OBFormat.MJPG:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    elif fmt == ob.OBFormat.YUYV:
        return cv2.cvtColor(np.frombuffer(data, np.uint8).reshape(h, w, 2),
                            cv2.COLOR_YUV2BGR_YUY2)
    arr = np.frombuffer(data, np.uint8)
    if arr.size == w * h * 3:
        return arr.reshape(h, w, 3)
    return None


def load_model():
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    in_ch = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_ch, NUM_CLASSES, kernel_size=1)
    if getattr(model, "aux_classifier", None) is not None:
        aux_in = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_in, NUM_CLASSES, kernel_size=1)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    try:
        model.load_state_dict(state, strict=True)
        print("✅ state_dict strict=True 匹配")
    except Exception as e:
        print(f"⚠ strict=True 加载失败：{e}\n   尝试 strict=False")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"   missing_keys: {missing}\n   unexpected_keys: {unexpected}")

    model.to(DEVICE).eval()
    print("✅ DeepLabv3-ResNet50 Loaded (aux_loss=True)")
    return model


SEG_MODEL = load_model()


def pose_to_matrix(pose):
    x, y, z, rx, ry, rz = pose
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-6:
        R = np.eye(3)
    else:
        kx, ky, kz = rx / theta, ry / theta, rz / theta
        K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def get_cam_rotation(cam_ori_id: int):
    if cam_ori_id == 0:
        return np.eye(3)
    elif cam_ori_id == 1:
        rx = -math.pi / 2
        return np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    elif cam_ori_id == 2:
        rx = math.pi / 2
        return np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    elif cam_ori_id == 3:
        ry = math.pi / 2
        return np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    elif cam_ori_id == 4:
        ry = -math.pi / 2
        return np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    else:
        return np.eye(3)


def bbox_iou(b1, b2):
    x1, y1, w1, h1 = b1;
    x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h1)
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = w1 * h1;
    area2 = w2 * h2
    union = area1 + area2 - inter + 1e-6
    return inter / union


def merge_into_acc(acc_list, det, iou_thr=0.5, ema=0.5):
    for a in acc_list:
        if bbox_iou(a["bbox"], det["bbox"]) > iou_thr:
            ax, ay, aw, ah = a["bbox"];
            dx, dy, dw, dh = det["bbox"]
            a["bbox"] = (int((1 - ema) * ax + ema * dx),
                         int((1 - ema) * ay + ema * dy),
                         int((1 - ema) * aw + ema * dw),
                         int((1 - ema) * ah + ema * dh))
            a["conf"] = max(a["conf"], det["conf"])
            ax, ay, az = a["xyz_cam"];
            dx, dy, dz = det["xyz_cam"]
            a["xyz_cam"] = ((1 - ema) * ax + ema * dx,
                            (1 - ema) * ay + ema * dy,
                            (1 - ema) * az + ema * dz)
            return
    acc_list.append(det)


# =============== 2 秒检测 + 时序多数投票 ===============

def collect_points_sorted(pipeline, align_filter, duration_sec=2.0):
    """
    返回每类跨帧合并后的实例（按 conf 降序）：
    { 1:[{bbox,uv,conf,xyz_cam}], 2:[...], 3:[...] }
    """
    acc = {1: [], 2: [], 3: []}
    t0 = time.time()

    while time.time() - t0 < duration_sec:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue

        align_filter.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            continue

        cw, ch = color_frame.get_width(), color_frame.get_height()
        color = convert_color_frame_to_bgr(color_frame)
        if color is None:
            continue
        if color.shape[1] != cw or color.shape[0] != ch:
            color = cv2.resize(color, (cw, ch))

        dw, dh = depth_frame.get_width(), depth_frame.get_height()
        depth_u16 = np.frombuffer(depth_frame.get_data(), np.uint16).reshape(dh, dw)
        if (dw, dh) != (cw, ch):
            depth_u16 = cv2.resize(depth_u16.astype(np.float32), (cw, ch),
                                   interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        depth = depth_u16.astype(np.float32)

        intr = color_frame.get_stream_profile().as_video_stream_profile().get_intrinsic()
        scale = float(depth_frame.get_depth_scale())

        img = cv2.resize(color, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]
        img = torch.from_numpy(img).to(DEVICE)
        with torch.no_grad():
            out = SEG_MODEL(img)["out"][0]
            prob_small = torch.softmax(out, dim=0).cpu().numpy()
        prob = np.stack([cv2.resize(prob_small[c], (cw, ch), cv2.INTER_LINEAR)
                         for c in range(NUM_CLASSES)], axis=0)
        pred = np.argmax(prob, axis=0).astype(np.uint8)

        frame_dets = {1: [], 2: [], 3: []}
        for cls in (1, 2, 3):
            mask = (pred == cls).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < CLS_AREA[cls]:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                inst = np.zeros_like(mask)
                cv2.drawContours(inst, [cnt], -1, 1, -1)
                mean_conf = float((prob[cls] * inst).sum() / (inst.sum() + 1e-6))
                if mean_conf < CLS_THRESH[cls]:
                    continue

                u = x + w // 2;
                v = y + h // 2
                region = depth[max(v - 2, 0):min(v + 3, ch), max(u - 2, 0):min(u + 3, cw)]
                region = region[region > 0]
                if region.size == 0:
                    continue

                Z_mm = float(region.mean()) * scale
                X_mm = (u - intr.cx) * Z_mm / intr.fx
                Y_mm = (v - intr.cy) * Z_mm / intr.fy

                det = {
                    "bbox": (x, y, w, h),
                    "uv": (u, v),
                    "conf": mean_conf,
                    "xyz_cam": (X_mm / 1000.0, Y_mm / 1000.0, Z_mm / 1000.0)
                }

                cx, cy = u, v
                slots = TEMPORAL_BUF[cls]
                key = None
                min_d, best_k = 1e9, None
                for k, (px, py, hist) in slots.items():
                    d = (px - cx) ** 2 + (py - cy) ** 2
                    if d < min_d:
                        min_d, best_k = d, k
                if min_d < (max(w, h) * 0.8) ** 2:
                    key = best_k
                else:
                    if len(slots) >= MAX_TRACKS_PER_CLASS:
                        oldest = list(slots.keys())[0]
                        slots.pop(oldest, None)
                    key = (time.time(), cx, cy)
                if key not in slots:
                    slots[key] = (cx, cy, deque(maxlen=HISTORY))
                _, _, hist = slots[key]
                slots[key] = (cx, cy, hist)
                hist.append(1)

                if sum(hist) < VOTE_K and len(hist) >= VOTE_K:
                    continue

                frame_dets[cls].append(det)

        # 类间互斥
        def iou(b1, b2):
            x1, y1, w1, h1 = b1;
            x2, y2, w2, h2 = b2
            xa, ya = max(x1, x2), max(y1, y2)
            xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h1)
            inter = max(0, xb - xa) * max(0, yb - ya)
            u = w1 * h1 + w2 * h2 - inter + 1e-6
            return inter / u

        flat = []
        for c in (1, 2, 3):
            for d in frame_dets[c]:
                flat.append((CLASS_PRIORITY[c], d["conf"], c, d))
        flat.sort(key=lambda x: (x[0], x[1]), reverse=True)

        kept = []
        for _, _, c, d in flat:
            ok = True
            for kc, kd in kept:
                if iou(d["bbox"], kd["bbox"]) > 0.5 and CLASS_PRIORITY[kc] >= CLASS_PRIORITY[c]:
                    ok = False
                    break
            if ok:
                kept.append((c, d))
        frame_dets = {1: [], 2: [], 3: []}
        for c, d in kept:
            frame_dets[c].append(d)

        # 预览
        vis = color.copy()
        for cls in (1, 2, 3):
            for i, d in enumerate(sorted(frame_dets[cls], key=lambda t: t["conf"], reverse=True), 1):
                x, y, w, h = d["bbox"]
                cv2.rectangle(vis, (x, y), (x + w, y + h), class_colors[cls], 2)
                cv2.putText(vis, f"{class_names[cls]} #{i} {d['conf']:.2f}", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_colors[cls], 2)
        cv2.imshow("collecting (frame-only)", vis)
        if cv2.waitKey(1) == 27:
            break

        for cls in (1, 2, 3):
            for d in frame_dets[cls]:
                merge_into_acc(acc[cls], d, iou_thr=0.5, ema=0.5)

    for cls in (1, 2, 3):
        acc[cls].sort(key=lambda d: d["conf"], reverse=True)
    return acc


# =============== 读 txt 的最新信号 ===============

def read_latest_signal():
    try:
        with open(TXT_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [ln.strip() for ln in lines if ln.strip()]
        if not lines:
            print("❌ TXT 文件为空")
            return None
        last_line = lines[-1]
        parts = last_line.replace(',', ' ').split()
        sig = float(parts[-1])
        return sig
    except Exception as e:
        print(f"⚠ 读取 TXT 出错: {e}")
        return None


# =============== 单个目标的 Z 轴粗/细调 + 最终信号 ===============

def run_z_tuning(rtde_r, rtde_c):
    """
    在当前位置做 Z 轴粗/细调：
      - 每次调用时，先粗调一步（不管当前 signal 是多少）
      - 粗调阶段：只根据阈值 THRESHOLD 判断何时切到微调，不用 Δsignal 判稳
      - 微调阶段：才使用 |Δsignal| <= DELTA_LIMIT 连续 N_STABLE 次来判定到达表面
    判定到达表面后，保持 FINAL_HOLD_SECONDS 秒采集最终信号。
    返回最终信号值（float 或 None）。
    """
    stable_count = 0
    total_dz = 0.0
    reached_surface = False
    final_signal = None

    # 每次调用都从“未进入微调”开始
    in_fine_mode = False

    print("⚙️ 开始 Z 轴粗/细调 ...")
    print(f"   阈值 THRESHOLD = {THRESHOLD}")
    print(f"   粗调步长 STEP_COARSE = {STEP_COARSE} m")
    print(f"   细调步长 STEP_FINE   = {STEP_FINE} m")
    print(f"   微调阶段稳定判定: |Δsignal| <= {DELTA_LIMIT}, 连续 {N_STABLE} 次")

    # 先读一个初始信号作为基准，不做判定，只打印
    last_signal = None
    last_read_time = 0.0
    while last_signal is None:
        now = time.time()
        if now - last_read_time < READ_INTERVAL:
            time.sleep(0.01)
            continue
        last_read_time = now

        s0 = read_latest_signal()
        if s0 is not None:
            last_signal = s0
            final_signal = s0
            print(f"\n📥 初始信号 = {s0:.3f}（仅作为起点，不用于稳定判定）")

    while True:
        # 1) 根据当前阶段，决定这一步用粗调还是微调
        if in_fine_mode:
            dz = STEP_FINE
            mode = "微调"
            vel_z = VEL_Z_FINE  # 细调用更慢速度
        else:
            dz = STEP_COARSE
            mode = "粗调"
            vel_z = VEL_Z_COARSE  # 粗调用中等速度

        # 2) 先移动，再读新信号
        tcp = rtde_r.getActualTCPPose()
        target = tcp.copy()
        target[2] += dz

        total_dz += dz

        print(f"\n模式: {mode}")
        print(f"当前 Z = {tcp[2]:.4f} m → 新 Z = {target[2]:.4f} m")
        print(f"本次下移 = {dz * 1000:.2f} mm，总下移 = {total_dz * 1000:.2f} mm")

        # Z 调整使用专用速度
        rtde_c.moveL(target, vel_z, ACC_Z)

        # 给传感器一点时间稳定
        time.sleep(0.1)

        # 等待一段时间再读新信号
        new_signal = None
        last_read_time = 0.0
        while new_signal is None:
            now = time.time()
            if now - last_read_time < READ_INTERVAL:
                time.sleep(0.01)
                continue
            last_read_time = now

            s = read_latest_signal()
            if s is not None:
                new_signal = s

        final_signal = new_signal
        print(f"📥 当前信号 = {new_signal:.3f}")

        # 3) 计算 Δsignal
        delta = new_signal - last_signal
        print(f"Δsignal = {delta:.3f}")

        # 4) 阶段逻辑
        if in_fine_mode:
            # —— 微调阶段：用 Δsignal 判稳 ——
            if abs(delta) <= DELTA_LIMIT:
                stable_count += 1
                print(f"  → |Δ| <= {DELTA_LIMIT}, stable_count = {stable_count}")
            else:
                stable_count = 0
                print(f"  → |Δ| > {DELTA_LIMIT}, stable_count 重置为 0")

            if stable_count >= N_STABLE:
                print(f"🟢 微调阶段：连续 {N_STABLE} 次 Δsignal 很小，判定已到达表面，停止下探")
                reached_surface = True

        else:
            # —— 粗调阶段：不判稳，只判断是否要切换到微调 ——
            if new_signal >= THRESHOLD:
                in_fine_mode = True
                stable_count = 0
                print(f"🔁 粗调阶段信号已达到阈值 {THRESHOLD}，切换到【微调阶段】")
            else:
                print(f"（粗调阶段）当前 signal = {new_signal:.3f} < 阈值 {THRESHOLD}，继续粗调")

        last_signal = new_signal

        if reached_surface:
            break

    # 5) 到达表面后保持一段时间，读取最终信号
    print(f"⏱ 保持当前位置 {FINAL_HOLD_SECONDS} 秒，采集最终信号 ...")
    t_end = time.time() + FINAL_HOLD_SECONDS
    while time.time() < t_end:
        sig = read_latest_signal()
        if sig is not None:
            final_signal = sig
        time.sleep(READ_INTERVAL)

    print(f"🔚 Z 调整结束，最终信号 ≈ {final_signal}")
    return final_signal


# =============== 一次多样本测量流程 ===============

def measure_all_samples_once(rtde_r, rtde_c, pipeline, align_filter, home_pose):
    """
    一次完整流程：
      1) 相机识别 2 秒；
      2) 对每个类别 sample1/2/3 取若干最高置信度目标（最多 MAX_INST_PER_CLASS）；
      3) 按类别顺序 1→2→3，类内按置信度从高到低：
           - moveL 到该目标对应的传感器位置上方
           - run_z_tuning 做粗/细调 + 2 秒最终信号
           - moveL 回 home_pose
    """
    print("🔍 开始 2 秒识别，用于多样本顺序测量 ...")
    for c in (1, 2, 3):
        TEMPORAL_BUF[c].clear()
    instances = collect_points_sorted(pipeline, align_filter)
    print("📦 collected:", {k: len(v) for k, v in instances.items()})

    tcp_pose0 = rtde_r.getActualTCPPose()
    T_base_tool0 = pose_to_matrix(tcp_pose0)
    R_tc = get_cam_rotation(CAM_ORI_ID)
    t_tc = np.array([0.0, 0.0, CAM_Z_DOWN])
    T_tool_cam = np.eye(4)
    T_tool_cam[:3, :3] = R_tc
    T_tool_cam[:3, 3] = t_tc
    T_base_cam0 = T_base_tool0 @ T_tool_cam

    targets = []  # {"cls": cls, "conf": conf, "xyz_base": np.array([x,y,z])}
    for cls in (1, 2, 3):
        insts = instances[cls]
        if not insts:
            continue

        top_insts = insts[:MAX_INST_PER_CLASS]
        print(f"📌 {class_names[cls]} 检测到 {len(insts)} 个，取前 {len(top_insts)} 个用于测量")

        for inst in top_insts:
            Xc, Yc, Zc = inst["xyz_cam"]
            p_cam = np.array([Xc, Yc, Zc, 1.0])
            p_base = T_base_cam0 @ p_cam
            xyz_base = p_base[:3]
            targets.append({
                "cls": cls,
                "conf": inst["conf"],
                "xyz_base": xyz_base
            })

    if not targets:
        print("⚠ 没有任何目标被稳定检测到，本轮测量结束")
        return

    # 按类别顺序 1→2→3，类内置信度从高到低
    targets.sort(key=lambda t: (t["cls"], -t["conf"]))

    for idx, tgt in enumerate(targets, start=1):
        cls = tgt["cls"]
        conf = tgt["conf"]
        xyz_base = tgt["xyz_base"]

        print(f"\n================= 目标 {idx}/{len(targets)} =================")
        print(f"目标类别: {class_names[cls]}   conf = {conf:.3f}")

        current_tcp = rtde_r.getActualTCPPose()
        T_now = pose_to_matrix(current_tcp)

        tool_X = T_now[:3, 0];
        tool_X /= (np.linalg.norm(tool_X) + 1e-9)
        tool_y = T_now[:3, 1];
        tool_y /= (np.linalg.norm(tool_y) + 1e-9)
        tool_z = T_now[:3, 2];
        tool_z /= (np.linalg.norm(tool_z) + 1e-9)

        # 目标点 = 物体中心 + 工具Y方向偏移 55mm + 工具Z方向安全下探
        target_xyz = xyz_base + tool_X * SENSOR_OFFSET_X + tool_y * SENSOR_OFFSET_y + tool_z * SAFE_OFFSET

        target_pose = [
            float(target_xyz[0]),
            float(target_xyz[1]),
            float(target_xyz[2]),
            current_tcp[3], current_tcp[4], current_tcp[5]
        ]

        print("➡ moveL 到该样本对应的传感器位置上方：")
        print("   target_pose =", target_pose)
        rtde_c.moveL(target_pose, VEL, ACC)
        print("✅ 已到达该样本上方，开始 Z 轴粗/细调 ...")

        final_signal = run_z_tuning(rtde_r, rtde_c)
        print(f"✅ {class_names[cls]} 此目标测量完成，最终信号 ≈ {final_signal}")

        print("↩ 回到 home_pose ...")
        rtde_c.moveL(home_pose, VEL, ACC)
        print("✅ 已回到 home_pose")

    print("\n🎉 本轮多样本顺序测量全部完成！")


# =============== 主流程 ===============

def main():
    # 机器人
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    rtde_c = RTDEControlInterface(ROBOT_IP)
    print("✅ robot connected")
    home_pose = rtde_r.getActualTCPPose()

    # 相机
    pipeline = ob.Pipeline()
    config = ob.Config()

    dp = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR) \
        .get_video_stream_profile(640, 400, ob.OBFormat.Y16, 30)
    cp = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR) \
        .get_video_stream_profile(640, 480, ob.OBFormat.MJPG, 30)
    config.enable_stream(dp);
    config.enable_stream(cp)

    pipeline.enable_frame_sync()
    pipeline.start(config)
    print("✅ camera started")

    align_filter = ob.AlignFilter(ob.OBStreamType.COLOR_STREAM)

    print("操作提示：")
    print("  s - 一次完整『多样本顺序测量』（识别2秒 → 依次移动+粗/细调+测量 → 回home）")
    print("  r - 只回初始点，不做识别和测量")
    print("  q - 退出程序")

    try:
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch().decode("utf-8", errors="ignore").lower()

                if ch == 's':
                    measure_all_samples_once(rtde_r, rtde_c, pipeline, align_filter, home_pose)

                elif ch == 'r':
                    print("↩ 回初始点 ...")
                    rtde_c.moveL(home_pose, VEL, ACC)
                    print("✅ 已回初始点")

                elif ch == 'q':
                    print("👋 退出程序")
                    break

            time.sleep(0.01)

    finally:
        # 关相机 / GUI / 机器人脚本
        try:
            pipeline.stop()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass
        try:
            rtde_c.stopScript()
        except:
            pass

        # 🔹 退出程序时删除 txt 文件
        try:
            if os.path.exists(TXT_PATH):
                os.remove(TXT_PATH)
                print(f"🗑 已删除信号日志文件: {TXT_PATH}")
            else:
                print(f"ℹ 未找到 TXT 文件，无需删除: {TXT_PATH}")
        except Exception as e:
            print(f"⚠ 删除 TXT 文件失败: {e}")

        print("stopped cleanly")


if __name__ == "__main__":
    main()
