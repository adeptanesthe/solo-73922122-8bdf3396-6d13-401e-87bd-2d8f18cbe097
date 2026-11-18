"""连拍图像坏点（传感器缺陷像素）候选定位。

判定原则：宁可漏报，绝不猜测。一个坐标只有同时满足下列全部条件才列为候选：

1. **持续**：在每一帧都相对局部邻域（5×5 去掉中心）中位数显著偏亮或偏暗，
   且偏离方向在各帧一致；
2. **孤立**：每一帧中该坐标的 8 邻域内都没有其他异常像素（异常不连成片）；
3. **邻域可变**：该坐标的邻域亮度在连拍过程中发生可辨变化
   （静止背景下无法区分“场景固有亮点”与“传感器坏点”）；
4. **无遮挡**：该坐标在每一帧都不落在任何检测框内。

对“接近但证据不足”的坐标按原因分类计数（背景静止 / 检测框遮挡 /
异常连成片），只提示、不给出坐标猜测。
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, UnidentifiedImageError

MIN_FRAMES = 3          # 连拍帧数下限
MAX_FRAMES = 6          # 连拍帧数上限
MAX_DIM = 512           # 单帧宽/高上限

RING = 2                    # 邻域窗口半径（5×5，去掉中心像素）
DEV_MIN = 12 / 255          # 偏差绝对下限（灰度级，0~255 归一化到 0~1）
DEV_SIGMA = 6.0             # 偏差相对帧噪声（MAD 估计）的倍数
CHANGE_MIN = 5 / 255        # 邻域亮度跨帧“可辨变化”下限
MAX_CANDIDATES = 200        # 返回的候选坐标数上限（按偏差强度截断）
MAX_SAMPLES = 5             # 每类“证据不足”原因附带的示例坐标数

LIMITS = {"min_frames": MIN_FRAMES, "max_frames": MAX_FRAMES, "max_dim": MAX_DIM}

_REASON_KEYS = ("static", "occluded", "patch")


# ---------------------------------------------------------------- 解码与前置校验

def _open_frame(data: bytes):
    """解码一帧为 [0,1] 灰度数组。失败抛出带中文原因的 ValueError。"""
    try:
        img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise ValueError("无法识别的图像格式（文件损坏或并非图像）") from None
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"图像头读取失败（{exc}）") from None
    w, h = img.size
    if w > MAX_DIM or h > MAX_DIM:
        raise ValueError(f"尺寸 {w}×{h} 超过 {MAX_DIM}×{MAX_DIM} 上限")
    try:
        gray = img.convert("L")
        gray.load()  # 强制真正解码像素，截断/损坏在此暴露
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"像素数据解码失败，文件可能不完整（{exc}）") from None
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    return arr, (w, h)


def _invalid(message: str, problems: list[str]) -> dict:
    return {
        "status": "invalid",
        "message": message,
        "problems": problems,
        "width": None,
        "height": None,
        "n_frames": 0,
        "candidates": [],
        "suppressed": {k: {"count": 0, "samples": []} for k in _REASON_KEYS},
    }


# ---------------------------------------------------------------- 像素级统计

def _ring_stats(img: np.ndarray):
    """每像素的 5×5 邻域（去掉中心）中位数与均值。"""
    k = 2 * RING + 1
    pad = np.pad(img, RING, mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(pad, (k, k))
    h, w = img.shape
    flat = win.reshape(h, w, k * k)
    ring = np.delete(flat, k * k // 2, axis=2)  # 去掉中心像素
    return np.median(ring, axis=2), ring.mean(axis=2)


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    """每像素 8 邻域内 True 的个数（边缘回绕不影响内部 RING 像素）。"""
    m = mask.astype(np.int16)
    cnt = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            cnt += np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return cnt


def _boxes_masks(detections, n: int, h: int, w: int) -> np.ndarray:
    """每帧的检测框覆盖掩码，形状 (n, h, w)。"""
    masks = np.zeros((n, h, w), dtype=bool)
    for d in detections:
        f = int(d["frame"])
        if not (0 <= f < n):
            continue
        x, y, bw, bh = (float(v) for v in d["rect"])
        x0 = max(0, int(math.floor(x)))
        y0 = max(0, int(math.floor(y)))
        x1 = min(w, int(math.ceil(x + bw)))
        y1 = min(h, int(math.ceil(y + bh)))
        if x1 > x0 and y1 > y0:
            masks[f, y0:y1, x0:x1] = True
    return masks


# ---------------------------------------------------------------- 主流程

def analyze_badpixels(frames, detections) -> dict:
    """frames: [(content_type, bytes)]；detections: 规范化检测框列表。

    返回 JSON 可序列化的结果字典，status ∈ ok / insufficient / clean / invalid。
    """
    n = len(frames)
    if not (MIN_FRAMES <= n <= MAX_FRAMES):
        return _invalid(
            f"坏点定位需要 {MIN_FRAMES}~{MAX_FRAMES} 张同尺寸连拍图像，"
            f"当前会话为 {n} 张，无法分析。",
            [f"帧数 {n} 不在 {MIN_FRAMES}~{MAX_FRAMES} 范围内"],
        )

    imgs, sizes, problems = [], [], []
    for f, (_ctype, data) in enumerate(frames):
        try:
            arr, size = _open_frame(data)
        except ValueError as exc:
            problems.append(f"第 {f} 帧：{exc}")
            continue
        imgs.append(arr)
        sizes.append(size)
    if problems:
        return _invalid(
            "有图像无法用于分析：" + "；".join(problems),
            problems,
        )
    if len(set(sizes)) != 1:
        detail = "、".join(f"第 {f} 帧 {w}×{h}" for f, (w, h) in enumerate(sizes))
        return _invalid(
            f"各帧尺寸不一致（{detail}），无法逐像素比对。",
            [f"尺寸不一致：{detail}"],
        )

    h, w = imgs[0].shape
    occ = _boxes_masks(detections, n, h, w)

    devs = np.empty((n, h, w), dtype=np.float32)
    ring_means = np.empty((n, h, w), dtype=np.float32)
    for f, img in enumerate(imgs):
        med, rmean = _ring_stats(img)
        devs[f] = img - med
        ring_means[f] = rmean

    # 边缘 RING 像素的邻域被反射填充、邻域计数会回绕，一律不参与判定
    interior = np.zeros((h, w), dtype=bool)
    interior[RING:h - RING, RING:w - RING] = True

    pos = np.zeros((n, h, w), dtype=bool)   # 显著偏亮
    neg = np.zeros((n, h, w), dtype=bool)   # 显著偏暗
    isolated = np.zeros((n, h, w), dtype=bool)  # 异常且 8 邻域无其他异常
    for f in range(n):
        dev = devs[f]
        sigma = 1.4826 * float(np.median(np.abs(dev[interior])))
        thr = max(DEV_MIN, DEV_SIGMA * sigma)
        p = (dev > thr) & interior
        g = (dev < -thr) & interior
        deviant = p | g
        pos[f] = p
        neg[f] = g
        isolated[f] = deviant & (_neighbor_count(deviant) == 0)

    occ_any = occ.any(axis=0)
    occ_all = occ.all(axis=0)
    visible = ~occ_any  # 每一帧都未被检测框覆盖

    pos_all = pos.all(axis=0)
    neg_all = neg.all(axis=0)
    iso_all = isolated.all(axis=0)
    changed = (ring_means.max(axis=0) - ring_means.min(axis=0)) > CHANGE_MIN

    persist = (pos_all | neg_all) & visible
    cand_mask = persist & iso_all & changed
    patch_mask = persist & ~iso_all            # 持续异常但连成片
    static_mask = persist & iso_all & ~changed  # 持续孤立但邻域静止

    # 部分帧被遮挡的坐标：仅在可见帧上要求同号持续异常 → 遮挡存疑
    partly = occ_any & ~occ_all & interior
    pos_vis = (pos | occ).all(axis=0)
    neg_vis = (neg | occ).all(axis=0)
    occl_mask = partly & (pos_vis | neg_vis)
    # 全部被遮挡的坐标无从判断，直接排除（不计数、不报告）

    ys, xs = np.nonzero(cand_mask)
    cands = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        cands.append({
            "x": x,
            "y": y,
            "kind": "bright" if pos_all[y, x] else "dark",
            "strength": round(float(np.median(np.abs(devs[:, y, x]))) * 255.0, 1),
        })
    cands.sort(key=lambda c: -c["strength"])
    truncated = len(cands) > MAX_CANDIDATES
    cands = cands[:MAX_CANDIDATES]

    suppressed = {}
    for key, mask in (("static", static_mask), ("occluded", occl_mask), ("patch", patch_mask)):
        sy, sx = np.nonzero(mask)
        samples = [[int(x), int(y)] for y, x in list(zip(sy.tolist(), sx.tolist()))[:MAX_SAMPLES]]
        suppressed[key] = {"count": int(mask.sum()), "samples": samples}

    labels = {"static": "背景静止", "occluded": "检测框遮挡", "patch": "异常连成片"}
    parts = [f"{labels[k]} {suppressed[k]['count']} 处" for k in _REASON_KEYS
             if suppressed[k]["count"] > 0]
    if cands:
        status = "ok"
        message = (f"发现 {len(cands)} 个坏点候选坐标，已在画面相同坐标上叠加标记（各帧一致）。"
                   + ("（按偏差强度仅列出前 " + str(MAX_CANDIDATES) + " 个）" if truncated else ""))
        if parts:
            message += "另有证据不足未列入的坐标：" + "、".join(parts) + "。"
    elif parts:
        status = "insufficient"
        message = ("证据不足，未给出坏点候选：" + "、".join(parts) +
                   "。建议让背景内容随帧变化、避开检测框遮挡或检查连片异常区域后重新连拍再查。")
    else:
        status = "clean"
        message = "未发现持续孤立的异常像素，未给出坏点候选。"

    return {
        "status": status,
        "message": message,
        "problems": [],
        "width": w,
        "height": h,
        "n_frames": n,
        "candidates": cands,
        "suppressed": suppressed,
    }
