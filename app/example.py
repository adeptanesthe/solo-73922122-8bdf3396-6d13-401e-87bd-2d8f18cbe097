"""内置“交叉遮挡”示例会话数据。

场景：6 帧 320x240 画面。两只外观不同的动物（A 偏红、B 偏蓝）
沿对角线相向而行，在第 3 帧（0 起）于画面中央交汇：
- B 在第 3 帧被 A 遮挡，缺失检测 → 需要跨 2 帧连接；
- A 在第 3 帧的检测框外观向量被 B 的颜色污染（余弦相似度下降），
  模拟遮挡时的表观退化；
- 两动物在第 2/4 帧空间位置非常接近，朴素逐帧贪心关联容易发生
  串号（ID switch），而全局最优解能给出正确的 2 条轨迹。

外观向量为 4 维（近似颜色直方图），均非零。
"""

from __future__ import annotations

import io
import math
import random

import numpy as np
from PIL import Image, ImageDraw

WIDTH, HEIGHT = 320, 240
N_FRAMES = 6

# 每只动物的中心轨迹（帧号 -> (cx, cy)），两条路径在画面中央呈 X 形交叉
A_CENTERS = [(40, 60), (82, 86), (124, 112), (166, 138), (208, 164), (250, 190)]
B_CENTERS = [(290, 60), (248, 86), (206, 112), (164, 138), (122, 164), (80, 190)]
B_OCCLUDED_FRAME = 3  # B 在该帧被 A 遮挡，无检测框

A_VEC = (0.90, 0.10, 0.05, 0.02)
B_VEC = (0.05, 0.10, 0.90, 0.02)
# A 在遮挡帧的表观被 B 污染
A_VEC_OCCLUDED = (0.60, 0.10, 0.35, 0.02)

BOX = 28  # 检测框边长（像素）


def _jittered(vec, rng, sigma=0.015):
    out = [max(0.01, v + rng.gauss(0, sigma)) for v in vec]
    return [round(v, 4) for v in out]


def build_example_detections():
    """返回示例检测框列表（dict）。"""
    rng = random.Random(20240611)
    dets = []
    for f, (cx, cy) in enumerate(A_CENTERS):
        vec = A_VEC_OCCLUDED if f == B_OCCLUDED_FRAME else A_VEC
        dets.append(
            {
                "id": f"A{f}",
                "frame": f,
                "rect": [cx - BOX // 2, cy - BOX // 2, BOX, BOX],
                "vector": _jittered(vec, rng),
            }
        )
    for f, (cx, cy) in enumerate(B_CENTERS):
        if f == B_OCCLUDED_FRAME:
            continue  # 被遮挡，无检测
        dets.append(
            {
                "id": f"B{f}",
                "frame": f,
                "rect": [cx - BOX // 2, cy - BOX // 2, BOX, BOX],
                "vector": _jittered(B_VEC, rng),
            }
        )
    dets.sort(key=lambda d: (d["frame"], d["id"]))
    return dets


def _draw_animal(draw, cx, cy, rgb, r=13):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb, outline=(20, 20, 20))


def build_example_frames():
    """返回 [(content_type, png_bytes), ...] 共 6 帧。"""
    rng = random.Random(7)
    frames = []
    for f in range(N_FRAMES):
        img = Image.new("RGB", (WIDTH, HEIGHT), (34, 52, 34))
        draw = ImageDraw.Draw(img)
        # 草地噪点
        for _ in range(450):
            x = rng.randrange(WIDTH)
            y = rng.randrange(HEIGHT)
            g = rng.randrange(45, 90)
            draw.point((x, y), fill=(g // 2, g, g // 3))
        # 先画 B（被遮挡时在下层），再画 A
        bx, by = B_CENTERS[f]
        if f == B_OCCLUDED_FRAME:
            # B 几乎完全被 A 挡住，只露出边缘一小角
            _draw_animal(draw, bx, by, (60, 60, 140), r=13)
        else:
            _draw_animal(draw, bx, by, (40, 40, 220), r=13)
        ax, ay = A_CENTERS[f]
        _draw_animal(draw, ax, ay, (220, 40, 40), r=14)
        draw.text((6, 4), f"frame {f}", fill=(240, 240, 240))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frames.append(("image/png", buf.getvalue()))
    return frames


EXAMPLE_DESCRIPTION = (
    "交叉遮挡示例：A（红）与 B（蓝）相向穿行，第 3 帧 B 被 A 遮挡缺失检测、"
    "A 的外观被污染。正确结果应为 2 条轨迹（2 只动物）；"
    "若只按最近邻逐帧贪心，容易在交汇点串号或把 B 断成两条而重复计数。"
)


# ---------------------------------------------------------------- 坏点定位示例

BP_W, BP_H, BP_N = 220, 160, 5
BP_HOT = (70, 50)      # 持续偏亮 → 应列为候选
BP_DARK = (160, 110)   # 持续偏暗 → 应列为候选
BP_PATCH = (120, 30)   # 2×2 连片亮斑 → 证据不足（异常连成片）
BP_OCC = (30, 130)     # 部分帧被检测框遮挡的亮点 → 证据不足（检测框遮挡）
BP_STATIC = (9, 9)     # 静止角落里的亮点 → 证据不足（背景静止）
BP_OCC_RECT = [24, 124, 14, 14]  # 第 1、3 帧遮挡 BP_OCC 的检测框


def build_badpixel_example():
    """5 帧 220×160 灰度连拍：背景随帧移动，埋入各类典型坐标。

    返回 (frames, detections)。用于在网页上直接验证坏点候选定位的
    各条判定路径（候选 / 背景静止 / 检测框遮挡 / 异常连成片）。
    """
    rng = np.random.default_rng(20261118)
    yy, xx = np.mgrid[0:BP_H, 0:BP_W]
    frames = []
    static_corner = None
    for f in range(BP_N):
        base = 110 + 40 * np.sin((xx + 9 * f) / 23.0) + 30 * np.cos((yy - 7 * f) / 29.0)
        base += rng.normal(0.0, 2.0, (BP_H, BP_W))
        if f == 0:
            static_corner = base[:18, :18].copy()
        base[:18, :18] = static_corner  # 左上角各帧完全一致（静止背景）
        arr = np.clip(base, 0, 255).astype(np.float32)
        arr[BP_STATIC[1], BP_STATIC[0]] = 240.0
        arr[BP_HOT[1], BP_HOT[0]] = 250.0
        arr[BP_DARK[1], BP_DARK[0]] = 4.0
        arr[BP_PATCH[1]:BP_PATCH[1] + 2, BP_PATCH[0]:BP_PATCH[0] + 2] = 246.0
        arr[BP_OCC[1], BP_OCC[0]] = 250.0
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
        if f in (1, 3):
            # 遮挡物经过 BP_OCC（对应检测框区域）
            x, y, w, h = BP_OCC_RECT
            draw = ImageDraw.Draw(img)
            draw.ellipse([x, y, x + w, y + h], fill=150)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frames.append(("image/png", buf.getvalue()))
    dets = [
        {"id": "occ1", "frame": 1, "rect": list(BP_OCC_RECT), "vector": [1.0, 0.0]},
        {"id": "occ2", "frame": 3, "rect": list(BP_OCC_RECT), "vector": [1.0, 0.0]},
    ]
    return frames, dets


BADPIXEL_EXAMPLE_DESCRIPTION = (
    "坏点定位示例：5 帧连拍，背景随帧变化。埋入 (70,50) 持续偏亮与 (160,110) "
    "持续偏暗两枚模拟坏点；另有静止角落亮点、2×2 连片亮斑、被检测框部分遮挡的"
    "亮点各一处（这三类应判为“证据不足”）。请点“5. 坏点候选定位”验证。"
)
