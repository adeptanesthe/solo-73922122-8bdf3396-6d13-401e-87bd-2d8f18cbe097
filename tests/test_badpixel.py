"""坏点候选定位测试：算法判据（持续孤立 / 邻域变化 / 遮挡 / 连片）、
前置校验（帧数、尺寸、解码失败）与 API 端到端（含迟到结果版本）。"""

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.badpixel import analyze_badpixels
from app.example import BP_DARK, BP_HOT, BP_OCC, BP_PATCH, BP_STATIC
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------- 工具

def make_burst(n=5, w=96, h=80, seed=1):
    """生成邻域随帧变化的连拍（移动渐变 + 轻噪声），返回 uint8 数组列表。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    frames = []
    for f in range(n):
        base = 110 + 40 * np.sin((xx + 9 * f) / 23.0) + 30 * np.cos((yy - 7 * f) / 29.0)
        base += rng.normal(0.0, 2.0, (h, w))
        frames.append(np.clip(base, 0, 255).astype(np.uint8))
    return frames


def to_frames(arrays):
    out = []
    for arr in arrays:
        buf = io.BytesIO()
        Image.fromarray(arr, mode="L").save(buf, format="PNG")
        out.append(("image/png", buf.getvalue()))
    return out


def analyze_arrays(arrays, dets=None):
    return analyze_badpixels(to_frames(arrays), dets or [])


def wait_badjob(job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/badjobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError("坏点分析任务超时未完成")


def upload_session(arrays, dets=None):
    files = [("images", (f"f{i}.png", data, "image/png"))
             for i, (_c, data) in enumerate(to_frames(arrays))]
    files.append(("detections", ("det.json", json.dumps({"detections": dets or []}),
                                "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- 算法：候选判定

def test_hot_and_dark_pixels_found():
    frames = make_burst()
    for arr in frames:
        arr[40, 50] = 255   # 持续偏亮
        arr[55, 60] = 0     # 持续偏暗
    res = analyze_arrays(frames)
    assert res["status"] == "ok", res["message"]
    got = {(c["x"], c["y"]): c["kind"] for c in res["candidates"]}
    assert got[(50, 40)] == "bright"
    assert got[(60, 55)] == "dark"
    for c in res["candidates"]:
        assert c["strength"] > 0
    assert res["width"] == 96 and res["height"] == 80 and res["n_frames"] == 5


def test_clean_burst_reports_nothing():
    res = analyze_arrays(make_burst())
    assert res["status"] == "clean"
    assert res["candidates"] == []
    assert all(v["count"] == 0 for v in res["suppressed"].values())


def test_static_background_is_insufficient_evidence():
    base = make_burst(n=1)[0]
    frames = [base.copy() for _ in range(5)]  # 完全静止的连拍
    for arr in frames:
        arr[40, 50] = 255
    res = analyze_arrays(frames)
    assert res["status"] == "insufficient"
    assert res["candidates"] == []
    assert res["suppressed"]["static"]["count"] >= 1
    assert [50, 40] in res["suppressed"]["static"]["samples"]
    assert "证据不足" in res["message"]


def test_connected_patch_is_not_isolated():
    frames = make_burst()
    for arr in frames:
        arr[30:32, 40:42] = 250  # 2×2 连片亮斑
    res = analyze_arrays(frames)
    assert res["candidates"] == []
    assert res["suppressed"]["patch"]["count"] >= 4
    assert "证据不足" in res["message"]


def test_occluded_pixel_is_suspicious_not_candidate():
    frames = make_burst()
    for arr in frames:
        arr[40, 50] = 255
    # 第 1、3 帧用检测框遮住该坐标
    dets = [
        {"id": "o1", "frame": 1, "rect": [46, 36, 10, 10], "vector": [1, 0]},
        {"id": "o2", "frame": 3, "rect": [46, 36, 10, 10], "vector": [1, 0]},
    ]
    res = analyze_arrays(frames, dets)
    assert all((c["x"], c["y"]) != (50, 40) for c in res["candidates"])
    assert res["suppressed"]["occluded"]["count"] >= 1
    assert [50, 40] in res["suppressed"]["occluded"]["samples"]


def test_fully_occluded_pixel_silently_excluded():
    frames = make_burst()
    for arr in frames:
        arr[40, 50] = 255
    dets = [{"id": f"o{f}", "frame": f, "rect": [46, 36, 10, 10], "vector": [1, 0]}
            for f in range(5)]
    res = analyze_arrays(frames, dets)
    assert all((c["x"], c["y"]) != (50, 40) for c in res["candidates"])
    assert res["suppressed"]["occluded"]["count"] == 0  # 无从判断，不计数


def test_sign_must_be_consistent():
    frames = make_burst()
    for f, arr in enumerate(frames):
        arr[40, 50] = 255 if f % 2 == 0 else 0  # 方向忽亮忽暗 → 非坏点表现
    res = analyze_arrays(frames)
    assert all((c["x"], c["y"]) != (50, 40) for c in res["candidates"])


# ---------------------------------------------------------------- 前置校验

def test_frame_count_limits():
    assert analyze_arrays(make_burst(n=2))["status"] == "invalid"
    assert analyze_arrays(make_burst(n=7))["status"] == "invalid"
    res = analyze_arrays(make_burst(n=2))
    assert "3~6" in res["message"]


def test_size_mismatch_reported():
    frames = make_burst(n=4)
    frames[2] = np.zeros((60, 96), dtype=np.uint8)  # 第 2 帧尺寸不同
    res = analyze_arrays(frames)
    assert res["status"] == "invalid"
    assert "尺寸不一致" in res["message"]


def test_oversize_frame_reported():
    frames = make_burst(n=3, w=96, h=80)
    frames[1] = np.zeros((80, 600), dtype=np.uint8)  # 宽超过 512
    res = analyze_arrays(frames)
    assert res["status"] == "invalid"
    assert "512" in res["message"] and "第 1 帧" in res["message"]


def test_undecodable_frame_explained():
    frames = to_frames(make_burst(n=3))
    frames[1] = ("image/png", b"this is not a real png file")
    res = analyze_badpixels(frames, [])
    assert res["status"] == "invalid"
    assert "第 1 帧" in res["message"]
    assert "无法识别" in res["message"] or "解码" in res["message"]


def test_truncated_image_explained():
    frames = to_frames(make_burst(n=3))
    frames[0] = ("image/png", frames[0][1][: len(frames[0][1]) // 3])  # 截断
    res = analyze_badpixels(frames, [])
    assert res["status"] == "invalid"
    assert "第 0 帧" in res["message"]


# ---------------------------------------------------------------- 内置坏点示例

def test_badpixel_example_hits_all_paths():
    r = client.post("/api/example_badpixels")
    assert r.status_code == 200
    p = r.json()
    assert p["n_frames"] == 5
    job = client.post(f"/api/sessions/{p['session_id']}/badpixels").json()
    done = wait_badjob(job["job_id"])
    assert done["status"] == "done"
    res = done["result"]
    assert res["status"] == "ok", res["message"]
    got = {(c["x"], c["y"]): c["kind"] for c in res["candidates"]}
    assert got[BP_HOT] == "bright"
    assert got[BP_DARK] == "dark"
    # 三类证据不足各至少一处，且坐标不混入候选
    assert res["suppressed"]["static"]["count"] >= 1
    assert res["suppressed"]["occluded"]["count"] >= 1
    assert res["suppressed"]["patch"]["count"] >= 1
    for pt in (BP_STATIC, BP_OCC, BP_PATCH):
        assert pt not in got


# ---------------------------------------------------------------- API 端到端

def test_api_end_to_end_and_version_guard():
    frames = make_burst()
    for arr in frames:
        arr[40, 50] = 255
    p = upload_session(frames)
    sid = p["session_id"]
    assert p["limits"]["badpixels"] == {"min_frames": 3, "max_frames": 6, "max_dim": 512}

    job = client.post(f"/api/sessions/{sid}/badpixels").json()
    assert job["version"] == p["version"]
    done = wait_badjob(job["job_id"])
    assert done["status"] == "done"
    res = done["result"]
    assert res["status"] == "ok"
    assert any(c["x"] == 50 and c["y"] == 40 for c in res["candidates"])

    # 分析期间修改输入 → 版本 +1，任务仍携带旧版本（前端据此丢弃迟到结果）
    r = client.put(f"/api/sessions/{sid}/params",
                   json={"w_dist": 2.0, "w_app": 5.0, "penalty": 10.0})
    assert r.json()["version"] == p["version"] + 1
    assert done["version"] != r.json()["version"]


def test_badpixel_analysis_does_not_change_session():
    frames = make_burst()
    for arr in frames:
        arr[40, 50] = 255
    p = upload_session(frames)
    sid = p["session_id"]
    before = client.get(f"/api/sessions/{sid}").json()
    done = wait_badjob(client.post(f"/api/sessions/{sid}/badpixels").json()["job_id"])
    assert done["status"] == "done"
    after = client.get(f"/api/sessions/{sid}").json()
    # 会话数据（版本、检测框、判断、参数）保持原样
    assert after["version"] == before["version"]
    assert after["detections"] == before["detections"]
    assert after["judgments"] == before["judgments"]
    assert after["params"] == before["params"]
    # 原图字节不变
    for i in range(5):
        assert client.get(f"/api/sessions/{sid}/frames/{i}").content == to_frames(frames)[i][1]


def test_badjob_unknown_404():
    assert client.get("/api/badjobs/nope").status_code == 404
    assert client.post("/api/sessions/nope/badpixels").status_code == 404


def test_badpixel_with_tracking_example_insufficient_or_clean():
    """交叉遮挡示例（6 帧）也可直接运行：不得误报候选（背景静止/噪声不持续）。"""
    p = client.post("/api/example").json()
    done = wait_badjob(client.post(f"/api/sessions/{p['session_id']}/badpixels").json()["job_id"])
    assert done["status"] == "done"
    res = done["result"]
    assert res["status"] in ("insufficient", "clean")
    assert res["candidates"] == []
