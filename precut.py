import os
import sys
import math
import json
import shutil
import subprocess as sp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm

# PySceneDetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# -----------------------------
# CONFIG (tune to your footage)
# -----------------------------
INPUT_ROOT = Path("videos")              # root containing subfolders/videos
OUTPUT_ROOT = Path("out")                # where we store intermediates/exports
FINAL_VIDEO = OUTPUT_ROOT / "output.mp4" # final concatenated result

# Scene detection (PySceneDetect ContentDetector threshold; 27–35 works for interiors)
SCENE_THRESHOLD = 27.0
MIN_SCENE_LEN_S = 1.0  # ignore scenes shorter than this

# Frame sampling within a scene for metrics
SAMPLE_FRAMES_FOR_SHAKE = 10        # evenly spaced frames to estimate motion
DOWNSCALE_WIDTH = 960               # speed-up for metrics (keep aspect)

# Hard drop thresholds
BLUR_VAR_LAPLACE_MIN = 120.0        # lower -> blurrier → drop
MEAN_BRIGHT_DIM_MAX = 60            # too dim if mean < this
MEAN_BRIGHT_BRIGHT_MIN = 200        # too bright if mean > this
TILT_DEGREES_MAX = 3.0              # crooked if > this (architecture)
SHAKE_STD_MAX = 1.8                 # optical-flow displacement std threshold

# Soft penalties (compose into score)
COLOR_CAST_ABS_MEAN_MAX = 6.0       # |a-128| or |b-128| > -> penalty
EDGE_RATIO_MAX = 0.18                # clutter proxy
ENTROPY_MAX = 5.3
BLOCKINESS_RATIO_MAX = 0.22
MIXED_LIGHT_STD_MAX = 12.0

# Keep if not hard-dropped and score >= this
KEEP_SCORE_MIN = 0.30

# Export settings
CRF = 18
PRESET = "medium"
AUDIO = True

# -----------------------------
# UTILITIES
# -----------------------------
def run(cmd: List[str]) -> None:
    sp.run(cmd, check=True)

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def resize_keep_ar(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv2.resize(img, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

def entropy_gray(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    return float(-(hist * np.log(hist + 1e-9)).sum())

def get_video_fps_nframes(path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n

def read_frame_at_time(path: str, t_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = max(0, int(t_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame at {t_s}s from {path}")
    return frame

def sample_times(start_s: float, end_s: float, k: int) -> List[float]:
    if k <= 1:
        return [ (start_s + end_s) / 2.0 ]
    span = max(1e-3, end_s - start_s)
    return [ start_s + span * i / (k - 1) for i in range(k) ]

def hough_vertical_tilt(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    if lines is None:
        return 0.0
    angs = []
    for rho_theta in lines[:200]:
        for _, theta in rho_theta:
            angs.append(np.degrees(theta))
    if not angs:
        return 0.0
    med = np.median(angs)
    # distance to nearest upright (0 or 90)
    return float(min(abs(med - 90.0), abs(med - 0.0)))

def exposure_flags(gray: np.ndarray) -> Tuple[bool, bool]:
    mean = gray.mean()
    p1, p99 = np.percentile(gray, [1, 99])
    too_dim    = (mean < MEAN_BRIGHT_DIM_MAX) or (p99 < 140)
    too_bright = (mean > MEAN_BRIGHT_BRIGHT_MIN) or (p1 > 40)
    return bool(too_dim), bool(too_bright)

def blur_flag(gray: np.ndarray) -> bool:
    var_lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return bool(var_lap < BLUR_VAR_LAPLACE_MIN)

def color_cast_flag(img_bgr: np.ndarray) -> bool:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a_mean = lab[:,:,1].mean() - 128
    b_mean = lab[:,:,2].mean() - 128
    return bool(abs(a_mean) > COLOR_CAST_ABS_MEAN_MAX or abs(b_mean) > COLOR_CAST_ABS_MEAN_MAX)

def clutter_flags(gray: np.ndarray) -> Tuple[bool, float, float]:
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = (edges > 0).mean()
    ent = entropy_gray(gray)
    cluttered = edge_ratio > EDGE_RATIO_MAX and ent > ENTROPY_MAX
    return bool(cluttered), float(edge_ratio), float(ent)

def blockiness_flag(gray: np.ndarray) -> bool:
    # crude: variance on 8px grid vs overall variance
    grid = gray[:, ::8].var() + gray[::8, :].var()
    nongrid = gray.var() + 1e-6
    return bool((grid / nongrid) > BLOCKINESS_RATIO_MAX)

def mixed_light_flag(img_bgr: np.ndarray) -> bool:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:,:,1].astype(np.float32) - 128
    b = lab[:,:,2].astype(np.float32) - 128
    return bool(((a.std() + b.std()) / 2.0) > MIXED_LIGHT_STD_MAX)

def shake_flag(video_path: str, start_s: float, end_s: float) -> bool:
    # Optical-flow based jitter over K frames
    times = sample_times(start_s, end_s, min(SAMPLE_FRAMES_FOR_SHAKE, 20))
    if len(times) < 2:
        return False
    frames = []
    for t in times:
        frm = read_frame_at_time(video_path, t)
        frames.append(resize_keep_ar(frm, DOWNSCALE_WIDTH))
    jitters = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for f in frames[1:]:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_gray, 400, 0.01, 8)
        if p0 is None:
            prev_gray = gray
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
        if p1 is None or st is None:
            prev_gray = gray
            continue
        valid = st.squeeze() == 1
        if valid.sum() == 0:
            prev_gray = gray
            continue
        disp = (p1[valid] - p0[valid]).reshape(-1, 2)
        jitters.append(np.std(disp, axis=0).mean())
        prev_gray = gray
    if not jitters:
        return False
    return bool(np.mean(jitters) > SHAKE_STD_MAX)

def phash(img_bgr: np.ndarray) -> str:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    return str(imagehash.phash(pil))

# -----------------------------
# SCENE DETECTION
# -----------------------------
@dataclass
class Scene:
    start_s: float
    end_s: float

def detect_scenes(video_path: str) -> List[Scene]:
    vm = VideoManager([video_path])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=SCENE_THRESHOLD))
    vm.set_downscale_factor()
    vm.start()
    sm.detect_scenes(frame_source=vm)
    scene_list = sm.get_scene_list()
    vm.release()

    scenes: List[Scene] = []
    for s in scene_list:
        start_s = s[0].get_seconds()
        end_s = s[1].get_seconds()
        if end_s - start_s >= MIN_SCENE_LEN_S:
            scenes.append(Scene(start_s, end_s))
    return scenes

# -----------------------------
# SCORING / FILTERING
# -----------------------------
def score_scene(video_path: str, scene: Scene) -> Dict:
    mid_t = (scene.start_s + scene.end_s) / 2.0
    frm = read_frame_at_time(video_path, mid_t)
    frm = resize_keep_ar(frm, DOWNSCALE_WIDTH)
    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

    # metrics
    too_dim, too_bright = exposure_flags(gray)
    is_blur = blur_flag(gray)
    tilt = hough_vertical_tilt(gray)
    is_crooked = tilt > TILT_DEGREES_MAX
    is_colorcast = color_cast_flag(frm)
    is_blocky = blockiness_flag(gray)
    cluttered, edge_ratio, ent = clutter_flags(gray)
    is_mixed = mixed_light_flag(frm)
    is_shaky = shake_flag(video_path, scene.start_s, scene.end_s)

    # hard drop
    hard_drop = any([too_dim, too_bright, is_blur, is_crooked, is_shaky])

    # soft score
    score = 1.0
    if is_colorcast: score -= 0.25
    if cluttered:    score -= 0.20
    if is_blocky:    score -= 0.20
    if is_mixed:     score -= 0.20

    keep = (not hard_drop) and (score >= KEEP_SCORE_MIN)

    return {
        "mid_hash": phash(frm),
        "too_dim": too_dim,
        "too_bright": too_bright,
        "blurry": is_blur,
        "tilt_deg": round(tilt, 2),
        "crooked": is_crooked,
        "color_cast": is_colorcast,
        "blocky": is_blocky,
        "cluttered": cluttered,
        "edge_ratio": round(edge_ratio, 4),
        "entropy": round(ent, 3),
        "mixed_light": is_mixed,
        "shaky": is_shaky,
        "score": round(score, 3),
        "keep": keep,
    }

# -----------------------------
# EXPORT / CONCAT
# -----------------------------
def export_clip(video_path: str, start_s: float, end_s: float, out_path: Path):
    ensure_dirs(out_path.parent)
    dur = max(0.1, end_s - start_s)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_s:.3f}",
        "-i", video_path,
        "-t", f"{dur:.3f}",
        "-vf", "scale='min(1920,iw)':-2",  # limit to 1080p-ish while preserving AR
        "-c:v", "libx264", "-crf", str(CRF), "-preset", PRESET,
    ]
    if AUDIO:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += [str(out_path)]
    run(cmd)

def concat_videos(filelist_path: Path, output_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(filelist_path),
        "-c", "copy",
        str(output_path),
    ]
    # If "copy" fails due to differing params (unlikely since we transcode),
    # fall back to re-encode:
    try:
        run(cmd)
    except sp.CalledProcessError:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(filelist_path),
            "-c:v", "libx264", "-crf", str(CRF), "-preset", PRESET,
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]
        run(cmd)

# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_dirs(OUTPUT_ROOT)

    videos = []
    for p in sorted(INPUT_ROOT.rglob("*.mp4")):
        videos.append(p)
    for p in sorted(INPUT_ROOT.rglob("*.MP4")):
        if p not in videos:
            videos.append(p)

    if not videos:
        print(f"No videos found under {INPUT_ROOT}")
        sys.exit(1)

    rows = []
    kept_exports = []
    concat_list_path = OUTPUT_ROOT / "concat_list.txt"
    if concat_list_path.exists():
        concat_list_path.unlink()

    print(f"Found {len(videos)} videos…")
    for vid in videos:
        print(f"\n==> Processing: {vid}")
        scenes = detect_scenes(str(vid))
        print(f"   Scenes detected: {len(scenes)}")

        for idx, sc in enumerate(tqdm(scenes, desc=f"Scoring {vid.name}")):
            res = score_scene(str(vid), sc)
            row = {
                "video": str(vid),
                "scene_index": idx,
                "start_s": round(sc.start_s, 3),
                "end_s": round(sc.end_s, 3),
                **res
            }
            rows.append(row)

            if res["keep"]:
                clip_path = OUTPUT_ROOT / "clips" / f"{vid.stem}_scene{idx:04d}.mp4"
                export_clip(str(vid), sc.start_s, sc.end_s, clip_path)
                kept_exports.append(clip_path)

    # Deduplicate near-duplicate clips by pHash (optional simple pass)
    uniq = []
    seen = []
    for r, pth in zip(rows, kept_exports):
        if not r.get("keep"):
            continue
        h = r.get("mid_hash")
        if any(imagehash.hex_to_hash(h) - imagehash.hex_to_hash(x) < 6 for x in seen):
            # duplicate → mark as dropped
            r["keep"] = False
            r["reason_dropped"] = "near-duplicate"
            continue
        seen.append(h)
        uniq.append(pth)
    kept_exports = uniq

    # Write CSV report
    df = pd.DataFrame(rows)
    df_path = OUTPUT_ROOT / "report.csv"
    df.to_csv(df_path, index=False)
    print(f"\nReport written: {df_path}")

    # Build concat file
    if not kept_exports:
        print("No clips kept. Adjust thresholds and re-run.")
        return
    with open(concat_list_path, "w") as f:
        for clip in kept_exports:
            f.write(f"file '{clip.resolve()}'\n")

    # Concat to final
    concat_videos(concat_list_path, FINAL_VIDEO)
    print(f"\nFinal video: {FINAL_VIDEO.resolve()}")

if __name__ == "__main__":
    main()

