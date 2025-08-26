# Video Dedup & Temporal Overlap Plan

**Goal:** Detect “almost duplicate” shots among 30–50 short clips (5–15 s each) of a house interior and output second-by-second overlap mappings between near-duplicate takes.

---

## 0) Inputs & Assumptions

* Clips: **30–50**, each **5–15 s**.
* Sampling: **3 FPS** keyframes per clip (adjustable).
* Embeddings: **CLIP** (e.g., ViT-L/14). **L2-normalize** every frame embedding.
* Output: Duplicate groups + per-pair aligned **time segments** where content overlaps.

---

## 1) Frame Extraction

Use ffmpeg to extract uniformly spaced frames and keep timestamps.

```bash
# 3 FPS, scaled for speed; keep original fps if already low
ffmpeg -i INPUT.mp4 -vf "fps=3,scale=768:-1" -q:v 2 frames/CLIP_ID/%06d.jpg
```

Store a simple index:

```
frames/CLIP_A/000001.jpg -> t=0.0s
frames/CLIP_A/000002.jpg -> t=0.333s
...
```

Optional prefilter: run **pHash** to drop exact/near-exact duplicates within each clip.

---

## 2) Embedding Generation (CLIP)

* Model: `ViT-L/14` (good tradeoff), or your preferred CLIP.
* For each image, compute embedding **e ∈ R^D**, then unit-normalize: `e := e / ||e||`.
* For clip V with T frames, collect `E_V ∈ R^{T×D}`.

---

## 3) Coarse Candidate Selection (Same-Room Guess)

Purpose: avoid O(N²·T²). Compare **clip-level centroids** first.

* Centroid per clip:  $\mu_V = \text{normalize}(\frac{1}{T}\sum_t e_t)$
* Pairwise centroid cosine:  $s_{coarse}(i,j) = \mu_i \cdot \mu_j$
* Keep candidates where **s\_coarse ≥ 0.84** (start threshold; tune 0.82–0.86).

This yields small candidate sets likely showing the **same room/area**.

---

## 4) Fine Alignment (Second-by-Second Overlap)

For each candidate pair (A, B):

**4.1 Build similarity matrix**

* `S = E_A @ E_B.T`  → frame-to-frame **cosine similarities** (since rows are unit-norm).

**4.2 Temporal alignment** (near-diagonal match)

* Search diagonals within a **band** ±20% of `min(T_A, T_B)` to handle speed drift.
* Smooth each diagonal with moving average **w=3–5**.
* Extract contiguous runs where similarity ≥ **τ\_fine = 0.86** and length ≥ **L\_min = 3 s** (→ 9 frames @3 FPS).
* Record segments as: `(A:tA_start–tA_end) ↔ (B:tB_start–tB_end, mean_sim)`.

**Alternative:** Use **DTW** (Sakoe–Chiba band) with cost = (1 − cosine), then slice contiguous high-sim stretches.

**4.3 Duplicate decision**
Mark A and B as **near-duplicate takes** if **all** hold:

* Overlap duration **≥ 3–4 s**;
* **Coverage ≥ 50%** of the shorter clip (overlapped seconds / len(shorter));
* Mean similarity across matched segments **≥ 0.88**.

---

## 5) Grouping & Representative Selection

* Build a graph: nodes = clips; edge if pair is near-duplicate (per §4.3).
* Connected components → **duplicate groups** (same room/shot attempts).
* Pick a **representative** per group (e.g., longest, least blur, best exposure).

---

## 6) Outputs

**6.1 Machine-readable**

CSV/JSON rows:

```
clip_A, tA_start_s, tA_end_s, clip_B, tB_start_s, tB_end_s, mean_sim, coverage
```

**6.2 Human-friendly preview (optional)**

* FFmpeg side-by-side with drawtext timestamps for quick QA.

```bash
ffmpeg -i A.mp4 -i B.mp4 -filter_complex \
  "[0:v]setpts=PTS-STARTPTS,scale=960:-1[va]; \
    [1:v]setpts=PTS-STARTPTS,scale=960:-1[vb]; \
    [va][vb]hstack=inputs=2,drawtext=fontfile=/usr/share/fonts/TTF/DejaVuSans.ttf: \
      text='%{pts\:hms}':x=10:y=H-th-10:box=1:boxcolor=black@0.5:boxborderw=5" \
  -an -y preview_side_by_side.mp4
```

---

## 7) Optional LLM Verification (Qwen2-VL)

Use Qwen2-VL for **semantic verification** and explanations on ambiguous segments:

**Prompt (multi-image, JSON output):**

```
You are comparing two short takes of the same shot recorded minutes apart.
Decide if they depict the SAME physical scene from slightly different angles.
Focus on static landmarks (wall patterns, window layout, floor texture, fixtures, signage).
Ignore transient elements (people, cars, shadows, reflections, white balance).

Respond as JSON:
{
  "same_scene": true|false,
  "confidence": 0.0-1.0,
  "shared_landmarks": ["...","..."],
  "camera_changes": "short description",
  "reasons": "one paragraph"
}
```

Feed 2–3 pairs (start/mid/end) from each matched segment.

Use Qwen only to **filter false positives** and to add human-readable **notes**.

---

## 8) Thresholds (starting points)

* Coarse centroid: **≥ 0.84** (same room candidate)
* Fine frame-level: **τ\_fine = 0.86**
* Segment min length: **L\_min = 3 s**
* Pair acceptance: mean-sim of matched segments **≥ 0.88**, coverage **≥ 50%** of shorter clip
* Band width for diagonal search: **±20%** of min(T\_A, T\_B)

Tune upward for stricter dedup; lower for varied lighting/exposure.

---

## 9) Minimal Core (NumPy)

```python
import numpy as np

def sim_matrix(A, B):
    # A: [TA x D], B: [TB x D], rows L2-normalized
    return A @ B.T  # cosine similarities

def diagonal_segments(S, fps=3, tau=0.86, min_sec=3, band_frac=0.2, smooth=3):
    TA, TB = S.shape
    K = int(band_frac * min(TA, TB))
    min_len = int(min_sec * fps)
    segs = []
    for k in range(-K, K+1):
        i0 = max(0, -k); j0 = max(0, k)
        L = min(TA - i0, TB - j0)
        if L <= 0: continue
        d = np.diag(S[i0:i0+L, j0:j0+L])
        if smooth > 1 and len(d) >= smooth:
            w = np.ones(smooth) / smooth
            d = np.convolve(d, w, mode='same')
        start = None
        for t, val in enumerate(d):
            if val >= tau and start is None:
                start = t
            elif (val < tau or t == len(d)-1) and start is not None:
                end = t if val < tau else t
                if end - start + 1 >= min_len:
                    ia, ib = i0 + start, j0 + start
                    ja, jb = i0 + end,   j0 + end
                    mean_sim = float(np.mean(d[start:end+1]))
                    segs.append(dict(A_start=ia, A_end=ja, B_start=ib, B_end=jb,
                                     mean_sim=mean_sim))
                start = None
    return segs

def duplicate_decision(A, B, fps=3):
    S = sim_matrix(A, B)
    segs = diagonal_segments(S, fps=fps, tau=0.86, min_sec=3)
    overlap_f = sum(s['A_end'] - s['A_start'] + 1 for s in segs)
    overlap_s = overlap_f / fps
    coverage = overlap_s / min(A.shape[0], B.shape[0])
    ok_len = overlap_s >= 3
    ok_cov = coverage >= 0.5
    ok_mean = (len(segs) > 0) and (np.mean([s['mean_sim'] for s in segs]) >= 0.88)
    return ok_len and ok_cov and ok_mean, segs, overlap_s, coverage
```

---

## 10) Practical Notes

* Keep images ≤ **768 px** long side for speed and VRAM.
* Normalize exposure a bit (gamma/CLAHE) to reduce lighting variance.
* Use **FAISS** (inner-product on unit vectors) to scale centroid matching.
* For very high precision, augment with local feature matching (SIFT/SuperPoint + LightGlue) as a final check.

---

## 11) Deliverables

* **JSON/CSV report** with matched segments and similarity stats.
* **Duplicate groups** (connected components).
* **Representative selection** rules logged.
* (Optional) **Side-by-side previews** for human spot-checking.

---

## 12) Next Steps

1. Implement frame extraction + embedding cache (parallelized).
2. Build centroid index; run coarse candidate selection.
3. Implement fine alignment + segment extraction.
4. Emit JSON/CSV + previews; review thresholds on a small subset.
5. Add Qwen2-VL verifier on borderline cases; iterate thresholds.
6. Integrate into your roadmap pipeline + UI (tagging, trimming, dedup).

