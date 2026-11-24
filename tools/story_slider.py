#!/usr/bin/env python3
"""Instagram story: photos full screen, one after another through a slider transition
(a white line with a round handle sweeps across and reveals the next photo). No flash, a calm slow push on each photo.

usage: python3 tools/story_slider.py out.mp4 logo.png photo1 photo2 ... [--hold 2.6 --wipe 1.1]
"""
import sys, subprocess, numpy as np, cv2
from PIL import Image

W, H, FPS = 1080, 1920, 30
args = sys.argv[1:]; opt = {}
for k in ("--hold", "--wipe"):
    if k in args: i = args.index(k); opt[k] = float(args[i + 1]); del args[i:i + 2]
OUT, LOGO, PH = args[0], args[1], args[2:]
HOLD, WIPE = opt.get("--hold", 2.6), opt.get("--wipe", 1.1)
def eio(x): x = min(1.0, max(0.0, x)); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def cover(img, z):
    ih, iw = img.shape[:2]; s = max(W / iw, H / ih) * z
    M = np.float32([[s, 0, W / 2 - iw * s / 2], [0, s, H / 2 - ih * s / 2]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255
IM = [cv2.imread(p)[..., ::-1].copy() for p in PH]
lg = np.asarray(Image.open(LOGO).convert("RGBA"))[..., 3].astype(np.float32) / 255
ys, xs = np.where(lg > 0.05); lg = lg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; lw = 220; lg = cv2.resize(lg, (lw, int(lg.shape[0] * lw / lg.shape[1])), interpolation=cv2.INTER_AREA)
def blend(fr, a, x, y, col, op):
    h, w = a.shape; m = a[..., None] * op; reg = fr[y:y + h, x:x + w]; reg *= 1 - m; reg += m * np.float32(col)
SEG = HOLD + WIPE; DUR = len(IM) * HOLD + (len(IM) - 1) * WIPE
def shot(k, t):
    """photo k at time t: a slow push-in across its whole time on screen"""
    t0 = k * SEG - (WIPE if k else 0); d = SEG + WIPE; return cover(IM[k], 1.0 + 0.045 * min(1, max(0, (t - t0) / d)))
def handle(fr, x):
    m = np.zeros((H, W), np.uint8); xi = int(round(x)); cv2.line(m, (xi, 0), (xi, H), 255, 5, cv2.LINE_AA)
    cv2.circle(m, (xi, 960), 46, 255, -1, cv2.LINE_AA); a = m.astype(np.float32) / 255
    sh = cv2.GaussianBlur(a, (0, 0), 8); fr *= 1 - sh[..., None] * 0.35; fr[:] = fr * (1 - a[..., None]) + a[..., None]
    ar = np.zeros((H, W), np.uint8)
    for s in (-1, 1): cv2.fillPoly(ar, [np.int32([(xi + s * 12, 948), (xi + s * 12, 972), (xi + s * 28, 960)])], 255, cv2.LINE_AA)
    a2 = ar.astype(np.float32)[..., None] / 255; fr[:] = fr * (1 - a2) + a2 * np.float32([0, 12, 40]) / 255
def frame(t):
    k = min(len(IM) - 1, int(t // SEG)); u = t - k * SEG
    if u < HOLD or k == len(IM) - 1: fr = shot(k, t)
    else:
        p = eio((u - HOLD) / WIPE); x = W * (1 - p) + (-30) * p; x = W - (W + 30) * p     # the line sweeps right → left
        fr = shot(k, t); nw = shot(k + 1, t); xi = int(max(0, min(W, x))); fr[:, xi:] = nw[:, xi:]; handle(fr, x)
    blend(fr, lg, W // 2 - lw // 2, 1560, (1, 1, 1), 0.85); return fr
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                      "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
for i in range(int(round(DUR * FPS))): p.stdin.write((np.clip(frame(i / FPS), 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
p.stdin.close(); p.wait(); print(f"{DUR:.1f}s")
