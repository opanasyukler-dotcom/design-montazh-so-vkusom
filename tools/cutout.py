"""Cut a person out of photos / a video (MediaPipe selfie segmenter, run in the mpenv): RGBA PNGs cropped to the person, soft edge.
  mpenv/bin/python cutout.py MODELS_DIR OUT_DIR photo1.jpg [photo2.heic ...]        → OUT_DIR/<name>.png
  mpenv/bin/python cutout.py MODELS_DIR OUT_DIR --video clip.mov SS DUR FPS HEIGHT  → OUT_DIR/<name>_0000.png … (same crop box for all)
"""
import os, subprocess, sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

def segmenter(models):
    return vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(models, "selfie_multiclass.tflite")), output_confidence_masks=True))

def alpha(seg, rgb):
    """person = 1 − background (multiclass: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 others)"""
    m = 1 - seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))).confidence_masks[0].numpy_view()
    m = cv2.morphologyEx((m > 0.5).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(np.float32)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))                 # keep the person, drop specks
    if n > 1: m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), max(1.0, rgb.shape[1] / 700))

def load(path):
    if path.lower().endswith((".heic", ".heif")):
        from pillow_heif import open_heif
        h = open_heif(path); im = np.asarray(h.to_pillow().convert("RGB"))
    else: im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return im

def save(rgb, a, box, out):
    x0, y0, x1, y1 = box; rgba = np.dstack([rgb, (a * 255).astype(np.uint8)])[y0:y1, x0:x1]
    cv2.imwrite(out, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

def bbox(a, pad=8):
    ys, xs = np.where(a > 0.1); h, w = a.shape
    return max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, w), min(ys.max() + pad, h)

def main(models, out, args):
    os.makedirs(out, exist_ok=True); seg = segmenter(models)
    if args[0] == "--video":
        src, ss, dur, fps, hh = args[1], float(args[2]), float(args[3]), float(args[4]), int(args[5])
        import tempfile, glob
        with tempfile.TemporaryDirectory() as td:                         # decode to stills (ffmpeg handles the phone rotation)
            subprocess.run(["ffmpeg", "-v", "error", "-ss", str(ss), "-t", str(dur), "-i", src, "-vf", f"fps={fps},scale=-2:{hh}",
                            os.path.join(td, "f_%04d.png")], check=True)
            frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in sorted(glob.glob(os.path.join(td, "f_*.png")))]
        alphas = [alpha(seg, f) for f in frames]; box = bbox(np.max(alphas, 0))
        name = os.path.splitext(os.path.basename(src))[0]
        for i, (f, a) in enumerate(zip(frames, alphas)): save(f, a, box, os.path.join(out, f"{name}_{i:04d}.png"))
    else:
        for p in args:
            rgb = load(p); s = 1600 / max(rgb.shape[:2])
            if s < 1: rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            a = alpha(seg, rgb); save(rgb, a, bbox(a), os.path.join(out, os.path.splitext(os.path.basename(p))[0] + ".png"))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
