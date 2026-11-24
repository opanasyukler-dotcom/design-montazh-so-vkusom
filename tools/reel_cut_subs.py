#!/usr/bin/env python3
"""Talking-head clean-up: cut every pause (and the sounds between words that are not words — «э», «а», breaths),
drop the false start, then burn in phrase captions (2–3 words, the spoken word lights up, key words pink, Montserrat).

usage:
  python3 tools/reel_cut_subs.py <dir> [--preview t1,t2]
  <dir>/V.mp4 (1080x1920, 30 fps, no audio), <dir>/V.wav (mono 48 kHz), <dir>/tr.txt (faster-whisper word list),
  <dir>/cfg.json {"start": s, "end": s, "fix": {...}, "key": [...], "sentence_end": [s, ...]}
  → <dir>/out.mp4
"""
import os, sys, json, wave, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1]; CFG = json.load(open(os.path.join(D, "cfg.json")))
W, H, FPS, SR = 1080, 1920, 30, 48000
PINK = np.float32([233, 197, 193]) / 255; WHITE = np.float32([1, 1, 1]); DEEP = np.float32([0, 5, 20]) / 255

# ---------- audio and the keep list: speech islands, every silence squeezed to a short breath
w = wave.open(os.path.join(D, "V.wav")); X = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
HOP = int(0.01 * SR); rms = np.sqrt(np.convolve(X ** 2, np.ones(2 * HOP) / (2 * HOP), "same")[::HOP])
WORDS = [(CFG["fix"].get(a, a), float(s), float(e)) for a, s, e in eval(open(os.path.join(D, "tr.txt")).read().replace("np.float64", ""))]
WORDS = [x for x in WORDS if x[0] and CFG["start"] - 0.05 <= x[1] and x[2] <= CFG["end"] + 0.05]
THR = CFG.get("thr", 0.018)
def speech_mask():
    m = rms > THR; m = np.convolve(m.astype(float), np.ones(7), "same") > 0          # bridge consonant gaps < 70 ms
    return m
M = speech_mask(); t0, t1 = CFG["start"], CFG["end"]
segs = []; i = int(t0 * 100)
while i < int(t1 * 100):
    if M[i]:
        j = i
        while j < int(t1 * 100) and M[j]: j += 1
        segs.append([i / 100, j / 100]); i = j
    else: i += 1
# islands with no word in them are noises / filler sounds → dropped
def has_word(a, b): return any(min(b, e) - max(a, s) > 0.04 for _, s, e in WORDS)
segs = [s for s in segs if has_word(*s)]
# pad each island, keep a short breath; a longer one where a sentence ends
KEEP = []
for k, (a, b) in enumerate(segs):
    a -= 0.03; b += 0.05
    if k + 1 < len(segs):
        nxt = segs[k + 1][0]; gap = nxt - b
        brk = any(abs(b - se) < 0.35 for se in CFG.get("sentence_end", []))
        b += min(max(0.0, gap - 0.04), 0.16 if brk else 0.04)
    KEEP.append([max(t0, a), min(t1 + 0.1, b)])
merged = []
for a, b in KEEP:
    if merged and a <= merged[-1][1] + 0.01: merged[-1][1] = max(merged[-1][1], b)
    else: merged.append([a, b])
KEEP = merged
for c0, c1 in CFG.get("cut", []):                                         # drawn-out vowels / stumbles cut by hand
    nk = []
    for a, b in KEEP:
        if c1 <= a or c0 >= b: nk.append([a, b])
        else:
            if a < c0: nk.append([a, c0])
            if c1 < b: nk.append([c1, b])
    KEEP = nk
def to_out(t):
    acc = 0.0
    for a, b in KEEP:
        if t < a: return acc
        if t <= b: return acc + t - a
        acc += b - a
    return acc
DUR = sum(b - a for a, b in KEEP)

def build_audio(path):
    out = []; f_ = int(0.012 * SR)
    for a, b in KEEP:
        s = X[int(a * SR):int(b * SR)].copy(); s[:f_] *= np.linspace(0, 1, f_); s[-f_:] *= np.linspace(1, 0, f_); out.append(s)
    y = np.concatenate(out); y = y / (np.sqrt(np.mean(np.sort(np.abs(y))[-len(y) // 3:] ** 2)) + 1e-9) * 0.16; y = np.clip(y, -0.98, 0.98)
    with wave.open(path, "wb") as o:
        o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR); o.writeframes((y * 32767).astype(np.int16).tobytes())

# ---------- captions
FD = os.path.join(ROOT, "fonts"); _FC = {}
def mont(size, wght):
    k = (size, wght)
    if k not in _FC:
        f = ImageFont.truetype(os.path.join(FD, "Montserrat-VariableFont_wght.ttf"), size); f.set_variation_by_axes([wght]); _FC[k] = f
    return _FC[k]
def alpha_img(txt, f):
    asc, desc = f.getmetrics(); P = int(f.size * 0.18); im = Image.new("L", (int(f.getlength(txt)) + 8, asc + desc + 4 + 2 * P), 0)
    ImageDraw.Draw(im).text((4, 2 + P + asc), txt, font=f, fill=255, anchor="ls"); return np.asarray(im, np.float32) / 255
def blend(fr, a, x, y, col, op=1.0):
    x, y = int(round(x)), int(round(y)); h, w_ = a.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w_), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    m = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * op; reg = fr[y0:y1, x0:x1]; reg *= 1 - m; reg += m * col
def eo(x): x = min(1.0, max(0.0, x)); return 1 - (1 - x) ** 3
KEY = set(CFG.get("key", []))
SW = [(w_, to_out(s), to_out(e)) for w_, s, e in WORDS]
PH, cur = [], []
for w_ in SW:
    cur.append(w_)
    if len(cur) >= 3 or w_[0][-1:] in ",.?!" or len(" ".join(x[0] for x in cur)) > 18: PH.append(cur); cur = []
if cur: PH.append(cur)
PH = [(ph, ph[0][1] - 0.06, (PH[k + 1][0][1] - 0.06) if k + 1 < len(PH) else DUR) for k, ph in enumerate(PH)]
CAP_Y = CFG.get("cap_y", 1450)
def captions(fr, t):
    for ph, a0, a1 in PH:
        if not (a0 <= t < a1): continue
        p = eo((t - a0) / 0.18); op = min(1, (a1 - t) / 0.08) * min(1, (t - a0) / 0.1); ims = []
        for w_, s, e in ph:
            ww = w_.strip(",.?!"); key = w_ in KEY or ww in KEY
            ims.append((alpha_img(ww, mont(58 if key else 56, 600 if key else 500)), PINK if key else WHITE, s))
        sp = 16; tw = sum(a.shape[1] for a, _, _ in ims) + sp * (len(ims) - 1); x = 468 - tw / 2; y = CAP_Y + 16 * (1 - p)
        for a, col, s in ims:
            lit = 1.0 if t >= s - 0.04 else 0.45
            sh = cv2.GaussianBlur(a, (0, 0), 3); blend(fr, sh, x + 2, y - a.shape[0] / 2 + 4, DEEP, 0.55 * op)
            blend(fr, a, x, y - a.shape[0] / 2, col, op * lit); x += a.shape[1] + sp

GT = np.repeat((np.clip((np.arange(H)[:, None] - 1180) / 520, 0, 1) ** 1.3 * 0.35)[..., None], W, 1).astype(np.float32)
def frames():
    cap = cv2.VideoCapture(os.path.join(D, "V.mp4")); want = []
    for k in range(int(round(DUR * FPS))):
        t = k / FPS; acc = 0.0
        for a, b in KEEP:
            if t < acc + b - a: want.append((k, a + t - acc)); break
            acc += b - a
    i = -1; f = None
    for k, st in want:
        j = int(round(st * FPS))
        while i < j:
            ok, g = cap.read()
            if not ok: break
            f = g; i += 1
        fr = f[..., ::-1].astype(np.float32) / 255; fr = fr * (1 - GT) + GT * 0.0
        captions(fr, k / FPS); yield k, fr

if __name__ == "__main__":
    print("keep", [(round(a, 2), round(b, 2)) for a, b in KEEP], "→", round(DUR, 2), "s")
    if "--preview" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]; want = {int(round(t * FPS)) for t in ts}
        for k, fr in frames():
            if k in want: cv2.imwrite(os.path.join(D, f"prev_{k / FPS:05.2f}.jpg"), (np.clip(fr, 0, 1)[..., ::-1] * 255).astype(np.uint8))
        sys.exit()
    build_audio(os.path.join(D, "mix.wav"))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", os.path.join(D, "mix.wav"),
                          "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                          os.path.join(D, "out.mp4")], stdin=subprocess.PIPE)
    for k, fr in frames(): p.stdin.write((np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait(); print("done")
