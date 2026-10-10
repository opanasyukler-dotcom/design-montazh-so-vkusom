"""Talking-head cleanup + kinetic subtitles in the house style (v3 look: thin italic Montserrat,
bold blush-glow accents, Biro Script handwritten words, Instagram-safe placement).
Work dir: in/src.mov, cuts.json (kept source intervals, frame-snapped), cwords.json (word times on the cut timeline), fonts.
  python3 -I talk_subs.py test 3.2,10   -> test_*.png
  python3 -I talk_subs.py render out.mp4   (video only, source resolution / fps, no grading)
"""
import json, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

FONT_MI, FONT_M, FONT_S = "Montserrat-Italic-VariableFont_wght.ttf", "Montserrat-VariableFont_wght.ttf", "BiroScriptUSPlus-Regular.ttf"
SRC = "in/src.mov"
W, H, FPS = 1080, 1920, 30
def hx(s): return np.float32([int(s[i:i + 2], 16) for i in (1, 3, 5)]) / 255
INK, ROSE, BLUSH = hx("#2F0600"), hx("#C5A29C"), hx("#F7DFDD")
WHITE = np.float32([1, 1, 1])

def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def ease_io(p): p = min(max(p, 0), 1); return p * p * (3 - 2 * p)
def back(p, c=1.6): p = min(max(p, 0), 1); return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2

def blend(dst, img, x, y, alpha=1.0):
    if alpha <= 0.003: return
    h, w = img.shape[:2]; x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0: return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]; d = dst[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4] * alpha; d += s[..., :3] * alpha

_fc = {}
def font(path, size, wght=None):
    k = (path, size, wght)
    if k not in _fc:
        f = ImageFont.truetype(path, size)
        if wght: f.set_variation_by_axes([wght])
        _fc[k] = f
    return _fc[k]

S = 46
def style_font(st):
    if st == "t": return font(FONT_MI, S, 540)
    if st == "b": return font(FONT_M, int(S * 1.4), 800)
    if st == "s": return font(FONT_S, int(S * 2.0))

class Word:
    def __init__(self, txt, st):
        self.st = st; f = style_font(st); pad = 12
        asc, desc = f.getmetrics(); w = int(f.getlength(txt)) + 2 * pad + 20
        im = Image.new("L", (w, asc + desc + 2 * pad), 0)
        ImageDraw.Draw(im).text((pad, pad + asc), txt, font=f, fill=255, anchor="ls")
        a = np.asarray(im).astype(np.float32) / 255
        if st == "s": a = cv2.dilate(a, np.ones((2, 2), np.uint8))
        col = BLUSH if st in ("b", "s") else WHITE
        self.img = np.zeros(a.shape + (4,), np.float32); self.img[..., :3] = col * a[..., None]; self.img[..., 3] = a
        s_ = np.minimum(1, cv2.GaussianBlur(a, (0, 0), 5) * 0.8)
        self.sh = np.zeros_like(self.img); self.sh[..., 3] = s_
        self.glow = None
        if st == "b":
            g = cv2.GaussianBlur(a, (0, 0), 9); self.glow = np.zeros_like(self.img)
            self.glow[..., :3] = ROSE * g[..., None]; self.glow[..., 3] = g; self.glow *= 0.85
        self.base, self.pad, self.adv = pad + asc, pad, f.getlength(txt)
        self.space = style_font("t").getlength(" ")

class Group:
    def __init__(self, lines, t_in, t_out):
        self.t_in, self.t_out, self.words = t_in, t_out, []
        for ln in lines:
            ws = [Word(t, st) for t, st in ln["words"]]
            total = sum(w.adv for w in ws) + sum(w.space for w in ws[:-1])
            al = ln["align"]; x = ln["x"] - (total / 2 if al == "c" else total if al == "r" else 0)
            for w, t0 in zip(ws, ln["t"]):
                w.x, w.y, w.t0 = x - w.pad, ln["y"] - w.base, t0; x += w.adv + w.space
                self.words.append(w)
    def draw(self, dst, tt):
        if tt < self.t_in - 0.01 or tt > self.t_out + 0.3: return
        out = ease((tt - self.t_out) / 0.25) if tt > self.t_out else 0
        for w in self.words:
            age = tt - w.t0
            if age < 0: continue
            al = min(1, age / 0.15) * (1 - out); img, sh, glow = w.img, w.sh, w.glow; dx = dy = 0
            if w.st == "s":
                p = ease_io(age / 0.45)
                if p < 1:
                    ww = img.shape[1]; ramp = np.clip((np.arange(ww) - p * ww * 1.15) / (-0.15 * ww), 0, 1)[None, :, None].astype(np.float32)
                    img = img * ramp; sh = sh * ramp
                al = 1 - out
            elif w.st == "b":
                sc = 0.7 + 0.3 * back(age / 0.3)
                if abs(sc - 1) > 0.01:
                    hh, ww = img.shape[:2]; nw, nh = max(2, int(ww * sc)), max(2, int(hh * sc))
                    img, sh, glow = (cv2.resize(a, (nw, nh)) for a in (img, sh, glow)); dx, dy = (ww - nw) / 2, (hh - nh) / 2
            else:
                e = ease(age / 0.28); dx = 16 * (1 - e); k = int(18 * (1 - e)) | 1
                if k > 2: img, sh = cv2.blur(img, (k, 1)), cv2.blur(sh, (k, 1))
            dy -= 14 * out
            blend(dst, sh, w.x + dx, w.y + dy + 3, al)
            if glow is not None: blend(dst, glow, w.x + dx, w.y + dy, al * (0.6 + 0.4 * np.sin(tt * 5) ** 2))
            blend(dst, img, w.x + dx, w.y + dy, al)

# ---------------------------------------------------------------- timeline
CUTS = json.load(open("cuts.json"))
STARTS = np.cumsum([0] + [round((b - a) * FPS) for a, b in CUTS]) / FPS
DUR = STARTS[-1]
WORDS = json.load(open("cwords.json"))
toks = []
for s, e, w in WORDS:
    w = w.strip()
    if w.startswith("-") and toks: toks[-1][2] += w; toks[-1][1] = e; continue
    toks.append([s, e, w])
chunks, cur = [], []
for t in toks:
    if cur and len(" ".join(x[2] for x in cur + [t])) > 22: chunks.append(cur); cur = []
    cur.append(t)
    if t[2][-1] in ",.?!" or len(cur) >= 3: chunks.append(cur); cur = []
if cur: chunks.append(cur)
merged = []
for ch in chunks:
    if merged and len(ch) == 1 and len(ch[0][2]) <= 6 and merged[-1][-1][2][-1] not in ",.?!" \
            and len(" ".join(x[2] for x in merged[-1] + ch)) <= 24:
        merged[-1] = merged[-1] + ch
    else: merged.append(ch)
chunks = merged

ACC_B = {"офигенная", "успех", "дожди", "ужасная", "зиму", "холода", "тяжело", "депрессивно", "сложно", "досыпать", "8-9", "5", "6"}
ACC_S = {"погода", "мадам", "помыть", "денёчки", "летом", "нормально", "приколы", "еле-еле"}
def clean(w): return w.lower().strip(",.?!«»")
SPOTS = [(110, 1250, "l"), (970, 1310, "r"), (540, 1370, "c")]     # Instagram-safe: above the caption area, clear of side buttons
subs = []
for ci, ch in enumerate(chunks):
    x, y, al = SPOTS[ci % len(SPOTS)]
    words = []
    for s, e, w in ch:
        c = clean(w); st = "b" if c in ACC_B else "s" if c in ACC_S else "t"
        words.append((w.rstrip(",.") if st != "t" or w[-1] in ",." else w, st, s))
    words = [(w.rstrip(",."), st, s) for w, st, s in words]
    txt = " ".join(w for w, _, _ in words)
    lines = [words]
    if len(txt) > 15 and len(words) > 1:
        k = (len(words) + 1) // 2; lines = [words[:k], words[k:]]
    L = []
    for li, ws in enumerate(lines):
        dx = (60 if al == "l" else -60 if al == "r" else 30) * li
        L.append(dict(words=[(w, st) for w, st, _ in ws], x=x + dx, y=y + li * int(S * 1.25), align=al, t=[t0 - 0.04 for _, _, t0 in ws]))
    t_out = min(chunks[ci + 1][0][0] - 0.08 if ci + 1 < len(chunks) else DUR, ch[-1][1] + 0.7)
    subs.append(Group(L, ch[0][0] - 0.06, t_out))

def zoom_at(tt):
    """gentle push per segment; alternate 1.00 / 1.07 framing so jump cuts read as intentional"""
    i = int(np.searchsorted(STARTS, tt, side="right") - 1); d = tt - STARTS[i]
    base = 1.0 if i % 2 == 0 else 1.07
    z = base * (1 + 0.02 * d / max(0.5, STARTS[i + 1] - STARTS[i]))
    if i > 0 and d < 0.2: z *= 1 + 0.03 * (1 - ease(d / 0.2))
    return z

def frame(src, tt):
    z = zoom_at(tt); fy = 700     # keep the face (upper third) stable while zooming
    M = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * fy]])
    fr = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT) if abs(z - 1) > 1e-4 else src
    fr = fr.astype(np.float32) / 255
    for g in subs: g.draw(fr, tt)
    return (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)

def source_frames():
    """yield (out_time, rgb) for every kept source frame, in order"""
    keep = []
    for a, b in CUTS: keep.append((round(a * FPS), round(b * FPS)))
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", SRC, "-vsync", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    idx = 0; out_i = 0; ki = 0
    while ki < len(keep):
        buf = dec.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        a, b = keep[ki]
        if a <= idx < b:
            yield out_i / FPS, np.frombuffer(buf, np.uint8).reshape(H, W, 3); out_i += 1
            if idx == b - 1: ki += 1
        idx += 1
    dec.kill()

if __name__ == "__main__":
    if sys.argv[1] == "test":
        want = [float(x) for x in sys.argv[2].split(",")]
        for tt, src in source_frames():
            if any(abs(tt - x) < 0.5 / FPS for x in want):
                cv2.imwrite(f"test_{tt:.2f}.png", cv2.cvtColor(cv2.resize(frame(src, tt), (540, 960), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
            if tt > max(want): break
        sys.exit()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-profile:v", "high",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", sys.argv[2]], stdin=subprocess.PIPE)
    n = 0
    for tt, src in source_frames():
        enc.stdin.write(frame(src, tt).tobytes()); n += 1
    enc.stdin.close(); enc.wait(); print("frames", n, "dur", DUR)
