"""Stroller clip: kinetic subtitles + hand-drawn animation + light SFX (from talk_subs).
Original: Talking-head cleanup + kinetic subtitles in the house style (v3 look: thin italic Montserrat,
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
W, H, FPS = 1080, 1920, 25
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
        s_ = np.minimum(1, cv2.GaussianBlur(cv2.dilate(a, np.ones((5, 5), np.uint8)) if st == "s" else a, (0, 0), 5) * (1.1 if st == "s" else 0.8))
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
NFR = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", SRC], capture_output=True, text=True).stdout.split()[0].strip(","))
CUTS = [(0, NFR / FPS)]
STARTS = np.array([0, NFR / FPS])
DUR = NFR / FPS
WORDS = json.load(open("words.json"))
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

ACC_B = {"заполненная", "корзина", "агат"}
ACC_S = {"шапку", "агате", "гуляем"}
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

def wt(i): return WORDS[i][0]

# ---------------------------------------------------------------- hand-drawn animation
def stroke(dst, pts, p, col, th, al=1.0):
    n = max(2, int(p * len(pts)))
    if p > 0: cv2.polylines(dst, [np.int32(pts[:n])], False, tuple(float(c) * al for c in col), th, cv2.LINE_AA)

class Scribble:
    """loose hand-drawn ellipse around something"""
    def __init__(self, cx, cy, rx, ry, t0, t1, col=BLUSH):
        u = np.linspace(-2.4, -2.4 + 2 * np.pi * 1.12, 140)
        wob = 1 + 0.04 * np.sin(u * 3.0)
        self.pts = np.c_[cx + rx * wob * np.cos(u), cy + ry * wob * np.sin(u)]
        self.t0, self.t1, self.col = t0, t1, col
    def draw(self, dst, tt):
        if tt < self.t0 or tt > self.t1 + 0.25: return
        al = 1 - (ease((tt - self.t1) / 0.25) if tt > self.t1 else 0)
        stroke(dst, self.pts + [0, 3], ease_io((tt - self.t0) / 0.5), INK * 0.6, 9, al * 0.5)
        stroke(dst, self.pts, ease_io((tt - self.t0) / 0.5), self.col, 7, al)

class Arrow:
    """curvy hand-drawn arrow from (x0,y0) to (x1,y1) with a head"""
    def __init__(self, x0, y0, x1, y1, t0, t1, bend=120):
        u = np.linspace(0, 1, 120)[:, None]; p0, p2 = np.float32([x0, y0]), np.float32([x1, y1])
        mid = (p0 + p2) / 2; nrm = np.float32([-(y1 - y0), x1 - x0]); nrm /= np.linalg.norm(nrm)
        p1 = mid + nrm * bend
        self.pts = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * p1 + u ** 2 * p2
        d = self.pts[-1] - self.pts[-8]; d /= np.linalg.norm(d); n = np.float32([-d[1], d[0]])
        tip = self.pts[-1]; self.head = [np.array([tip - d * 38 + n * 22, tip]), np.array([tip - d * 38 - n * 22, tip])]
        self.t0, self.t1 = t0, t1
    def draw(self, dst, tt):
        if tt < self.t0 or tt > self.t1 + 0.25: return
        al = 1 - (ease((tt - self.t1) / 0.25) if tt > self.t1 else 0); p = ease_io((tt - self.t0) / 0.4)
        stroke(dst, self.pts + [0, 3], p, INK * 0.6, 9, al * 0.5); stroke(dst, self.pts, p, BLUSH, 7, al)
        q = ease_io((tt - self.t0 - 0.38) / 0.15)
        for h in self.head: stroke(dst, h, q, BLUSH, 7, al)

def heart_img(size, col):
    im = Image.new("L", (size * 4, size * 4), 0); d = ImageDraw.Draw(im)
    u = np.linspace(0, 2 * np.pi, 120); s = size * 4 / 36
    pts = [(size * 2 + 16 * np.sin(t) ** 3 * s, size * 1.9 - (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) * s) for t in u]
    d.polygon(pts, fill=255); a = cv2.resize(np.asarray(im).astype(np.float32) / 255, (size, size), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 4), np.float32); out[..., :3] = col * a[..., None]; out[..., 3] = a; return out

class Hearts:
    """a little burst of hearts floating up from a point"""
    def __init__(self, cx, cy, t0, n=6, seed=1):
        r = np.random.default_rng(seed); self.t0 = t0
        self.h = [(heart_img(int(r.uniform(52, 92)), BLUSH if k % 2 else ROSE), cx + r.uniform(-90, 90), cy + r.uniform(-30, 30),
                   r.uniform(-40, 40), r.uniform(150, 260), t0 + k * 0.08, r.uniform(1.2, 1.7)) for k in range(n)]
    def draw(self, dst, tt):
        for img, x, y, vx, vy, t0, life in self.h:
            a = tt - t0
            if a < 0 or a > life: continue
            sc = back(a / 0.3); al = min(1, a / 0.1) * (1 - ease((a - life + 0.4) / 0.4))
            w = max(2, int(img.shape[1] * sc)); im = cv2.resize(img, (w, w))
            blend(dst, im, x + vx * a + 12 * np.sin(a * 6) - w / 2, y - vy * a - w / 2, al)

class Chip:
    def __init__(self, txt, t0, t1, x=70, y=250):
        f = font(FONT_M, 34, 800); tw = int(f.getlength(txt)); hgt = 74; pw = tw + 110
        im = Image.new("RGBA", (pw, hgt), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, pw - 1, hgt - 1), radius=hgt // 2, fill=(247, 223, 221, 240))
        d.ellipse((26, hgt // 2 - 9, 44, hgt // 2 + 9), fill=(130, 93, 77, 255))
        d.text((62, hgt // 2), txt, font=f, fill=(47, 6, 0, 255), anchor="lm")
        a = np.asarray(im).astype(np.float32) / 255; a[..., :3] *= a[..., 3:4]; self.img = a
        self.x, self.y, self.t0, self.t1 = x, y, t0, t1
    def draw(self, dst, tt):
        if tt < self.t0 or tt > self.t1 + 0.3: return
        e = ease((tt - self.t0) / 0.35); out = ease((tt - self.t1) / 0.25) if tt > self.t1 else 0
        blend(dst, self.img, self.x - 300 * (1 - e) - 300 * out, self.y, min(1, (tt - self.t0) / 0.15) * (1 - out))

ZOOM_T = (wt(13), 7.7)     # «полная корзина»: push in on the basket
ZOOM_F = (380, 920)
ANIM = [
    Chip("ПРОГУЛКА", 0.15, 3.6),
    Scribble(560, 1010, 230, 170, wt(5) - 0.05, 3.9),                       # шапку
    Hearts(600, 430, wt(9), n=7, seed=2),                                     # Агате
    Arrow(820, 560, 520, 860, wt(13) - 0.1, 7.6),                            # полная корзина
    Scribble(370, 920, 270, 200, wt(14), 7.6, col=ROSE),
    Hearts(420, 700, wt(20), n=9, seed=5),                                    # да, Агат?
]

def zoom_at(tt):
    z = 1 + 0.03 * tt / DUR
    p = ease_io((tt - ZOOM_T[0]) / 0.45) - ease_io((tt - ZOOM_T[1]) / 0.5)
    return z * (1 + 0.12 * p), p

def frame(src, tt):
    z, p = zoom_at(tt)
    fx, fy = W / 2 + (ZOOM_F[0] - W / 2) * p, H / 2 + (ZOOM_F[1] - H / 2) * p
    M = np.float32([[z, 0, (1 - z) * fx], [0, z, (1 - z) * fy]])
    fr = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    fr = fr.astype(np.float32) / 255
    for e in ANIM: e.draw(fr, tt)
    for g in subs: g.draw(fr, tt)
    return (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)

def sfx(path):
    import wave
    SR = 48000; rng = np.random.default_rng(7); n = int((DUR + 0.2) * SR); trk = np.zeros(n)
    def onepole(x, cut):
        y = np.empty_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * cut / SR)
        for i in range(len(x)): s += a * (x[i] - s); y[i] = s
        return y
    def norm(y): return y / (np.abs(y).max() + 1e-9)
    def put(sig, t, g):
        i = int(max(0, t) * SR); seg = trk[i:i + len(sig)]; seg += sig[:len(seg)] * g
    def pop():
        m = int(0.09 * SR); t = np.arange(m) / SR; f = 300 + 700 * np.exp(-t * 55)
        return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 40))
    def scribble(d=0.45):
        m = int(d * SR); t = np.arange(m) / SR
        x = onepole(rng.standard_normal(m), 3500) - onepole(rng.standard_normal(m), 600)
        return norm(x * (0.55 + 0.45 * np.sin(2 * np.pi * 11 * t) ** 2) * np.sin(np.pi * t / d) ** 0.6)
    def sparkle():
        m = int(0.8 * SR); t = np.arange(m) / SR
        y = sum(np.sin(2 * np.pi * f * t) * np.exp(-(t - k * 0.06).clip(0) * 9) * (t >= k * 0.06) for k, f in enumerate([1568, 1976, 2349, 3136]))
        return norm(y)
    def whoosh(d=0.4):
        m = int(d * SR); t = np.linspace(0, 1, m); x = rng.standard_normal(m)
        y = onepole(x, 3000) - onepole(x, 500); return norm(y * np.sin(np.pi * t) ** 2)
    POP, SCR = pop(), scribble()
    for e in ANIM:
        if isinstance(e, Chip): put(POP, e.t0 + 0.05, 0.25)
        elif isinstance(e, (Scribble, Arrow)): put(SCR, e.t0, 0.16)
        elif isinstance(e, Hearts): put(sparkle(), e.t0, 0.12)
    put(whoosh(), ZOOM_T[0] - 0.2, 0.18)
    for g in subs:
        for w in g.words:
            if w.st == "b": put(POP, w.t0, 0.14)
            elif w.st == "s": put(SCR, w.t0, 0.06)
    w = wave.open(path, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(trk, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()

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
    if sys.argv[1] == "sfx": sfx(sys.argv[2]); sys.exit()
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
