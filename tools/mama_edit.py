"""Reels edit: brand-palette text inserts, white Montserrat subtitles, zooms, transitions, SFX.
Inputs (cwd): cut.mov (already-trimmed 1080x1920 edit), cwords.json (word timings), cuts.json.
Outputs: video_only.mp4, sfx.wav
"""
import json, subprocess, sys, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 25
FONT = sys.argv[1] if len(sys.argv) > 1 else "Montserrat-VariableFont_wght.ttf"
FONT_I = FONT.replace("Montserrat-", "Montserrat-Italic-")
LIMIT = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
import os
TEST = [float(x) for x in os.environ.get("TEST", "").split(",") if x]

def hx(s): return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))
INK, COCOA, ROSE, MIST, BLUSH = hx("#2F0600"), hx("#825D4D"), hx("#C5A29C"), hx("#C5B0AD"), hx("#F7DFDD")
WHITE = (255, 255, 255)

_fc = {}
def font(size, wght, italic=False):
    k = (size, wght, italic)
    if k not in _fc:
        f = ImageFont.truetype(FONT_I if italic else FONT, size); f.set_variation_by_axes([wght]); _fc[k] = f
    return _fc[k]

def premul(im):
    a = np.asarray(im).astype(np.float32) / 255
    a[..., :3] *= a[..., 3:4]; return a

def text_img(txt, f, col):
    asc, desc = f.getmetrics(); w = int(f.getlength(txt)) + 12
    im = Image.new("RGBA", (w, asc + desc + 12), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((6, 6 + asc), txt, font=f, fill=col + (255,), anchor="ls")
    return premul(im), 6 + asc

def blend(dst, img, x, y, alpha=1.0):
    if alpha <= 0.003: return
    h, w = img.shape[:2]; x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0: return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]; d = dst[y0:y1, x0:x1]
    if dst.shape[2] == 4:
        d *= 1 - s[..., 3:4] * alpha; d += s * alpha
    else:
        d *= 1 - s[..., 3:4] * alpha; d += s[..., :3] * alpha

def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def ease_io(p): p = min(max(p, 0), 1); return p * p * (3 - 2 * p)
def back(p, c=1.7):
    p = min(max(p, 0), 1); return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2

# ---------------- timeline ----------------
WORDS = json.load(open("cwords.json"))
cuts = json.load(open("cuts.json"))
acc, JOINS = 0.0, []
for i, (a, b) in enumerate(cuts):
    if a in (38.40, 76.10, 97.45, 120.00): JOINS.append(acc)   # section changes (source jumps between kept blocks)
    acc += b - a
DUR = acc
def wt(i): return WORDS[i][0]

# ---------------- subtitles ----------------
FIX = {"бесил,": "без сил,"}
toks = []
for s, e, w in WORDS:
    w = w.strip(); w = FIX.get(w, w)
    if w in ("–", "-"): continue
    if w.startswith("-") and toks: toks[-1][2] += w; toks[-1][1] = e; continue
    toks.append([s, e, w])
chunks, cur = [], []
for t in toks:
    if cur and len(" ".join(x[2] for x in cur + [t])) > 22:
        chunks.append(cur); cur = []
    cur.append(t)
    if t[2][-1] in ",.?!" or len(cur) >= 3:
        chunks.append(cur); cur = []
if cur: chunks.append(cur)
merged = []
for ch in chunks:   # glue a dangling short tail word ("так,") back onto its phrase
    if merged and len(ch) == 1 and len(ch[0][2]) <= 6 and merged[-1][-1][2][-1] not in ",.?!" \
            and len(" ".join(x[2] for x in merged[-1] + ch)) <= 24:
        merged[-1] = merged[-1] + ch
    else: merged.append(ch)
chunks = merged
SUB_Y = 1535
SUBF = font(66, 700)
subs = []
for ci, ch in enumerate(chunks):
    words = [w for _, _, w in ch]
    disp = [w.rstrip(",.") for w in words]
    imgs = [text_img(w, SUBF, WHITE) for w in disp]
    space = SUBF.getlength(" ")
    total = sum(im.shape[1] - 12 for im, _ in imgs) + space * (len(imgs) - 1)
    x = (W - total) / 2; items = []
    for (im, base), (s, e, _) in zip(imgs, ch):
        sh = np.minimum(1, cv2.GaussianBlur(im[..., 3], (0, 0), 5) * 0.85)
        shi = np.zeros(sh.shape + (4,), np.float32); shi[..., 3] = sh
        items.append(dict(t0=s, img=im, sh=shi, x=x - 6, y=SUB_Y - base)); x += im.shape[1] - 12 + space
    start = ch[0][0] - 0.05
    nxt = chunks[ci + 1][0][0] - 0.05 if ci + 1 < len(chunks) else DUR
    end = min(nxt, ch[-1][1] + 0.6)
    subs.append(dict(start=start, end=end, items=items))

def draw_subs(frame, tt):
    for s in subs:
        if not (s["start"] <= tt < s["end"]): continue
        out = 1 - min(1, max(0, (s["end"] - tt) / 0.08)) if s["end"] - tt < 0.08 else 0
        for it in s["items"]:
            age = tt - it["t0"] + 0.05
            if age < 0: continue
            e = ease(age / 0.18); al = min(1, age / 0.1) * (1 - out)
            blend(frame, it["sh"], it["x"], it["y"] + 4 + 18 * (1 - e), al)
            img = it["img"]
            if e < 0.999:
                sc = 0.86 + 0.14 * back(age / 0.22)
                h, w = img.shape[:2]; nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
                img = cv2.resize(img, (nw, nh)); dx, dy = (w - nw) / 2, (h - nh) / 2
            else: dx = dy = 0
            blend(frame, img, it["x"] + dx, it["y"] + dy + 18 * (1 - e), al)

# ---------------- insert cards ----------------
# line = list of (text, kind); kinds: n normal, e emphasis, l label(light), q italic quote
LIGHT = dict(bg=BLUSH, n=INK, e=COCOA, l=COCOA, q=INK, deco=ROSE)
DARK = dict(bg=INK, n=BLUSH, e=ROSE, l=MIST, q=BLUSH, deco=ROSE)

class Card:
    def __init__(self, lines, theme, t_in, t_out, top, line_times=None, size=56, deco=None, shake=False):
        self.theme, self.t_in, self.t_out, self.top, self.deco, self.shake = theme, t_in, t_out, top, deco, shake
        padx, pady, lh = 48, 36, int(size * 1.26)
        maxw = W - 2 * 50 - 2 * padx
        while True:
            widths = [sum(font(size, 600 if k == "l" else 800, k == "q").getlength(t + " ") for t, k in ln) for ln in lines]
            if max(widths) <= maxw: break
            size -= 2; lh = int(size * 1.26)
        self.size = size
        cw = int(max(widths) + 2 * padx); chh = int(lh * len(lines) + 2 * pady - (lh - size) * 0.6)
        self.w, self.h = cw, chh
        bg = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
        ImageDraw.Draw(bg).rounded_rectangle((0, 0, cw - 1, chh - 1), radius=34, fill=theme["bg"] + (240,))
        self.bg = premul(bg)
        sh = np.zeros((chh + 80, cw + 80), np.float32); sh[40:40 + chh, 40:40 + cw] = self.bg[..., 3]
        sh = cv2.GaussianBlur(sh, (0, 0), 16) * 0.45
        self.shadow = np.zeros(sh.shape + (4,), np.float32); self.shadow[..., 3] = sh
        self.words = []; self.spans = {}
        for li, ln in enumerate(lines):
            y_base = pady + size * 0.98 + li * lh
            x = padx + (maxw - widths[li]) / 2 if False else (cw - widths[li]) / 2 + 4
            lt = (line_times[li] if line_times else t_in + 0.25 + li * 0.35)
            k_idx = 0
            for t, k in ln:
                for wd in t.split(" "):
                    f = font(size, 600 if k == "l" else 800, k == "q")
                    im, b = text_img(wd, f, theme[k])
                    self.words.append(dict(img=im, x=x - 6, y=y_base - b, t0=lt + 0.055 * k_idx, kind=k))
                    if k == "e": self.spans.setdefault(li, [x, x]); self.spans[li][1] = x + f.getlength(wd)
                    x += f.getlength(wd + " "); k_idx += 1
            self.spans.setdefault(li, None)
        self.line_y = [pady + size * 0.98 + li * lh for li in range(len(lines))]
        self.x = (W - cw) / 2

    def draw(self, frame, tt):
        if tt < self.t_in or tt > self.t_out + 0.35: return
        age = tt - self.t_in; out = ease((tt - self.t_out) / 0.3) if tt > self.t_out else 0
        cv = self.bg.copy()
        # hand-drawn deco on emphasis
        if self.deco:
            kind, lis, td = self.deco
            for n_, li in enumerate(lis):
              p = ease_io((tt - td - 0.3 * n_) / 0.45)
              if p > 0 and self.spans.get(li):
                  x0, x1 = self.spans[li]; yb = self.line_y[li]
                  ov = np.zeros((self.h, self.w), np.uint8)
                  if kind == "underline":
                      pts = [(int(x0 + (x1 - x0) * u), int(yb + 14 + 5 * np.sin(u * 9))) for u in np.linspace(0, p, 40)]
                      cv2.polylines(ov, [np.int32(pts)], False, 255, 7, cv2.LINE_AA)
                  else:
                      cx, cy = (x0 + x1) / 2, yb - self.size * 0.36
                      rx, ry = (x1 - x0) / 2 + 26, self.size * 0.78
                      pts = [(int(cx + rx * np.cos(a)), int(cy + ry * np.sin(a))) for a in np.linspace(-2.6, -2.6 + 2 * np.pi * 1.08 * p, 60)]
                      cv2.polylines(ov, [np.int32(pts)], False, 255, 6, cv2.LINE_AA)
                  m = ov.astype(np.float32)[..., None] / 255
                  col = np.float32(self.theme["deco"]) / 255
                  cv[..., :3] = cv[..., :3] * (1 - m) + col * m; cv[..., 3:4] = np.maximum(cv[..., 3:4], m)
        for wd in self.words:
            a = tt - wd["t0"]
            if a < 0: continue
            e = ease(a / 0.22); img = wd["img"]
            if e < 0.99: img = cv2.blur(img, (int(24 * (1 - e)) | 1, 1))
            blend(cv, img, wd["x"] + 22 * (1 - e), wd["y"], min(1, a / 0.12))
        sc = 0.9 + 0.1 * back(age / 0.45); al = min(1, age / 0.15) * (1 - out)
        dy = 50 * (1 - ease(age / 0.4)) - 30 * out
        dx = 0
        if self.shake and 0 < age < 0.5: dx = 14 * np.sin(age * 60) * (1 - age / 0.5)
        img = cv
        if abs(sc - 1) > 0.002:
            img = cv2.resize(cv, (int(self.w * sc), int(self.h * sc)))
        ox = self.x + (self.w - img.shape[1]) / 2 + dx; oy = self.top + (self.h - img.shape[0]) / 2 + dy
        blend(frame, self.shadow, ox - 40, oy - 40 + 14, al)
        blend(frame, img, ox, oy, al)

CARD_TOP = 985
cards = [
    Card([[("Можно очень любить", "n")], [("своего ребёнка", "n")], [("и заебаться", "e")], [("от материнства", "e")]],
         DARK, 0.10, 7.0, CARD_TOP - 40, line_times=[0.3, 0.75, 1.3, 1.7], size=66, deco=("underline", [2, 3], 2.2)),
    Card([[("Пытаюсь быть", "n")], [("не только мамой.", "n")], [("Пока получается", "e")], [("так себе", "e")]],
         LIGHT, wt(35) - 0.15, wt(45) + 0.95, CARD_TOP, line_times=[wt(35), wt(37), wt(41), wt(44)], size=60),
    Card([[("А что на самом деле", "n")], [("происходит", "n")], [("ЗА", "e"), ("этими", "n")], [("фотографиями?", "n")]],
         LIGHT, wt(74) - 0.15, wt(94) + 0.4, CARD_TOP, line_times=[wt(74), wt(76), wt(77), wt(78)], size=62, deco=("circle", [2], wt(78))),
    Card([[("И в какой-то момент", "n")], [("начинаешь думать:", "n")], [("«А почему у меня", "q")], [("не так?»", "q")]],
         DARK, wt(117) - 0.15, wt(135) + 0.12, CARD_TOP, line_times=[wt(117), wt(119), wt(126) - 0.2, wt(126) + 0.3], size=60),
    Card([[("Pinterest:", "l"), ("эстетичное", "n")], [("материнство", "n")]],
         LIGHT, wt(136) + 0.2, wt(157) + 0.45, CARD_TOP, line_times=[wt(136) + 0.35, wt(136) + 0.7], size=56),
]
cards.append(Card([[("Я:", "l"), ("у меня горят", "n")], [("вареники", "e")]],
         DARK, wt(145) - 0.1, wt(157) + 0.45, CARD_TOP + cards[-1].h + 18, line_times=[wt(144), wt(146)], size=56, shake=True))

# ---------------- camera ----------------
ZOOM_IN, ZOOM_OUT = wt(144), wt(158) - 0.6   # "меня горят вареники" ... before next section
FOCUS = (540, 560)
def camera(tt):
    sec_start = max([0] + [j for j in JOINS if j <= tt]); nxt = min([j for j in JOINS if j > tt] + [DUR])
    z = 1.0 + 0.045 * (tt - sec_start) / max(1, nxt - sec_start)
    since = tt - sec_start
    if since < 0.4 and sec_start > 0: z *= 1 + 0.09 * (1 - ease(since / 0.4))     # punch-in on cut
    p = ease_io((tt - ZOOM_IN) / 0.7) - ease_io((tt - ZOOM_OUT) / 0.5)
    z *= 1 + 0.2 * p
    return z, p

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.42) / (H * 0.62)) ** 2)
VIG = np.clip((r - 0.55) / 0.6, 0, 1) ** 1.6 * 0.55
VIG3 = VIG[..., None]
VIGCOL = np.float32(INK) / 255
LEAK = np.exp(-(((xx - W * 0.85) / 520) ** 2 + ((yy - H * 0.2) / 700) ** 2))[..., None] * (np.float32(BLUSH) / 255)
rng = np.random.default_rng(3)
GRAIN = [(rng.standard_normal((H // 2, W // 2)).astype(np.float32) * 0.018) for _ in range(6)]
GRAIN = [cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] for g in GRAIN]

yb = np.arange(H, dtype=np.float32)
BAND = (np.exp(-((yb - (SUB_Y - 20)) / 150) ** 2) * 0.42)[:, None, None]   # soft ink band under subtitles

def grade(fr, tt, fi):
    fr = fr * (1 - VIG3) + VIGCOL * VIG3
    fr = fr * (1 - BAND) + VIGCOL * BAND
    for j in [0.0] + JOINS:
        d = tt - j
        if 0 <= d < 0.55:
            fr = fr + LEAK * 0.45 * (1 - ease(d / 0.55))
    fr += GRAIN[fi % 6]
    return fr

def progress(fr, tt):
    p = tt / DUR; y0 = 26
    fr[y0:y0 + 6, 60:W - 60] = fr[y0:y0 + 6, 60:W - 60] * 0.55 + np.float32(MIST) / 255 * 0.45
    x1 = int(60 + (W - 120) * p)
    fr[y0:y0 + 6, 60:x1] = np.float32(ROSE) / 255

dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "cut.mov", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
enc = None if TEST else subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "video_only.mp4"], stdin=subprocess.PIPE)
fi = 0
while True:
    buf = dec.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    tt = fi / FPS
    if tt > LIMIT: break
    if TEST and not any(abs(tt - x) < 0.02 for x in TEST): fi += 1; continue
    src = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    z, zp = camera(tt)
    fx, fy = W / 2 + (FOCUS[0] - W / 2) * zp, H / 2 + (FOCUS[1] - H / 2) * zp
    M = np.float32([[z, 0, fx - z * fx], [0, z, fy - z * fy]])
    fr = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255
    fr = grade(fr, tt, fi)
    for c in cards: c.draw(fr, tt)
    draw_subs(fr, tt)
    progress(fr, tt)
    tail = DUR - tt
    if tail < 0.5: fr = fr * (tail / 0.5) + np.float32(INK) / 255 * (1 - tail / 0.5)
    if TEST: cv2.imwrite(f"test_{tt:.2f}.png", cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    else: enc.stdin.write((np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    fi += 1
dec.kill()
if TEST: sys.exit()
enc.stdin.close(); enc.wait()

# ---------------- sound design ----------------
SR = 48000
rng = np.random.default_rng(7)
def onepole(x, cut):
    y = np.empty_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * np.asarray(cut) / SR) * np.ones(len(x))
    for i in range(len(x)): s += a[i] * (x[i] - s); y[i] = s
    return y
def norm(y): return y / (np.abs(y).max() + 1e-9)
def whoosh(d=0.5, lo=300, hi=5000, peak=0.6):
    n = int(d * SR); t = np.linspace(0, 1, n)
    env = np.where(t < peak, (t / peak) ** 2, ((1 - t) / (1 - peak)) ** 1.5)
    cut = lo + (hi - lo) * np.sin(np.pi * np.clip(t / (peak * 2), 0, 1)) ** 2
    x = rng.standard_normal(n); y = onepole(x, cut) - onepole(x, cut * 0.25)
    return norm(y * env)
def pop():
    n = int(0.09 * SR); t = np.arange(n) / SR
    f = 280 + 650 * np.exp(-t * 55); return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 42))
def tick():
    n = int(0.03 * SR); t = np.arange(n) / SR
    return norm(np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 260) + 0.3 * rng.standard_normal(n) * np.exp(-t * 600))
def boom():
    n = int(1.2 * SR); t = np.arange(n) / SR
    f = 42 + 70 * np.exp(-t * 18)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2) * np.minimum(1, t / 0.004)
    return norm(y + 0.15 * onepole(rng.standard_normal(n), 900) * np.exp(-t * 9))
def chime():
    n = int(1.8 * SR); t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * k) for f, a, k in [(1046.5, 1, 2.6), (1568, .5, 3.4), (2093, .25, 5), (523.25, .35, 2)])
    return norm(y * np.minimum(1, t / 0.005))
def scribble(d=0.45):
    n = int(d * SR); t = np.arange(n) / SR
    x = onepole(rng.standard_normal(n), 3500) - onepole(rng.standard_normal(n), 600)
    am = 0.55 + 0.45 * np.sin(2 * np.pi * 11 * t) ** 2
    return norm(x * am * np.sin(np.pi * t / d) ** 0.6)
def sizzle(d=2.2):
    n = int(d * SR); t = np.arange(n) / SR
    x = rng.standard_normal(n); x = x - onepole(x, 2500)
    crack = (rng.random(n) < 0.0009).astype(float) * rng.uniform(0.5, 1.5, n)
    crack = np.convolve(crack, np.exp(-np.arange(200) / 25), "same")
    env = np.minimum(1, t / 0.15) * np.minimum(1, (d - t) / 0.6)
    return norm((x * 0.5 * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t)) + crack) * env)
def riser(d=0.7):
    n = int(d * SR); t = np.linspace(0, 1, n)
    y = whoosh(d, 400, 9000, 0.92); tone = np.sin(2 * np.pi * np.cumsum(300 + 500 * t ** 2) / SR)
    return norm(y + 0.25 * tone * t ** 2)

trk = np.zeros(int((DUR + 3) * SR))
def put(sig, t, g):
    i = int(max(0, t) * SR); seg = trk[i:i + len(sig)]; seg += sig[:len(seg)] * g

WB, WS, POP, TICK = whoosh(0.6), whoosh(0.32, 600, 7000, 0.5), pop(), tick()
put(boom(), 0.08, 0.42); put(WS, 0.05, 0.16)
for j in JOINS: put(WB, j - 0.36, 0.26)
for c in cards:
    put(WS, c.t_in - 0.08, 0.15); put(POP, c.t_in + 0.12, 0.22)
    for k, wd in enumerate(c.words):
        put(TICK, wd["t0"], 0.05 * rng.uniform(0.7, 1))
    put(whoosh(0.3, 500, 4000, 0.3), c.t_out, 0.08)
    if c.deco: put(scribble(), c.deco[2], 0.12)
put(riser(), ZOOM_IN - 0.45, 0.17)
put(sizzle(), wt(145), 0.13)
put(chime(), wt(173) - 0.1, 0.09)     # «улыбается»
put(chime(), DUR - 0.55, 0.07)
trk = trk[:int((DUR + 0.04) * SR)]
with wave.open("sfx.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(trk, -1, 1) * 32767).astype(np.int16).tobytes())
print("DUR", round(DUR, 2), "JOINS", [round(j, 2) for j in JOINS], "frames", fi)
