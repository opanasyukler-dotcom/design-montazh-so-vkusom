"""Reels edit v3 — v2 reference style, rendered at 4K (2160x3840) 50 fps, with user materials.
Work dir needs: cut4k.mov (trimmed 4K edit, 25 fps), cwords.json, fonts (FONT_*), mat/ (photos + icons).
All layout numbers are in 1080x1920 design units and scaled by K at draw time.
Usage:  python3 mama_edit_v3.py render <start_frame> <end_frame> <out.mp4>
        python3 mama_edit_v3.py sfx            -> sfx.wav
        TEST=1.5,20 python3 mama_edit_v3.py test   -> test_*.png (downscaled previews)
Text never goes below y≈1440 (design) so Instagram's caption/buttons don't cover it.
"""
import json, os, subprocess, sys, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920            # design units
K = 2                         # 4K output
OW, OH = W * K, H * K
SRC_FPS, FPS = 25, 50
FONT_M, FONT_MI, FONT_S = "Montserrat-VariableFont_wght.ttf", "Montserrat-Italic-VariableFont_wght.ttf", "GreatVibes-Regular.ttf"
TEST = [float(x) for x in os.environ.get("TEST", "").split(",") if x]
MAT = "mat/"
PHOTOS = ["IMG_6878.JPG", "IMG_6879.JPG", "IMG_6880.JPG", "IMG_6881.JPG"]
ICON_IG, ICON_PIN = "1DD39ED4-C755-48D6-B799-1AF0239AD715.png", "D9C1F2FC-9B96-40D4-9DA7-CA2C1B644E64.png"

def hx(s): return np.float32([int(s[i:i + 2], 16) for i in (1, 3, 5)]) / 255
INK, COCOA, ROSE, MIST, BLUSH = hx("#2F0600"), hx("#825D4D"), hx("#C5A29C"), hx("#C5B0AD"), hx("#F7DFDD")
WHITE = np.float32([1, 1, 1])

def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def ease_io(p): p = min(max(p, 0), 1); return p * p * (3 - 2 * p)
def back(p, c=1.6): p = min(max(p, 0), 1); return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2
def lerp(a, b, e): return a + (b - a) * e
def P(v): return int(round(v * K))

def blend(dst, img, x, y, alpha=1.0):
    """premultiplied RGBA img (output px) over dst at design position x,y"""
    if alpha <= 0.003: return
    h, w = img.shape[:2]; x, y = P(x), P(y)
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0: return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]; d = dst[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4] * alpha
    d += (s if dst.shape[2] == 4 else s[..., :3]) * alpha

def premul_rgba(arr):
    a = arr.astype(np.float32) / 255; a[..., :3] *= a[..., 3:4]; return a

def soft_shadow(alpha, sigma, strength, margin):
    """blurred shadow (computed at 1/4 res for speed); alpha in output px; returns RGBA with margin (output px)"""
    h, w = alpha.shape
    sm = np.zeros((h + 2 * margin, w + 2 * margin), np.float32); sm[margin:margin + h, margin:margin + w] = alpha
    q = cv2.resize(sm, (max(2, sm.shape[1] // 4), max(2, sm.shape[0] // 4)), interpolation=cv2.INTER_AREA)
    q = cv2.GaussianBlur(q, (0, 0), max(0.5, sigma / 4))
    sm = cv2.resize(q, (sm.shape[1], sm.shape[0])) * strength
    out = np.zeros(sm.shape + (4,), np.float32); out[..., 3] = sm; return out

# ---------------------------------------------------------------- text
_fc = {}
def font(kind, size, wght=400):
    k = (kind, size, wght)
    if k not in _fc:
        f = ImageFont.truetype({"m": FONT_M, "i": FONT_MI, "s": FONT_S}[kind], size)
        if kind != "s": f.set_variation_by_axes([wght])
        _fc[k] = f
    return _fc[k]

def raster(txt, f, col):
    pad = 10 * K
    asc, desc = f.getmetrics(); w = int(f.getlength(txt)) + 2 * pad + 20 * K
    im = Image.new("L", (w, asc + desc + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad, pad + asc), txt, font=f, fill=255, anchor="ls")
    a = np.asarray(im).astype(np.float32) / 255
    out = np.zeros(a.shape + (4,), np.float32); out[..., :3] = col * a[..., None]; out[..., 3] = a
    return out, (pad + asc) / K, pad / K, f.getlength(txt) / K

THEMES = {
    "dark": dict(t=WHITE, b=BLUSH, s=BLUSH, k=MIST, g=WHITE, glow=ROSE, shadow=True, hl=ROSE),
    "light": dict(t=INK, b=COCOA, s=COCOA, k=COCOA, g=ROSE, glow=None, shadow=False, hl=ROSE),
}
S = 42

def style_font(st, size):
    size = size * K
    if st == "t": return font("i", size, 520)
    if st == "b": return font("m", int(size * 1.45), 800)
    if st == "s": return font("s", int(size * 1.75))
    if st == "k": return font("i", int(size * 0.72), 350)
    if st == "g": return font("m", int(size * 2.3), 900)
    if st == "T": return font("m", size, 800)

class Word:
    def __init__(self, txt, st, theme, size):
        self.st = st; th = THEMES[theme]
        col = th[st] if st in th else (INK if theme == "light" else BLUSH)
        if st == "g": txt = txt.upper()
        self.img, self.base, self.pad, self.adv = raster(txt, style_font(st, size), col)
        self.alpha_mul = 0.33 if st == "g" else 1.0
        self.glow = None
        if st == "b" and th["glow"] is not None:
            g = cv2.GaussianBlur(self.img[..., 3], (0, 0), 9 * K)
            gl = np.zeros_like(self.img); gl[..., :3] = th["glow"] * g[..., None]; gl[..., 3] = g
            self.glow = gl * 0.9
        self.sh = None
        if th["shadow"] and st != "g":
            s_ = np.minimum(1, cv2.GaussianBlur(self.img[..., 3], (0, 0), 5 * K) * 0.75)
            self.sh = np.zeros_like(self.img); self.sh[..., 3] = s_
        self.space = style_font("t", size).getlength(" ") / K

class Group:
    def __init__(self, lines, theme, t_in, t_out, size=S, deco=(), vertical=False, stagger=0.07):
        self.t_in, self.t_out, self.vertical, self.theme = t_in, t_out, vertical, theme
        self.words = []
        for li, ln in enumerate(lines):
            ws = [Word(t, st, theme, ln.get("size", size)) for t, st in ln["words"]]
            xs, x = [], 0.0
            for k, w in enumerate(ws):
                xs.append(x); x += w.adv * (0.55 if w.st == "g" else 1) + (w.space if w.st != "g" else 0)
            total = x - (ws[-1].space if ws[-1].st != "g" else 0)
            al = ln.get("align", "l")
            x0 = ln["x"] - (total / 2 if al == "c" else total if al == "r" else 0)
            times = ln.get("t")
            for k, w in enumerate(ws):
                dy = S * 0.32 if (k and ws[k - 1].st == "g") else 0
                if w.st == "g": dy = -S * 0.25
                w.x = x0 + xs[k] - w.pad; w.y = ln["y"] + dy - w.base
                w.t0 = times[k] if isinstance(times, list) else (times if times is not None else t_in) + stagger * k
                w.line, w.idx = li, k
                self.words.append(w)
        self.deco = []
        for kind, li, a, b, td in deco:
            sel = [w for w in self.words if w.line == li and a <= w.idx <= b]
            x0 = min(w.x + w.pad for w in sel); x1 = max(w.x + w.pad + w.adv for w in sel)
            yb = max(w.y + w.base for w in sel); hgt = max(w.base - w.pad for w in sel)
            self.deco.append((kind, x0, x1, yb, hgt, td))

    def active(self, tt): return self.t_in - 0.01 <= tt <= self.t_out + 0.3

    def draw(self, dst, tt):
        out = ease((tt - self.t_out) / 0.28) if tt > self.t_out else 0
        th = THEMES[self.theme]; nch = dst.shape[2]
        for kind, x0, x1, yb, hgt, td in self.deco:
            p = ease_io((tt - td) / (0.3 if kind == "hl" else 0.5))
            if p <= 0: continue
            al = 1 - out
            if kind == "hl":
                bx0, bx1, by0, by1 = x0 - 8, x0 - 8 + (x1 - x0 + 16) * p, yb - hgt - 8, yb + 12
                sl = dst[P(by0):P(by1), P(bx0):P(bx1)]
                sl[..., :3] = sl[..., :3] * (1 - 0.55 * al) + th["hl"] * 0.55 * al
                if nch == 4: sl[..., 3] = np.maximum(sl[..., 3], 0.55 * al)
                if p >= 1:
                    col = (tuple(float(c) * al for c in COCOA) + (al,))[:nch]
                    for hx_, hy0, hy1, dyc in ((bx0, by0 - 14, by1, by0 - 14), (bx1, by0, by1 + 14, by1 + 14)):
                        cv2.line(dst, (P(hx_), P(hy0)), (P(hx_), P(hy1)), col, 3 * K, cv2.LINE_AA)
                        cv2.circle(dst, (P(hx_), P(dyc)), 8 * K, col, -1, cv2.LINE_AA)
            else:
                n = max(2, int(60 * p))
                pts = np.int32([(P(x0 + (x1 - x0) * u), P(yb + 14 + 6 * np.sin(u * (x1 - x0) / 22))) for u in np.linspace(0, p, n)])
                c = (tuple(float(v) * al for v in th["s"]) + (al,))[:nch]
                cv2.polylines(dst, [pts], False, c, 4 * K, cv2.LINE_AA)
        for w in self.words:
            age = tt - w.t0
            if age < 0: continue
            al = min(1, age / 0.16) * (1 - out) * w.alpha_mul
            img, glow, sh = w.img, w.glow, w.sh
            dx = dy = 0
            if w.st == "s":
                p = ease_io(age / 0.45)
                if p < 1:
                    ww = img.shape[1]; ramp = np.clip((np.arange(ww) - p * ww * 1.15) / (-0.15 * ww), 0, 1)[None, :, None].astype(np.float32)
                    img = img * ramp
                    if sh is not None: sh = sh * ramp
                al = (1 - out)
            elif w.st == "b":
                sc = 0.7 + 0.3 * back(age / 0.3)
                if abs(sc - 1) > 0.01:
                    hh, ww = img.shape[:2]; nw, nh = max(2, int(ww * sc)), max(2, int(hh * sc))
                    img = cv2.resize(img, (nw, nh)); dx, dy = (ww - nw) / 2 / K, (hh - nh) / 2 / K
                    glow = cv2.resize(glow, (nw, nh)) if glow is not None else None
                    sh = cv2.resize(sh, (nw, nh)) if sh is not None else None
            else:
                e = ease(age / 0.3); dx = 16 * (1 - e)
                k = int(18 * K * (1 - e)) | 1
                if k > 2:
                    img = cv2.blur(img, (k, 1)); sh = cv2.blur(sh, (k, 1)) if sh is not None else None
            if out > 0: dy -= 14 * out
            if sh is not None: blend(dst, sh, w.x + dx, w.y + dy + 3, al)
            if glow is not None: blend(dst, glow, w.x + dx, w.y + dy, al * (0.6 + 0.4 * np.sin(tt * 5) ** 2))
            blend(dst, img, w.x + dx, w.y + dy, al)

# ---------------------------------------------------------------- timeline
WORDS = json.load(open("cwords_4k.json"))
N_SRC = 2351
DUR = N_SRC / SRC_FPS
def wt(i): return WORDS[i][0]

FULL = dict(rect=(0, 0, W, H), rad=0, bg=INK, gray=0, blur=0, dim=0, zoom=1.0, fy=960)
def L(**k): d = dict(FULL); d.update(k); return d
LAY = {
    "full": FULL,
    "gray": L(gray=1),
    "framed": L(rect=(150, 400, 780, 1130), rad=30, bg=BLUSH, fy=700),
    "split": L(rect=(300, 0, 780, H), bg=BLUSH, fy=900),
    "collA": L(rect=(560, 690, 420, 600), rad=26, bg=BLUSH, fy=620),
    "blur": L(gray=1, blur=1, dim=0.4),
    "circle": L(rect=(310, 190, 460, 460), rad=230, bg=INK, fy=600),
    "collB": L(rect=(480, 690, 500, 740), rad=28, bg=BLUSH, fy=620),
    "zoom": L(zoom=1.16, fy=640),
}
SCENES = [
    (0.0, "gray", "hook"), (2.35, "full", "hook"), (7.0, "full", "full"),
    (14.75, "split", "split"), (19.70, "framed", "above"), (27.35, "collA", "collA"),
    (34.35, "full", "full"), (37.15, "blur", "low"), (44.35, "full", "full"),
    (48.25, "gray", "full"), (50.75, "full", "full"), (59.05, "circle", "ink"),
    (64.95, "full", "full"), (69.85, "collB", "collB"), (76.75, "zoom", "full"),
    (78.30, "framed", "above"), (84.45, "full", "full"),
]
TR = 0.45
def scene_at(t): return max(k for k, s in enumerate(SCENES) if s[0] <= t)
def layout(t):
    i = scene_at(t); cur = LAY[SCENES[i][1]]
    if i == 0: return dict(cur)
    prev = LAY[SCENES[i - 1][1]]; e = ease_io((t - SCENES[i][0]) / TR)
    return {k: (tuple(lerp(x, y, e) for x, y in zip(prev[k], cur[k])) if isinstance(cur[k], tuple) else lerp(prev[k], cur[k], e)) for k in cur}

# ---------------------------------------------------------------- subtitles
FIX = {"бесил,": "без сил,"}
toks = []
for s, e, w in WORDS:
    w = FIX.get(w.strip(), w.strip())
    if w in ("–", "-"): continue
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

ACC_B = {"сложно", "тяжело", "слабо", "pinterest", "instagram", "идеальное", "эстетичными", "скрывается", "картинка",
         "никогда", "загоняет", "вареники", "поесть", "огромная", "меняетесь", "улыбается", "без сил"}
ACC_S = {"люблю", "жизнь", "счастлив", "красиво", "прекрасно", "чувства", "материнство", "вдохновляет"}
def clean(w): return w.lower().strip(",.?!«»")
# Instagram safe zone: first baseline ≤ ~1390 (second line +50), right edge ≤ ~920 (action buttons)
ZONES = {
    "hook": ("dark", [(540, 1370, "c")]),
    "full": ("dark", [(100, 1250, "l"), (900, 1310, "r"), (540, 1370, "c")]),
    "split": ("dark", [(380, 1310, "l"), (900, 1360, "r")]),
    "above": ("light", [(540, 270, "c")]),
    "collA": ("light", [(80, 1330, "l"), (540, 1380, "c")]),
    "low": ("dark", [(540, 1340, "c")]),
    "ink": ("dark", [(540, 1380, "c")]),
    "collB": ("light", [(80, 1370, "l")]),
}
subs, zc = [], {}
for ci, ch in enumerate(chunks):
    zone = SCENES[scene_at(ch[0][0] + 0.05)][2]
    theme, spots = ZONES[zone]; n = zc.get(zone, 0); zc[zone] = n + 1
    x, y, al = spots[n % len(spots)]
    words = []
    for k, (s, e, w) in enumerate(ch):
        c = clean(w); st = "t"
        if c in ACC_B: st = "b"
        elif c in ACC_S: st = "s"
        elif c == "но" and k == 0: st = "g"
        words.append((w.rstrip(",.") if st != "g" else w.rstrip(","), st, s))
    if " ".join(w for w, _, _ in words) == "надолго ли":
        words = [("[надолго ли]", "k", ch[0][0])]
    txt = " ".join(w for w, _, _ in words)
    lines = [words]
    if len(txt) > 15 and len(words) > 1:
        cut = (len(words) + 1) // 2
        lines = [words[:cut], words[cut:]]
    L_ = []
    for li, ws in enumerate(lines):
        dx = (60 if al == "l" else -60 if al == "r" else 30) * li
        L_.append(dict(words=[(w, st) for w, st, _ in ws], x=x + dx, y=y + li * int(S * 1.2), align=al, t=[t0 for _, _, t0 in ws]))
    deco = []
    if "не так" in txt:
        for li, ws in enumerate(lines):
            names = [w for w, _, _ in ws]
            if "не" in names and "так" in names:
                a = names.index("не"); deco.append(("hl", li, a, a + 1, ws[a + 1][2] + 0.2))
    t_out = min(chunks[ci + 1][0][0] - 0.08 if ci + 1 < len(chunks) else DUR, ch[-1][1] + 0.7)
    subs.append(Group(L_, theme, ch[0][0] - 0.05, t_out, deco=deco))

# ---------------------------------------------------------------- inserts (user's text, verbatim)
ins = []
ins.append(Group([
    dict(words=[("Можно очень", "t"), ("любить", "s")], x=110, y=990, t=0.25, size=50),
    dict(words=[("своего ребёнка", "t")], x=170, y=1060, t=0.9, size=50),
    dict(words=[("и", "t"), ("заебаться", "b")], x=110, y=1165, t=[2.45, 2.55], size=56),
    dict(words=[("от материнства", "t")], x=900, y=1245, align="r", t=3.0, size=50),
], "dark", 0.2, 6.9, deco=[("wave", 3, 0, 0, 3.4)]))
ins.append(Group([
    dict(words=[("Пытаюсь быть", "t")], x=1700, y=110, align="r", t=wt(35), size=46),
    dict(words=[("не только мамой.", "T")], x=1700, y=205, align="r", t=wt(36) + 0.2, size=78),
], "light", wt(35) - 0.1, 19.6, vertical=True))
ins.append(Group([
    dict(words=[("Пока получается", "t")], x=380, y=1090, t=wt(41), size=48),
    dict(words=[("так себе", "b")], x=440, y=1180, t=wt(44), size=48),
], "dark", wt(41) - 0.1, 19.6, deco=[("hl", 1, 0, 1, wt(44) + 0.35)]))
ins.append(Group([
    dict(words=[("А что на самом деле", "t")], x=540, y=700, align="c", t=wt(74), size=50),
    dict(words=[("происходит", "s")], x=540, y=800, align="c", t=wt(75) + 0.15, size=50),
    dict(words=[("ЗА", "b")], x=540, y=960, align="c", t=wt(77), size=110),
    dict(words=[("этими фотографиями?", "t")], x=540, y=1050, align="c", t=wt(78), size=50),
], "dark", wt(74) - 0.1, 44.3, deco=[("wave", 3, 0, 1, wt(79) + 0.2)]))
ins.append(Group([
    dict(words=[("И в какой-то момент", "t")], x=540, y=820, align="c", t=wt(117), size=50),
    dict(words=[("начинаешь", "t"), ("думать:", "s")], x=540, y=920, align="c", t=wt(118) + 0.1, size=50),
    dict(words=[("«А почему у меня", "t")], x=540, y=1070, align="c", t=wt(124) - 0.1, size=56),
    dict(words=[("не так?»", "b")], x=540, y=1185, align="c", t=wt(126), size=56),
], "dark", wt(117) - 0.1, 64.8, deco=[("hl", 3, 0, 1, wt(127) + 0.1)]))
ins.append(Group([
    dict(words=[("Pinterest:", "T")], x=520, y=300, t=wt(142), size=44),
    dict(words=[("эстетичное", "s")], x=520, y=400, t=wt(142) + 0.3, size=46),
    dict(words=[("материнство", "s")], x=560, y=480, t=wt(142) + 0.6, size=46),
], "light", wt(142) - 0.1, 76.6))
ins.append(Group([
    dict(words=[("Я:", "T")], x=80, y=930, t=wt(144), size=48),
    dict(words=[("у меня", "t")], x=80, y=1010, t=wt(144) + 0.1, size=48),
    dict(words=[("горят", "t")], x=80, y=1080, t=wt(145), size=48),
    dict(words=[("вареники", "b")], x=60, y=1180, t=wt(146), size=40),
], "light", wt(144) - 0.1, 76.6, deco=[("hl", 3, 0, 0, wt(146) + 0.45)]))
GROUPS = ins + subs

# ---------------------------------------------------------------- pictures: photos, icons, tiles, paths
def load_rgb(path):
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255

def tile(img_f, w, h, fy=0.4, rad=22, border=10, rot=0.0):
    """img_f float RGB any size -> framed rounded tile at output res (+ shadow). w,h design"""
    ih, iw = img_f.shape[:2]; a = w / h
    cw, ch = iw, iw / a
    if ch > ih: ch = ih; cw = ch * a
    y0 = int(np.clip(fy * ih - ch / 2, 0, ih - ch)); x0 = int((iw - cw) / 2)
    pw, ph = P(w), P(h); b = P(border); r = P(rad)
    crop = cv2.resize(img_f[y0:y0 + int(ch), x0:x0 + int(cw)], (pw, ph), interpolation=cv2.INTER_AREA)
    tw, th = pw + 2 * b, ph + 2 * b
    m = np.zeros((th, tw), np.uint8); cv2.rectangle(m, (r, 0), (tw - r, th), 255, -1); cv2.rectangle(m, (0, r), (tw, th - r), 255, -1)
    for cx, cy in ((r, r), (tw - r, r), (r, th - r), (tw - r, th - r)): cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    out = np.zeros((th, tw, 4), np.float32); out[..., :3] = WHITE; out[b:b + ph, b:b + pw, :3] = crop
    out[..., 3] = m / 255; out[..., :3] *= out[..., 3:4]
    if rot:
        M = cv2.getRotationMatrix2D((tw / 2, th / 2), rot, 1); c, s = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(th * s + tw * c) + 2, int(th * c + tw * s) + 2; M[0, 2] += nw / 2 - tw / 2; M[1, 2] += nh / 2 - th / 2
        out = cv2.warpAffine(out, M, (nw, nh), flags=cv2.INTER_LINEAR)
    return out, soft_shadow(out[..., 3], 12 * K, 0.3, 30 * K)

class Sprite:
    """tile/sticker that pops in (back-ease), floats slightly, and fades up on exit. x,y = design top-left"""
    def __init__(self, img, sh, x, y, t_in, t_out, wobble=0.0):
        self.img, self.sh, self.x, self.y, self.t_in, self.t_out, self.wobble = img, sh, x, y, t_in, t_out, wobble
    def draw(self, dst, tt):
        if tt < self.t_in or tt > self.t_out + 0.35: return
        age = tt - self.t_in; out = ease((tt - self.t_out) / 0.3) if tt > self.t_out else 0
        sc = (0.6 + 0.4 * back(age / 0.4)) * (1 - 0.15 * out); al = min(1, age / 0.12) * (1 - out)
        img, sh = self.img, self.sh
        if abs(sc - 1) > 0.01:
            img = cv2.resize(img, (max(2, int(img.shape[1] * sc)), max(2, int(img.shape[0] * sc))))
            sh = cv2.resize(sh, (max(2, int(sh.shape[1] * sc)), max(2, int(sh.shape[0] * sc))))
        fl = self.wobble * np.sin((tt - self.t_in) * 2.2)
        cx = self.x + self.img.shape[1] / K / 2; cy = self.y + self.img.shape[0] / K / 2 - 40 * out + fl
        blend(dst, sh, cx - sh.shape[1] / K / 2, cy - sh.shape[0] / K / 2 + 10, al)
        blend(dst, img, cx - img.shape[1] / K / 2, cy - img.shape[0] / K / 2, al)

def icon(path, size, rot=0):
    im = Image.open(path).convert("RGBA").resize((P(size), P(size)), Image.LANCZOS)
    if rot: im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    a = premul_rgba(np.asarray(im)); return a, soft_shadow(a[..., 3], 10 * K, 0.35, 24 * K)

class Path:
    def __init__(self, pts, t0, t1, t_out, avatar=False):
        p0, p1, p2, p3 = [np.float32(p) for p in pts]
        u = np.linspace(0, 1, 400)[:, None]
        self.c = (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.c, axis=0), axis=1))]
        self.t0, self.t1, self.t_out, self.avatar = t0, t1, t_out, avatar
    def draw(self, dst, tt, src):
        if tt < self.t0 or tt > self.t_out + 0.3: return
        p = ease_io((tt - self.t0) / (self.t1 - self.t0)); al = 1 - (ease((tt - self.t_out) / 0.3) if tt > self.t_out else 0)
        n = int(p * (len(self.c) - 1)); col = tuple(float(v) for v in (ROSE * al + BLUSH * (1 - al)))
        for i in range(n):
            if int(self.s[i] / 16) % 2 == 0:
                cv2.line(dst, (P(self.c[i][0]), P(self.c[i][1])), (P(self.c[i + 1][0]), P(self.c[i + 1][1])), col, 4 * K, cv2.LINE_AA)
        if self.avatar and n > 0:
            cx, cy = self.c[n]; r = P(62); bd = P(6)
            face = cv2.resize(src[P(300):P(920), P(230):P(850)], (2 * r, 2 * r)).astype(np.float32) / 255
            sz = 2 * r + 2 * bd
            m = np.zeros((sz, sz), np.float32); cv2.circle(m, (sz // 2, sz // 2), r + bd, 1.0, -1, cv2.LINE_AA)
            av = np.zeros((sz, sz, 4), np.float32); av[..., :3] = INK * m[..., None]; av[..., 3] = m
            mi = np.zeros((2 * r, 2 * r), np.float32); cv2.circle(mi, (r, r), r - 2, 1.0, -1, cv2.LINE_AA)
            av[bd:-bd, bd:-bd, :3] = av[bd:-bd, bd:-bd, :3] * (1 - mi[..., None]) + face * mi[..., None]
            blend(dst, av, cx - sz / K / 2, cy - sz / K / 2, al)

PH = [load_rgb(MAT + p) for p in PHOTOS]
SPRITES = []
# collage of the user's "ideal motherhood" photos
for (img, x, y, w, h, rot, t) in [
        (PH[0], 50, 170, 300, 400, -4, wt(58)), (PH[1], 380, 120, 290, 390, 3, wt(59)),
        (PH[2], 700, 200, 300, 400, -2, wt(62)), (PH[3], 60, 700, 320, 430, 2, wt(63))]:
    SPRITES.append(Sprite(*tile(img, w, h, fy=0.45, rot=rot), x, y, t, 34.3, wobble=3))
# Pinterest vs me: kitchen-with-baby photo + pin badge
SPRITES.append(Sprite(*tile(PH[3], 340, 470, fy=0.45, rot=-3), 90, 230, wt(142), 76.6, wobble=2))
SPRITES.append(Sprite(*icon(MAT + ICON_PIN, 110, rot=-8), 370, 190, wt(142) + 0.25, 76.6))
# stickers on «Pinterest», «Instagram»
SPRITES.append(Sprite(*icon(MAT + ICON_PIN, 170, rot=-10), 60, 330, wt(48), 27.2, wobble=6))
SPRITES.append(Sprite(*icon(MAT + ICON_IG, 180, rot=9), 840, 460, wt(49), 27.2, wobble=6))
PATHS = [
    Path([(200, 600), (230, 760), (420, 640), (560, 760)], wt(58) + 0.3, 33.6, 34.3, avatar=True),
    Path([(300, 740), (330, 940), (420, 900), (490, 1060)], wt(144), wt(146), 76.6),
]

def rr_mask(w, h, rad):
    m = np.zeros((h, w), np.uint8); rad = int(min(rad, w / 2, h / 2))
    if rad < 1: return None
    cv2.rectangle(m, (rad, 0), (w - rad - 1, h - 1), 255, -1); cv2.rectangle(m, (0, rad), (w - 1, h - rad - 1), 255, -1)
    for cx, cy in ((rad, rad), (w - rad - 1, rad), (rad, h - rad - 1), (w - rad - 1, h - rad - 1)):
        cv2.circle(m, (cx, cy), rad, 255, -1, cv2.LINE_AA)
    return m.astype(np.float32) / 255

def make_ui():
    s = K
    pl = Image.new("RGBA", (220 * s, 220 * s), (0, 0, 0, 0)); d = ImageDraw.Draw(pl)
    d.ellipse((4 * s, 4 * s, 216 * s, 216 * s), fill=(255, 255, 255, 70), outline=(255, 255, 255, 200), width=4 * s)
    d.polygon([(85 * s, 62 * s), (85 * s, 158 * s), (165 * s, 110 * s)], fill=(255, 255, 255, 235))
    cu = Image.new("RGBA", (90 * s, 120 * s), (0, 0, 0, 0)); d = ImageDraw.Draw(cu)
    pts = [(x * s, y * s) for x, y in [(6, 4), (6, 92), (28, 72), (44, 108), (58, 102), (42, 66), (72, 66)]]
    d.polygon(pts, fill=(255, 255, 255, 255)); d.line(pts + [pts[0]], fill=(47, 6, 0, 255), width=3 * s)
    return premul_rgba(np.asarray(pl)), premul_rgba(np.asarray(cu))
PLAY, CURSOR = make_ui()
CLICK = 2.3
def draw_ui(dst, tt):
    if tt > CLICK + 0.5: return
    al = 1 - ease((tt - CLICK) / 0.35) if tt > CLICK else min(1, tt / 0.2)
    sc = 1 - 0.12 * np.exp(-((tt - CLICK) / 0.08) ** 2)
    pl = cv2.resize(PLAY, (int(PLAY.shape[1] * sc), int(PLAY.shape[0] * sc)))
    blend(dst, pl, 540 - pl.shape[1] / K / 2, 760 - pl.shape[0] / K / 2, al)
    p = ease_io((tt - 0.6) / 1.4); cx, cy = lerp(860, 560, p), lerp(1100, 790, p)
    cs = 1 - 0.18 * np.exp(-((tt - CLICK) / 0.07) ** 2)
    cu = cv2.resize(CURSOR, (int(CURSOR.shape[1] * cs), int(CURSOR.shape[0] * cs)))
    blend(dst, cu, cx, cy, al)

# ---------------------------------------------------------------- frame compose
yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32) / K
r_ = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.42) / (H * 0.62)) ** 2)
VIG = (np.clip((r_ - 0.55) / 0.6, 0, 1) ** 1.6 * 0.5)[..., None].astype(np.float32)
VIG_INK = (VIG * INK).astype(np.float32); VIG_KEEP = (1 - VIG).astype(np.float32)
del yy, xx, r_
rng = np.random.default_rng(3)
GRAIN = [cv2.resize(rng.standard_normal((OH // 2, OW // 2)).astype(np.float32) * 0.014, (OW, OH))[..., None] for _ in range(4)]
VAR_ZOOM = wt(145)

def compose(src, tt):
    lay = layout(tt)
    fr = np.empty((OH, OW, 3), np.float32); fr[:] = lay["bg"]
    x, y, w, h = lay["rect"]; wi, hi = max(2, P(w)), max(2, P(h))
    zoom = lay["zoom"] * (1 + 0.035 * ((tt - SCENES[scene_at(tt)][0]) / 8))
    if SCENES[scene_at(tt)][1] == "collB": zoom *= 1 + 0.22 * ease_io((tt - VAR_ZOOM) / 0.6)
    a = w / h; cw = OW / zoom; ch = cw / a
    if ch > OH / zoom: ch = OH / zoom; cw = ch * a
    cy0 = np.clip(lay["fy"] * K - ch / 2, 0, OH - ch); cx0 = (OW - cw) / 2
    M = np.float32([[wi / cw, 0, -cx0 * wi / cw], [0, hi / ch, -cy0 * hi / ch]])
    vid = cv2.warpAffine(src, M, (wi, hi), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if lay["blur"] > 0.02:
        q = cv2.resize(vid, (wi // 4, hi // 4), interpolation=cv2.INTER_AREA)
        q = cv2.GaussianBlur(q, (0, 0), (1 + 20 * lay["blur"]) * K / 4)
        vid = cv2.resize(q, (wi, hi), interpolation=cv2.INTER_LINEAR)
    vid = vid.astype(np.float32) * np.float32(1 / 255)
    if lay["gray"] > 0.01:
        g = vid @ np.float32([0.299, 0.587, 0.114]); vid *= 1 - lay["gray"]; vid += g[..., None] * lay["gray"]
    if lay["dim"] > 0: vid *= 1 - lay["dim"]
    full = w > W - 2 and h > H - 2
    if full:
        fr[:] = vid[:OH, :OW]; fr *= VIG_KEEP; fr += VIG_INK
    else:
        m = rr_mask(wi, hi, P(lay["rad"]))
        blend(fr, soft_shadow(m if m is not None else np.ones((hi, wi), np.float32), 18 * K, 0.35, 40 * K), x - 40, y - 40 + 16)
        if m is None: m = np.ones((hi, wi), np.float32)
        v4 = np.concatenate([vid * m[..., None], m[..., None]], 2)
        blend(fr, v4, x, y)
    for st, _, _ in SCENES[1:]:
        d = tt - st
        if 0 <= d < 0.3: fr += BLUSH * np.float32(0.25 * (1 - d / 0.3))
    return fr

def render_frame(src, tt, fo):
    fr = compose(src, tt)
    for sp in SPRITES: sp.draw(fr, tt)
    for pth in PATHS: pth.draw(fr, tt, src)
    draw_ui(fr, tt)
    for g in GROUPS:
        if not g.active(tt): continue
        if g.vertical:
            cv = np.zeros((P(320), OH, 4), np.float32); g.draw(cv, tt)
            blend(fr, np.ascontiguousarray(np.rot90(cv, 1)), 0, 0)
        else: g.draw(fr, tt)
    fr += GRAIN[fo % 4]
    tail = DUR - tt
    if tail < 0.5: fr = fr * np.float32(tail / 0.5) + INK * np.float32(1 - tail / 0.5)
    np.clip(fr, 0, 1, out=fr)
    return (fr * 255 + 0.5).astype(np.uint8)

def decoder(start_src):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{start_src / SRC_FPS:.3f}", "-i", "cut4k.mov", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE, bufsize=OW * OH * 3)

def sfx():
    SR = 48000; rng = np.random.default_rng(7)
    def onepole(x, cut):
        y = np.empty_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * np.asarray(cut) / SR) * np.ones(len(x))
        for i in range(len(x)): s += a[i] * (x[i] - s); y[i] = s
        return y
    def norm(y): return y / (np.abs(y).max() + 1e-9)
    def whoosh(d=0.5, lo=300, hi=5000, peak=0.6):
        n = int(d * SR); t = np.linspace(0, 1, n)
        env = np.where(t < peak, (t / peak) ** 2, ((1 - t) / (1 - peak)) ** 1.5)
        cut = lo + (hi - lo) * np.sin(np.pi * np.clip(t / (peak * 2), 0, 1)) ** 2
        x = rng.standard_normal(n); return norm((onepole(x, cut) - onepole(x, cut * 0.25)) * env)
    def pop():
        n = int(0.09 * SR); t = np.arange(n) / SR; f = 280 + 650 * np.exp(-t * 55)
        return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 42))
    def bubble():
        n = int(0.14 * SR); t = np.arange(n) / SR; f = 500 + 900 * t / 0.14
        return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / 0.14) ** 2)
    def tick():
        n = int(0.03 * SR); t = np.arange(n) / SR
        return norm(np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 260) + 0.3 * rng.standard_normal(n) * np.exp(-t * 600))
    def mclick():
        n = int(0.05 * SR); t = np.arange(n) / SR; m = int(0.025 * SR)
        return norm(rng.standard_normal(n) * np.exp(-t * 400) + 0.6 * np.sin(2 * np.pi * 3200 * t) * np.exp(-t * 300)
                    + 0.5 * np.r_[np.zeros(m), rng.standard_normal(n - m) * np.exp(-np.arange(n - m) / SR * 500)])
    def shutter():
        n = int(0.22 * SR); t = np.arange(n) / SR; x = rng.standard_normal(n)
        return norm(onepole(x * (np.exp(-t * 80) + 0.8 * np.exp(-np.maximum(0, t - 0.09) * 70) * (t > 0.09)), 6000))
    def chime():
        n = int(1.8 * SR); t = np.arange(n) / SR
        y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * k) for f, a, k in [(1046.5, 1, 2.6), (1568, .5, 3.4), (2093, .25, 5), (523.25, .35, 2)])
        return norm(y * np.minimum(1, t / 0.005))
    def scribble(d=0.45):
        n = int(d * SR); t = np.arange(n) / SR
        x = onepole(rng.standard_normal(n), 3500) - onepole(rng.standard_normal(n), 600)
        return norm(x * (0.55 + 0.45 * np.sin(2 * np.pi * 11 * t) ** 2) * np.sin(np.pi * t / d) ** 0.6)
    def sizzle(d=2.0):
        n = int(d * SR); t = np.arange(n) / SR; x = rng.standard_normal(n); x = x - onepole(x, 2500)
        crack = np.convolve((rng.random(n) < 0.0009) * rng.uniform(0.5, 1.5, n), np.exp(-np.arange(200) / 25), "same")
        return norm((x * 0.5 + crack) * np.minimum(1, t / 0.15) * np.minimum(1, (d - t) / 0.6))
    def boom():
        n = int(1.0 * SR); t = np.arange(n) / SR; f = 42 + 70 * np.exp(-t * 18)
        return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.5) * np.minimum(1, t / 0.004))
    trk = np.zeros(int((DUR + 3) * SR))
    def put(sig, t, g):
        i = int(max(0, t) * SR); seg = trk[i:i + len(sig)]; seg += sig[:len(seg)] * g
    WB, WS, POP, TICK, SCR = whoosh(0.55), whoosh(0.3, 600, 7000, 0.5), pop(), tick(), scribble()
    put(mclick(), CLICK, 0.35); put(boom(), CLICK + 0.02, 0.32); put(WS, CLICK - 0.1, 0.14)
    for st, _, _ in SCENES[1:]:
        if abs(st - CLICK) > 0.1: put(WB, st - 0.3, 0.2)
    for g in GROUPS:
        for wd in g.words:
            if wd.st == "b": put(POP, wd.t0, 0.16)
            elif wd.st == "s": put(SCR, wd.t0, 0.05)
            elif g in ins: put(TICK, wd.t0, 0.035)
        for d in g.deco: put(SCR, d[5], 0.1)
    for sp in SPRITES:
        is_icon = sp.img.shape[0] < P(260)
        put(bubble() if is_icon else shutter(), sp.t_in, 0.2 if is_icon else 0.22)
    put(sizzle(), wt(145), 0.12)
    put(chime(), wt(173) - 0.1, 0.08); put(chime(), DUR - 0.6, 0.07)
    trk = trk[:int((DUR + 0.04) * SR)]
    with wave.open("sfx.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(trk, -1, 1) * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sfx": sfx(); sys.exit()
    if mode == "test":
        for t in TEST:
            fs = int(t * SRC_FPS); dec = decoder(fs)
            src = np.frombuffer(dec.stdout.read(OW * OH * 3), np.uint8).reshape(OH, OW, 3); dec.kill()
            o = render_frame(src, t, int(t * FPS))
            cv2.imwrite(f"test_{t:.2f}.png", cv2.cvtColor(cv2.resize(o, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        sys.exit()
    f0, f1, outp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]      # output-frame range [f0, f1)
    s0 = f0 // 2
    dec = decoder(s0)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-g", "100", "-pix_fmt", "yuv420p",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", outp], stdin=subprocess.PIPE)
    cur_s, src = s0 - 1, None
    for fo in range(f0, f1):
        need = min(fo // 2, N_SRC - 1)
        while cur_s < need:
            buf = dec.stdout.read(OW * OH * 3)
            if len(buf) < OW * OH * 3: break
            src = np.frombuffer(buf, np.uint8).reshape(OH, OW, 3); cur_s += 1
        enc.stdin.write(render_frame(src, fo / FPS, fo).tobytes())
        if fo % 100 == 0: print(outp, fo, flush=True)
    dec.kill(); enc.stdin.close(); enc.wait()
    print("done", outp)
