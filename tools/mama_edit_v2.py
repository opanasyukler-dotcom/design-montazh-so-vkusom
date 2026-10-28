"""Reels edit v2 — reference-style kinetic typography + layout inserts, brand palette.
Run in a work dir containing: cut.mov (trimmed 1080x1920 edit), cwords.json (word timings), fonts (see FONT_*).
Outputs: video_only.mp4, sfx.wav.   TEST=1.5,20 env renders only those frames to test_*.png
"""
import json, os, subprocess, sys, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 25
FONT_M, FONT_MI, FONT_S = "Montserrat-VariableFont_wght.ttf", "Montserrat-Italic-VariableFont_wght.ttf", "GreatVibes-Regular.ttf"
TEST = [float(x) for x in os.environ.get("TEST", "").split(",") if x]

def hx(s): return np.float32([int(s[i:i + 2], 16) for i in (1, 3, 5)]) / 255
INK, COCOA, ROSE, MIST, BLUSH = hx("#2F0600"), hx("#825D4D"), hx("#C5A29C"), hx("#C5B0AD"), hx("#F7DFDD")
WHITE = np.float32([1, 1, 1])

def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def ease_io(p): p = min(max(p, 0), 1); return p * p * (3 - 2 * p)
def back(p, c=1.6): p = min(max(p, 0), 1); return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2
def lerp(a, b, e): return a + (b - a) * e

def blend(dst, img, x, y, alpha=1.0):
    """premultiplied RGBA img over dst (RGB or RGBA)"""
    if alpha <= 0.003: return
    h, w = img.shape[:2]; x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0: return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]; d = dst[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4] * alpha
    d += (s if dst.shape[2] == 4 else s[..., :3]) * alpha

# ---------------------------------------------------------------- text tokens
_fc = {}
def font(kind, size, wght=400):
    k = (kind, size, wght)
    if k not in _fc:
        f = ImageFont.truetype({"m": FONT_M, "i": FONT_MI, "s": FONT_S}[kind], size)
        if kind != "s": f.set_variation_by_axes([wght])
        _fc[k] = f
    return _fc[k]

def raster(txt, f, col, pad=10):
    asc, desc = f.getmetrics(); w = int(f.getlength(txt)) + 2 * pad + 20
    im = Image.new("L", (w, asc + desc + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad, pad + asc), txt, font=f, fill=255, anchor="ls")
    a = np.asarray(im).astype(np.float32) / 255
    out = np.zeros(a.shape + (4,), np.float32); out[..., :3] = col * a[..., None]; out[..., 3] = a
    return out, pad + asc, pad, f.getlength(txt)

THEMES = {
    "dark": dict(t=WHITE, b=BLUSH, s=BLUSH, k=MIST, g=WHITE, glow=ROSE, shadow=True, hl=ROSE, dot=BLUSH),
    "light": dict(t=INK, b=COCOA, s=COCOA, k=COCOA, g=ROSE, glow=None, shadow=False, hl=ROSE, dot=COCOA),
}
S = 42  # base subtitle size (smaller than v1's 66)

def style_font(st, size):
    if st == "t": return font("i", size, 520)
    if st == "b": return font("m", int(size * 1.45), 800)
    if st == "s": return font("s", int(size * 1.75))
    if st == "k": return font("i", int(size * 0.72), 350)
    if st == "g": return font("m", int(size * 2.3), 900)
    if st == "T": return font("m", size, 800)          # heavy upright, for vertical / insert titles

class Word:
    def __init__(self, txt, st, theme, size):
        self.st = st; th = THEMES[theme]
        col = th[st] if st in th else th["t"] if st != "T" else (INK if theme == "light" else BLUSH)
        if st == "g": txt = txt.upper()
        f = style_font(st, size)
        self.img, self.base, self.pad, self.adv = raster(txt, f, col)
        self.alpha_mul = 0.33 if st == "g" else 1.0
        self.glow = None
        if st == "b" and th["glow"] is not None:
            g = cv2.GaussianBlur(self.img[..., 3], (0, 0), 9)
            gl = np.zeros_like(self.img); gl[..., :3] = th["glow"] * g[..., None]; gl[..., 3] = g
            self.glow = gl * 0.9
        self.sh = None
        if th["shadow"] and st != "g":
            s_ = np.minimum(1, cv2.GaussianBlur(self.img[..., 3], (0, 0), 5) * 0.75)
            self.sh = np.zeros_like(self.img); self.sh[..., 3] = s_
        self.space = style_font("t", size).getlength(" ")

class Group:
    """lines: [dict(words=[(txt, st)], x, y(baseline), align, t=[per-word times] or t0)]"""
    def __init__(self, lines, theme, t_in, t_out, size=S, deco=(), vertical=False, stagger=0.07):
        self.t_in, self.t_out, self.vertical, self.theme = t_in, t_out, vertical, theme
        self.words, self.boxes = [], {}
        for li, ln in enumerate(lines):
            ws = [Word(t, st, theme, ln.get("size", size)) for t, st in ln["words"]]
            gap = [w.space for w in ws]
            # ghost word overlaps following words
            xs, x = [], 0.0
            for k, w in enumerate(ws):
                xs.append(x)
                x += w.adv * (0.55 if w.st == "g" else 1) + (gap[k] if w.st != "g" else 0)
            total = x - (gap[-1] if ws[-1].st != "g" else 0)
            x0 = ln["x"] - (total / 2 if ln.get("align", "l") == "c" else total if ln.get("align") == "r" else 0)
            times = ln.get("t")
            for k, w in enumerate(ws):
                w.x = x0 + xs[k] - w.pad; dy = S * 0.32 if (k and ws[k - 1].st == "g") else 0
                if w.st == "g": dy = -S * 0.25
                w.y = ln["y"] + dy - w.base
                w.t0 = times[k] if isinstance(times, list) else (times if times is not None else t_in) + stagger * k
                w.line, w.idx = li, k
                self.words.append(w)
        self.deco = []
        for kind, li, a, b, td in deco:
            sel = [w for w in self.words if w.line == li and a <= w.idx <= b]
            x0 = min(w.x + w.pad for w in sel); x1 = max(w.x + w.pad + w.adv for w in sel)
            yb = max(w.y + w.base for w in sel); hgt = max(w.base - w.pad for w in sel)
            self.deco.append((kind, x0, x1, yb, hgt, td))

    def draw(self, dst, tt):
        if tt < self.t_in - 0.01 or tt > self.t_out + 0.3: return
        out = ease((tt - self.t_out) / 0.28) if tt > self.t_out else 0
        th = THEMES[self.theme]
        for kind, x0, x1, yb, hgt, td in self.deco:
            p = ease_io((tt - td) / (0.3 if kind == "hl" else 0.5))
            if p <= 0: continue
            al = 1 - out
            if kind == "hl":
                bx0, bx1, by0, by1 = x0 - 8, x0 - 8 + (x1 - x0 + 16) * p, yb - hgt - 8, yb + 12
                dst_slice = dst[int(by0):int(by1), int(bx0):int(bx1)]
                dst_slice[..., :3] = dst_slice[..., :3] * (1 - 0.55 * al) + th["hl"] * 0.55 * al
                if dst.shape[2] == 4: dst_slice[..., 3] = np.maximum(dst_slice[..., 3], 0.55 * al)
                if p >= 1:   # selection handles like a text-selection UI
                    col = tuple(float(c) for c in COCOA) + (1.0,)
                    for (hx_, hy0, hy1, dyc) in ((bx0, by0 - 14, by1, by0 - 14), (bx1, by0, by1 + 14, by1 + 14)):
                        cv2.line(dst, (int(hx_), int(hy0)), (int(hx_), int(hy1)), col[:dst.shape[2]], 3, cv2.LINE_AA)
                        cv2.circle(dst, (int(hx_), int(dyc)), 8, col[:dst.shape[2]], -1, cv2.LINE_AA)
            else:  # hand-drawn wave underline
                n = max(2, int(60 * p))
                pts = np.int32([(x0 + (x1 - x0) * u, yb + 14 + 6 * np.sin(u * (x1 - x0) / 22)) for u in np.linspace(0, p, n)])
                col = th["s"] if kind == "wave" else th["hl"]
                c = tuple(float(v) * al for v in col) + ((al,) if dst.shape[2] == 4 else ())
                cv2.polylines(dst, [pts], False, c, 4, cv2.LINE_AA)
        for w in self.words:
            age = tt - w.t0
            if age < 0: continue
            al = min(1, age / 0.16) * (1 - out) * w.alpha_mul
            img, glow, sh = w.img, w.glow, w.sh
            dx = dy = 0
            if w.st == "s":          # handwriting wipe
                p = ease_io(age / 0.45)
                if p < 1:
                    hh, ww = img.shape[:2]; ramp = np.clip((np.arange(ww) - p * ww * 1.15) / (-0.15 * ww), 0, 1)[None, :, None]
                    img = img * ramp
                    if sh is not None: sh = sh * ramp
                al = (1 - out)
            elif w.st == "b":
                e = back(age / 0.3); sc = 0.7 + 0.3 * e
                if abs(sc - 1) > 0.01:
                    hh, ww = img.shape[:2]; nw, nh = max(2, int(ww * sc)), max(2, int(hh * sc))
                    img = cv2.resize(img, (nw, nh)); dx, dy = (ww - nw) / 2, (hh - nh) / 2
                    glow = cv2.resize(glow, (nw, nh)) if glow is not None else None
                    sh = cv2.resize(sh, (nw, nh)) if sh is not None else None
            else:                    # soft blur-fade from the right
                e = ease(age / 0.3); dx = 16 * (1 - e)
                k = int(18 * (1 - e)) | 1
                if k > 2:
                    img = cv2.blur(img, (k, 1)); sh = cv2.blur(sh, (k, 1)) if sh is not None else None
            if out > 0:
                dy -= 14 * out
            if sh is not None: blend(dst, sh, w.x + dx, w.y + dy + 3, al)
            if glow is not None: blend(dst, glow, w.x + dx, w.y + dy, al * (0.6 + 0.4 * np.sin(tt * 5) ** 2))
            blend(dst, img, w.x + dx, w.y + dy, al)

# ---------------------------------------------------------------- timeline
WORDS = json.load(open("cwords.json"))
DUR = 2355 / FPS
def wt(i): return WORDS[i][0]
def we(i): return WORDS[i][1]

# layouts: rect (x,y,w,h), radius, bg, gray, blur, dim, zoom, focus-y for crop
FULL = dict(rect=(0, 0, W, H), rad=0, bg=INK, gray=0, blur=0, dim=0, zoom=1.0, fy=960)
def L(**k): d = dict(FULL); d.update(k); return d
LAY = {
    "full": FULL,
    "gray": L(gray=1),
    "framed": L(rect=(150, 380, 780, 1150), rad=30, bg=BLUSH, fy=700),
    "split": L(rect=(300, 0, 780, H), bg=BLUSH, fy=900),
    "collA": L(rect=(560, 1010, 430, 650), rad=26, bg=BLUSH, fy=620),
    "blur": L(gray=1, blur=1, dim=0.4),
    "circle": L(rect=(310, 190, 460, 460), rad=230, bg=INK, fy=600),
    "collB": L(rect=(470, 760, 530, 820), rad=28, bg=BLUSH, fy=620),
    "zoom": L(zoom=1.16, fy=640),
}
SCENES = [  # (start, layout, subtitle zone)
    (0.0, "gray", "hook"), (2.35, "full", "hook"), (7.0, "full", "full"),
    (14.75, "split", "split"), (19.70, "framed", "above"), (27.35, "collA", "collA"),
    (34.35, "full", "full"), (37.15, "blur", "low"), (44.35, "full", "full"),
    (48.25, "gray", "full"), (50.75, "full", "full"), (59.05, "circle", "ink"),
    (64.95, "full", "full"), (69.85, "collB", "collB"), (76.75, "zoom", "full"),
    (78.30, "framed", "above"), (84.45, "full", "full"),
]
TR = 0.45
def scene_at(t):
    i = max(k for k, s in enumerate(SCENES) if s[0] <= t); return i
def layout(t):
    i = scene_at(t); cur = LAY[SCENES[i][1]]
    if i == 0: return dict(cur), 1.0
    prev = LAY[SCENES[i - 1][1]]; e = ease_io((t - SCENES[i][0]) / TR)
    out = {}
    for k in cur:
        a, b = prev[k], cur[k]
        out[k] = tuple(lerp(x, y, e) for x, y in zip(a, b)) if isinstance(a, tuple) else lerp(a, b, e)
    return out, e

# ---------------------------------------------------------------- subtitles (kinetic, small)
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
ZONES = {   # theme, list of (x, y, align) cycled
    "hook": ("dark", [(540, 1600, "c")]),
    "full": ("dark", [(100, 1330, "l"), (980, 1400, "r"), (540, 1480, "c")]),
    "split": ("dark", [(380, 1520, "l"), (1020, 1600, "r")]),
    "above": ("light", [(540, 250, "c")]),
    "collA": ("light", [(100, 1790, "l"), (540, 1830, "c")]),
    "low": ("dark", [(540, 1560, "c")]),
    "ink": ("dark", [(540, 1640, "c")]),
    "collB": ("light", [(80, 1720, "l"), (540, 1790, "c")]),
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
    # two lines if long, second line indented (staggered, like the reference)
    txt = " ".join(w for w, _, _ in words)
    lines = [words]
    if len(txt) > 15 and len(words) > 1:
        cut = max(1, len(words) // 2 if len(words) > 2 else 1)
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
# 1 hook
ins.append(Group([
    dict(words=[("Можно очень", "t"), ("любить", "s")], x=110, y=1070, t=0.25, size=50),
    dict(words=[("своего ребёнка", "t")], x=170, y=1140, t=0.9, size=50),
    dict(words=[("и", "t"), ("заебаться", "b")], x=110, y=1250, t=[2.45, 2.55], size=56),
    dict(words=[("от материнства", "t")], x=980, y=1330, align="r", t=3.0, size=50),
], "dark", 0.2, 6.9, deco=[("wave", 3, 0, 0, 3.4)]))
# 2 split: vertical title in the blush strip + highlighted tail
ins.append(Group([
    dict(words=[("Пытаюсь быть", "t")], x=1820, y=110, align="r", t=wt(35), size=46),
    dict(words=[("не только мамой.", "T")], x=1820, y=205, align="r", t=wt(36) + 0.2, size=78),
], "light", wt(35) - 0.1, 19.6, vertical=True))
ins.append(Group([
    dict(words=[("Пока получается", "t")], x=380, y=1230, t=wt(41), size=48),
    dict(words=[("так себе", "b")], x=440, y=1320, t=wt(44), size=48),
], "dark", wt(41) - 0.1, 19.6, deco=[("hl", 1, 0, 1, wt(44) + 0.35)]))
# 3 blur scene
ins.append(Group([
    dict(words=[("А что на самом деле", "t")], x=540, y=700, align="c", t=wt(74), size=50),
    dict(words=[("происходит", "s")], x=540, y=800, align="c", t=wt(75) + 0.15, size=50),
    dict(words=[("ЗА", "b")], x=540, y=960, align="c", t=wt(77), size=110),
    dict(words=[("этими фотографиями?", "t")], x=540, y=1050, align="c", t=wt(78), size=50),
], "dark", wt(74) - 0.1, 44.3, deco=[("wave", 3, 0, 1, wt(79) + 0.2)]))
# 4 ink screen with circle video
ins.append(Group([
    dict(words=[("И в какой-то момент", "t")], x=540, y=860, align="c", t=wt(117), size=50),
    dict(words=[("начинаешь", "t"), ("думать:", "s")], x=540, y=960, align="c", t=wt(118) + 0.1, size=50),
    dict(words=[("«А почему у меня", "t")], x=540, y=1120, align="c", t=wt(124) - 0.1, size=56),
    dict(words=[("не так?»", "b")], x=540, y=1240, align="c", t=wt(126), size=56),
], "dark", wt(117) - 0.1, 64.8, deco=[("hl", 3, 0, 1, wt(127) + 0.1)]))
# 5 collage: Pinterest vs me
ins.append(Group([
    dict(words=[("Pinterest:", "T")], x=520, y=300, t=wt(142), size=44),
    dict(words=[("эстетичное", "s")], x=520, y=400, t=wt(142) + 0.3, size=46),
    dict(words=[("материнство", "s")], x=560, y=480, t=wt(142) + 0.6, size=46),
], "light", wt(142) - 0.1, 76.6))
ins.append(Group([
    dict(words=[("Я:", "T")], x=80, y=1000, t=wt(144), size=48),
    dict(words=[("у меня", "t")], x=80, y=1080, t=wt(144) + 0.1, size=48),
    dict(words=[("горят", "t")], x=80, y=1150, t=wt(145), size=48),
    dict(words=[("вареники", "b")], x=60, y=1250, t=wt(146), size=40),
], "light", wt(144) - 0.1, 76.6, deco=[("hl", 3, 0, 0, wt(146) + 0.45)]))
GROUPS = ins + subs

# ---------------------------------------------------------------- decorations
def grab(t):
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", "cut.mov", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True).stdout
    return np.frombuffer(p, np.uint8).reshape(H, W, 3)
def pinterest_look(img):
    f = img.astype(np.float32) / 255
    f = np.clip((f - 0.03) * 1.25 + 0.06, 0, 1) ** 0.8          # lift & brighten
    f = f * 0.82 + BLUSH * 0.18                                   # creamy blush wash
    glow = cv2.GaussianBlur(f, (0, 0), 18); f = 1 - (1 - f) * (1 - glow * 0.35)
    return f
def tile(img_f, w, h, fy=620, rad=22, border=10, rot=0.0):
    a = w / h; cw = W; ch = cw / a
    if ch > H: ch = H; cw = ch * a
    y0 = int(np.clip(fy - ch / 2, 0, H - ch)); x0 = int((W - cw) / 2)
    crop = cv2.resize(img_f[y0:y0 + int(ch), x0:x0 + int(cw)], (w, h), interpolation=cv2.INTER_AREA)
    tw, th = w + 2 * border, h + 2 * border
    m = np.zeros((th, tw), np.uint8); cv2.rectangle(m, (rad, 0), (tw - rad, th), 255, -1); cv2.rectangle(m, (0, rad), (tw, th - rad), 255, -1)
    for cx, cy in ((rad, rad), (tw - rad, rad), (rad, th - rad), (tw - rad, th - rad)): cv2.circle(m, (cx, cy), rad, 255, -1, cv2.LINE_AA)
    out = np.zeros((th, tw, 4), np.float32); out[..., :3] = WHITE; out[border:border + h, border:border + w, :3] = crop
    out[..., 3] = m / 255; out[..., :3] *= out[..., 3:4]
    if rot:
        M = cv2.getRotationMatrix2D((tw / 2, th / 2), rot, 1); c, s = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(th * s + tw * c) + 2, int(th * c + tw * s) + 2; M[0, 2] += nw / 2 - tw / 2; M[1, 2] += nh / 2 - th / 2
        out = cv2.warpAffine(out, M, (nw, nh), flags=cv2.INTER_LINEAR)
    sh = np.zeros((out.shape[0] + 60, out.shape[1] + 60, 4), np.float32)
    sh[30:-30, 30:-30, 3] = out[..., 3]; sh[..., 3] = cv2.GaussianBlur(sh[..., 3], (0, 0), 12) * 0.3
    return out, sh

class Tile:
    def __init__(self, img, x, y, w, h, t_in, t_out, rot=0):
        self.img, self.sh = tile(img, w, h, rot=rot); self.x, self.y, self.t_in, self.t_out = x, y, t_in, t_out
    def draw(self, dst, tt):
        if tt < self.t_in or tt > self.t_out + 0.35: return
        age = tt - self.t_in; out = ease((tt - self.t_out) / 0.3) if tt > self.t_out else 0
        sc = 0.6 + 0.4 * back(age / 0.4); al = min(1, age / 0.12) * (1 - out)
        img, sh = self.img, self.sh
        if abs(sc - 1) > 0.01:
            img = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)))
            sh = cv2.resize(sh, (int(sh.shape[1] * sc), int(sh.shape[0] * sc)))
        cx, cy = self.x + self.img.shape[1] / 2, self.y + self.img.shape[0] / 2 - 40 * out
        blend(dst, sh, cx - sh.shape[1] / 2, cy - sh.shape[0] / 2 + 10, al)
        blend(dst, img, cx - img.shape[1] / 2, cy - img.shape[0] / 2, al)

class Path:
    """dashed bezier drawn progressively; optional live avatar riding it"""
    def __init__(self, pts, t0, t1, t_out, avatar=False):
        p0, p1, p2, p3 = [np.float32(p) for p in pts]
        u = np.linspace(0, 1, 400)[:, None]
        self.c = (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.c, axis=0), axis=1))]; self.s = seg
        self.t0, self.t1, self.t_out, self.avatar = t0, t1, t_out, avatar
    def draw(self, dst, tt, src):
        if tt < self.t0 or tt > self.t_out + 0.3: return
        p = ease_io((tt - self.t0) / (self.t1 - self.t0)); al = 1 - (ease((tt - self.t_out) / 0.3) if tt > self.t_out else 0)
        n = int(p * (len(self.c) - 1))
        col = tuple(float(v) for v in (ROSE * al + BLUSH * (1 - al)))
        for i in range(0, n):
            if int(self.s[i] / 16) % 2 == 0:
                cv2.line(dst, tuple(np.int32(self.c[i])), tuple(np.int32(self.c[i + 1])), col, 4, cv2.LINE_AA)
        if self.avatar and n > 0:
            cx, cy = self.c[n]; r = 62
            face = src[300:300 + 620, 230:230 + 620]
            face = cv2.resize(face, (2 * r, 2 * r)).astype(np.float32) / 255
            m = np.zeros((2 * r + 12, 2 * r + 12), np.float32); cv2.circle(m, (r + 6, r + 6), r + 6, 1.0, -1, cv2.LINE_AA)
            av = np.zeros((2 * r + 12, 2 * r + 12, 4), np.float32); av[..., :3] = INK; av[..., 3] = m
            av[..., :3] *= m[..., None]
            mi = np.zeros((2 * r, 2 * r), np.float32); cv2.circle(mi, (r, r), r - 2, 1.0, -1, cv2.LINE_AA)
            av[6:-6, 6:-6, :3] = av[6:-6, 6:-6, :3] * (1 - mi[..., None]) + face * mi[..., None]
            blend(dst, av, cx - r - 6, cy - r - 6, al)

print("grabbing stills")
STILLS = {t: pinterest_look(grab(t)) for t in (1.2, 9.5, 31.0, 50.0, 56.0)}
TILES = [
    Tile(STILLS[1.2], 70, 150, 300, 420, wt(58), 34.3, rot=-4),
    Tile(STILLS[9.5], 400, 330, 300, 420, wt(62), 34.3, rot=3),
    Tile(STILLS[56.0], 80, 640, 300, 420, wt(63), 34.3, rot=-2),
    Tile(STILLS[50.0], 90, 250, 340, 470, wt(142), 76.6, rot=-3),
]
PATHS = [
    Path([(220, 600), (240, 900), (520, 700), (560, 1060)], wt(58) + 0.2, 33.6, 34.3, avatar=True),
    Path([(300, 760), (330, 1000), (420, 980), (500, 1150)], wt(144), wt(146), 76.6),
]

def rr_mask(w, h, rad):
    m = np.zeros((h, w), np.uint8); rad = int(min(rad, w / 2, h / 2))
    if rad < 1: m[:] = 255; return m.astype(np.float32) / 255
    cv2.rectangle(m, (rad, 0), (w - rad - 1, h - 1), 255, -1); cv2.rectangle(m, (0, rad), (w - 1, h - rad - 1), 255, -1)
    for cx, cy in ((rad, rad), (w - rad - 1, rad), (rad, h - rad - 1), (w - rad - 1, h - rad - 1)):
        cv2.circle(m, (cx, cy), rad, 255, -1, cv2.LINE_AA)
    return m.astype(np.float32) / 255

# play button + cursor for the hook (reference-style fake player UI)
def make_ui():
    pl = Image.new("RGBA", (220, 220), (0, 0, 0, 0)); d = ImageDraw.Draw(pl)
    d.ellipse((4, 4, 216, 216), fill=(255, 255, 255, 70), outline=(255, 255, 255, 200), width=4)
    d.polygon([(85, 62), (85, 158), (165, 110)], fill=(255, 255, 255, 235))
    cu = Image.new("RGBA", (90, 120), (0, 0, 0, 0)); d = ImageDraw.Draw(cu)
    pts = [(6, 4), (6, 92), (28, 72), (44, 108), (58, 102), (42, 66), (72, 66)]
    d.polygon(pts, fill=(255, 255, 255, 255), outline=(47, 6, 0, 255)); d.line(pts + [pts[0]], fill=(47, 6, 0, 255), width=3)
    def pm(im): a = np.asarray(im).astype(np.float32) / 255; a[..., :3] *= a[..., 3:4]; return a
    return pm(pl), pm(cu)
PLAY, CURSOR = make_ui()
CLICK = 2.3
def draw_ui(dst, tt):
    if tt > CLICK + 0.5: return
    al = 1 - ease((tt - CLICK) / 0.35) if tt > CLICK else min(1, tt / 0.2)
    sc = 1 - 0.12 * np.exp(-((tt - CLICK) / 0.08) ** 2)
    pl = cv2.resize(PLAY, (int(220 * sc), int(220 * sc)))
    blend(dst, pl, 540 - pl.shape[1] / 2, 760 - pl.shape[0] / 2, al)
    p = ease_io((tt - 0.6) / 1.4)
    cx, cy = lerp(860, 560, p), lerp(1100, 790, p)
    cs = 1 - 0.18 * np.exp(-((tt - CLICK) / 0.07) ** 2)
    cu = cv2.resize(CURSOR, (int(90 * cs), int(120 * cs)))
    blend(dst, cu, cx, cy, al)

# ---------------------------------------------------------------- frame compose
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r_ = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.42) / (H * 0.62)) ** 2)
VIG = (np.clip((r_ - 0.55) / 0.6, 0, 1) ** 1.6 * 0.5)[..., None]
rng = np.random.default_rng(3)
GRAIN = [cv2.resize(rng.standard_normal((H // 2, W // 2)).astype(np.float32) * 0.016, (W, H))[..., None] for _ in range(6)]
VAR_ZOOM = (wt(145), 72.0)      # small push-in inside the "Я" tile on «горят вареники»

def compose(src, tt, fi):
    lay, e = layout(tt)
    fr = np.empty((H, W, 3), np.float32); fr[:] = lay["bg"]
    x, y, w, h = lay["rect"]; wi, hi = max(2, int(round(w))), max(2, int(round(h)))
    zoom = lay["zoom"] * (1 + 0.035 * ((tt - SCENES[scene_at(tt)][0]) / 8))
    if SCENES[scene_at(tt)][1] == "collB": zoom *= 1 + 0.22 * ease_io((tt - VAR_ZOOM[0]) / 0.6)
    a = w / h; cw = W / zoom; ch = cw / a
    if ch > H / zoom: ch = H / zoom; cw = ch * a
    fy = lay["fy"]; cy0 = np.clip(fy - ch / 2, 0, H - ch); cx0 = (W - cw) / 2
    sx = wi / cw
    M = np.float32([[sx, 0, -cx0 * sx], [0, hi / ch, -cy0 * hi / ch]])
    vid = cv2.warpAffine(src, M, (wi, hi), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255
    if lay["blur"] > 0.02:
        vid = cv2.GaussianBlur(vid, (0, 0), 1 + 20 * lay["blur"])
    if lay["gray"] > 0.01:
        g = vid @ np.float32([0.299, 0.587, 0.114]); vid = vid * (1 - lay["gray"]) + g[..., None] * lay["gray"]
    if lay["dim"] > 0: vid *= 1 - lay["dim"]
    full = w > W - 4 and h > H - 4
    xi, yi = int(round(x)), int(round(y))
    if full:
        fr[:] = vid[:H, :W] if vid.shape[0] >= H else cv2.resize(vid, (W, H))
        fr *= 1 - VIG; fr += INK * VIG
    else:
        m = rr_mask(wi, hi, lay["rad"])[..., None]
        shd = np.zeros((hi + 80, wi + 80, 4), np.float32); shd[40:-40, 40:-40, 3] = m[..., 0]
        shd[..., 3] = cv2.GaussianBlur(shd[..., 3], (0, 0), 18) * 0.35
        blend(fr, shd, xi - 40, yi - 40 + 16)
        v4 = np.concatenate([vid * m, m], 2)
        blend(fr, v4, xi, yi)
    # flash on layout change
    for st, _, _ in SCENES[1:]:
        d = tt - st
        if 0 <= d < 0.3: fr += (BLUSH * 0.25 * (1 - d / 0.3))
    return fr, full

if __name__ == "__main__":
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "cut.mov", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = None if TEST else subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                              "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "video_only.mp4"], stdin=subprocess.PIPE)
    fi = 0
    while True:
        buf = dec.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        tt = fi / FPS
        if TEST and not any(abs(tt - x) < 0.02 for x in TEST):
            fi += 1
            if tt > max(TEST): break
            continue
        src = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        fr, full = compose(src, tt, fi)
        for tl in TILES: tl.draw(fr, tt)
        for pth in PATHS: pth.draw(fr, tt, src)
        draw_ui(fr, tt)
        for g in GROUPS:
            if g.t_in - 0.01 <= tt <= g.t_out + 0.3:
                if g.vertical:
                    cv = np.zeros((W, H, 4), np.float32); g.draw(cv, tt)
                    blend(fr, np.ascontiguousarray(np.rot90(cv, 1)), 0, 0)
                else:
                    g.draw(fr, tt)
        fr += GRAIN[fi % 6]
        tail = DUR - tt
        if tail < 0.5: fr = fr * (tail / 0.5) + INK * (1 - tail / 0.5)
        o = (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)
        if TEST: cv2.imwrite(f"test_{tt:.2f}.png", cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
        else: enc.stdin.write(o.tobytes())
        fi += 1
    dec.kill()
    if TEST: sys.exit()
    enc.stdin.close(); enc.wait()

    # ------------------------------------------------------------ sound design
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
    def tick():
        n = int(0.03 * SR); t = np.arange(n) / SR
        return norm(np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 260) + 0.3 * rng.standard_normal(n) * np.exp(-t * 600))
    def mclick():
        n = int(0.05 * SR); t = np.arange(n) / SR
        return norm(rng.standard_normal(n) * np.exp(-t * 400) + 0.6 * np.sin(2 * np.pi * 3200 * t) * np.exp(-t * 300)
                    + 0.5 * np.r_[np.zeros(int(0.025 * SR)), rng.standard_normal(n - int(0.025 * SR)) * np.exp(-np.arange(n - int(0.025 * SR)) / SR * 500)])
    def shutter():
        n = int(0.22 * SR); t = np.arange(n) / SR; x = rng.standard_normal(n)
        y = x * (np.exp(-t * 80) + 0.8 * np.exp(-np.maximum(0, t - 0.09) * 70) * (t > 0.09))
        return norm(onepole(y, 6000))
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
    for st, lay_, _ in SCENES[1:]:
        if abs(st - CLICK) > 0.1: put(WB, st - 0.3, 0.2)
    for g in GROUPS:
        for wd in g.words:
            if wd.st == "b": put(POP, wd.t0, 0.16)
            elif wd.st == "s": put(SCR, wd.t0, 0.05)
            elif g in ins: put(TICK, wd.t0, 0.035)
        for d in g.deco: put(SCR, d[5], 0.1)
    for tl in TILES: put(shutter(), tl.t_in, 0.22)
    put(sizzle(), wt(145), 0.12)
    put(chime(), wt(173) - 0.1, 0.08); put(chime(), DUR - 0.6, 0.07)
    trk = trk[:int((DUR + 0.04) * SR)]
    with wave.open("sfx.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(trk, -1, 1) * 32767).astype(np.int16).tobytes())
    print("done", fi, "frames")
