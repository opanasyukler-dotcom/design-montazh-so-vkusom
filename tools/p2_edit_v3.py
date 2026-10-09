"""«Я не люблю материнство. Часть 2» v3 — richer, «дороже» look on top of v2:
colour grade + vignette, white text with brand-colour marker highlights, Noto emoji stickers tied to words,
word-support animations (×100 counter, confetti, bubbles, fire pulse), richer but soft SFX.
Work dir: same as p2_edit.py / p2_edit_v2.py plus emoji/<codepoint>.png (Noto 512 px).
  python3 -I p2_edit_v3.py test 3,33
  python3 -I p2_edit_v3.py render f0 f1 out.mp4
  python3 -I p2_edit_v3.py sfx sfx.wav
"""
import os, subprocess, sys, wave
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv; sys.argv = [sys.argv[0], "lib"]
import p2_edit as B
import p2_edit_v2 as V
sys.argv = _argv
import numpy as np, cv2
from PIL import Image

W, H, K, FPS, OW, OH = B.W, B.H, B.K, B.FPS, B.OW, B.OH
ease, ease_io, back, P, blend, font, raster = B.ease, B.ease_io, B.back, B.P, B.blend, B.font, B.raster
INK, COCOA, ROSE, BLUSH, WHITE = B.INK, B.COCOA, B.ROSE, B.BLUSH, B.WHITE
BLACK = np.float32([0, 0, 0])
ws, sec_end, DUR = B.ws, B.sec_end, V.DUR
FONT_M = B.FONT_M
EMO = os.environ.get("EMOJI_DIR", "emoji")

# ---------------------------------------------------------------- colour grade («дорогая» картинка)
def _curve(x, lift, gain, contrast):
    x = x / 255.0
    s = x + contrast * (x - 0.5) * (1 - np.abs(2 * x - 1))          # gentle S
    s = lift + (gain - lift) * np.clip(s, 0, 1)
    return np.clip(s * 255 + 0.5, 0, 255).astype(np.uint8)
_x = np.arange(256, dtype=np.float32)
LUT = np.stack([_curve(_x, 0.022, 1.000, 0.22),                  # R: warm highlights
                _curve(_x, 0.020, 0.985, 0.20),                  # G
                _curve(_x, 0.035, 0.955, 0.18)], -1)             # B: slightly lifted, cooler shadows / warmer whites
LUT = LUT.reshape(256, 1, 3)
_yy, _xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
_r = np.sqrt(((_xx - OW / 2) / (OW * 0.7)) ** 2 + ((_yy - OH * 0.45) / (OH * 0.7)) ** 2)
VIG = np.repeat((255 * (1 - 0.32 * np.clip((_r - 0.55) / 0.55, 0, 1) ** 1.8)).astype(np.uint8)[..., None], 3, 2)
del _yy, _xx, _r
def grade(fr):
    fr = cv2.LUT(fr, LUT)
    g = cv2.cvtColor(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    fr = cv2.addWeighted(fr, 1.10, g, -0.10, 0)                  # +10 % saturation
    return cv2.multiply(fr, VIG, scale=1 / 255)

# ---------------------------------------------------------------- text with brand-colour marker highlights
HL = {"c": (COCOA, WHITE), "r": (ROSE, WHITE), "b": (BLUSH, INK)}
def seg_img(txt, size, wght, hl):
    f = font(FONT_M, int(size * K), wght)
    fg = WHITE if hl is None else HL[hl][1]
    img, sh, base, pad, adv = raster(txt, f, fg, halo=None if hl else BLACK, halo_r=7, halo_a=0.55)
    return img, sh, base, pad, adv
class RichLine:
    """[(text, hl)] where hl None = plain white, 'c'/'r'/'b' = marker pill in brand colour (wipes in)"""
    def __init__(self, segs, size, wght, x, y, t0, t1, align="c", anim="rise", hl_delay=0.25):
        self.parts = []; xs = 0.0; sp = font(FONT_M, int(size * K), wght).getlength(" ") / K
        for txt, hl in segs:
            img, sh, base, pad, adv = seg_img(txt, size, wght if hl is None else max(wght, 800), hl)
            self.parts.append(dict(img=img, sh=sh, base=base, pad=pad, adv=adv, hl=hl, dx=xs)); xs += adv + sp
        total = xs - sp; x0 = x - (total / 2 if align == "c" else total if align == "r" else 0)
        for p in self.parts:
            p["x"] = x0 + p["dx"]
            if p["hl"]:
                pw, ph = p["adv"] + 30, size * 1.08
                m = np.zeros((P(ph), P(pw)), np.uint8)
                r = P(ph * 0.32); cv2.rectangle(m, (r, 0), (P(pw) - r, P(ph)), 255, -1); cv2.rectangle(m, (0, r), (P(pw), P(ph) - r), 255, -1)
                for cx, cy in ((r, r), (P(pw) - r - 1, r), (r, P(ph) - r - 1), (P(pw) - r - 1, P(ph) - r - 1)): cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
                a = m.astype(np.float32) / 255 * 0.94; pill = np.zeros(m.shape + (4,), np.float32)
                pill[..., :3] = HL[p["hl"]][0] * a[..., None]; pill[..., 3] = a; p["pill"] = pill; p["pw"], p["ph"] = pw, ph
        self.y, self.size, self.t0, self.t1, self.anim, self.hl_delay = y, size, t0, t1, anim, hl_delay
    def active(self, tt): return self.t0 <= tt <= self.t1 + 0.3
    def draw(self, dst, tt):
        age = tt - self.t0; out = ease((tt - self.t1) / 0.25) if tt > self.t1 else 0
        e = ease(age / 0.35); dy = 26 * (1 - e) - 16 * out; al = min(1, age / 0.18) * (1 - out)
        for p in self.parts:
            top = self.y - p["base"]
            if p["hl"]:
                q = ease_io((age - self.hl_delay) / 0.3)
                if q > 0:
                    w = max(2, int(p["pill"].shape[1] * q))
                    blend(dst, p["pill"][:, :w], p["x"] - 15, self.y - self.size * 0.80 + dy, al)
            img, sh = p["img"], p["sh"]
            k = int(20 * K * (1 - e)) | 1
            if k > 2:
                img = cv2.blur(img, (1, k)); sh = cv2.blur(sh, (1, k)) if sh is not None else None
            if sh is not None: blend(dst, sh, p["x"] - p["pad"], top + dy, al)
            blend(dst, img, p["x"] - p["pad"], top + dy, al)

# subtitles: white; one accent word per chunk sits on a cocoa marker pill
S = 50
class SubWord:
    def __init__(self, txt, accent):
        f = font(FONT_M, int(S * K), 800 if accent else 600)
        self.img, self.sh, self.base, self.pad, self.adv = raster(txt, f, WHITE, halo=None if accent else BLACK, halo_r=6, halo_a=0.6)
        self.space = font(FONT_M, S * K, 600).getlength(" ") / K; self.accent = accent
        if accent:
            pw, ph = self.adv + 26, S * 1.12
            m = np.zeros((P(ph), P(pw)), np.uint8); r = P(ph * 0.3)
            cv2.rectangle(m, (r, 0), (P(pw) - r, P(ph)), 255, -1); cv2.rectangle(m, (0, r), (P(pw), P(ph) - r), 255, -1)
            for cx, cy in ((r, r), (P(pw) - r - 1, r), (r, P(ph) - r - 1), (P(pw) - r - 1, P(ph) - r - 1)): cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
            a = m.astype(np.float32) / 255 * 0.93; self.pill = np.zeros(m.shape + (4,), np.float32)
            self.pill[..., :3] = COCOA * a[..., None]; self.pill[..., 3] = a
class Sub:
    def __init__(self, lines, t_in, t_out):
        self.t_in, self.t_out, self.words = t_in, t_out, []
        for ln in lines:
            ws_ = [SubWord(t, a) for t, a in ln["words"]]
            total = sum(w.adv for w in ws_) + sum(w.space * (1.4 if w.accent else 1) for w in ws_[:-1])
            al = ln["align"]; x = ln["x"] - (total / 2 if al == "c" else total if al == "r" else 0)
            for w, t0 in zip(ws_, ln["t"]):
                w.x, w.ly, w.t0 = x, ln["y"], t0; x += w.adv + w.space * (1.4 if w.accent else 1); self.words.append(w)
    def active(self, tt): return self.t_in - 0.01 <= tt <= self.t_out + 0.3
    def draw(self, dst, tt):
        out = ease((tt - self.t_out) / 0.22) if tt > self.t_out else 0
        for w in self.words:
            age = tt - w.t0
            if age < 0: continue
            e = ease(age / 0.25); al = min(1, age / 0.12) * (1 - out); dy = 14 * (1 - e) - 12 * out
            if w.accent:
                q = ease_io(age / 0.22); pw = max(2, int(w.pill.shape[1] * q))
                blend(dst, w.pill[:, :pw], w.x - 13, w.ly - S * 0.83 + dy, al)
            if w.sh is not None: blend(dst, w.sh, w.x - w.pad, w.ly - w.base + dy, al)
            blend(dst, w.img, w.x - w.pad, w.ly - w.base + dy, al)

ACC = {"уволиться", "квест", "мамы", "триггерит", "поликлинику", "ад", "шизофрению", "больную", "стрелки", "помыться",
       "событие", "свихнулась", "пижаме", "косметики", "успех", "телефон", "дочь", "сто", "стабильность", "времени", "чемоданов"}
SPOTS = [(110, 1270, "l"), (970, 1330, "r"), (540, 1390, "c")]
subs = []
for ci, ch in enumerate(B.chunks):
    x, y, al = SPOTS[ci % 3]
    seen = False; words = []
    for s, e, w in ch:
        c = B.clean(w); acc = (c in ACC) and not seen; seen = seen or acc
        words.append((w.strip("«»").rstrip(",."), acc, s))
    txt = " ".join(w for w, _, _ in words); lines = [words]
    if len(txt) > 15 and len(words) > 1:
        k = (len(words) + 1) // 2; lines = [words[:k], words[k:]]
    L = []
    for li, wl in enumerate(lines):
        dx = (60 if al == "l" else -60 if al == "r" else 30) * li
        L.append(dict(words=[(w, a) for w, a, _ in wl], x=x + dx, y=y + li * int(S * 1.35), align=al, t=[t0 - 0.04 for _, _, t0 in wl]))
    nxt = B.chunks[ci + 1][0][0] - 0.08 if ci + 1 < len(B.chunks) else DUR
    subs.append(Sub(L, ch[0][0] - 0.06, min(nxt, ch[-1][1] + 0.7)))

# inserts: same texts as v2, white, with brand-colour highlights on the key part
TEXT = [
    RichLine([("ЧАСТЬ 2", "c")], 30, 800, 540, 300, 0.1, V.T_HOOK_END, hl_delay=0.05),
    RichLine([("Я НЕ ЛЮБЛЮ", None)], 70, 900, 540, 400, 0.25, V.T_HOOK_END),
    RichLine([("МАТЕРИНСТВО", "c")], 80, 900, 540, 500, 0.6, V.T_HOOK_END, hl_delay=0.35),
    RichLine([("Что конкретно меня", None), ("бесит?", "r")], 42, 700, 540, 590, 1.3, V.T_HOOK_END),
    RichLine([("Каждый день хочу", None), ("уволиться.", "c")], 46, 800, 540, 250, ws(99), V.T_WORK[1]),
    RichLine([("Каждый день", None), ("передумываю.", "r")], 46, 700, 540, 330, ws(106), V.T_WORK[1]),
    RichLine([("Закончила одну смену.", None)], 50, 700, 540, 330, ws(164), sec_end("two") - 0.15),
    RichLine([("Поехала на", None), ("вторую.", "c")], 62, 900, 540, 425, V.T_TWO2, sec_end("two") - 0.15, hl_delay=0.1),
    RichLine([("Выйти с ребёнком из дома", None)], 46, 700, 540, 330, ws(273), V.T_REC[1]),
    RichLine([("= спецоперация", "c")], 64, 900, 540, 425, ws(276), V.T_REC[1], hl_delay=0.1),
    RichLine([("это мой", None), ("муж", "c")], 54, 800, 520, 560, V.T_HUSB + 0.1, V.T_HUSB + V.HUSB_LEN - 0.05, hl_delay=0.15),
    RichLine([("Раньше: стрелки, укладка, макияж.", None)], 40, 700, 540, 330, ws(376), sec_end("self") - 0.15),
    RichLine([("Сейчас:", None), ("успеть помыться.", "c")], 52, 900, 540, 415, ws(400), sec_end("self") - 0.15, hl_delay=0.15),
    RichLine([("Иногда просто хочется снова", None)], 44, 600, 540, 330, ws(478), V.T_DREAM[1]),
    RichLine([("почувствовать", None), ("себя собой.", "b")], 52, 800, 540, 410, ws(484), V.T_DREAM[1], hl_delay=0.3),
]

# ---------------------------------------------------------------- emoji stickers
_emo = {}
def sticker(code, size):
    k = (code, size)
    if k not in _emo:
        im = np.asarray(Image.open(f"{EMO}/{code}.png").convert("RGBA")).astype(np.float32) / 255
        pad = 40; im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)))
        a = im[..., 3]
        border = cv2.dilate((a > 0.08).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
        border = cv2.GaussianBlur(border.astype(np.float32) / 255, (0, 0), 1.2)
        out = np.zeros_like(im); out[..., :3] = WHITE * border[..., None]; out[..., 3] = border      # white die-cut outline
        rgb = im[..., :3] * a[..., None]; out[..., :3] = out[..., :3] * (1 - a[..., None]) + rgb; out[..., 3] = np.maximum(out[..., 3], a)
        px = P(size); out = cv2.resize(out, (px, px), interpolation=cv2.INTER_AREA)
        sh = cv2.GaussianBlur(out[..., 3], (0, 0), 8 * K) * 0.35; shi = np.zeros_like(out); shi[..., 3] = sh
        _emo[k] = (out, shi)
    return _emo[k]

class Sticker:
    def __init__(self, code, x, y, t0, dur=1.9, size=200, rot=0.0):
        self.code, self.x, self.y, self.t0, self.t1, self.size, self.rot = code, x, y, t0, t0 + dur, size, rot
    def active(self, tt): return self.t0 <= tt <= self.t1 + 0.25
    def draw(self, dst, tt):
        img, sh = sticker(self.code, self.size); age = tt - self.t0
        out = ease((tt - self.t1) / 0.22) if tt > self.t1 else 0
        sc = back(age / 0.38, 2.2) * (1 - 0.4 * out); al = min(1, age / 0.08) * (1 - out)
        ang = self.rot + 6 * np.sin(age * 3.2) * (1 - out) + (1 - min(1, age / 0.3)) * -18
        dy = -8 * np.sin(age * 2.4)
        if sc < 0.02: return
        n = max(4, int(img.shape[1] * sc)); c = (n / 2, n / 2)
        M = cv2.getRotationMatrix2D(c, ang, 1)
        im2 = cv2.warpAffine(cv2.resize(img, (n, n)), M, (n, n)); sh2 = cv2.warpAffine(cv2.resize(sh, (n, n)), M, (n, n))
        blend(dst, sh2, self.x - n / K / 2 + 6, self.y - n / K / 2 + 14 + dy, al)
        blend(dst, im2, self.x - n / K / 2, self.y - n / K / 2 + dy, al)

L_, R_ = 170, 905          # left / right sticker lanes (face sits in the middle)
STK = [
    Sticker("1f644", R_, 900, ws(33), rot=8),                     # мне абсолютно не нравятся
    Sticker("1f90d", L_, 980, ws(21), rot=-8, size=150),          # дочь
    Sticker("1f4bb", L_, 950, ws(54), rot=-6),                    # работать
    Sticker("1f4bc", R_, 980, ws(66), rot=10),                    # уволиться
    Sticker("1f3e0", L_, 940, ws(86), rot=-5, size=155),          # часть жизни привычную
    Sticker("1f635_200d_1f4ab", R_, 930, ws(105), rot=8),         # уволиться (борюсь)
    Sticker("1f9f8", L_, 950, ws(125), rot=-8, size=155),         # стабильность
    Sticker("1f3a5", R_, 930, ws(145), rot=8),                    # съёмки
    Sticker("1f5fa_fe0f", L_, 960, ws(150), rot=-6),              # квест
    Sticker("1f469_200d_1f37c", R_, 950, ws(177), rot=6, size=180),  # работу мамы
    Sticker("1f3e5", L_, 960, ws(272), rot=-6),                   # поликлинику
    Sticker("1f9f3", R_, 960, ws(276), rot=8),                    # чемоданов
    Sticker("1f634", L_, 980, ws(289), rot=-8, size=150),         # сон
    Sticker("1f37c", R_, 990, ws(293), rot=10, size=150),         # еду
    Sticker("1f525", L_, 950, ws(309), rot=-4, size=190),         # ад
    Sticker("1f300", L_, 640, ws(316), rot=0, size=150),          # шизофрению (left/top: husband is on the right)
    Sticker("1f475", L_, 960, ws(329), rot=-6),                   # бабушки
    Sticker("1f440", R_, 960, ws(335), rot=6, size=150),          # смотрит
    Sticker("1f484", R_, 940, ws(380), rot=10),                   # краситься
    Sticker("2728", L_, 900, ws(382), rot=0, size=150),           # стрелки
    Sticker("1f6bf", R_, 960, ws(386), rot=-6),                   # душе
    Sticker("1f487_200d_2640_fe0f", L_, 960, ws(388), rot=-6),    # причёску
    Sticker("1f9fc", L_, 980, ws(404), rot=-8),                   # помыться
    Sticker("1f6cc", R_, 950, ws(471), rot=6, size=160),          # пижаме
    Sticker("1fae0", L_, 960, ws(474), rot=-6, size=150),         # грязной головой
    Sticker("1f457", R_, 950, ws(482), rot=8),                    # наряжаться
    Sticker("1f635_200d_1f4ab", L_, 960, ws(492), rot=-6),        # свихнулась
    Sticker("1f9f3", R_, 950, ws(515), rot=8, size=150),          # чемодан
    Sticker("1f484", R_ - 60, 1060, ws(516), rot=-12, size=120),  # косметики
    Sticker("23f0", L_, 960, ws(534), rot=-8),                    # нет времени
    Sticker("1f9f4", R_, 960, ws(541), rot=8),                    # крем
    Sticker("1f4f1", L_, 900, ws(550), rot=-12, dur=1.0),         # телефон
    Sticker("1f4a5", L_ + 40, 860, ws(550) + 0.35, rot=0, dur=0.9, size=200),
]

# ---------------------------------------------------------------- word-support animations
class Counter100:
    """«сто раз»: a quick ×1 → ×100 counter on a cocoa pill + 💯"""
    def __init__(self, t0, x=540, y=560):
        self.t0, self.t1, self.x, self.y = t0, t0 + 2.0, x, y; self.cache = {}
    def active(self, tt): return self.t0 <= tt <= self.t1 + 0.25
    def img(self, n):
        if n not in self.cache:
            f = font(FONT_M, int(96 * K), 900); im, sh, base, pad, adv = raster(f"×{n}", f, WHITE, halo=BLACK, halo_r=8, halo_a=0.5)
            self.cache[n] = (im, sh)
        return self.cache[n]
    def draw(self, dst, tt):
        age = tt - self.t0; out = ease((tt - self.t1) / 0.22) if tt > self.t1 else 0
        n = int(1 + 99 * ease(min(1, age / 0.9))); im, sh = self.img(n)
        sc = 1 + 0.18 * np.exp(-((age - 0.9) / 0.08) ** 2)
        if abs(sc - 1) > 0.01: im = cv2.resize(im, (int(im.shape[1] * sc), int(im.shape[0] * sc))); sh = cv2.resize(sh, im.shape[1::-1])
        al = min(1, age / 0.1) * (1 - out)
        blend(dst, sh, self.x - im.shape[1] / K / 2, self.y - im.shape[0] / K / 2, al)
        blend(dst, im, self.x - im.shape[1] / K / 2, self.y - im.shape[0] / K / 2, al)
        if age > 0.9:
            s = Sticker("1f4af", self.x + 230, self.y - 40, self.t0 + 0.9, dur=self.t1 - self.t0 - 0.9, size=150, rot=10); s.draw(dst, tt)

class Confetti:
    """brand-colour confetti burst («отдельное событие»)"""
    def __init__(self, t0, n=110, seed=3):
        r = np.random.default_rng(seed); self.t0, self.t1 = t0, t0 + 2.4
        cols = [BLUSH, ROSE, COCOA, WHITE]
        self.p = [(r.uniform(150, 930), r.uniform(-120, 200), r.uniform(-90, 90), r.uniform(-300, -80), r.uniform(22, 40), r.uniform(12, 20),
                   cols[k % 4], r.uniform(0, 6.28), r.uniform(-9, 9)) for k in range(n)]
    def active(self, tt): return self.t0 <= tt <= self.t1
    def draw(self, dst, tt):
        a = tt - self.t0; fade = 1 - ease((tt - self.t1 + 0.6) / 0.6)
        for x0, y0, vx, vy, w, h, c, ph, spin in self.p:
            x = x0 + vx * a; y = y0 + 520 + vy * a + 420 * a * a
            ang = ph + spin * a; ww = w * abs(np.cos(ang))
            pts = cv2.boxPoints(((P(x), P(y)), (max(1, P(ww)), P(h)), np.degrees(ang) * 0.5)).astype(np.int32)
            col = tuple(int(v * 255) for v in c)
            if fade < 0.999:
                ov = dst.copy(); cv2.fillConvexPoly(ov, pts, col, cv2.LINE_AA); cv2.addWeighted(ov, fade, dst, 1 - fade, 0, dst)
            else: cv2.fillConvexPoly(dst, pts, col, cv2.LINE_AA)

class Bubbles:
    """soap bubbles floating up («помыться»)"""
    def __init__(self, t0, n=14, seed=5):
        r = np.random.default_rng(seed); self.t0, self.t1 = t0, t0 + 2.6
        self.b = [(r.uniform(120, 960), r.uniform(1150, 1500), r.uniform(18, 46), r.uniform(120, 260), r.uniform(0, 0.8), r.uniform(0, 6)) for _ in range(n)]
    def active(self, tt): return self.t0 <= tt <= self.t1
    def draw(self, dst, tt):
        a = tt - self.t0
        for x, y, rad, v, delay, ph in self.b:
            age = a - delay
            if age < 0: continue
            al = min(1, age / 0.2) * (1 - ease((tt - self.t1 + 0.6) / 0.6))
            cx, cy = x + 16 * np.sin(age * 2.5 + ph), y - v * age
            ov = dst.copy()
            cv2.circle(ov, (P(cx), P(cy)), P(rad), (255, 255, 255), max(1, int(2.5 * K)), cv2.LINE_AA)
            cv2.circle(ov, (P(cx - rad * 0.35), P(cy - rad * 0.35)), max(1, P(rad * 0.18)), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.75 * al, dst, 1 - 0.75 * al, 0, dst)

class FirePulse:
    """warm red-orange edge pulse + micro-shake («личный ад»)"""
    def __init__(self, t0): self.t0, self.t1 = t0, t0 + 1.4
    def active(self, tt): return self.t0 <= tt <= self.t1
    def draw(self, dst, tt): pass   # handled in render_frame (needs whole-frame access)

EFX = [Counter100(ws(64)), Confetti(ws(410) - 0.05), Bubbles(ws(404) - 0.1)]
FIRE = FirePulse(ws(309) - 0.05)
_edge = (np.clip((np.sqrt(((np.mgrid[0:OH // 4, 0:OW // 4][1] - OW / 8) / (OW / 8)) ** 2 + ((np.mgrid[0:OH // 4, 0:OW // 4][0] - OH / 8) / (OH / 8)) ** 2) - 0.6) / 0.6, 0, 1) ** 1.5).astype(np.float32)
EDGE = cv2.resize(_edge, (OW, OH))[..., None]

# ---------------------------------------------------------------- frame
RET = V.RET

def base_layers(src, tt):
    """v2 picture pipeline (retouch, zoom/freeze, framed, b&w, dream, REC, flashes) without v2 text"""
    saved_text, saved_subs = V.TEXT, V.subs
    V.TEXT, V.subs = [], []
    try: fr = V.render_frame(src, tt)
    finally: V.TEXT, V.subs = saved_text, saved_subs
    return fr

def render_frame(src, tt):
    if FIRE.active(tt):
        a = np.sin(np.pi * (tt - FIRE.t0) / (FIRE.t1 - FIRE.t0))
        dx, dy = 6 * K * np.sin(tt * 70) * a, 4 * K * np.cos(tt * 55) * a
        src = cv2.warpAffine(src, np.float32([[1, 0, dx], [0, 1, dy]]), (OW, OH), borderMode=cv2.BORDER_REFLECT)
    fr = base_layers(src, tt)
    if tt < DUR - 0.45: fr = grade(fr)
    if FIRE.active(tt):
        a = np.sin(np.pi * (tt - FIRE.t0) / (FIRE.t1 - FIRE.t0)) * 0.20
        f = fr.astype(np.float32); f = f * (1 - EDGE * a) + np.float32([255, 90, 30]) * EDGE * a
        fr = np.clip(f, 0, 255).astype(np.uint8)
    for s in STK:
        if s.active(tt): s.draw(fr, tt)
    for e in EFX:
        if e.active(tt): e.draw(fr, tt)
    for t in TEXT:
        if t.active(tt): t.draw(fr, tt)
    for g in subs:
        if g.active(tt) and not (V.T_HUSB <= tt < V.T_HUSB + V.HUSB_LEN): g.draw(fr, tt)
    return fr

# ---------------------------------------------------------------- sound design (soft, tasteful levels)
def sfx(path):
    SR = 48000; rng = np.random.default_rng(11); n = int((DUR + 0.5) * SR); trk = np.zeros(n)
    def onepole(x, cut):
        y = np.empty_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * cut / SR)
        for i in range(len(x)): s += a * (x[i] - s); y[i] = s
        return y
    def norm(y): return y / (np.abs(y).max() + 1e-9)
    def put(sig, t, g):
        i = int(max(0, t) * SR); seg = trk[i:i + len(sig)]; seg += sig[:len(seg)] * g
    def env(m, att, dec): t = np.arange(m) / SR; return np.minimum(1, t / att) * np.exp(-t * dec)
    def tone(f, d, dec, att=0.003): m = int(d * SR); t = np.arange(m) / SR; return np.sin(2 * np.pi * f * t) * env(m, att, dec)
    def softpop(f0=520):       # rounded bubble pop for stickers
        m = int(0.12 * SR); t = np.arange(m) / SR; f = f0 + 900 * np.exp(-t * 60)
        return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * env(m, 0.002, 38))
    def whoosh(d=0.45, lo=400, hi=3500):
        m = int(d * SR); t = np.linspace(0, 1, m); x = rng.standard_normal(m)
        return norm((onepole(x, hi) - onepole(x, lo)) * np.sin(np.pi * t) ** 2)
    def tick(): return norm(tone(2600, 0.025, 250) + 0.2 * rng.standard_normal(int(0.025 * SR)) * env(int(0.025 * SR), 0.001, 400))
    def chime(f0=1046.5, d=1.6):
        m = int(d * SR); t = np.arange(m) / SR
        return norm(sum(a * np.sin(2 * np.pi * f0 * r * t) * np.exp(-t * k) for r, a, k in [(1, 1, 2.8), (2, .35, 4), (3, .15, 6), (4.2, .06, 9)]) * np.minimum(1, t / 0.004))
    def sparkle():
        m = int(1.0 * SR); t = np.arange(m) / SR
        return norm(sum(np.sin(2 * np.pi * f * t) * np.exp(-np.clip(t - k * 0.06, 0, None) * 7) * (t >= k * 0.06) for k, f in enumerate([1568, 2093, 2637, 3136, 4186])))
    def party():
        m = int(0.6 * SR); x = rng.standard_normal(m); t = np.arange(m) / SR
        return norm(onepole(x, 5000) * np.exp(-t * 9) + 0.6 * tone(220, 0.6, 12)[:m])
    def bubble(f0):
        m = int(0.09 * SR); t = np.arange(m) / SR; f = f0 * (1 + 2.5 * t / 0.09)
        return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / 0.09))
    def fire():
        m = int(1.4 * SR); x = rng.standard_normal(m); t = np.arange(m) / SR
        crack = np.convolve((rng.random(m) < 0.002) * rng.uniform(0.3, 1, m), np.exp(-np.arange(150) / 20), "same")
        return norm((onepole(x, 1500) - onepole(x, 150)) * np.sin(np.pi * t / 1.4) + 0.5 * crack)
    def boom():
        m = int(1.0 * SR); t = np.arange(m) / SR; f = 45 + 60 * np.exp(-t * 16)
        return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * env(m, 0.004, 3.5))
    def shutter():
        m = int(0.22 * SR); t = np.arange(m) / SR; x = rng.standard_normal(m)
        return norm(onepole(x * (np.exp(-t * 80) + 0.8 * np.exp(-np.maximum(0, t - 0.09) * 70) * (t > 0.09)), 6000))
    def rec_beep(): return norm(np.r_[tone(1760, 0.08, 10), np.zeros(int(0.05 * SR)), tone(1760, 0.08, 10)])
    def scratch():
        m = int(0.32 * SR); t = np.arange(m) / SR; f = 900 * np.exp(-t * 6) + 120
        return norm((np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.6 + onepole(rng.standard_normal(m), 2500)) * np.exp(-t * 7))
    def crack():
        m = int(0.5 * SR); t = np.arange(m) / SR; x = rng.standard_normal(m)
        return norm(onepole(x, 7000) * np.exp(-t * 14) + tone(90, 0.5, 10)[:m] * 0.8)
    WH = whoosh()
    put(boom(), 0.1, 0.22); put(chime(784, 1.8), 0.6, 0.06)
    for t in TEXT: put(whoosh(0.35, 700, 5000), t.t0 - 0.05, 0.05)
    for sc in V.SEC_CHANGES: put(WH, sc - 0.25, 0.09)
    put(WH, V.T_WORK[0] - 0.1, 0.10)
    put(whoosh(0.5, 600, 7000), V.T_TWO2 - 0.45, 0.10); put(boom(), V.T_TWO2, 0.12)
    put(rec_beep(), V.T_REC[0] + 0.05, 0.07)
    put(scratch(), V.T_HUSB - 0.05, 0.10); put(shutter(), V.T_HUSB + 0.05, 0.15)
    put(whoosh(0.5, 300, 2500), V.T_BW[0] - 0.2, 0.08); put(sparkle(), V.T_BW[1], 0.06)
    put(sparkle(), V.T_DREAM[0] + 0.1, 0.06)
    for s in STK: put(softpop(rng.uniform(450, 650)), s.t0, 0.09)
    c = EFX[0]
    for k in range(14): put(tick(), c.t0 + 0.9 * ease(k / 14) ** 0.6, 0.05)
    put(chime(1318.5, 1.2), c.t0 + 0.9, 0.06)
    put(party(), EFX[1].t0, 0.12); put(sparkle(), EFX[1].t0 + 0.1, 0.06)
    for k in range(8): put(bubble(rng.uniform(500, 900)), EFX[2].t0 + 0.2 + k * 0.22, 0.05)
    put(fire(), FIRE.t0, 0.10)
    put(crack(), ws(550) + 0.35, 0.14)
    w = wave.open(path, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(trk, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sfx": sfx(sys.argv[2]); sys.exit()
    if mode == "test":
        for t in [float(x) for x in sys.argv[2].split(",")]:
            fo = int(round(t * FPS)); dec = B.decoder(fo)
            src = np.frombuffer(dec.stdout.read(OW * OH * 3), np.uint8).reshape(OH, OW, 3); dec.kill()
            RET.n = 0; RET.box = None; V._freeze = None
            if V.T_HUSB <= t < V.T_HUSB + V.HUSB_LEN:
                d2 = B.decoder(int(round(V.T_HUSB * FPS))); f_ = np.frombuffer(d2.stdout.read(OW * OH * 3), np.uint8).reshape(OH, OW, 3); d2.kill()
                V._freeze = RET(f_); RET.n = 0
            o = render_frame(src, t)
            cv2.imwrite(f"test_{t:.2f}.png", cv2.cvtColor(cv2.resize(o, (540, 960), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        sys.exit()
    f0, f1, outp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    if int(V.T_HUSB * FPS) < f0 <= int((V.T_HUSB + V.HUSB_LEN) * FPS):
        raise SystemExit("chunk must not start inside the husband freeze")
    dec = B.decoder(f0)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-g", "120", "-pix_fmt", "yuv420p",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", outp], stdin=subprocess.PIPE)
    for fo in range(f0, f1):
        buf = dec.stdout.read(OW * OH * 3)
        if len(buf) < OW * OH * 3: break
        enc.stdin.write(render_frame(np.frombuffer(buf, np.uint8).reshape(OH, OW, 3), fo / FPS).tobytes())
    dec.kill(); enc.stdin.close(); enc.wait(); print("done", outp)
