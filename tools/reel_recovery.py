#!/usr/bin/env python3
"""Reels «А если я немного нарушу режим?» — почему восстановление после операции это часть результата, Анна Тваури.

One talking-head take, cut for short attention spans: every sentence is its own shot, the surgeon alternates between
full screen (punch-ins) and a big circle above animated infographics — a «хирург | вы» result bar, a deck of «что
нельзя» cards with icons, the patient's chat bubble «мне уже хорошо», the key line set word by word, a checkbox «не для
галочки», three healing steps, a ДА/НЕТ question and the closing card. Brand system of the other reels (navy, powder
pink, Bebas Neue + thin Montserrat, logo); whole sentences only.

usage:
  REEL_SRC=<folder with 15.mp4, logo.png, song.wav> python3 tools/reel_recovery.py [--preview t1,t2 | --cover | --audio-only]
The take is first normalised into $REEL_SRC/_work:
  ffmpeg -i 15.mp4 -vf "scale=1080:1920:flags=lanczos,fps=30,format=yuv420p" -an -c:v libx264 -crf 14 _work/IMG_15.mp4
  ffmpeg -i 15.mp4 -vn -ac 1 -ar 48000 _work/IMG_15.wav
"""
import os, sys, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_recovery"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_recovery_regime"
W, H, FPS, SR = 1080, 1920, 30, 48000
NAVY = np.float32([0, 18, 58]) / 255; PINK = np.float32([233, 197, 193]) / 255; WHITE = np.float32([1, 1, 1])
DEEP = np.float32([0, 8, 30]) / 255

def cl(x): return min(1.0, max(0.0, x))
def eo(x): x = cl(x); return 1 - (1 - x) ** 3
def eio(x): x = cl(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x): x = cl(x); c = 1.7; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p

# ---------------- voice edit: (block, source, [(in, out), …]) — whole sentences from the one take; the pauses between them are tightened
VO = [("S1", "15", [(0.36, 3.10)]),     # что будет, если не соблюдать рекомендации?
      ("S2", "15", [(3.56, 6.60)]),     # результат операции зависит не только от хирурга
      ("S3", "15", [(6.94, 10.24)]),    # хирург делает свою часть работы в операционной
      ("S4", "15", [(10.46, 15.52)]),   # но после этого начинается этап, где очень многое зависит от пациента
      ("S5", "15", [(15.70, 26.10)]),   # если не носить компрессионное бельё, … восстановление может идти хуже
      ("S6", "15", [(26.40, 30.36)]),   # я понимаю, что всем хочется быстрее вернуться к обычной жизни
      ("S7", "15", [(30.58, 33.64)]),   # но организм не ускоряется от нашего желания
      ("S8", "15", [(33.92, 36.16)]),   # рекомендации даются не для галочки
      ("S9", "15", [(36.52, 42.58)]),   # они нужны, чтобы ткани спокойно заживали, отёк уходил, а результат формировался правильно
      ("S10", "15", [(42.96, 48.30)]),  # поэтому перед операцией я всегда спрашиваю, сможете ли вы реально соблюдать режим
      ("S11", "15", [(48.62, 51.62)])]  # если нет, лучше выбрать другое время
GAP = {"S2": 0.15, "S3": 0.12, "S4": 0.15, "S5": 0.15, "S6": 0.2, "S7": 0.35, "S8": 0.6, "S9": 0.15, "S10": 0.2, "S11": 0.15}
SEG = []; _t = 0.04                                                    # the reel starts on the first word
for _b, _s, _parts in VO:
    if SEG: _t += GAP[_b]
    _d = sum(o - a for a, o in _parts); SEG.append((_b, _s, _parts, _t, _d)); _t += _d
def vstart(b): return next(s[3] for s in SEG if s[0] == b)
def vend(b): return next(s[3] + s[4] for s in SEG if s[0] == b)
def at_src(name, src_t):
    for b, s, parts, t0, d in SEG:
        if s != name: continue
        acc = 0
        for a, o in parts:
            if a <= src_t <= o: return t0 + acc + src_t - a
            acc += o - a
    raise ValueError(src_t)
def src_of(t, name):
    """source time for footage of `name`; between its lines the picture keeps running"""
    best = None
    for b, s, parts, t0, d in SEG:
        if s != name: continue
        a0 = parts[0][0]; a1 = parts[-1][1]
        if t0 <= t <= t0 + d: return a0 + t - t0
        if t < t0: return a0 - (t0 - t) if best is None else min(best[0] + t - best[1], a0)
        best = (a1, t0 + d)
    return best[0] + t - best[1]

def T(s_): return at_src("15", s_)
T_S = {b: vstart(b) - 0.08 for b, *_ in VO}; T_END = vend("S11") + 0.35; DUR = T_END + 3.4
T_FILL1 = T(8.22); T_FILL2 = T(14.44)                                  # «часть работы» · «зависит от пациента»
T_CARDS = [T(16.72), T(18.48), T(20.10), T(21.30), T(22.80), T(24.40)]  # бельё · нагрузки · тяжести · осмотры · ограничения · хуже
T_KEY = [T(30.84), T(31.38), T(32.40)]                                 # организм · не ускоряется · от нашего желания
T_NOT = T(35.26)                                                       # «не для галочки»
T_HEAL = [T(37.48), T(39.60), T(40.76)]                                # ткани · отёк · результат
T_ASK = T(45.82); T_NO = T(49.06); T_TIME = T(50.32)                   # «сможете ли вы» · «нет» · «другое время»

# ---------------- fonts & text bitmaps ----------------
FD = os.path.join(ROOT, "fonts"); _FC = {}
def bebas(size, bold=True):
    k = ("b", size, bold)
    if k not in _FC: _FC[k] = ImageFont.truetype(os.path.join(FD, f"bebas-neue-{700 if bold else 400}-normal.woff"), size)
    return _FC[k]
def mont(size, wght=300):
    k = ("m", size, wght)
    if k not in _FC:
        f = ImageFont.truetype(os.path.join(FD, "Montserrat-VariableFont_wght.ttf"), size); f.set_variation_by_axes([wght]); _FC[k] = f
    return _FC[k]
def alpha_img(txt, f):
    asc, desc = f.getmetrics(); w = int(f.getlength(txt)) + 8
    im = Image.new("L", (w, asc + desc + 4), 0); ImageDraw.Draw(im).text((4, 2 + asc), txt, font=f, fill=255, anchor="ls")
    return np.asarray(im, np.float32) / 255
def cap_top(f): asc, _ = f.getmetrics(); return 2 + asc + f.getbbox("Н", anchor="ls")[1]
def blend(fr, a, x, y, col, op=1.0):
    x, y = int(round(x)), int(round(y)); h, w = a.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0 or op <= 0.003: return
    m = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * op
    reg = fr[y0:y1, x0:x1]; reg *= 1 - m; reg += m * col
def scaled(a, s):
    s = max(s, 0.02); return cv2.resize(a, (max(1, int(a.shape[1] * s)), max(1, int(a.shape[0] * s))), interpolation=cv2.INTER_LINEAR)

class Drop:
    """Bebas line(s), letter by letter: every glyph drops in from 1.7x size and fades in (*word* = pink)"""
    def __init__(s, lines, size, x, y, maxw=940, lh=0.93, stag=0.03):
        while True:
            f = bebas(size); rows = []; pink = False
            for ln in lines:
                glyphs, cx = [], 0
                for wd in ln.split(" "):
                    if wd.startswith("*"): pink = True
                    w0 = wd.strip("*")
                    for ch in w0:
                        glyphs.append((alpha_img(ch, f), PINK if pink else WHITE, cx)); cx += f.getlength(ch)
                    if wd.endswith("*"): pink = False
                    cx += f.getlength(" ")
                rows.append((glyphs, cx - f.getlength(" ")))
            if max(r[1] for r in rows) <= maxw or size < 40: break
            size -= 4
        s.rows, s.size, s.x, s.y, s.stag, s.lh, s.top = rows, size, x, y, stag, int(size * lh), cap_top(f)
        s.h = s.lh * (len(rows) - 1) + size * 0.72; s.n = sum(len(g) for g, _ in rows)
    def draw(s, fr, t, t0, t1=1e9, dur=0.32):
        if t < t0 or t > t1 + 0.4: return
        q = eio((t - t1) / 0.3) if t > t1 else 0.0; k = 0
        for r, (glyphs, _) in enumerate(s.rows):
            for a, col, dx in glyphs:
                p = eo((t - t0 - k * s.stag) / dur); k += 1
                if p <= 0: continue
                sc = 1 + 0.7 * (1 - p); g = scaled(a, sc)
                cx = s.x + dx + a.shape[1] / 2 - 4; cy = s.y + r * s.lh - s.top + a.shape[0] / 2
                blend(fr, g, cx - g.shape[1] / 2, cy - g.shape[0] / 2 - q * 30, col, min(1, p * 1.6) * (1 - q))

class Track:
    """small Montserrat line that condenses from wide letter-spacing while fading in"""
    def __init__(s, lines, size=36, x=72, y=0, wght=300, col=WHITE, center=False):
        f = mont(size, wght); s.lines = []
        for ln in lines: s.lines.append([(alpha_img(ch, f), f.getlength(ln[:i])) for i, ch in enumerate(ln) if ch != " "] + [(None, f.getlength(ln))])
        s.x, s.y, s.lh, s.col, s.top, s.center = x, y, int(size * 1.5), col, cap_top(f), center
    def draw(s, fr, t, t0, t1=1e9):
        if t < t0 or t > t1 + 0.4: return
        p = eo((t - t0) / 0.8); op = min(1, (t - t0) / 0.4) * (1 - eio((t - t1) / 0.3) if t > t1 else 1)
        for k, ln in enumerate(s.lines):
            wtot = ln[-1][1]; sp = 14 * (1 - p); ox = s.x - (wtot + sp * len(ln)) / 2 if s.center else s.x
            for i, (a, dx) in enumerate(ln[:-1]): blend(fr, a, ox + dx + i * sp - 4, s.y + k * s.lh - s.top, s.col, op)

class Marker:
    """Bebas line where one word sits on a pink marker bar that sweeps in (word turns navy)"""
    def __init__(s, pre, key, size, x, y):
        f = bebas(size); s.a = alpha_img(pre + " ", f); s.k = alpha_img(key, f); s.x, s.y, s.top, s.size = x, y, cap_top(f), size
        s.kx = x + f.getlength(pre + " ")
    def draw(s, fr, t, t0, t1=1e9):
        if t < t0 or t > t1 + 0.4: return
        op = (1 - eio((t - t1) / 0.3)) if t > t1 else 1.0; p = eo((t - t0) / 0.4)
        a = s.a.copy(); a[:, int(a.shape[1] * p):] = 0; blend(fr, a, s.x - 4, s.y - s.top, WHITE, op)
        pb = eio((t - t0 - 0.3) / 0.35)
        if pb > 0:
            bw = int((s.k.shape[1] + 6) * pb); bar = np.ones((int(s.size * 0.86), max(1, bw)), np.float32)
            blend(fr, bar, s.kx - 10, s.y - s.size * 0.07, PINK, op)
        kc = WHITE * (1 - pb) + NAVY * pb; blend(fr, s.k, s.kx - 4, s.y - s.top, kc, op * eo((t - t0 - 0.2) / 0.3))

class Roll:
    """slot-machine swap: the old word rolls up out of its mask, the new one rolls in from below"""
    def __init__(s, pre, w_old, w_new, size, x, y, maxw=940):
        while True:
            f = bebas(size); px = f.getlength(pre + " ")
            if px + max(f.getlength(w_old), f.getlength(w_new)) <= maxw or size < 40: break
            size -= 4
        s.pre, s.o, s.n = alpha_img(pre, f) if pre else np.zeros((2, 2), np.float32), alpha_img(w_old, f), alpha_img(w_new, f)
        s.x, s.y, s.top, s.px, s.size = x, y, cap_top(f), (px if pre else 0), size
    def draw(s, fr, t, t0, tswap, t1=1e9):
        if t < t0 or t > t1 + 0.4: return
        op = (1 - eio((t - t1) / 0.3)) if t > t1 else 1.0; p = eo((t - t0) / 0.45)
        def masked(a, dy):
            b = np.zeros_like(a); h = a.shape[0]; dy = int(dy)
            if 0 <= dy < h: b[dy:] = a[:h - dy]
            elif -h < dy < 0: b[:h + dy] = a[-dy:]
            return b
        hh = s.o.shape[0]
        blend(fr, masked(s.pre, (1 - p) * hh), s.x - 4, s.y - s.top, WHITE, op)
        q = eio((t - tswap) / 0.45) if t > tswap else 0.0
        blend(fr, masked(s.o, (1 - p) * hh - q * hh), s.x + s.px - 4, s.y - s.top, WHITE if not s.pre.any() else PINK, op)
        if q > 0: blend(fr, masked(s.n, (1 - q) * hh), s.x + s.px - 4, s.y - s.top, PINK, op)

# ---------------- line drawing ----------------
def path_draw(fr, pts, p, col, th=3, closed=False, op=1.0):
    if p <= 0 or op <= 0: return
    P = [tuple(map(float, q)) for q in pts] + ([tuple(map(float, pts[0]))] if closed else [])
    L = [np.hypot(P[i + 1][0] - P[i][0], P[i + 1][1] - P[i][1]) for i in range(len(P) - 1)]
    left = p * sum(L); xs = [q[0] for q in P]; ys = [q[1] for q in P]
    x0, y0 = int(max(0, min(xs) - th - 2)), int(max(0, min(ys) - th - 2)); x1, y1 = int(min(W, max(xs) + th + 3)), int(min(H, max(ys) + th + 3))
    if x1 <= x0 or y1 <= y0: return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for i in range(len(L)):
        if left <= 0: break
        a = np.float32(P[i]); b = np.float32(P[i + 1]); f = min(1, left / max(L[i], 1e-6)); e = a + (b - a) * f
        cv2.line(m, (int((a[0] - x0) * 16), int((a[1] - y0) * 16)), (int((e[0] - x0) * 16), int((e[1] - y0) * 16)), 255, th, cv2.LINE_AA, shift=4); left -= L[i]
    blend(fr, m.astype(np.float32) / 255, x0, y0, col, op)
def fill_poly(fr, pts, col, op=1.0):
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(np.float32(pts) * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
    blend(fr, m.astype(np.float32) / 255, 0, 0, col, op)
def star_pts(cx, cy, r, n=5, inner=0.45, rot=-np.pi / 2):
    return [(cx + (r if i % 2 == 0 else r * inner) * np.cos(rot + i * np.pi / n), cy + (r if i % 2 == 0 else r * inner) * np.sin(rot + i * np.pi / n)) for i in range(2 * n)]
def heart_pts(cx, cy, s):
    u = np.linspace(0, 2 * np.pi, 60)
    return list(zip(cx + s * 16 * np.sin(u) ** 3 / 16, cy - s * (13 * np.cos(u) - 5 * np.cos(2 * u) - 2 * np.cos(3 * u) - np.cos(4 * u)) / 16))
def sparkle(fr, cx, cy, r, t, t0, col=PINK):
    """4-point twinkle: pops in, breathes, fades"""
    lt = t - t0
    if lt < 0 or lt > 1.6: return
    s = eback(lt / 0.35) * (1 - eio((lt - 1.1) / 0.5)) * (1 + 0.12 * np.sin(lt * 12))
    if s > 0.02: fill_poly(fr, star_pts(cx, cy, r * s, 4, 0.22, -np.pi / 2 + lt * 0.8), col)

# ---------------- sources ----------------
class Src:
    def __init__(s, path): s.path = path; s.cap = None; s.i = -1; s.f = None
    def get(s, i):
        i = max(0, int(i))
        if s.cap is None or i < s.i: s.cap = cv2.VideoCapture(s.path); s.i = -1
        while s.i < i:
            ok, f = s.cap.read()
            if not ok: break
            s.i += 1; s.f = f
        return cv2.cvtColor(s.f, cv2.COLOR_BGR2RGB)
    def at(s, t): return s.get(round(t * FPS))
V = {"15": Src(os.path.join(WORK, "IMG_15.mp4"))}

def cover(img, w, h, fx=0.5, fy=0.5, z=1.0):
    ih, iw = img.shape[:2]; s = max(w / iw, h / ih) * z
    tx = min(0, max(w - iw * s, w / 2 - fx * iw * s)); ty = min(0, max(h - ih * s, h / 2 - fy * ih * s))
    M = np.float32([[s, 0, tx], [0, s, ty]])
    return cv2.warpAffine(img, M, (int(w), int(h)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255, M
def full(img, fx=0.5, fy=0.5, z=1.0): return cover(img, W, H, fx, fy, z)[0]

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
BG = np.clip(NAVY * (1 + 0.35 * np.exp(-(((xx - 540) / 900) ** 2 + ((yy - 520) / 900) ** 2)))[..., None], 0, 1).astype(np.float32)
def ramp(y0, y1): return np.clip((yy[:, :1] - y0) / (y1 - y0), 0, 1)[..., None]
GBOT = (ramp(950, 1650) ** 1.3 * 0.92).astype(np.float32); GTOP = ((1 - ramp(330, 900)) ** 1.4 * 0.92).astype(np.float32)
VIG = (np.clip(((xx - 540) / 760) ** 2 + ((yy - 960) / 1250) ** 2, 0, 1) ** 1.5 * 0.35)[..., None].astype(np.float32)
def shade(fr, g, col=DEEP): fr *= 1 - g; fr += g * col

_lg = np.asarray(Image.open(os.path.join(SRC, "logo.png")).convert("RGBA"))[..., 3].astype(np.float32) / 255
_ys, _xs = np.where(_lg > 0.05); _lg = _lg[_ys.min():_ys.max() + 1, _xs.min():_xs.max() + 1]
_cols = np.where(_lg.max(0) > 0.05)[0]; _gap = np.argmax(np.diff(_cols)); _split = (_cols[_gap] + _cols[_gap + 1]) // 2
def logo_alpha(width): return cv2.resize(_lg, (width, int(_lg.shape[0] * width / _lg.shape[1])), interpolation=cv2.INTER_AREA)

def chip(fr, txt, x, y, op=1.0):
    a = alpha_img(txt, mont(30, 600)); w, h = a.shape[1] + 34, 58
    box = np.zeros((h + 4, w + 4), np.float32); box[2:h + 2, 2:w + 2] = 1; blend(fr, box, x, y, NAVY, 0.8 * op)
    ln = np.zeros((h + 4, w + 4), np.uint8); cv2.rectangle(ln, (2, 2), (w + 1, h + 1), 255, 2, cv2.LINE_AA); blend(fr, ln.astype(np.float32) / 255, x, y, PINK, op)
    blend(fr, a, x + 17, y + 2 + (h - a.shape[0]) / 2 + 2, WHITE, op)

def notch_poly(x0, y0, x1, y1, nw, nh, o=0):
    """brand frame, mirrored for this reel: the step is cut out of the top-left corner"""
    return np.float32([(x0 + nw - o, y0 - o), (x1 + o, y0 - o), (x1 + o, y1 + o), (x0 - o, y1 + o), (x0 - o, y0 + nh - o), (x0 + nw - o, y0 + nh - o)])
def poly_mask(pts):
    m = np.zeros((H * 2, W * 2), np.uint8); cv2.fillPoly(m, [np.round(pts * 2).astype(np.int32)], 255, cv2.LINE_AA)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
def card(fr, img, box, nw, nh, p_in, t_line_p, dx=0, dy=0, fx=0.5, fy=0.45, z=1.0):
    """photo inside a notched card that slides in; the outline draws itself"""
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
    im, _ = cover(img, w, h, fx, fy, z); m = poly_mask(notch_poly(x0 + dx, y0 + dy, x1 + dx, y1 + dy, nw, nh))[..., None] * p_in
    sub = np.zeros_like(fr); xi, yi = int(x0 + dx), int(y0 + dy)
    xa, xb, ya, yb = max(0, xi), min(W, xi + w), max(0, yi), min(H, yi + h)
    if xb > xa and yb > ya: sub[ya:yb, xa:xb] = im[ya - yi:yb - yi, xa - xi:xb - xi]
    fr[:] = fr * (1 - m) + sub * m
    path_draw(fr, notch_poly(x0, y0, x1, y1, nw, nh, 12), t_line_p, WHITE, 2, True)

def iris(old, new, p, cx=540, cy=760):
    r = eio(p) * 1500; d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2); m = np.clip((r - d) / 3 + 0.5, 0, 1)[..., None]
    fr = old * (1 - m) + new * m; ring_ = np.clip(1 - np.abs(d - r) / 4, 0, 1)[..., None] * (0 < p < 1)
    return (fr * (1 - ring_) + PINK * ring_).astype(np.float32)
def zoom_through(old, new, p):
    e = eio(p); o = cv2.resize(old, None, fx=1 + 0.35 * e, fy=1 + 0.35 * e); oy, ox = (o.shape[0] - H) // 2, (o.shape[1] - W) // 2
    o = o[oy:oy + H, ox:ox + W]; s = 0.88 + 0.12 * e; n = cv2.resize(new, None, fx=s, fy=s); nn = np.zeros_like(new) + DEEP
    py, px = (H - n.shape[0]) // 2, (W - n.shape[1]) // 2; nn[py:py + n.shape[0], px:px + n.shape[1]] = n
    return (o * (1 - e) + nn * e).astype(np.float32)
def diag_wipe(old, new, p):
    e = eio(p); d = (xx * 0.45 + yy) - lerp(-500, H + 600, e); m = np.clip(-d / 3 + 0.5, 0, 1)[..., None]
    ln = np.clip(1 - np.abs(d) / 4, 0, 1)[..., None] * (0 < p < 1)
    fr = old * (1 - m) + new * m; return (fr * (1 - ln) + PINK * ln).astype(np.float32)


# ---------------- subtitles: hand-checked words, timings from speech recognition (relative to each line's start)
SUBS = {
    'S1': [('Что', 0.04, 0.24), ('будет,', 0.24, 0.64), ('если', 0.76, 1.18), ('не', 1.18, 1.42), ('соблюдать', 1.42, 1.88), ('рекомендации?', 1.88, 2.72)],
    'S2': [('Результат', 0.1, 0.6), ('операции', 0.6, 1.22), ('зависит', 1.22, 1.7), ('не', 1.7, 1.96), ('только', 1.96, 2.22), ('от', 2.22, 2.46), ('хирурга.', 2.46, 3.18)],
    'S3': [('Хирург', 0.12, 0.58), ('делает', 0.58, 1.0), ('свою', 1.0, 1.28), ('часть', 1.28, 1.82), ('работы', 1.82, 2.26), ('в', 2.26, 2.44), ('операционной.', 2.44, 3.22)],
    'S4': [('Но', 0.04, 0.28), ('после', 0.28, 0.78), ('этого', 0.78, 1.3), ('начинается', 1.3, 2.04), ('этап,', 2.04, 2.54), ('где', 2.62, 2.94), ('очень', 2.94, 3.44), ('многое', 3.44, 3.98), ('зависит', 3.98, 4.24), ('от', 4.24, 4.4), ('пациента.', 4.4, 5.02)],
    'S5': [('Если', 0.16, 0.52), ('не', 0.52, 0.72), ('носить', 0.72, 1.02), ('компрессионное', 1.02, 1.72), ('бельё,', 1.72, 2.26), ('рано', 2.56, 2.78), ('вернуться', 2.78, 3.18), ('к', 3.18, 3.38), ('нагрузкам,', 3.38, 4.1), ('поднимать', 4.4, 4.9), ('тяжести,', 4.9, 5.52), ('пропускать', 5.6, 6.14), ('осмотры,', 6.14, 7.0), ('игнорировать', 7.1, 7.72), ('ограничения,', 7.72, 8.56), ('восстановление', 8.7, 9.34), ('может', 9.34, 9.6), ('идти', 9.6, 9.9), ('хуже.', 9.9, 10.34)],
    'S6': [('Я', 0.06, 0.38), ('понимаю,', 0.38, 0.9), ('что', 1.08, 1.3), ('всем', 1.3, 1.54), ('хочется', 1.54, 2.06), ('быстрее', 2.06, 2.52), ('вернуться', 2.52, 2.92), ('к', 2.92, 3.14), ('обычной', 3.14, 3.52), ('жизни.', 3.52, 3.88)],
    'S7': [('Но', 0.04, 0.26), ('организм', 0.26, 0.8), ('не', 0.8, 1.0), ('ускоряется', 1.0, 1.82), ('от', 1.82, 2.12), ('нашего', 2.12, 2.44), ('желания.', 2.44, 3.12)],
    'S8': [('Рекомендации', 0.08, 0.9), ('даются', 0.9, 1.34), ('не', 1.34, 1.54), ('для', 1.54, 1.76), ('галочки.', 1.76, 2.32)],
    'S9': [('Они', 0.06, 0.26), ('нужны,', 0.26, 0.7), ('чтобы', 0.78, 0.96), ('ткани', 0.96, 1.44), ('спокойно', 1.44, 2.16), ('заживали.', 2.16, 2.86), ('Отёк', 3.08, 3.4), ('уходил,', 3.4, 3.96), ('а', 4.08, 4.24), ('результат', 4.24, 4.66), ('формировался', 4.66, 5.48), ('правильно.', 5.48, 5.98)],
    'S10': [('Поэтому', 0.08, 0.5), ('перед', 0.5, 0.68), ('операцией', 0.68, 1.38), ('я', 1.38, 1.58), ('всегда', 1.58, 1.96), ('спрашиваю,', 1.96, 2.62), ('сможете', 2.86, 3.46), ('ли', 3.46, 3.62), ('вы', 3.62, 3.88), ('реально', 3.88, 4.36), ('соблюдать', 4.36, 4.98), ('режим.', 4.98, 5.28)],
    'S11': [('Если', 0.04, 0.44), ('нет,', 0.44, 0.94), ('лучше', 1.38, 1.7), ('выбрать', 1.7, 2.16), ('другое', 2.16, 2.68), ('время.', 2.68, 2.94)],}
SUB_Y = 1420                                                           # Instagram Reels safe zone: x <= 940, y 250-1500
def _sub_words():
    """one word at a time: each stays until the next one starts (or a beat after it ends)"""
    out = [(w.strip(",."), t0 + a, t0 + e) for b, s_, parts, t0, d in SEG for w, a, e in SUBS[b]]
    return [(w, a, min(e + 0.3, out[i + 1][1] - 0.04) if i + 1 < len(out) else e + 0.3) for i, (w, a, e) in enumerate(out)]
SUB_WORDS = _sub_words()
def subtitles(fr, t):
    for w, a, e in SUB_WORDS:
        t0 = a - 0.04
        if not (t0 <= t < e): continue
        p = eo((t - t0) / 0.12); im = scaled(alpha_img(w, mont(60, 300)), 0.85 + 0.15 * p)   # thin Montserrat, pops in
        h = 84; bw = im.shape[1] + 40; op = min(1, (e - t) / 0.06)
        box = np.zeros((h, int(bw)), np.uint8); cv2.rectangle(box, (0, 0), (box.shape[1] - 1, h - 1), 255, -1)
        blend(fr, cv2.GaussianBlur(box, (0, 0), 1.2).astype(np.float32) / 255, 500 - bw / 2, SUB_Y - h / 2, DEEP, 0.8 * op)
        blend(fr, im, 500 - im.shape[1] / 2, SUB_Y - im.shape[0] / 2 + 3, WHITE, op)
def lift(hd, bottom=1230):
    """bottom-anchor a headline inside the safe zone"""
    hd.y = bottom - hd.h; return hd

# ---------------- building blocks ----------------
def SUR(t): return V["15"].at(src_of(t, "15"))
FACE = (560, 760); CC, CR, CS = (540, 560), 280, 0.737                 # face in the take; circle centre, radius, scale (a down-scale)
def ring(fr, cx, cy, r, op=1.0, th=5):
    path_draw(fr, [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 120)], 1, PINK, th, op=op)
_rng = np.random.default_rng(5); DOTS = [(_rng.uniform(60, 1020), _rng.uniform(0, 1920), _rng.uniform(2, 6), _rng.uniform(20, 60)) for _ in range(22)]
def gbg(t):
    """navy backdrop with slow pink dust drifting up"""
    fr = BG.copy(); m = np.zeros((H, W), np.uint8)
    for x, y, r, v in DOTS: cv2.circle(m, (int(x + 12 * np.sin(t + y)), int((y - v * t) % H)), int(r), 255, -1, cv2.LINE_AA)
    blend(fr, m.astype(np.float32) / 255, 0, 0, PINK, 0.22); return fr
def circle_layout(t, t0, morph=True):
    """surgeon in a big circle at the top over the backdrop; on entry the full frame shrinks into the circle"""
    fr = gbg(t); img = SUR(t); p = eio((t - t0) / 0.45) if morph else 1.0
    s = lerp(1.0, CS, p); fx, fy = FACE; cx, cy = lerp(fx, CC[0], p), lerp(fy, CC[1], p); r = lerp(1150, CR, p)
    M = np.float32([[s, 0, cx - fx * s], [0, s, cy - fy * s]])
    sub = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2); m = np.clip(r - d + 0.5, 0, 1)[..., None]
    sh = np.clip((r + 26 - d) / 26, 0, 1)[..., None] * 0.5 * p; fr = fr * (1 - sh) + DEEP * sh
    fr = fr * (1 - m) + sub * m; ring(fr, cx, cy, r + 3, cl((p - 0.5) * 2)); return fr
def fullshot(t, z=1.0, dz=0.0, t0=0.0, dur=3.0):
    """the take full screen; z = punch-in (≤ 1.14), dz = slow drift"""
    return full(SUR(t), FACE[0] / W, FACE[1] / H, z + dz * eio((t - t0) / dur))
def pull(t, t0, z0=1.16, z1=1.0, d=1.4):
    """camera pull-back: the shot starts tight and eases out to the wide framing"""
    return lerp(z0, z1, eo(cl((t - t0) / d)))
def bump(t, t0, a=0.05, d=0.3): return a * (1 - eo((t - t0) / d)) if t >= t0 else 0.0
def shake(fr, t, t0, amp=10, d=0.35):
    if not (t0 <= t < t0 + d): return fr
    k = 1 - (t - t0) / d; dx, dy = int(amp * k * np.sin((t - t0) * 90)), int(amp * k * np.cos((t - t0) * 70))
    return np.roll(fr, (dy, dx), (0, 1))
def rrect(x0, y0, x1, y1, r):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1, r = int(x0), int(y0), int(x1), int(y1), int(r)
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), 255, -1); cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)): cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m.astype(np.float32) / 255
def rrect_line(fr, x0, y0, x1, y1, r, col, th=3, op=1.0):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1, r = int(x0), int(y0), int(x1), int(y1), int(r)
    for a, b in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)), ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))): cv2.line(m, a, b, 255, th, cv2.LINE_AA)
    for (cx, cy), a0 in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270), ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)): cv2.ellipse(m, (cx, cy), (r, r), 0, a0, a0 + 90, 255, th, cv2.LINE_AA)
    blend(fr, m.astype(np.float32) / 255, 0, 0, col, op)
def text_c(fr, txt, cx, cy, f, col, op=1.0):
    a = alpha_img(txt, f); blend(fr, a, cx - a.shape[1] / 2, cy - a.shape[0] / 2, col, op); return a.shape[1]

# icons, drawn in a ±100 box around (cx, cy) at scale s; p = draw-on progress
def icon(fr, kind, cx, cy, s=1.0, p=1.0, col=WHITE, th=6, t=0.0):
    P_ = lambda pts: [(cx + x * s, cy + y * s) for x, y in pts]; th = max(2, int(th * min(1, s * 1.2)))
    if kind == "shorts":                                                 # compression garment
        path_draw(fr, P_([(-80, -82), (80, -82), (80, -56), (-80, -56)]), p, col, th, True)
        path_draw(fr, P_([(-80, -56), (80, -56), (96, 72), (16, 82), (0, 6), (-16, 82), (-96, 72)]), p, col, th, True)
        for y in (-20, 14, 46): path_draw(fr, P_([(-70 - y * 0.1, y), (-12, y)]), p, PINK, max(2, th - 2)); path_draw(fr, P_([(12, y), (70 + y * 0.1, y)]), p, PINK, max(2, th - 2))
    elif kind == "dumbbell":
        path_draw(fr, P_([(-60, 0), (60, 0)]), p, col, th + 4)
        for sx in (-1, 1):
            path_draw(fr, P_([(sx * 46, -48), (sx * 70, -48), (sx * 70, 48), (sx * 46, 48)]), p, col, th, True)
            path_draw(fr, P_([(sx * 70, -30), (sx * 92, -30), (sx * 92, 30), (sx * 70, 30)]), p, PINK, th, True)
    elif kind == "weight":                                               # kettlebell
        path_draw(fr, P_([(64 * np.cos(a), 28 + 64 * np.sin(a)) for a in np.linspace(-0.95 * np.pi / 2 - 0.8, 1.5 * np.pi + 0.8, 50)]), p, col, th)
        path_draw(fr, P_([(42 * np.cos(a), -40 + 42 * np.sin(a)) for a in np.linspace(np.pi * 1.05, np.pi * 1.95, 30)]), p, col, th)
        path_draw(fr, P_([(-40, -36), (-40, -14)]), p, col, th); path_draw(fr, P_([(40, -36), (40, -14)]), p, col, th)
        if p > 0.7: text_c(fr, "10 КГ", cx, cy + 32 * s, bebas(max(12, int(46 * s))), PINK, cl((p - 0.7) / 0.3))
    elif kind == "calendar":
        path_draw(fr, P_([(-82, -64), (82, -64), (82, 86), (-82, 86)]), p, col, th, True)
        if p > 0.3: fill_poly(fr, P_([(-82, -64), (82, -64), (82, -30), (-82, -30)]), PINK, cl((p - 0.3) / 0.3))
        for x in (-44, 44): path_draw(fr, P_([(x, -84), (x, -50)]), p, col, th)
        for i in range(3):
            for j in range(4):
                if p > 0.5: fill_poly(fr, P_([(-54 + j * 36 + 6 * np.cos(a), -2 + i * 30 + 6 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8)]), col, 0.7 * cl((p - 0.5) / 0.3))
    elif kind == "stop":
        path_draw(fr, P_([(86 * np.cos(np.pi / 8 + k * np.pi / 4), 86 * np.sin(np.pi / 8 + k * np.pi / 4)) for k in range(8)]), p, PINK, th, True)
        path_draw(fr, P_([(0, -44), (0, 18)]), p, col, th + 4)
        if p > 0.8: fill_poly(fr, P_([(8 * np.cos(a), 46 + 8 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 12)]), col)
    elif kind == "down":                                                 # recovery curve goes the wrong way
        path_draw(fr, P_([(-86, -84), (-86, 80), (90, 80)]), p, col, th - 1)
        path_draw(fr, P_([(-70, -50), (-30, -20), (0, -36), (34, 10), (70, 50)]), p, PINK, th + 1)
        if p > 0.9: fill_poly(fr, P_([(80, 62), (50, 52), (72, 30)]), PINK)
    elif kind == "plus":
        path_draw(fr, P_([(72 * np.cos(a), 72 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 60)]), p, col, th)
        path_draw(fr, P_([(0, -38), (0, 38)]), p, PINK, th + 6); path_draw(fr, P_([(-38, 0), (38, 0)]), p, PINK, th + 6)
    elif kind == "drop":                                                 # swelling drop that shrinks
        k = 1 - 0.35 * cl(t); pts = [(0, -80 * k)] + [(52 * k * np.cos(a), 22 * k + 52 * k * np.sin(a)) for a in np.linspace(-0.25 * np.pi, 1.25 * np.pi, 40)]
        path_draw(fr, P_(pts), p, col, th, True); path_draw(fr, P_([(0, -20), (0, 60)]), p * cl(t * 2), PINK, th); fill_poly(fr, P_([(-16, 46), (16, 46), (0, 68)]), PINK, cl(t * 2) * p)
    elif kind == "spark":
        fill_poly(fr, P_(star_pts(0, 0, 80, 4, 0.28)), PINK, p); fill_poly(fr, P_(star_pts(58, -52, 26, 4, 0.3)), col, cl((p - 0.5) * 2))
    elif kind == "warn":
        path_draw(fr, P_([(0, -84), (92, 72), (-92, 72)]), p, PINK, th, True)
        path_draw(fr, P_([(0, -30), (0, 24)]), p, col, th + 3)
        if p > 0.8: fill_poly(fr, P_([(7 * np.cos(a), 48 + 7 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 12)]), col)
def xmark(fr, cx, cy, r, p, col=PINK, th=8):
    path_draw(fr, [(cx - r, cy - r), (cx + r, cy + r)], eio(p * 2), col, th); path_draw(fr, [(cx + r, cy - r), (cx - r, cy + r)], eio(p * 2 - 1), col, th)
def card(fr, kind, cx, cy, s, op, t_icon, t=0.0):
    hw = 125 * s
    if op <= 0: return
    blend(fr, rrect(cx - hw, cy - hw, cx + hw, cy + hw, 34 * s), 0, 0, NAVY, 0.9 * op)
    rrect_line(fr, cx - hw, cy - hw, cx + hw, cy + hw, 34 * s, PINK, max(2, int(4 * s)), op)
    icon(fr, kind, cx, cy, 0.95 * s, t_icon, t=t)

# ---------------- scenes ----------------
# S1 — straight into the question
H1 = lift(Drop(["А ЕСЛИ Я НЕМНОГО", "*НАРУШУ РЕЖИМ?*"], 96, 70, 0, maxw=860), 1320)   # below the chin (≈ y 940 at the tightest punch-in)
def s1(t):
    fr = fullshot(t, pull(t, 0, 1.16, 1.0, 2.6)); shade(fr, GBOT)
    k = eback((t - 0.35) / 0.35); wob = 6 * np.sin(t * 9) * (1 - cl((t - 0.35) / 1.5))
    if k > 0: icon(fr, "warn", 820 + wob, 430, 0.75 * k, eo((t - 0.35) / 0.4))
    H1.draw(fr, t, 0.05); return shake(fr, t, 0.35, 8)

# S2–S3 — full screen pulling back; the result bar: «хирург» fills in the operating room, «вы» stays open
HB = Drop(["РЕЗУЛЬТАТ ОПЕРАЦИИ ="], 72, 0, 1092, maxw=780)
HB.x = 540 - max(r[1] for r in HB.rows) / 2
BX0, BX1, BY0, BY1 = 150, 930, 1180, 1262                              # inside the Reels safe zone (x ≤ 940)
def result_bar(fr, t, t0, fill1, fill2, y0=BY0, y1=BY1, x0=BX0, x1=BX1, f_lab=None, small=False):
    xm = (x0 + x1) / 2; p_out = eio((t - t0) / 0.6)
    rr = 22 if not small else 14
    if fill1 > 0: blend(fr, rrect(x0, y0, x0 + (xm - x0) * fill1, y1, rr), 0, 0, PINK, 0.95)
    if fill2 > 0: blend(fr, rrect(xm, y0, xm + (x1 - xm) * fill2, y1, rr), 0, 0, WHITE, 0.95)
    if p_out > 0:
        rrect_line(fr, x0, y0, x0 + (x1 - x0) * p_out, y1, rr, WHITE, 3, 0.9)
        if p_out >= 1: path_draw(fr, [(xm, y0 + 6), (xm, y1 - 6)], 1, WHITE, 3, op=0.8)
    f = f_lab or bebas(56 if not small else 34)
    if fill1 > 0.5: text_c(fr, "ХИРУРГ", (x0 + xm) / 2, (y0 + y1) / 2 + 3, f, NAVY, cl((fill1 - 0.5) * 2))
    if fill2 > 0.5: text_c(fr, "ВЫ", (xm + x1) / 2, (y0 + y1) / 2 + 3, f, NAVY, cl((fill2 - 0.5) * 2))
    elif p_out >= 1 and not small: text_c(fr, "?", (xm + x1) / 2, (y0 + y1) / 2 + 3, f, WHITE, 0.6 + 0.4 * np.sin(t * 6) ** 2)
def s2(t):
    t0 = T_S["S2"]; t3 = T_S["S3"]                                       # full screen, pulling back; S3 is a jump cut that pulls back again
    fr = fullshot(t, pull(t, t3, 1.14, 1.02, 1.4) if t >= t3 else pull(t, t0, 1.16, 1.0, 1.6)); shade(fr, GBOT); shade(fr, GBOT * 0.5)
    HB.draw(fr, t, t0 + 0.25); f1 = eio((t - T_FILL1) / 0.7)
    result_bar(fr, t, t0 + 0.5, f1, 0)
    lab = Track(["операционная"], 30, 0, 0, 300, center=True); lab.x, lab.y = (BX0 + 540) / 2, BY1 + 30 + lab.top; lab.draw(fr, t, T_FILL1 + 0.3)
    lab2 = Track(["восстановление"], 30, 0, 0, 300, center=True); lab2.x, lab2.y = (540 + BX1) / 2, BY1 + 30 + lab2.top; lab2.draw(fr, t, T_FILL1 + 0.7)
    return fr

# S4 — «после этого начинается этап…»: full screen, the headline, the bar's second half fills on «пациента»
H4 = lift(Drop(["ПОСЛЕ ОПЕРАЦИИ ВАША ЧАСТЬ", "РАБОТЫ *ТОЛЬКО НАЧИНАЕТСЯ*"], 72, 70, 0, maxw=860), 1255)
def s4(t):
    t0 = T_S["S4"]; fr = fullshot(t, pull(t, t0, 1.14, 1.0, 1.5) + bump(t, T_FILL2, 0.04)); shade(fr, GBOT); shade(fr, GBOT * 0.4)
    H4.draw(fr, t, t0 + 0.15)
    if t > t0 + 0.6: result_bar(fr, t, t0 + 0.6, 1.0, eio((t - T_FILL2) / 0.6), 1280, 1330, 70, 590, small=True)
    return fr

# S5 — the deck of «что мешает»: each card pops, gets its pink ✕ and files into the row under the circle
CARDS = [("shorts", "СНЯТЬ КОМПРЕССИЮ РАНЬШЕ"), ("dumbbell", "РАНО ВЕРНУТЬСЯ В СПОРТ"), ("weight", "ПОДНИМАТЬ ТЯЖЕСТИ"),
         ("calendar", "ПРОПУСТИТЬ ОСМОТР"), ("stop", "ИГНОРИРОВАТЬ ОГРАНИЧЕНИЯ"), ("down", "ВОССТАНОВЛЕНИЕ ИДЁТ ХУЖЕ")]
BIG = (540, 1085); SLOT = lambda i: (540 + (i - 2) * 150, 908)
def s5(t):
    t0 = T_S["S5"]; fr = circle_layout(t, t0)
    for i, (kind, lab) in enumerate(CARDS):
        ti = T_CARDS[i] - 0.12
        if t < ti: continue
        nxt = T_CARDS[i + 1] - 0.12 if i + 1 < len(CARDS) else 1e9
        if t < nxt:
            s = eback((t - ti) / 0.3); card(fr, kind, BIG[0], BIG[1], s, 1, eo((t - ti - 0.05) / 0.45))
            if i < 5: xmark(fr, BIG[0] + 104, BIG[1] - 104, 22, (t - ti - 0.5) / 0.35)
            sz = fit_size(lab, 780, 64); text_c(fr, lab, 540, 1268, bebas(sz), PINK if i == 5 else WHITE, eo((t - ti - 0.1) / 0.3))
        else:
            q = eio((t - nxt) / 0.35); (sx, sy) = SLOT(i); cx, cy = lerp(BIG[0], sx, q), lerp(BIG[1], sy, q); s = lerp(1, 0.4, q)
            card(fr, kind, cx, cy, s, 1, 1); xmark(fr, cx + 104 * s, cy - 104 * s, 22 * s + 4, 1, th=max(4, int(8 * s)))
    return fr
def fit_size(txt, maxw, size):
    while bebas(size).getlength(txt) > maxw and size > 30: size -= 2
    return size

# S6 — «всем хочется быстрее…»: the patient's message types itself
def s6(t):
    t0 = T_S["S6"]; fr = fullshot(t, 1.08, -0.05, t0, 4.0); shade(fr, GBOT)
    lt = t - t0 - 0.2
    if lt > 0:
        typing = lt < 0.9; s = eback(lt / 0.3)
        if typing:
            w, h = 170 * s, 84 * s; x0, y0 = 110, 1300 - h
        else:
            q = eo((lt - 0.9) / 0.3); w, h = lerp(170, 590, q), lerp(84, 170, q); x0, y0 = 110, 1300 - h
        blend(fr, rrect(x0, y0, x0 + w, y0 + h, 40), 0, 0, WHITE, 0.97)
        fill_poly(fr, [(x0 + 30, y0 + h - 4), (x0 + 74, y0 + h - 4), (x0 + 18, y0 + h + 34)], WHITE, 0.97)
        if typing:
            for k in range(3):
                b = 0.5 + 0.5 * np.sin(lt * 9 - k * 0.9); m = np.zeros((H, W), np.uint8)
                cv2.circle(m, (int(x0 + 46 + k * 40), int(y0 + h / 2 - 6 * b)), int(10 * s), 255, -1, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, NAVY, 0.4 + 0.6 * b)
        elif lt > 1.05:
            op = eo((lt - 1.05) / 0.25); f = bebas(60)
            a1 = alpha_img("МНЕ УЖЕ ХОРОШО.", f); a2 = alpha_img("МОЖЕТ, УЖЕ МОЖНО?", f)
            blend(fr, a1, x0 + 34, y0 + 20, NAVY, op); blend(fr, a2, x0 + 34, y0 + 20 + 70, NAVY, op)
        tr = Track(["сообщение пациентки"], 28, 112, 0, 300); tr.y = 1300 - 170 - 46 + tr.top; tr.draw(fr, t, t0 + 1.2)
    return fr

# S7 — the key line, word by word, with a punch on every line
K1 = Drop(["ОРГАНИЗМ НЕ"], 92, 70, 1095, maxw=860); K2 = Drop(["*УСКОРЯЕТСЯ*"], 108, 70, 1095 + 88, maxw=860); K3 = Drop(["ОТ НАШЕГО ЖЕЛАНИЯ"], 92, 70, 1095 + 88 + 102, maxw=860)
def s7(t):
    z = 1.12 + sum(bump(t, tk, 0.035, 0.35) for tk in T_KEY); fr = fullshot(t, z); shade(fr, GBOT); fr *= 0.78; fr += 0.22 * DEEP
    K1.draw(fr, t, T_KEY[0] - 0.15, dur=0.25); K2.draw(fr, t, T(31.58) - 0.1, dur=0.25); K3.draw(fr, t, T_KEY[2] - 0.1, dur=0.25)
    for tk in T_KEY:
        u = t - tk
        if 0 < u < 0.5: ring(fr, 540, 1150, 200 + 900 * eo(u / 0.5), 0.5 * (1 - u / 0.5), 4)
    return shake(fr, t, T(31.58), 7, 0.3)

# S8 — «не для галочки»: full screen pulling back; a tick is drawn — and struck out
H8 = Drop(["НЕ ДЛЯ *ГАЛОЧКИ*"], 96, 300, 1182, maxw=630)
T8 = Track(["рекомендации"], 32, 300, 0, 300)
def s8(t):
    t0 = T_S["S8"]; fr = fullshot(t, pull(t, t0, 1.16, 1.0, 1.4)); shade(fr, GBOT); shade(fr, GBOT * 0.5)
    T8.y = 1120 + T8.top; T8.draw(fr, t, t0 + 0.2)
    sx = 10 * np.sin((t - T_NOT) * 40) * (1 - cl((t - T_NOT) / 0.4)) if t > T_NOT else 0; cx, cy, b, k = 190 + sx, 1222, 66, 0.66
    p = eio((t - t0 - 0.25) / 0.4)
    if p > 0:
        blend(fr, rrect(cx - b, cy - b, cx + b, cy + b, 18), 0, 0, NAVY, 0.85 * p); rrect_line(fr, cx - b, cy - b, cx + b, cy + b, 18, WHITE, 5, p)
        path_draw(fr, [(cx - 52 * k, cy + 4 * k), (cx - 14 * k, cy + 44 * k), (cx + 60 * k, cy - 50 * k)], eio((t - t0 - 0.6) / 0.35), PINK, 12)
        if t > T_NOT: path_draw(fr, [(cx - 92, cy + 60), (cx + 92, cy - 60)], eio((t - T_NOT) / 0.25), WHITE, 9)
    H8.draw(fr, t, T_NOT - 0.1)
    return fr

# S9 — what the regime is for: three healing steps tick in under the headline
H9 = Drop(["РЕЖИМ ПОСЛЕ ОПЕРАЦИИ", "ЭТО *ЧАСТЬ РЕЗУЛЬТАТА*"], 76, 70, 885, maxw=860)
HEAL = [("plus", "ТКАНИ СПОКОЙНО ЗАЖИВАЮТ"), ("drop", "ОТЁК УХОДИТ"), ("spark", "РЕЗУЛЬТАТ ФОРМИРУЕТСЯ ПРАВИЛЬНО")]
def s9(t):
    t0 = T_S["S9"]; fr = circle_layout(t, t0); H9.draw(fr, t, t0 + 0.1)
    for k, (kind, lab) in enumerate(HEAL):
        ti = T_HEAL[k] - 0.12
        if t < ti: continue
        y = 1100 + k * 95; s = eback((t - ti) / 0.3)
        icon(fr, kind, 120, y, 0.36 * s, eo((t - ti) / 0.4), t=(t - ti) / 1.5)
        a = alpha_img(lab, bebas(fit_size(lab, 740, 64))); p = eo((t - ti - 0.1) / 0.35); a2 = a.copy(); a2[:, int(a.shape[1] * p):] = 0
        blend(fr, a2, 190, y - a.shape[0] / 2 + 6, WHITE)
    return fr

# S10–S11 — the question with ДА / НЕТ; «нет» gets pressed, the calendar moves on
H10 = lift(Drop(["СМОЖЕТЕ ЛИ ВЫ РЕАЛЬНО", "*СОБЛЮДАТЬ РЕЖИМ?*"], 78, 70, 0, maxw=860), 1222)
def button(fr, txt, cx, cy, op, on=0.0, sc=1.0):
    w, h = 250 * sc, 92 * sc; x0, y0 = cx - w / 2, cy - h / 2
    blend(fr, rrect(x0, y0, x0 + w, y0 + h, 46 * sc), 0, 0, NAVY * (1 - on) + PINK * on, 0.88 * op)
    rrect_line(fr, x0, y0, x0 + w, y0 + h, 46 * sc, PINK if not on else WHITE, 3, op)
    text_c(fr, txt, cx, cy + 4, bebas(int(64 * sc)), WHITE * (1 - on) + NAVY * on, op)
def flip_calendar(fr, t, t0, cx=820, cy=440):
    lt = t - t0
    if lt < 0: return
    s = eback(lt / 0.3) * 0.8; icon(fr, "calendar", cx, cy, s, eo(lt / 0.35))
    fl = (lt - 0.35) / 0.18
    if 0 < fl < 4:
        u = fl % 1; y_top = cy - 30 * s; y = lerp(cy + 86 * s, y_top, eio(u))
        fill_poly(fr, [(cx - 78 * s, y_top), (cx + 78 * s, y_top), (cx + 78 * s, y), (cx - 78 * s, y)], WHITE, 0.35 * (1 - u))
def s10(t):
    t0 = T_S["S10"]; late = t >= T_S["S11"]                             # S11 is a jump cut: tighter framing
    fr = fullshot(t, pull(t, T_S["S11"], 1.14, 1.04, 2.0) if late else pull(t, t0, 1.16, 1.0, 1.8)); shade(fr, GBOT); shade(fr, GBOT * 0.4)
    H10.draw(fr, t, T_ASK - 0.1)
    op = eo((t - T_ASK - 0.7) / 0.3); on = eo((t - T_NO) / 0.2) if t > T_NO else 0.0
    if op > 0:
        button(fr, "ДА", 190, 1285, op * (1 - 0.6 * on), 0.0, 0.8); button(fr, "НЕТ", 420, 1285, op, on, 0.8 + bump(t, T_NO, 0.12, 0.3))
    flip_calendar(fr, t, T_TIME - 0.1)
    return fr

# END — closing card
HZ = Drop(["ЛУЧШЕ ПЕРЕНЕСТИ ОПЕРАЦИЮ,", "ЧЕМ НЕ ДАТЬ СЕБЕ", "*НОРМАЛЬНО ВОССТАНОВИТЬСЯ*"], 104, 70, 760, maxw=860)
def s_end(t):
    lt = t - T_END; fr = gbg(t)
    lg = logo_alpha(320); blend(fr, lg, 540 - 160, 420, PINK, eo((lt - 0.1) / 0.5))
    HZ.draw(fr, t, T_END + 0.2, dur=0.3)
    tr = Track(["режим — это часть результата"], 34, 72, 0, 300); tr.y = HZ.y + HZ.h + 60 + tr.top; tr.draw(fr, t, T_END + 1.3)
    return fr

SCENES = [(0, s1), (T_S["S2"], s2), (T_S["S4"], s4), (T_S["S5"], s5), (T_S["S6"], s6), (T_S["S7"], s7), (T_S["S8"], s8), (T_S["S9"], s9), (T_S["S10"], s10), (T_END, s_end)]
TRANS = {T_S["S8"]: ("flash", 0.18), T_S["S4"]: ("zoom", 0.35), T_S["S6"]: ("zoom", 0.35), T_S["S7"]: ("flash", 0.2), T_S["S10"]: ("zoom", 0.35), T_END: ("iris", 0.5)}
def push(old, new, p):
    e = eio(p); x = int(W * e); fr = np.empty_like(old); fr[:, :W - x] = old[:, x:]; fr[:, W - x:] = new[:, :x]
    if 0 < x < W: fr[:, max(0, W - x - 2):W - x + 2] = PINK
    return fr
def render(t):
    k = max(i for i, (s0, _) in enumerate(SCENES) if t >= s0); s0, fn = SCENES[k]
    kind, d = TRANS.get(s0, ("none", 0))
    if kind != "none" and t < s0 + d:
        p = (t - s0) / d
        if kind == "flash":
            fr = fn(t); f = 1 - eo(p); return (fr * (1 - f) + f).astype(np.float32)
        old = SCENES[k - 1][1](t); new = fn(t)
        if kind == "iris": return iris(old, new, p, 540, 900)
        if kind == "zoom": return zoom_through(old, new, p)
        if kind == "push": return push(old, new, p)
        if kind == "diag": return diag_wipe(old, new, p)
    return fn(t)
def render(t, _base=render):
    fr = np.asarray(_base(t), np.float32).copy(); subtitles(fr, t); return fr

# ---------------- audio ----------------
def lp(x, fc):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); return np.fft.irfft(X / np.sqrt(1 + (f / fc) ** 2), len(x))
def hp(x, fc): return x - lp(x, fc)
rng = np.random.default_rng(11)
def env(n, a, d): t = np.arange(n) / SR; return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / d)
def whoosh(dur=0.6, f0=300, f1=5000):
    n = int(dur * SR); x = rng.standard_normal(n); t = np.arange(n) / SR; X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1 / SR); out = np.zeros(n)
    for k in range(8):
        c = f0 * (f1 / f0) ** (k / 7); bp = np.exp(-((np.log(f + 1) - np.log(c)) ** 2) / 0.25)
        out += np.fft.irfft(X * bp, n) * np.exp(-((t / dur - k / 7) ** 2) / 0.02)
    out *= np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5; return out / np.abs(out).max()
def pop(f0=1100, f1=380, dur=0.09):
    n = int(dur * SR); t = np.arange(n) / SR; ph = 2 * np.pi * np.cumsum(f1 + (f0 - f1) * np.exp(-t / 0.015)) / SR
    return np.sin(ph) * env(n, 0.002, 0.03)
def tick(): n = int(0.012 * SR); return hp(rng.standard_normal(n), 3000) * env(n, 0.0005, 0.003)
def boom(dur=0.9):
    n = int(dur * SR); t = np.arange(n) / SR; ph = 2 * np.pi * np.cumsum(42 + 70 * np.exp(-t / 0.05)) / SR
    return np.sin(ph) * env(n, 0.002, 0.28) + 0.25 * lp(rng.standard_normal(n), 900) * env(n, 0.001, 0.04)
def chime(f=1318.5, dur=1.6):
    n = int(dur * SR); t = np.arange(n) / SR
    return sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / (0.5 / r ** 0.5)) for r, a in [(1, 1), (2.01, 0.4), (3.02, 0.2), (4.2, 0.1)]) * env(n, 0.003, 9)
def shutter():
    n = int(0.16 * SR); x = hp(rng.standard_normal(n), 1800); e = np.zeros(n); e[:int(0.02 * SR)] = 1; e[int(0.07 * SR):int(0.1 * SR)] = 0.7
    return x * lp(e, 300) * 3
def flash_hit(): x = boom(0.5) * 0.7; x[:int(0.25 * SR)] += whoosh(0.25, 2000, 12000) * 0.6; return x
def heartbeat():
    n = int(0.5 * SR); t = np.arange(n) / SR; b = lambda t0: np.sin(2 * np.pi * 55 * (t - t0)) * env(n, 0.001, 0.06) * (t >= t0)
    return lp(np.roll(b(0), 0) + 0.7 * np.roll(b(0), int(0.18 * SR)), 400)
def sparkle_snd(): n = int(0.5 * SR); t = np.arange(n) / SR; return sum(np.sin(2 * np.pi * f * t) for f in (2637, 3520, 4186)) * env(n, 0.002, 0.12) / 3
def riser(dur=1.2):
    n = int(dur * SR); t = np.arange(n) / SR; x = hp(rng.standard_normal(n), 1500) * (t / dur) ** 2
    return x / np.abs(x).max() + 0.3 * np.sin(2 * np.pi * np.cumsum(300 + 900 * (t / dur) ** 2) / SR) * (t / dur) ** 2

# this reel's own sound palette: wood, water, paper, felt, glass
def marimba(f=587.3, dur=0.5):
    n = int(dur * SR); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 3.93 * t) * np.exp(-t / 0.03)) * env(n, 0.001, 0.16)
def water(f0=500, f1=1700):
    n = int(0.12 * SR); t = np.arange(n) / SR; ph = 2 * np.pi * np.cumsum(f0 + (f1 - f0) * (1 - np.exp(-t / 0.015))) / SR
    return np.sin(ph) * env(n, 0.001, 0.035)
def paper(dur=0.28, fc=3500):
    n = int(dur * SR); t = np.arange(n) / SR; x = rng.standard_normal(n)
    x = lp(hp(x, fc * 0.5), fc * 2.2) * (np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.7)          # smooth, no flutter
    return x / (np.abs(x).max() + 1e-9)
def felt(f=95, dur=0.3):
    n = int(dur * SR); t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t) * env(n, 0.002, 0.07) + 0.5 * lp(rng.standard_normal(n), 600) * env(n, 0.001, 0.018)
def glass(f=1760, dur=1.4):
    n = int(dur * SR); t = np.arange(n) / SR
    return sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / d) for r, a, d in [(1, 1, 0.6), (2.76, 0.45, 0.3), (5.4, 0.2, 0.15)]) * env(n, 0.002, 9) / 1.6
def swell(dur=1.0):
    n = int(dur * SR); t = np.arange(n) / SR; x = lp(rng.standard_normal(n), 1800) * (t / dur) ** 2.2
    return x / (np.abs(x).max() + 1e-9)
def click():
    n = int(0.09 * SR); t = np.arange(n) / SR; x = hp(rng.standard_normal(n), 2500)
    e = np.exp(-t / 0.004) + 0.7 * np.exp(-np.clip(t - 0.045, 0, None) / 0.005) * (t > 0.045)
    x = x * e / 2; k = felt(160, 0.08); x[:len(k)] += k * 0.4; return x
PENTA = [587.3, 659.3, 740.0, 880.0, 987.8, 1174.7]                    # D major pentatonic, matches the music

MUSIC = os.path.join(SRC, "song.wav")                                  # «Champagne Coast — Instrumental» (audio of the screen recording, 48 kHz stereo)
M0 = 3.5                                                               # skip the recording's fade-in; the track outlasts the reel, no loop needed
G_VO, G_GAP = 0.022, 0.075
def track(n):
    w = wave.open(MUSIC); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64).reshape(-1, w.getnchannels()) / 32768
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    seg = x[int(M0 * SR):int(M0 * SR) + n]; seg = seg / np.sqrt(np.mean(seg.mean(1) ** 2))
    return np.pad(seg, ((0, n - len(seg)), (0, 0)))

def build_audio(path):
    n = int(DUR * SR); vo = np.zeros(n); act = np.zeros(n); srcs = {}
    ir_t = np.arange(int(0.45 * SR)) / SR; IR = rng.standard_normal(len(ir_t)) * np.exp(-ir_t / 0.11); IR = lp(IR, 3500); IR /= np.sqrt(np.sum(IR ** 2))
    for b, s_, parts, t0, d in SEG:
        if s_ not in srcs:
            w = wave.open(os.path.join(WORK, f"IMG_{s_}.wav")); srcs[s_] = hp(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768, 90)
        x = srcs[s_]; seg = x[int(parts[0][0] * SR):int(parts[0][1] * SR)].copy()
        for a, o in parts[1:]:                                          # joined parts: 15 ms crossfade
            nxt = x[int(a * SR):int(o * SR)]; f = int(0.015 * SR); r = np.linspace(0, 1, f)
            seg = np.concatenate([seg[:-f], seg[-f:] * (1 - r) + nxt[:f] * r, nxt[f:]])
        fi, fo = int(0.03 * SR), int(0.11 * SR)
        seg[:fi] *= np.sin(np.linspace(0, np.pi / 2, fi)) ** 2; seg[-fo:] *= np.cos(np.linspace(0, np.pi / 2, fo)) ** 2
        fr_rms = np.sqrt(np.mean(seg[:len(seg) // 960 * 960].reshape(-1, 960) ** 2, 1)); g = 1 / (np.sqrt(np.mean(np.sort(fr_rms)[-max(1, len(fr_rms) * 2 // 5):] ** 2)) + 1e-6)
        tail = np.convolve(seg[-int(0.25 * SR):], IR)[int(0.25 * SR):] * 0.22
        seg = np.concatenate([seg, tail]) * g
        i = int(t0 * SR); vo[i:i + len(seg)] += seg[:n - i]; act[max(0, i - int(0.25 * SR)):i + int(d * SR) + int(0.1 * SR)] = 1
    vo = np.tanh(vo * 0.16 * 1.6) / 1.6
    dd = np.convolve(act, np.ones(int(0.25 * SR)) / int(0.25 * SR), "same")
    tt = np.arange(n) / SR; fade = np.clip((DUR - tt) / 1.8, 0, 1) ** 1.5 * np.minimum(1, tt / 0.15)
    mus = track(n) * ((G_GAP - (G_GAP - G_VO) * np.clip(dd, 0, 1)) * fade)[:, None]
    sfx = np.zeros(n + 2 * SR)
    def put(x, t, g): i = int(max(0, t) * SR); seg = sfx[i:i + len(x)]; seg += x[:len(seg)] * g
    put(pop(1300, 500, 0.1), 0.35, 0.12); put(felt(90, 0.3), 0.36, 0.25)                             # warning sign
    put(marimba(PENTA[2], 0.45), T_FILL1, 0.2); put(marimba(PENTA[4], 0.45), T_FILL2, 0.2)             # result bar halves
    for k, tk in enumerate(T_CARDS):
        put(pop(1200 + 60 * k, 450, 0.09), tk - 0.12, 0.12); put(marimba(PENTA[k % 6], 0.35), tk - 0.1, 0.12)
        if k < 5: put(felt(140, 0.12), tk + 0.38, 0.2)                                                # the ✕
    for k in range(6): put(tick(), T_S["S6"] + 0.25 + 0.15 * k, 0.25)                                 # typing
    put(water(600, 1500), T_S["S6"] + 1.1, 0.25)                                                       # message lands
    for tk in T_KEY: put(felt(70, 0.4), tk - 0.05, 0.45)                                              # key line punches
    put(marimba(PENTA[3], 0.4), T_S["S8"] + 0.6, 0.2); put(felt(110, 0.25), T_NOT, 0.35)               # tick · struck out
    for k, tk in enumerate(T_HEAL): put(marimba(PENTA[k + 2], 0.45), tk - 0.1, 0.2)
    put(pop(1000, 500, 0.08), T_ASK + 0.7, 0.12); put(click(), T_NO, 0.9)
    for k in range(4): put(paper(0.12, 4500), T_TIME + 0.25 + 0.18 * k, 0.14)
    put(glass(1174.7), T_END + 0.3, 0.16)
    for i, f in enumerate([PENTA[0], PENTA[2], PENTA[3], PENTA[5]]): put(marimba(f, 0.6), T_END + 1.2 + i * 0.12, 0.12)
    sfx[:n] *= 1 - 0.8 * np.clip(dd, 0, 1)
    mix = vo[:, None] + mus + sfx[:n, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (1.0 / 20); mix /= max(1, np.abs(mix).max() / 0.95)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

def make_cover(path):
    """everything inside the centre 3:4 (y 240-1680) that the profile grid shows; a deeper navy than the video"""
    CBG = np.clip(np.float32([0, 10, 36]) / 255 * (1 + 0.3 * np.exp(-(((xx - 540) / 900) ** 2 + ((yy - 520) / 900) ** 2)))[..., None], 0, 1).astype(np.float32)
    fr = CBG.copy(); img = V["15"].at(20.3); cx, cy, r, s = 540, 660, 300, 0.8
    M = np.float32([[s, 0, cx - FACE[0] * s], [0, s, cy - FACE[1] * s]])
    sub = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2); m = np.clip(r - d + 0.5, 0, 1)[..., None]; fr = fr * (1 - m) + sub * m; ring(fr, cx, cy, r + 3)
    lg = logo_alpha(300); blend(fr, lg, 540 - 150, 262, PINK)
    icon(fr, "warn", 790, 450, 0.6)
    hd = Drop(["А ЕСЛИ Я НЕМНОГО", "НАРУШУ *РЕКОМЕНДАЦИИ?*"], 110, 70, 1040, maxw=860); hd.draw(fr, 99, 0)
    ty = Track(["что реально может повлиять", "на восстановление"], 34, 72, 0, 400); ty.y = 1040 + hd.h + 56 + ty.top; ty.draw(fr, 99, 0)
    cv2.imwrite(path, cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

def to8(fr): return (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)
if __name__ == "__main__":
    print(f"duration {DUR:.2f}s; scenes", [round(s, 2) for s, _ in SCENES])
    if "--preview" in sys.argv:
        for t in [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]:
            cv2.imwrite(os.path.join(WORK, f"prev_{t:05.2f}.jpg"), cv2.cvtColor(to8(render(t)), cv2.COLOR_RGB2BGR))
        sys.exit()
    if "--cover" in sys.argv: make_cover(os.path.join(ROOT, f"{NAME}_cover.jpg")); sys.exit()
    build_audio(os.path.join(WORK, "mix.wav"))
    vid_path = os.path.join(WORK, "video.mp4")
    if "--audio-only" not in sys.argv:
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                              "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", vid_path], stdin=subprocess.PIPE)
        N = int(round(DUR * FPS))
        for i in range(N):
            p.stdin.write(to8(render(i / FPS)).tobytes())
            if i % 60 == 0: print(f"{i}/{N}", flush=True)
        p.stdin.close(); p.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid_path, "-i", os.path.join(WORK, "mix.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", os.path.join(ROOT, f"{NAME}.mp4")], check=True)
    make_cover(os.path.join(ROOT, f"{NAME}_cover.jpg"))
    print("done")
