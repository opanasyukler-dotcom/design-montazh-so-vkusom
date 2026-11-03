#!/usr/bin/env python3
"""Reels «Сколько жира можно ввести в грудь за один раз?» — breast lipofilling volumes, voiced by the surgeon's voice
message, animated over the clinic's footage, Анна Тваури.

The voice (telegram_audio.ogg) is cut by whole clauses; the picture is the clinic's B-roll (fat syringes, the canister,
the injection, the ДО marking) under ADHD-friendly explainers: a breast «vessel» filling to its capacity, a small and
a large vessel side by side, a syringe counter running to 200 ml, then to 400–450 ml, donor zones lighting up, a
ДО → ПОСЛЕ slider, and the ПОСЛЕ/ДО collages (ПОСЛЕ colour-matched to ДО). One-word subtitles; whole clauses.

usage:
  REEL_SRC=<folder with telegram_audio.ogg, C0492-C0552.MP4, IMG_0539/0548/0549/9436/9437/9441.heic, logo.png, song3.wav> python3 tools/reel_fat_volume.py [--preview t1,t2 | --cover | --audio-only]
Prepared in $REEL_SRC/_work: voice.wav (48 kHz mono), C05xx.mp4 / C049x.mp4 (1080x1920, 30 fps), IMG_xxxx.png (the photos).
"""
import os, sys, json, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_fat_volume"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_fat_volume_lipofilling"
W, H, FPS, SR = 1080, 1920, 30, 48000
NAVY = np.float32([0, 12, 40]) / 255; PINK = np.float32([233, 197, 193]) / 255; WHITE = np.float32([1, 1, 1])
DEEP = np.float32([0, 5, 20]) / 255

def cl(x): return min(1.0, max(0.0, x))
def eo(x): x = cl(x); return 1 - (1 - x) ** 3
def eio(x): x = cl(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x): x = cl(x); c = 1.7; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p

# ---------------- voice edit: (block, source, [(in, out), …]) — whole sentences from the one take; the pauses between them are tightened
VO = [("A", "voice", [(6.58, 10.56)]),     # в каждую пациентку можно вводить разное количество жира
      ("B", "voice", [(11.40, 14.80)]),    # имеет значение ёмкость, то есть воспринимающий объём
      ("C", "voice", [(14.96, 19.70)]),    # а у какой-то пациентки небольшой объём железы, соответственно туда немного можно ввести
      ("D", "voice", [(19.88, 22.80)]),    # минимум, который я обычно ввожу, это 200 мл
      ("E", "voice", [(23.90, 28.46)]),    # максимально я вводила и 400, и 450 мл в железу
      ("F", "voice", [(46.10, 50.70)]),    # у пациентки должны быть запасы жировой ткани в донорских зонах
      ("G", "voice", [(56.24, 64.12)])]    # поэтому каждой пациентке вводится разный объём жира, исходя из её анатомии
GAP = {"B": 0.35, "C": 0.35, "D": 0.45, "E": 0.45, "F": 0.55, "G": 0.5}
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

T_S = {b: vstart(b) - 0.08 for b, *_ in VO}
def R(b, rel): return vstart(b) + rel                                   # a word inside line b (relative time from SUBS)
T_K = [vend("G") + 0.5 + 1.9 * k for k in range(3)]                     # collages: front, 3/4, side
DUR = T_K[-1] + 2.6

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
V = {k: Src(os.path.join(WORK, f"{k}.mp4")) for k in ["C0492", "C0498", "C0535", "C0537", "C0544", "C0552"]}

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
    'A': [('В', 0.04, 0.2), ('каждой', 0.2, 0.7), ('пациентки', 0.7, 1.42), ('можно', 1.42, 1.78), ('разное', 1.78, 2.14), ('количество', 2.14, 2.9), ('вводить', 2.9, 3.6), ('жира.', 3.6, 3.9)],
    'B': [('Имеет', 0.06, 0.26), ('значение', 0.26, 0.82), ('ёмкость,', 0.82, 1.28), ('то', 1.3, 1.34), ('есть', 1.34, 1.54), ('воспринимающий', 1.54, 2.38), ('объём.', 2.38, 3.34)],
    'C': [('А', 0.04, 0.26), ('у', 0.26, 0.32), ('какой-то', 0.32, 0.68), ('пациентки', 0.68, 1.4), ('небольшой', 1.4, 2.0), ('объём', 2.0, 2.34), ('железы,', 2.34, 2.86), ('соответственно', 3.0, 3.3), ('туда', 3.3, 3.6), ('немного', 3.6, 4.0), ('можно', 4.0, 4.4), ('ввести.', 4.4, 4.7)],
    'D': [('Минимум,', 0.04, 0.54), ('который', 0.72, 0.84), ('я', 0.84, 1.14), ('обычно', 1.14, 1.48), ('ввожу', 1.48, 1.76), ('это', 1.76, 1.98), ('200', 1.98, 2.32), ('мл.', 2.32, 2.84)],
    'E': [('Максимально', 0.06, 0.6), ('я', 0.6, 0.86), ('вводила', 0.86, 1.28), ('и', 1.28, 1.56), ('400', 1.56, 1.92), ('и', 1.92, 2.06), ('450', 2.06, 2.78), ('мл', 2.78, 3.66), ('в железу.', 3.66, 4.5)],
    'F': [('У', 0.06, 0.36), ('пациентки', 0.36, 0.88), ('должны', 0.88, 1.18), ('быть', 1.18, 1.82), ('запасы', 1.82, 2.3), ('жировой', 2.3, 2.6), ('ткани', 2.6, 2.88), ('в', 2.88, 2.98), ('донорских', 2.98, 3.5), ('зонах.', 3.5, 4.54)],
    'G': [('Поэтому', 0.04, 0.44), ('у', 0.44, 0.44), ('каждой', 0.44, 0.78), ('пациентки', 0.78, 1.28), ('вводится', 1.28, 1.7), ('разный', 1.7, 2.16), ('объём', 2.16, 2.48), ('жира,', 2.48, 2.8), ('исходя', 2.98, 3.26), ('из', 3.26, 3.4), ('того,', 3.4, 3.7), ('какой', 3.94, 4.04), ('объём', 4.04, 4.46), ('железы', 4.46, 5.14), ('у', 5.14, 5.24), ('неё', 5.24, 5.46), ('имеется,', 5.46, 5.94), ('то', 6.04, 6.08), ('есть', 6.08, 6.22), ('исходя', 6.22, 6.64), ('из', 6.64, 6.82), ('её', 6.82, 7.04), ('анатомии.', 7.04, 7.82)],
}
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
        p = eo((t - t0) / 0.12); im = scaled(alpha_img(w, mont(44, 300)), 0.85 + 0.15 * p)   # thin Montserrat, pops in
        h = 64; bw = im.shape[1] + 34; op = min(1, (e - t) / 0.06)
        box = np.zeros((h, int(bw)), np.uint8); cv2.rectangle(box, (0, 0), (box.shape[1] - 1, h - 1), 255, -1)
        blend(fr, cv2.GaussianBlur(box, (0, 0), 1.2).astype(np.float32) / 255, 500 - bw / 2, SUB_Y - h / 2, DEEP, 0.8 * op)
        blend(fr, im, 500 - im.shape[1] / 2, SUB_Y - im.shape[0] / 2 + 3, WHITE, op)
def lift(hd, bottom=1230):
    """bottom-anchor a headline inside the safe zone"""
    hd.y = bottom - hd.h; return hd

# ---------------- building blocks ----------------
def ring(fr, cx, cy, r, op=1.0, th=5):
    path_draw(fr, [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 120)], 1, PINK, th, op=op)
_rng = np.random.default_rng(5); DOTS = [(_rng.uniform(60, 1020), _rng.uniform(0, 1920), _rng.uniform(2, 6), _rng.uniform(20, 60)) for _ in range(22)]
def gbg(t):
    """navy backdrop with slow pink dust drifting up"""
    fr = BG.copy(); m = np.zeros((H, W), np.uint8)
    for x, y, r, v in DOTS: cv2.circle(m, (int(x + 12 * np.sin(t + y)), int((y - v * t) % H)), int(r), 255, -1, cv2.LINE_AA)
    blend(fr, m.astype(np.float32) / 255, 0, 0, PINK, 0.22); return fr
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
def fit_size(txt, maxw, size):
    while bebas(size).getlength(txt) > maxw and size > 30: size -= 2
    return size
def pillx(fr, txt, x, y, p, tick_p=0.0, size=30):
    if p <= 0: return
    f = mont(size, 600); a = alpha_img(txt, f); s = eback(p); w, h = a.shape[1] + 86, 64
    blend(fr, rrect(x, y, x + w * s, y + h * s, 32 * s), 0, 0, NAVY, 0.9 * min(1, p * 2))
    rrect_line(fr, x, y, x + w * s, y + h * s, 32 * s, PINK, 2, min(1, p * 2))
    if s > 0.9: blend(fr, a, x + 64, y + h / 2 - a.shape[0] / 2 + 2, WHITE, cl((p - 0.5) * 2)); path_draw(fr, [(x + 22, y + 33), (x + 32, y + 43), (x + 48, y + 22)], tick_p, PINK, 5)
def arrow(fr, p0, p1, p, col=PINK, th=6):
    if p <= 0: return
    x0, y0 = p0; x1, y1 = p1; xe, ye = lerp(x0, x1, p), lerp(y0, y1, p); path_draw(fr, [(x0, y0), (xe, ye)], 1, col, th)
    a = np.arctan2(ye - y0, xe - x0); hs = 22
    fill_poly(fr, [(xe, ye), (xe - hs * np.cos(a - 0.45), ye - hs * np.sin(a - 0.45)), (xe - hs * np.cos(a + 0.45), ye - hs * np.sin(a + 0.45))], col)
def anon_top(img, y=150):
    top = img[:y + 80].astype(np.float32); bl = cv2.resize(cv2.GaussianBlur(cv2.resize(top, None, fx=0.1, fy=0.1), (0, 0), 3), (img.shape[1], y + 80))
    m = np.clip((y + 40 - np.arange(y + 80)) / 80, 0, 1)[:, None, None]; out = img.copy(); out[:y + 80] = (top * (1 - m) + bl * m).astype(np.uint8); return out
def vid(k, src_t, top=0):
    f = V[k].at(src_t); return anon_top(f, top) if top else f
def line_src(b, t):
    """source time of block b's own footage at reel time t (the picture keeps running before and after the line)"""
    _, _, parts, t0, d = next(x for x in SEG if x[0] == b); return parts[0][0] + (t - t0)
def anon_face(f, y=280): return anon_top(f, y)

import pillow_heif; pillow_heif.register_heif_opener()
from PIL import ImageOps
def photo(n): return np.asarray(ImageOps.exif_transpose(Image.open(os.path.join(SRC, n + ".heic"))).convert("RGB")).copy()
def qmarks(fr, t, t0, pts=((860, 430), (200, 560), (880, 700))):
    for k, (x, y) in enumerate(pts):
        lt = t - t0 - 0.25 * k
        if lt > 0: text_c(fr, "?", x, y - 8 * np.sin(lt * 4), bebas(max(8, int(110 * eback(lt / 0.3)))), PINK, min(1, lt * 3))
def counter(fr, t, v, cx=540, cy=1110, size=190, unit="МЛ", prefix=""):
    text_c(fr, f"{prefix}{int(round(v))} {unit}", cx, cy, bebas(size), PINK)
def vessel(fr, cx, cy, r, level, cap=1.0, op=1.0, glow=0.0):
    """a breast «vessel»: circle outline, pink fill rising to `level` (0..1 of the circle), dashed cap line at `cap`"""
    m = np.zeros((H, W), np.uint8); cv2.circle(m, (int(cx), int(cy)), int(r), 255, -1, cv2.LINE_AA); circ = m.astype(np.float32) / 255
    ytop = cy + r - 2 * r * level; band = (yy[:, :1] >= ytop).astype(np.float32)
    wave_ = 6 * np.sin(xx[:1] / 30 + level * 20); band = np.clip((yy - (ytop + wave_)) / 3, 0, 1)
    blend(fr, circ * band, 0, 0, PINK, 0.75 * op); blend(fr, circ, 0, 0, NAVY, 0.25 * op * (1 - level))
    ring(fr, cx, cy, r, op, 4)
    yc = cy + r - 2 * r * cap; dx = np.sqrt(max(0, r * r - (yc - cy) ** 2))
    for j in range(0, 14, 2):
        a0 = cx - dx + 2 * dx * j / 14; a1 = cx - dx + 2 * dx * (j + 1) / 14; path_draw(fr, [(a0, yc), (a1, yc)], 1, WHITE, 3, op=0.8 * op)
    if glow > 0: ring(fr, cx, cy, r + 10 + 20 * glow, (1 - glow) * op, 3)

# A — the question over the fat syringes
HA = lift(Drop(["СКОЛЬКО ЖИРА МОЖНО ВВЕСТИ", "*В ГРУДЬ ЗА ОДИН РАЗ?*"], 92, 70, 0, maxw=880), 1300)
def s_a(t):
    fr = full(V["C0544"].at(3.0 + t * 0.8), 0.5, 0.5, pull(t, 0, 1.25, 1.05, 3.0)); shade(fr, GBOT); shade(fr, GBOT * 0.5)
    qmarks(fr, t, 0.6); HA.draw(fr, t, 0.05); return shake(fr, t, 0.6, 6)

# B — capacity: the ДО breasts become vessels that fill to their own line
HB = lift(Drop(["ОДНОЙ ЦИФРЫ", "*ДЛЯ ВСЕХ НЕТ*"], 120, 70, 0, maxw=860), 1260)
TB = Track(["всё зависит от анатомии"], 32, 72, 0, 300)
def s_b(t):
    lt = t - T_S["B"]; fr = full(anon_face(V["C0492"].at(1.0 + lt * 0.8), 300), 0.5, 0.42, lerp(1.08, 1.14, eio(lt / 4))); shade(fr, VIG); shade(fr, GBOT)
    lv = 0.62 * eio((t - R("B", 0.82)) / 1.6); op = eo((t - R("B", 0.6)) / 0.4)
    for cx in (390, 700):
        if op > 0: vessel(fr, cx, 700, 120, lv, 0.62, op, cl((t - R("B", 2.4)) / 0.6) if t > R("B", 2.4) else 0)
    if t > R("B", 1.54): pillx(fr, "воспринимающий объём", 330, 860, eo((t - R("B", 1.54)) / 0.3), eio((t - R("B", 1.8)) / 0.3))
    chip(fr, "ДО", 40, 290); HB.draw(fr, t, T_S["B"] + 0.1); TB.y = HB.y + HB.h + 40 + TB.top; TB.draw(fr, t, T_S["B"] + 0.7); return fr

# C — a small and a large gland side by side: the small one is full far sooner
HC = Drop(["ЧЕМ МЕНЬШЕ ОБЪЁМ ТКАНЕЙ,", "ТЕМ *МЕНЬШЕ МОЖНО ВВЕСТИ*", "ЗА ОДИН ЭТАП"], 80, 80, 300, maxw=880)
def s_c(t):
    lt = t - T_S["C"]; fr = gbg(t); HC.draw(fr, t, T_S["C"] + 0.1)
    p = eio((t - R("C", 1.4)) / 2.6); op = eo((t - T_S["C"] - 0.3) / 0.4)
    l_small = min(0.9, p * 1.6); l_big = 0.9 * p
    full_ = t > R("C", 1.4) + 1.6
    vessel(fr, 300, 930, 110, l_small, 0.9, op, cl((t - R("C", 1.4) - 1.6) / 0.6) if full_ else 0)
    vessel(fr, 690, 900, 190, l_big, 0.9, op)
    if op > 0:
        text_c(fr, "МЕНЬШЕ", 300, 1100, bebas(52), WHITE, op); text_c(fr, "БОЛЬШЕ", 690, 1130, bebas(52), WHITE, op)
        text_c(fr, "железа", 300, 1150, mont(28, 400), PINK, op); text_c(fr, "железа", 690, 1180, mont(28, 400), PINK, op)
    if full_: pillx(fr, "уже полный", 170, 760, eo((t - R("C", 1.4) - 1.6) / 0.3), eio((t - R("C", 1.4) - 1.85) / 0.3), 26)
    return fr

# D — about 200 ml: the canister, the counter runs to 200 on «200»
HD = lift(Drop(["ОКОЛО *200 МЛ*"], 140, 70, 0, maxw=860), 1290)
TD = Track(["один из частых рабочих объёмов"], 32, 72, 0, 300)
def s_d(t):
    lt = t - T_S["D"]; fr = full(V["C0537"].at(1.5 + lt), 0.5, 0.45, pull(t, T_S["D"], 1.18, 1.04, 2.0)); shade(fr, GBOT); shade(fr, GBOT * 0.5)
    tn = R("D", 1.98); v = 200 * eo((t - tn + 1.0) / 1.2) if t > tn - 1.0 else 0
    if t > tn - 1.0:
        f = 1 - eo((t - tn) / 0.2) if t > tn else 0; text_c(fr, f"{int(round(v))}", 540, 780, bebas(int(230 * (1 + 0.1 * f))), WHITE); text_c(fr, "мл", 540, 905, mont(40, 400), PINK)
    if t > tn: HD.draw(fr, t, tn - 0.05); TD.y = HD.y + HD.h + 40 + TD.top; TD.draw(fr, t, tn + 0.5)
    return fr

# E — up to 400–450 ml: the injection, a bar climbs past 400 to 450
HE = lift(Drop(["В НЕКОТОРЫХ СЛУЧАЯХ", "*400-450 МЛ*"], 104, 70, 0, maxw=860), 1240)
TE = Track(["если позволяет анатомия", "и есть донорская жировая ткань"], 30, 72, 0, 300)
def bar(fr, t, v, y0=430, x0=120, x1=900, vmax=500):
    blend(fr, rrect(x0, y0, x1, y0 + 40, 20), 0, 0, WHITE, 0.2)
    xe = x0 + (x1 - x0) * v / vmax
    if v > 5: blend(fr, rrect(x0, y0, max(x0 + 40, xe), y0 + 40, 20), 0, 0, PINK, 0.95)
    for mk in (200, 400, 450):
        xm = x0 + (x1 - x0) * mk / vmax; path_draw(fr, [(xm, y0 - 14), (xm, y0 + 54)], 1, WHITE, 3, op=0.8)
        text_c(fr, str(mk), xm, y0 - 40, bebas(44), WHITE if v < mk else PINK)
def s_e(t):
    lt = t - T_S["E"]; fr = full(V["C0552"].at(1.5 + lt * 0.7), 0.5, 0.5, pull(t, T_S["E"], 1.15, 1.02, 2.0)); shade(fr, GBOT); shade(fr, ((1 - ramp(250, 620)) * 0.6).astype(np.float32))
    v = 200 + 200 * eo((t - R("E", 1.56)) / 0.5) * (t > R("E", 1.56)) + 50 * eo((t - R("E", 2.06)) / 0.5) * (t > R("E", 2.06))
    bar(fr, t, v * eo((t - T_S["E"]) / 0.4))
    if t > R("E", 1.56): text_c(fr, f"{int(round(v))} МЛ", 540, 1020, bebas(150), WHITE)
    HE.draw(fr, t, R("E", 1.56)); TE.y = HE.y + HE.h + 36 + TE.top; TE.draw(fr, t, R("E", 2.4)); return fr

# F — donor zones: belly, flanks, thighs light up on the ДО body
HF1 = Drop(["ВАЖНО НЕ ТОЛЬКО, СКОЛЬКО", "*МОЖЕТ ПРИНЯТЬ ГРУДЬ*"], 72, 70, 300, maxw=880)
HF2 = Drop(["ВАЖНО, СКОЛЬКО ЖИРА МОЖНО", "*БЕЗОПАСНО ЗАБРАТЬ*"], 72, 70, 460, maxw=880)
ZONES = [("живот", 590, 960, 160, 70), ("бока", 345, 980, 50, 100), ("бока", 850, 980, 50, 100), ("бёдра", 320, 1260, 60, 120), ("бёдра", 880, 1260, 60, 120)]   # C0492 frame px
def s_f(t):
    lt = t - T_S["F"]; z = lerp(0.92, 0.96, eio(lt / 5))
    f = anon_face(V["C0492"].at(4.0 + lt * 0.6), 300); M = np.float32([[z, 0, 540 * (1 - z)], [0, z, 0]])
    fr = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255; shade(fr, VIG)
    tz = R("F", 1.82)
    for k, (lab, x, y, rx, ry) in enumerate(ZONES):
        q = eio((t - tz - 0.2 * k) / 0.5)
        if q <= 0: continue
        X, Y = M[0, 0] * x + M[0, 2], M[1, 1] * y + M[1, 2]; br = 1 + 0.05 * np.sin((t - tz) * 7)
        m = np.zeros((H, W), np.uint8); cv2.ellipse(m, (int(X), int(Y)), (int(rx * z * br), int(ry * z * br)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        blend(fr, cv2.GaussianBlur(m, (0, 0), 10).astype(np.float32) / 255, 0, 0, PINK, 0.3 * q)
        e_ = cv2.ellipse2Poly((int(X), int(Y)), (int(rx * z * br), int(ry * z * br)), 0, 0, 360, 12)
        for j in range(0, len(e_) - 1, 2): path_draw(fr, [tuple(e_[j]), tuple(e_[j + 1])], q, PINK, 4)
    shade(fr, ((1 - ramp(560, 760)) * 0.8).astype(np.float32))
    HF1.draw(fr, t, T_S["F"] + 0.1); HF2.draw(fr, t, R("F", 2.98))
    pillx(fr, "донорские зоны", 80, 1330, eo((t - R("F", 2.98)) / 0.3), eio((t - R("F", 3.2)) / 0.3)); return fr

# G — individually: ДО → ПОСЛЕ slider on the front photos, then the closing line
HG = Drop(["ОБЪЁМ ПОДБИРАЕТСЯ", "*ИНДИВИДУАЛЬНО*"], 110, 70, 300, maxw=860)
ALIGN = {"IMG_9436": (360, 1800, 1160), "IMG_0539": (470, 1840, 1080), "IMG_9437": (590, 1830, 1120), "IMG_0548": (690, 1860, 1120),
         "IMG_9441": (540, 1850, 1140), "IMG_0549": (660, 1900, 1160)}           # (neck, navel / waist row, body centre x) in photo px
def skin(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); return (hsv[..., 2] > 90) & (hsv[..., 1] > 25) & (hsv[..., 1] < 170) & ((hsv[..., 0] < 25) | (hsv[..., 0] > 165))
def match_colour(src, ref):
    a = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32); b = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    ms, mr = skin(src), skin(ref)
    for c in range(3):
        s0, s1_ = a[..., c][ms].mean(), a[..., c][ms].std() + 1e-3; r0, r1 = b[..., c][mr].mean(), b[..., c][mr].std() + 1e-3
        a[..., c] = (a[..., c] - s0) * float(np.clip(r1 / s1_, 0.75, 1.35)) + r0
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
def place(im, n, ph=H, neck_at=300, span=900, black=False):
    neck, navel, cx = ALIGN[n]; s_ = span / (navel - neck); r = cv2.resize(anon_top(im, neck - 40), None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    bgc = np.median(r[:40, :40].reshape(-1, 3), 0); out = np.zeros((ph, W, 3), np.float32) + (0 if black else bgc)
    ox, oy = int(cx * s_ - W / 2), int(neck * s_ - neck_at); xs0, ys0 = max(0, ox), max(0, oy); xs1, ys1 = min(r.shape[1], ox + W), min(r.shape[0], oy + ph)
    if not black:                                                         # backdrop kept: extend the photo's own edges instead of a flat fill
        P_ = 1400; rr = cv2.copyMakeBorder(r, P_, P_, P_, P_, cv2.BORDER_REPLICATE); rr = cv2.blur(rr, (1, 1))
        out = rr[P_ + oy:P_ + oy + ph, P_ + ox:P_ + ox + W].copy()
    else: out[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox] = r[ys0:ys1, xs0:xs1]
    if black:
        out *= np.clip((neck_at + 1.75 * span - np.arange(ph)) / 80, 0, 1)[:, None, None]          # down to mid-thigh: no floor
        lum = out.mean(2); k = np.clip((lum - 0.12) / 0.14, 0, 1); k = k * k * (3 - 2 * k)
        k = np.maximum(k, np.clip((out[..., 2] - out[..., 0] - 0.04) / 0.05, 0, 1)); out *= k[..., None]
    return out
PH = {n: photo(n) for n in ALIGN}
for a_, d_ in (("IMG_0539", "IMG_9436"), ("IMG_0548", "IMG_9437"), ("IMG_0549", "IMG_9441")): PH[a_] = match_colour(PH[a_], PH[d_])
SL_B, SL_A = place(PH["IMG_9436"], "IMG_9436", neck_at=470), place(PH["IMG_0539"], "IMG_0539", neck_at=470)
T_SW = (vstart("G") + 0.3, vstart("G") + 1.9)
def s_g(t):
    xd = int(W * (1 - eio((t - T_SW[0]) / (T_SW[1] - T_SW[0]))))
    fr = SL_B.copy(); fr[:, xd:] = SL_A[:, xd:]; shade(fr, ((1 - ramp(250, 620)) * 0.75).astype(np.float32)); shade(fr, GBOT * 0.6)
    if 0 < xd < W:
        fr[:, max(0, xd - 3):xd + 3] = PINK
        m = np.zeros((H, W), np.uint8); cv2.circle(m, (xd, 960), 32, 255, -1, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, PINK)
        ar = alpha_img("‹ ›", mont(34, 600)); blend(fr, ar, xd - ar.shape[1] / 2, 960 - ar.shape[0] / 2 - 2, NAVY)
    chip(fr, "ДО", 40, 1300, cl((xd - 200) / 100)); chip(fr, "ПОСЛЕ", 40, 1300, cl((160 - xd) / 100))
    HG.draw(fr, t, R("G", 4.04)); return fr

# K — collages: ПОСЛЕ on top, ДО below, on black; shutter on every cut
COLL8 = [(np.clip(np.vstack([place(PH[a_], a_, 960, 40, 470), place(PH[d_], d_, 960, 40, 470)]), 0, 1) * 255).astype(np.uint8)   # the photos' own backdrop kept
         for a_, d_ in (("IMG_0539", "IMG_9436"), ("IMG_0548", "IMG_9437"), ("IMG_0549", "IMG_9441"))]
def s_k(t):
    k = max(i for i, tk in enumerate(T_K) if t >= tk - 1e-6); lt = t - T_K[k]
    fr = full(COLL8[k], 0.5, 0.5, 1 + 0.03 * (1 - eo(lt / 0.35))); fr[957:963] = PINK
    chip(fr, "ПОСЛЕ", 40, 290); chip(fr, "ДО", 40, 1000)
    f = 1 - eo(lt / 0.18) if k else 0; return fr * (1 - f) + f

SCENES = [(0, s_a), (T_S["B"], s_b), (T_S["C"], s_c), (T_S["D"], s_d), (T_S["E"], s_e), (T_S["F"], s_f), (T_S["G"], s_g), (T_K[0], s_k)]
TRANS = {T_S["B"]: ("zoom", 0.35), T_S["C"]: ("iris", 0.45), T_S["D"]: ("zoom", 0.35), T_S["E"]: ("flash", 0.2), T_S["F"]: ("diag", 0.45), T_S["G"]: ("zoom", 0.35), T_K[0]: ("iris", 0.5)}
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

MUSIC = os.path.join(SRC, "song3.wav")                                 # the client's track (audio of the screen recording, 48 kHz stereo)
M0 = 0.3                                                               # the track outlasts the reel, no loop needed
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
            w = wave.open(os.path.join(WORK, f"{s_}.wav")); srcs[s_] = hp(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768, 90)
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
    for k in range(3): put(pop(1300 - 150 * k, 500, 0.1), 0.6 + 0.25 * k, 0.11)                                   # question marks
    put(swell(1.4), R("B", 0.7), 0.06); put(water(500, 1300), R("B", 0.82), 0.2); put(marimba(PENTA[3], 0.5), R("B", 2.4), 0.16)
    put(swell(2.4), R("C", 1.2), 0.05); put(felt(140, 0.2), R("C", 3.0), 0.25); put(marimba(PENTA[2], 0.5), R("C", 3.05), 0.16)
    for j in range(10): put(tick(), R("D", 1.98) - 1.0 + 0.11 * j, 0.22)                                         # counter
    put(marimba(PENTA[4], 0.6), R("D", 1.98), 0.2)
    put(pop(1000, 450, 0.09), R("E", 1.56), 0.14); put(marimba(PENTA[5], 0.6), R("E", 2.06), 0.2)
    for k in range(len(ZONES)): put(water(450 + 100 * k, 1300 + 100 * k), R("F", 1.82) + 0.2 * k, 0.16)
    put(marimba(PENTA[3], 0.5), R("F", 2.98), 0.16); put(swell(1.0), T_SW[0], 0.05); put(marimba(PENTA[4], 0.6), T_SW[1], 0.16)
    for tk in T_K: put(click(), tk, 0.9)
    sfx[:n] *= 1 - 0.8 * np.clip(dd, 0, 1)
    mix = vo[:, None] + mus + sfx[:n, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (1.7 / 20); mix /= max(1, np.abs(mix).max() / 0.95)                           # ≈ -16.5 LUFS like the other reels
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

def make_cover(path):
    """everything inside the centre 3:4 (y 240-1680) that the profile grid shows; ДО | ПОСЛЕ front, a deeper navy"""
    CBG = np.clip(np.float32([0, 8, 28]) / 255 * (1 + 0.3 * np.exp(-(((xx - 540) / 900) ** 2 + ((yy - 520) / 900) ** 2)))[..., None], 0, 1).astype(np.float32)
    fr = CBG.copy()
    for k, n in enumerate(("IMG_9436", "IMG_0539")):
        half = place(PH[n], n, 800, 40, 520, True)[:, 270:810]; fr[340:1140, k * 540:(k + 1) * 540] = half
    fr[340:1140, 538:542] = PINK; chip(fr, "ДО", 40, 370); chip(fr, "ПОСЛЕ", 580, 370)
    g = np.clip((yy[:, :1] - 950) / 190, 0, 1)[..., None]; fr = fr * (1 - g) + CBG * g
    blend(fr, logo_alpha(300), 540 - 150, 262, PINK)
    hd = Drop(["СКОЛЬКО ЖИРА МОЖНО", "*ВВЕСТИ В ГРУДЬ?*"], 104, 70, 1170, maxw=880); hd.draw(fr, 99, 0)
    ty = Track(["200, 400 или 450 мл?"], 36, 72, 0, 400); ty.y = 1170 + hd.h + 50 + ty.top; ty.draw(fr, 99, 0)
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
