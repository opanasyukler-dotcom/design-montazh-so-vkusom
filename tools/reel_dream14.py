#!/usr/bin/env python3
"""Reels «Я мечтала об этом 14 лет» — not a list of procedures but one woman's long-held wish: «практически всю жизнь
мечтала» → «лет 14 этой мечте» → «не могу на себя насмотреться». Анна Тваури.

ДО with her voice, the three zones that stayed with her after the baby, the plan ticking as the surgeon names it, ПОСЛЕ with a
0→14 years counter, the zones confirmed on «грудь · животик · талия», the clinic's ПОСЛЕ/ДО collages with hearts and a shutter,
and the closing card. ADHD-friendly animation, one-word subtitles, whole sentences.

usage:
  REEL_SRC=<folder with IMG_9531/9532/9533/9536.MP4, IMG_9706/9707.JPG, logo.png, song4.wav, shutter_client.wav> python3 tools/reel_dream14.py [--preview t1,t2 | --cover | --audio-only]
Prepared in $REEL_SRC/_work: <name>.mp4 (1080x1920, 30 fps) and <name>.wav (48 kHz mono).
"""
import os, sys, json, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_dream14"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_dream_14_years"
W, H, FPS, SR = 1080, 1920, 30, 48000
NAVY = np.float32([0, 12, 40]) / 255; PINK = np.float32([233, 197, 193]) / 255; WHITE = np.float32([1, 1, 1])
DEEP = np.float32([0, 5, 20]) / 255

def cl(x): return min(1.0, max(0.0, x))
def eo(x): x = cl(x); return 1 - (1 - x) ** 3
def eio(x): x = cl(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x): x = cl(x); c = 1.7; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p

# ---------------- voice edit: (block, source, [(in, out), …]) — whole sentences from the one take; the pauses between them are tightened
VO = [("A", "IMG_9531", [(0.00, 6.15)]),     # я уже очень давно хотела, я мечтала об этом, практически всю жизнь мечтала
      ("C", "IMG_9533", [(0.00, 6.95)]),     # будем делать абдоминопластику, подтягивать грудь и липофилинг
      ("D", "IMG_9536", [(1.55, 6.88)]),     # самое главное, что у меня маленькая грудь. это была моя мечта, не знаю, лет 14 этой мечте
      ("E", "IMG_9536", [(11.18, 16.50)]),   # грудь очень мне нравится, животик плоский, талия, всё чудесно
      ("F", "IMG_9536", [(20.14, 24.48)])]   # я очень-очень рада, не могу на себя насмотреться
GAP = {"C": 0.4 + 3.2, "D": 0.5, "E": 0.45, "F": 0.55}      # the gap before C holds the silent ДО-zones scene
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
def R(b, rel): return vstart(b) + rel
T_B = vend("A") + 0.35                                    # ДО zones (silent, over IMG_9532)
T_K = [vend("F") + 0.4, vend("F") + 2.3]                 # collages: front, side
T_Z = T_K[1] + 2.2; DUR = T_Z + 3.0

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
V = {k: Src(os.path.join(WORK, f"{k}.mp4")) for k in ["IMG_9531", "IMG_9532", "IMG_9533", "IMG_9536"]}

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
    'A': [('Я', 0.04, 0.62), ('уже', 0.62, 0.88), ('очень', 0.88, 1.16), ('давно', 1.16, 1.38), ('хотела,', 1.38, 2.02), ('я', 2.36, 2.42), ('мечтала', 2.42, 2.82), ('об', 2.82, 2.98), ('этом,', 2.98, 4.12), ('практически', 4.12, 4.8), ('всю', 4.8, 5.02), ('жизнь', 5.02, 5.3), ('мечтала.', 5.3, 6.08)],
    'C': [('Будем', 0.04, 0.84), ('делать', 0.84, 2.24), ('абдоминопластику,', 2.24, 3.76), ('подтягивать', 3.98, 4.6), ('грудь', 4.6, 5.18), ('и', 5.18, 5.46), ('липофилинг.', 5.46, 6.9)],
    'D': [('Самое', 0.05, 0.21), ('главное,', 0.21, 0.43), ('что', 0.45, 0.61), ('у', 0.61, 0.62), ('меня', 0.62, 0.73), ('маленькая', 0.73, 1.27), ('грудь.', 1.27, 1.53), ('Это', 1.57, 1.77), ('была', 1.77, 1.99), ('моя', 1.99, 2.23), ('мечта,', 2.23, 2.99), ('не', 2.99, 3.17), ('знаю,', 3.17, 3.37), ('лет', 3.53, 3.61), ('14', 3.61, 4.35), ('этой', 4.35, 4.67), ('мечте.', 4.67, 5.27)],
    'E': [('Грудь', 0.04, 0.52), ('очень', 0.52, 0.98), ('мне', 0.98, 1.26), ('нравится,', 1.26, 1.52), ('животик', 1.68, 2.32), ('плоский,', 2.32, 3.2), ('талия,', 3.52, 4.04), ('всё', 4.22, 4.6), ('чудесно.', 4.6, 5.28)],
    'F': [('Я', 0.04, 0.28), ('очень-очень', 0.28, 1.3), ('рада,', 1.3, 1.98), ('не', 2.14, 2.34), ('могу', 2.34, 2.64), ('на', 2.64, 2.84), ('себя', 2.84, 3.22), ('насмотреться.', 3.22, 4.28)],
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

def src_t(k, t):
    """source time of video k at reel time t (keeps running between that video's own lines)"""
    best = None
    for b_, s_, parts, t0, d in SEG:
        if s_ != k: continue
        if t < t0: return parts[0][0] - (t0 - t) if best is None else min(best[0] + t - best[1], parts[0][0])
        acc = 0
        for a, o in parts:
            if t <= t0 + acc + o - a: return a + t - t0 - acc
            acc += o - a
        best = (parts[-1][1], t0 + d)
    return best[0] + t - best[1]
def frame(k, t, top=0):
    f = V[k].at(max(0.0, src_t(k, t))); return anon_top(f, top) if top else f
def tick_list(fr, t, items, x=80, y0=1080, dy=82, size=62):
    for k, (lab, t0) in enumerate(items):
        lt = t - t0
        if lt < 0: continue
        y = y0 + k * dy; s_ = eback(lt / 0.3); b = 22 * s_
        path_draw(fr, [(x + 22 - b, y + 22 - b), (x + 22 + b, y + 22 - b), (x + 22 + b, y + 22 + b), (x + 22 - b, y + 22 + b)], 1, PINK, 4, True)
        path_draw(fr, [(x + 10, y + 23), (x + 19, y + 33), (x + 36, y + 10)], eio((lt - 0.15) / 0.3), PINK, 5)
        a = alpha_img(lab, bebas(fit_size(lab, 760, size))); p = eo((lt - 0.1) / 0.35); a2 = a.copy(); a2[:, int(a.shape[1] * p):] = 0
        blend(fr, a2, x + 64, y + 22 - a.shape[0] / 2 + 5, WHITE)
def hearts(fr, t, t0, pts=((820, 560, 0.0), (260, 640, 0.35), (700, 420, 0.7))):
    for x, y, d in pts:
        u = t - t0 - d
        if 0 <= u < 1.8: fill_poly(fr, heart_pts(x, y - 120 * u, eback(u / 0.3) * 3.2), PINK, 0.9 * (1 - eio((u - 1.2) / 0.6)))

# A — the hook: ДО, her voice; a «life» line runs under the quote on «всю жизнь»
HA = lift(Drop(["Я МЕЧТАЛА ОБ ЭТОМ", "*ПРАКТИЧЕСКИ ВСЮ ЖИЗНЬ*"], 100, 70, 0, maxw=860), 1290)
def s_a(t):
    fr = full(frame("IMG_9531", t, 60), 0.5, 0.45, pull(t, 0, 1.16, 1.02, 3.0)); shade(fr, VIG); shade(fr, GBOT)
    chip(fr, "ДО", 40, 290); HA.draw(fr, t, 0.05)
    p = eio((t - R("A", 4.8)) / 0.9)
    if p > 0:
        y = HA.y + HA.h + 30; path_draw(fr, [(80, y), (lerp(80, 900, p), y)], 1, PINK, 6)
        for k, x in enumerate((80, 490, 900)):
            if p * 820 + 80 >= x - 1:
                m = np.zeros((H, W), np.uint8); cv2.circle(m, (x, int(y)), 11, 255, -1, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, PINK)
    for k, (x, y) in enumerate([(820, 420), (250, 520), (860, 640)]): sparkle(fr, x, y, 26, t, R("A", 2.42) + 0.2 * k)
    return fr

# B — after the baby the thought stayed: the three zones pinned on the ДО (silent)
HB = Drop(["ПОСЛЕ РОЖДЕНИЯ РЕБЁНКА", "ЭТА МЫСЛЬ *НИКУДА НЕ УШЛА*"], 80, 70, 1150, maxw=880)
PINS = [("грудь", 690, 620, 0.5), ("живот", 640, 940, 0.9), ("талия", 140, 900, 1.3)]
def s_b(t):
    lt = t - T_B; fr = full(V["IMG_9532"].at(0.3 + lt), 0.5, 0.45, lerp(1.0, 1.05, eio(lt / 3.4))); shade(fr, VIG)
    shade(fr, (ramp(1050, 1250) * 0.85).astype(np.float32)); chip(fr, "ДО", 40, 290)
    for lab, x, y, d in PINS:
        tk = T_B + d; u = t - tk
        if u < 0: continue
        m = np.zeros((H, W), np.uint8); cv2.circle(m, (x, y), int(14 * eback(u / 0.3)), 255, -1, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, PINK)
        uu = (u * 1.1) % 1; ring(fr, x, y, 22 + 24 * uu, 1 - uu, 3)
        pillx(fr, lab, x + 34 if x < 600 else x - 250, y - 32, eo((u - 0.1) / 0.3), eio((u - 0.3) / 0.3), 28)
    HB.draw(fr, t, T_B + 0.1); return fr

# C — the plan ticks as the surgeon names it
HC = lift(Drop(["В ИТОГЕ РЕШИЛИСЬ НА", "*КОМПЛЕКСНУЮ КОРРЕКЦИЮ*"], 86, 70, 0, maxw=880), 1010)
PLAN = [("АБДОМИНОПЛАСТИКА", R("C", 2.24)), ("+ ПОДТЯЖКА ГРУДИ", R("C", 3.98)), ("+ ЛИПОФИЛИНГ", R("C", 5.46))]
def s_c(t):
    fr = full(frame("IMG_9533", t, 40), 0.5, 0.45, pull(t, T_S["C"], 1.12, 1.0, 2.0)); shade(fr, VIG); shade(fr, (ramp(780, 1050) * 0.8).astype(np.float32))
    chip(fr, "ДО", 40, 290); HC.draw(fr, t, T_S["C"] + 0.1); tick_list(fr, t, PLAN, y0=1060); return fr

# D — ПОСЛЕ: «лет 14 этой мечте», the counter runs to 14
HD = lift(Drop(["ЭТО БЫЛА МОЯ МЕЧТА", "*ЛЕТ 14*"], 110, 70, 0, maxw=860), 1290)
def s_d(t):
    fr = full(frame("IMG_9536", t), 0.5, 0.45, pull(t, T_S["D"], 1.14, 1.02, 2.4)); shade(fr, VIG); shade(fr, GBOT)
    chip(fr, "ПОСЛЕ", 40, 290)
    tn = R("D", 3.61)
    if t > tn - 1.0:
        v = 14 * eo((t - tn + 1.0) / 1.1); f = 1 - eo((t - tn) / 0.2) if t > tn else 0
        text_c(fr, f"{int(round(v))}", 760, 640, bebas(int(260 * (1 + 0.12 * f))), PINK); text_c(fr, "ЛЕТ", 760, 790, bebas(70), WHITE)
        ring(fr, 760, 690, 190 + 10 * np.sin(t * 5), 0.8, 4)
    HD.draw(fr, t, R("D", 1.57)); return fr

# E — «грудь · животик плоский · талия»: the same three zones, now ticked
HE = lift(Drop(["ТО, ЧТО ОНА ХОТЕЛА ВИДЕТЬ", "*В ЗЕРКАЛЕ МНОГО ЛЕТ*"], 82, 70, 0, maxw=880), 1010)
GOT = [("ГРУДЬ", R("E", 0.04)), ("ПЛОСКИЙ ЖИВОТ", R("E", 1.68)), ("ТАЛИЯ", R("E", 3.52))]
def s_e(t):
    fr = full(frame("IMG_9536", t), 0.5, 0.45, pull(t, T_S["E"], 1.12, 1.0, 2.0)); shade(fr, VIG); shade(fr, (ramp(780, 1050) * 0.8).astype(np.float32))
    chip(fr, "ПОСЛЕ", 40, 290); HE.draw(fr, t, T_S["E"] + 0.1); tick_list(fr, t, GOT, y0=1060)
    for k, (x, y) in enumerate([(300, 560), (820, 520), (560, 420)]): sparkle(fr, x, y, 26, t, R("E", 4.6) + 0.18 * k)
    return fr

# F — the closing line over ПОСЛЕ (standing, the part of the take between her lines), hearts
def s_f(t):
    lt = t - T_S["F"]; fr = full(V["IMG_9536"].at(7.0 + lt * 0.9), 0.5, 0.45, lerp(1.14, 1.02, eo(lt / 2.6))); shade(fr, VIG); shade(fr, GBOT)
    chip(fr, "ПОСЛЕ", 40, 290); HF.draw(fr, t, R("F", 2.14)); hearts(fr, t, R("F", 0.28))
    for k, (x, y) in enumerate([(300, 560), (820, 520)]): sparkle(fr, x, y, 26, t, R("F", 3.2) + 0.2 * k)
    return fr
# K — the clinic's ПОСЛЕ/ДО collages full screen, shutter on each
HF = lift(Drop(["НЕ МОГУ НА СЕБЯ", "*НАСМОТРЕТЬСЯ*"], 120, 70, 0, maxw=860), 1290)
from PIL import ImageOps
COLL = [np.asarray(ImageOps.exif_transpose(Image.open(os.path.join(SRC, n))).convert("RGB")) for n in ("IMG_9706.JPG", "IMG_9707.JPG")]
def s_k(t):
    k = 0 if t < T_K[1] else 1; lt = t - T_K[k]
    fr = full(COLL[k], 0.5, 0.5, 1 + 0.03 * (1 - eo(lt / 0.35))); fr[957:963] = PINK
    chip(fr, "ПОСЛЕ", 40, 290); chip(fr, "ДО", 40, 1000)
    f = 1 - eo(lt / 0.18) if k else 0; return fr * (1 - f) + f

# Z — the closing card
HZ = Drop(["14 ЛЕТ МЕЧТАТЬ", "*И НАКОНЕЦ РЕШИТЬСЯ*"], 116, 70, 760, maxw=880)
TZ = Track(["комплексная коррекция груди и живота"], 32, 72, 0, 300)
def s_z(t):
    lt = t - T_Z; fr = gbg(t)
    blend(fr, logo_alpha(300), 540 - 150, 420, PINK, eo((lt - 0.1) / 0.5))
    HZ.draw(fr, t, T_Z + 0.2); TZ.y = HZ.y + HZ.h + 50 + TZ.top; TZ.draw(fr, t, T_Z + 0.9)
    return fr

SCENES = [(0, s_a), (T_B, s_b), (T_S["C"], s_c), (T_S["D"], s_d), (T_S["E"], s_e), (T_S["F"], s_f), (T_K[0], s_k), (T_Z, s_z)]
TRANS = {T_B: ("flash", 0.18), T_S["C"]: ("zoom", 0.35), T_S["D"]: ("iris", 0.5), T_S["E"]: ("zoom", 0.35), T_S["F"]: ("zoom", 0.35), T_K[0]: ("iris", 0.5), T_Z: ("iris", 0.5)}
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

MUSIC = os.path.join(SRC, "song5.wav")                                 # the client's track for this reel (audio of the screen recording)
M0 = 0.5
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
    w_ = wave.open(os.path.join(SRC, "shutter_client.wav")); SHOT = np.frombuffer(w_.readframes(w_.getnframes()), np.int16).astype(np.float64) / 32768
    SHOT = SHOT / (np.abs(SHOT).max() + 1e-9)
    put(swell(1.0), R("A", 4.5), 0.06); put(marimba(PENTA[4], 0.5), R("A", 5.6), 0.16)
    for k, (_, _, _, d) in enumerate(PINS): put(water(500 + 120 * k, 1400 + 120 * k), T_B + d, 0.2)
    for k, (_, t0) in enumerate(PLAN): put(marimba(PENTA[k + 2], 0.45), t0, 0.2)
    for j in range(10): put(tick(), R("D", 3.61) - 1.0 + 0.1 * j, 0.22)
    put(marimba(PENTA[5], 0.6), R("D", 3.61), 0.2)
    for k, (_, t0) in enumerate(GOT): put(marimba(PENTA[k + 2], 0.45), t0, 0.2)
    for tk in T_K: put(SHOT, tk, 0.5)
    for k in range(3): put(pop(1300, 600, 0.08), R("F", 0.28) + 0.35 * k, 0.07)
    put(glass(1174.7), T_Z + 0.2, 0.14)
    sfx[:n] *= 1 - 0.8 * np.clip(dd, 0, 1)
    mix = vo[:, None] + mus + sfx[:n, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (1.7 / 20); mix /= max(1, np.abs(mix).max() / 0.95)                           # ≈ -16.5 LUFS like the other reels
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

def make_cover(path):
    """everything inside the centre 3:4 (y 240-1680) that the profile grid shows; ДО | ПОСЛЕ from the front collage"""
    CBG = np.clip(np.float32([0, 8, 28]) / 255 * (1 + 0.3 * np.exp(-(((xx - 540) / 900) ** 2 + ((yy - 520) / 900) ** 2)))[..., None], 0, 1).astype(np.float32)
    fr = CBG.copy(); im = COLL[0]; h2 = im.shape[0] // 2
    for k, half in enumerate([im[h2 + 10:], im[:h2 - 10]]):                    # ДО (bottom half) left, ПОСЛЕ (top half) right
        c = half[:, int(half.shape[1] * 0.18):int(half.shape[1] * 0.82)]; s_ = 540 / c.shape[1]; c = cv2.resize(c, (540, int(c.shape[0] * s_)), interpolation=cv2.INTER_AREA)
        c = c[:760]; fr[340:340 + c.shape[0], k * 540:(k + 1) * 540] = c.astype(np.float32) / 255
    fr[340:340 + c.shape[0], 538:542] = PINK; chip(fr, "ДО", 40, 370); chip(fr, "ПОСЛЕ", 580, 370)
    g = np.clip((yy[:, :1] - 900) / 200, 0, 1)[..., None]; fr = fr * (1 - g) + CBG * g
    blend(fr, logo_alpha(300), 540 - 150, 262, PINK)
    hd = Drop(["Я МЕЧТАЛА ОБ ЭТОМ", "*14 ЛЕТ*"], 120, 70, 1160, maxw=880); hd.draw(fr, 99, 0)
    ty = Track(["а потом решилась на комплексную операцию"], 32, 72, 0, 400); ty.y = 1160 + hd.h + 50 + ty.top; ty.draw(fr, 99, 0)
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
