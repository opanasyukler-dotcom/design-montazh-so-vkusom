#!/usr/bin/env python3
"""Reels «Можно ли за одну операцию изменить сразу всю фигуру?» — Анна Тваури.

One operation, the proportions of the whole body: abdominoplasty for the stretched skin (the dumbbell crossed out), liposuction 360
(she turns all the way round inside an orbit with a 0→360 counter, the zones pop, the arm), the fat that is not thrown away (drops
flow from the belly and the back to the buttock and the breast on the side ДО photo), the breast with her own fat (no lift, no
implants, the outline swells, «+1 · 1,5»), more volume at the hips (the line bows out, a round contour, a projection scheme),
ПОСЛЕ angle after angle and the ПОСЛЕ/ДО grid. The surgeon's consultation runs in a circle. One-word subtitles, whole sentences,
one warm grade for all the photos.

usage:
  REEL_SRC=<folder with IMG_0610/0611/0612, IMG_0588/0594/0597/0599 (ДО), IMG_2356/2359/2362/2369 (ПОСЛЕ), IMG_0625 (arm), logo.png,
  shutter_client.wav, keys/, optional music.wav> python3 tools/reel_whole_figure.py [--preview t1,t2 | --cover | --audio-only]
Prepared in $REEL_SRC/_work: <name>.mp4 (1080x1920, 30 fps), <name>.wav (48 kHz mono), photos as <name>.png.
"""
import os, sys, json, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_whole_figure"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_whole_figure"
W, H, FPS, SR = 1080, 1920, 30, 48000
NAVY = np.float32([0, 12, 40]) / 255; PINK = np.float32([233, 197, 193]) / 255; WHITE = np.float32([1, 1, 1])
DEEP = np.float32([0, 5, 20]) / 255

def cl(x): return min(1.0, max(0.0, x))
def eo(x): x = cl(x); return 1 - (1 - x) ** 3
def eio(x): x = cl(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x): x = cl(x); c = 1.7; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p

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
                o_ = min(1, p * 1.6) * (1 - q); blend(fr, g, cx - g.shape[1] / 2 + 3, cy - g.shape[0] / 2 - q * 30 + 4, DEEP, 0.55 * o_)   # soft drop shadow: readable on skin and walls
                blend(fr, g, cx - g.shape[1] / 2, cy - g.shape[0] / 2 - q * 30, col, o_)

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
            for i, (a, dx) in enumerate(ln[:-1]):
                blend(fr, a, ox + dx + i * sp - 2, s.y + k * s.lh - s.top + 2, DEEP, 0.6 * op); blend(fr, a, ox + dx + i * sp - 4, s.y + k * s.lh - s.top, s.col, op)

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
    if s > 0.9: blend(fr, a, x + (64 if tick_p > 0 else 43), y + h / 2 - a.shape[0] / 2 + 2, WHITE, cl((p - 0.5) * 2))
    if s > 0.9 and tick_p > 0: path_draw(fr, [(x + 22, y + 33), (x + 32, y + 43), (x + 48, y + 22)], tick_p, PINK, 5)
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
_STK = None
def sticker_rgba(r):
    global _STK
    if _STK is None:
        im = cv2.imread(os.path.join(SRC, "sticker.png"), cv2.IMREAD_UNCHANGED); _STK = np.dstack([im[..., 2::-1], im[..., 3]]).astype(np.float32) / 255
    sz = 2 * int(r) + 2; return cv2.resize(_STK, (sz, sz), interpolation=cv2.INTER_AREA)
def put_sticker(img, cx, cy, r):
    s_ = sticker_rgba(r); h, w = s_.shape[:2]; x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
    if xb <= xa or yb <= ya: return
    a = s_[ya - y0:yb - y0, xa - x0:xb - x0, 3:4]; c = s_[ya - y0:yb - y0, xa - x0:xb - x0, :3]
    reg = img[ya:yb, xa:xb].astype(np.float32) / 255; img[ya:yb, xa:xb] = ((reg * (1 - a) + c * a) * 255 + 0.5).astype(np.uint8)
def tick_list(fr, t, items, x=80, y0=1080, dy=82, size=62):
    for k, (lab, t0) in enumerate(items):
        lt = t - t0
        if lt < 0: continue
        y = y0 + k * dy                                                   # each line pops in on the word (no tick marks)
        a = alpha_img(lab, bebas(fit_size(lab, 860, size))); s_ = 0.85 + 0.15 * eback(lt / 0.3); a2 = scaled(a, s_)
        blend(fr, a2, x, y + 22 - a2.shape[0] / 2 + 5, WHITE, min(1, lt / 0.12))
def hearts(fr, t, t0, pts=((820, 560, 0.0), (260, 640, 0.35), (700, 420, 0.7))):
    for x, y, d in pts:
        u = t - t0 - d
        if 0 <= u < 1.8: fill_poly(fr, heart_pts(x, y - 120 * u, eback(u / 0.3) * 3.2), PINK, 0.9 * (1 - eio((u - 1.2) / 0.6)))



# ---------------- voice edit: (block, source, [(in, out), …]) — whole sentences from the consultation; the pauses between them are tightened
VO = [("A1", "IMG_0612", [(5.60, 7.62)]),                     # сегодня у нас комплексная операция
      ("B", "IMG_0612", [(10.45, 11.12), (11.58, 13.15), (14.32, 19.64)]),   # будем делать (без «ей») абдоминопластику. здесь достаточно большое количество растяжек, поэтому спортом, к сожалению, это не уберётся
      ("C", "IMG_0612", [(23.70, 30.85)]),                    # сделаем липосакцию полностью живота, спинки. липосакция у нас 360. руки мы будем делать
      ("D", "IMG_0612", [(33.55, 37.48), (40.62, 41.88)]),    # мы весь забранный жир пересадим в область ягодицы … и в область груди
      ("E", "IMG_0612", [(42.56, 44.66), (125.52, 127.95)]),  # с грудью подтяжку мы не планируем. липофилинг увеличивает на размер, на полтора
      ("F", "IMG_0612", [(91.40, 97.26), (97.52, 99.45)])]    # немножко расширим бёдра в этой зоне, чтобы здесь был более округлый контур. и добавим, конечно же, проекцию
T0 = 2.9                                                      # the hook stands alone first, the surgeon comes in under it
GAP = {"B": 0.45, "C": 0.45, "D": 0.45, "E": 0.45, "F": 0.45}
SEG = []; _t = T0
for _b, _s, _parts in VO:
    if SEG: _t += GAP[_b]
    _d = sum(o - a for a, o in _parts); SEG.append((_b, _s, _parts, _t, _d)); _t += _d
def vstart(b): return next(s[3] for s in SEG if s[0] == b)
def vend(b): return next(s[3] + s[4] for s in SEG if s[0] == b)
def R(b, rel): return vstart(b) + rel
T_S = {b: vstart(b) - 0.08 for b, *_ in VO}
T_A1 = vstart("A1") - 0.12                                   # the cut to the surgeon
T_ARM = R("C", 5.78)                                          # «руки» — the arm photo
T_G = vend("F") + 0.5                                         # ПОСЛЕ, angle after angle
T_K = T_G + 2.2 + 3 * 0.95                                    # ПОСЛЕ / ДО collages, one per angle (after the slider and three more angles)
DUR = T_K + 4 * 1.6

V = {k: Src(os.path.join(WORK, f"{k}.mp4")) for k in ["IMG_0610", "IMG_0611", "IMG_0612"]}
def consult(t):
    """the consultation at the moment the surgeon says the line (it runs on between lines)"""
    return anon_top(V["IMG_0612"].at(max(0.0, src_t("IMG_0612", t))), 250)

# ---------------- photos: ДО and ПОСЛЕ, one warm grade for all eight
def photo(n): return cv2.imread(os.path.join(WORK, n + ".png"))[..., ::-1].copy()
ALIGN = {"do_front": (288, 1688, 1123), "do_34": (345, 1700, 1094), "do_side": (317, 1700, 991), "do_back": (403, 1728, 1066),
         "posle_front": (90, 988, 722), "posle_34": (115, 922, 745), "posle_side": (115, 945, 633), "posle_back": (-80, 773, 789)}
                                                              # (neck, navel / underwear row, body centre x); the ПОСЛЕ back is matched on the buttock fold and the heels
FACE = {"do_front": 340, "do_34": 420, "do_side": 470, "do_back": 0, "posle_front": 110, "posle_34": 140, "posle_side": 130, "posle_back": 0}
def skin(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); return (hsv[..., 2] > 90) & (hsv[..., 1] > 25) & (hsv[..., 1] < 170) & ((hsv[..., 0] < 25) | (hsv[..., 0] > 165))
def blackdrop(im):
    lum = im.astype(np.float32).mean(2) / 255; k = np.clip((lum - 0.16) / 0.12, 0, 1); k = cv2.GaussianBlur(k * k * (3 - 2 * k), (0, 0), 2)
    k = np.maximum(k, np.clip((im[..., 2].astype(np.float32) - im[..., 0] - 10) / 12, 0, 1)); return k[..., None]
WARM = (2.5, 7.0)                                                       # Lab a/b lift (OpenCV units): a touch pinker, clearly warmer
def skin_stats(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); m = skin(rgb)
    return np.float32([[lab[..., c][m].mean(), lab[..., c][m].std()] for c in range(3)])
def grade(rgb, st, tgt):
    """one grade for all photos: skin statistics moved to the common target; the black backdrop stays black"""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        k = float(np.clip(tgt[c, 1] / (st[c, 1] + 1e-3), 0.75, 1.35)); lab[..., c] = (lab[..., c] - st[c, 0]) * k + tgt[c, 0]
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    w = blackdrop(rgb); return np.clip(out * w + rgb.astype(np.float32) * (1 - w) * (rgb.max(2, keepdims=True) > 70), 0, 255).astype(np.uint8)
NAMES = {"do_front": "IMG_0588", "do_34": "IMG_0594", "do_side": "IMG_0597", "do_back": "IMG_0599",
         "posle_front": "IMG_P1", "posle_34": "IMG_P2", "posle_side": "IMG_P3", "posle_back": "IMG_P4"}   # ПОСЛЕ front / 3/4 / side: the client's later set
RAW = {k: photo(n) for k, n in NAMES.items()}
STATS = {k: skin_stats(v) for k, v in RAW.items()}
TGT = np.mean([v for v in STATS.values()], 0); TGT[1, 0] += WARM[0]; TGT[2, 0] += WARM[1]
PH = {}
for k, v in RAW.items():
    g_ = grade(v, STATS[k], TGT); st2 = skin_stats(g_); st2[:, 1] = TGT[:, 1]; g_ = grade(g_, st2, TGT)   # second pass, means only
    PH[k] = anon_top(g_, FACE[k]) if FACE[k] else g_
FLOOR = {"posle_34": 1960, "posle_side": 1940, "posle_back": 1640}            # where the light floor / skirting comes up behind the legs (photo px)
for k, y0 in FLOOR.items():                                                     # below that line keep only the skin: the floor goes black, the legs stay
    sk = cv2.GaussianBlur(cv2.dilate(cv2.morphologyEx(skin(RAW[k]).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)), np.ones((5, 5), np.uint8)).astype(np.float32) / 255, (0, 0), 3)
    PH[k][y0:] = (PH[k][y0:] * sk[y0:, :, None]).astype(np.uint8)
ARM = photo("IMG_0625")
def panel(im, n, pw, ph, neck_at=40, span=250, key=True):
    neck, navel, cx = ALIGN[n]; s_ = span / (navel - neck); r = cv2.resize(im, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    out = np.zeros((ph, pw, 3), np.float32); ox, oy = int(cx * s_ - pw / 2), int(neck * s_ - neck_at)
    xs0, ys0 = max(0, ox), max(0, oy); xs1, ys1 = min(r.shape[1], ox + pw), min(r.shape[0], oy + ph)
    if xs1 > xs0 and ys1 > ys0: out[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox] = r[ys0:ys1, xs0:xs1]
    if not key:                                                         # the photo as shot: only its own edges fade into black, the body untouched
        m = np.zeros((ph, pw), np.float32); m[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox] = 1
        m = cv2.GaussianBlur(cv2.erode(m, np.ones((41, 41), np.uint8)), (0, 0), 12); return out * m[..., None]
    lum = out.mean(2); k = np.clip((lum - 0.2) / 0.14, 0, 1); k = k * k * (3 - 2 * k)
    k = np.maximum(k, np.clip((out[..., 2] - out[..., 0] - 0.04) / 0.05, 0, 1)); return out * k[..., None]
def onphoto(img, fx, fy, z, t, t0, z0=None, d=2.5):
    """a photo full screen with a slow pull-back; returns the frame and a mapping photo px → screen"""
    zz = z if z0 is None else lerp(z0, z, eo((t - t0) / d)); im, M = cover(img, W, H, fx, fy, zz)
    return im, (lambda x, y: (M[0, 0] * x + M[0, 2], M[1, 1] * y + M[1, 2]))

# ---------------- subtitles
SUBS = {
    'A1': [('Сегодня', 0.04, 0.32), ('у', 0.32, 0.42), ('нас', 0.42, 0.64), ('комплексная', 0.64, 1.22), ('операция.', 1.22, 1.96)],
    'B': [('Будем', 0.02, 0.31), ('делать', 0.31, 0.67), ('абдоминопластику.', 0.83, 2.21), ('Здесь', 2.3, 2.74), ('достаточно', 2.74, 3.18),
          ('большое', 3.18, 3.6), ('количество', 3.6, 4.12), ('растяжек,', 4.12, 4.72), ('поэтому', 4.92, 5.62), ('спортом,', 5.62, 6.08), ('к', 6.08, 6.28),
          ('сожалению,', 6.28, 6.6), ('это', 6.7, 6.82), ('не', 6.82, 7.02), ('уберётся.', 7.02, 7.52)],
    'C': [('Сделаем', 0.02, 0.42), ('липосакцию', 0.42, 1.16), ('полностью', 1.16, 1.7), ('живота,', 2.0, 3.38), ('спинки.', 3.38, 3.94), ('Липосакция', 4.22, 4.76),
          ('у', 4.76, 4.82), ('нас', 4.82, 5.02), ('360.', 5.02, 5.72), ('Руки', 5.86, 6.24), ('мы', 6.24, 6.5), ('будем', 6.5, 6.72), ('делать.', 6.72, 7.12)],
    'D': [('Мы', 0.15, 0.27), ('весь', 0.27, 0.37), ('забранный', 0.37, 0.87), ('жир', 0.87, 1.13), ('пересадим', 1.13, 2.07), ('в', 2.07, 2.29), ('область', 2.29, 3.13),
          ('ягодиц', 3.13, 3.89), ('и', 3.97, 4.09), ('в', 4.09, 4.27), ('область', 4.27, 4.51), ('груди.', 4.51, 5.13)],
    'E': [('С', 0.0, 0.18), ('грудью', 0.18, 0.48), ('подтяжку', 0.48, 1.06), ('мы', 1.06, 1.26), ('не', 1.26, 1.5), ('планируем.', 1.5, 2.06), ('Липофилинг', 2.1, 2.76),
          ('увеличивает', 2.76, 3.4), ('на', 3.4, 3.62), ('размер,', 3.62, 3.92), ('на', 3.92, 4.2), ('полтора.', 4.2, 4.48)],
    'F': [('Немножко', 0.02, 0.58), ('расширим', 0.58, 1.06), ('бёдра', 1.06, 1.4), ('в', 1.4, 1.6), ('этой', 1.6, 1.74), ('зоне,', 1.74, 2.0), ('чтобы', 2.4, 2.62),
          ('здесь', 2.62, 3.04), ('был', 3.04, 3.54), ('более', 3.54, 4.56), ('округлый', 4.56, 5.18), ('контур.', 5.18, 5.84), ('И', 5.9, 6.14), ('добавим,', 6.14, 6.46),
          ('конечно', 6.46, 6.9), ('же,', 6.9, 7.12), ('проекцию.', 7.22, 7.76)],
}
def _sub_words():
    out = [(w.strip(",.?"), t0 + a, t0 + e) for b, s_, parts, t0, d in SEG for w, a, e in SUBS[b]]
    return [(w, a, min(e + 0.3, out[i + 1][1] - 0.04) if i + 1 < len(out) else e + 0.3) for i, (w, a, e) in enumerate(out)]
SUB_WORDS = _sub_words()
SUB_Y = 1420
def subtitles(fr, t):
    for w, a, e in SUB_WORDS:
        t0 = a - 0.04
        if not (t0 <= t < e): continue
        p = eo((t - t0) / 0.12); im = scaled(alpha_img(w, mont(44, 300)), 0.85 + 0.15 * p)
        h = 64; bw = im.shape[1] + 34; op = min(1, (e - t) / 0.06)
        box = np.zeros((h, int(bw)), np.uint8); cv2.rectangle(box, (0, 0), (box.shape[1] - 1, h - 1), 255, -1)
        blend(fr, cv2.GaussianBlur(box, (0, 0), 1.2).astype(np.float32) / 255, 500 - bw / 2, SUB_Y - h / 2, DEEP, 0.8 * op)
        blend(fr, im, 500 - im.shape[1] / 2, SUB_Y - im.shape[0] / 2 + 3, WHITE, op)

def top(hd, y=330): hd.y = y; return hd
def under(hd, gap=26): return hd.y + hd.h + gap
def dot(fr, x, y, r, col=PINK, op=1.0):
    if r <= 0.5: return
    m = np.zeros((H, W), np.uint8); cv2.circle(m, (int(x * 4), int(y * 4)), int(r * 4), 255, -1, cv2.LINE_AA, shift=2); blend(fr, m.astype(np.float32) / 255, 0, 0, col, op)
def dashed(fr, pts, p, col=PINK, th=4, dash=18, op=1.0):
    P = np.float32(pts); L = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; tot = L[-1] * p; s = 0.0
    while s < tot:
        e = min(s + dash, tot); a = [np.interp(s, L, P[:, 0]), np.interp(s, L, P[:, 1])]; b = [np.interp(e, L, P[:, 0]), np.interp(e, L, P[:, 1])]
        path_draw(fr, [a, b], 1, col, th, op=op); s += dash * 1.9
def drop_pts(cx, cy, r):
    return [(cx, cy - 1.9 * r)] + [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(-0.2 * np.pi, 1.2 * np.pi, 20)]
def ellipse_pts(cx, cy, rx, ry, rot=0.0, a0=0.0, a1=2 * np.pi, n=80):
    a = np.linspace(a0, a1, n); c, s = np.cos(rot), np.sin(rot); x, y = rx * np.cos(a), ry * np.sin(a)
    return list(zip(cx + x * c - y * s, cy + x * s + y * c))
def pip(fr, img, cx, cy, r, p, fx=0.5, fy=0.5, z=1.0):
    """the surgeon in a circle: the consultation shot inside a round window with a pink rim; pops in with p"""
    if p <= 0: return
    rr = max(4, r * eback(p)); d = int(2 * rr); sub, _ = cover(img, d, d, fx, fy, z)
    m = np.zeros((d, d), np.uint8); cv2.circle(m, (d // 2, d // 2), int(rr) - 1, 255, -1, cv2.LINE_AA); m = (m.astype(np.float32) / 255)[..., None] * min(1, p * 2)
    x0, y0 = int(cx - d / 2), int(cy - d / 2); xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + d), min(H, y0 + d)
    if xb <= xa or yb <= ya: return
    reg = fr[ya:yb, xa:xb]; mm = m[ya - y0:yb - y0, xa - x0:xb - x0]; reg[:] = reg * (1 - mm) + sub[ya - y0:yb - y0, xa - x0:xb - x0] * mm
    ring(fr, cx, cy, rr, min(1, p * 2), 5)
def pills_row(fr, t, items, x0, y0, maxw=940, size=28, gap=14, dy=74):
    """pills that pop one by one, wrapping into rows"""
    x, y = x0, y0
    for lab, t0 in items:
        w = alpha_img(lab, mont(size, 600)).shape[1] + 86
        if x + w > x0 + maxw: x, y = x0, y + dy
        pillx(fr, lab, x, y, eo((t - t0) / 0.3), 0.0, size); x += w + gap
def pip_on(t, t_in, t_out): return eo((t - t_in) / 0.35) * (1 - eio((t - t_out + 0.3) / 0.3))

# Instagram's safe zone for this client (her template): text and animation stay inside x 72-1018, y 234-1600, and out of the
# bottom-right notch (x > 864 below y 1347). The footage itself stays full frame (the client's call): only text and animation move.
SX0, SX1, SY0, SY1, NX, NY = 72, 1018, 234, 1600, 864, 1347
LX = 80                                                               # left edge for headlines, pills and chips
INS = 1.0                                                             # no pull-back of the footage (set < 1 to shrink the shot to the bottom)
def inset(fr, s=INS, black=False):
    if s >= 1: return fr
    """the whole frame scaled down, bottom-centred; around it a soft blur of the same frame (black for the studio photos)"""
    h, w = int(H * s), int(W * s); fg = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
    if black: out = np.zeros_like(fr)
    else: out = cv2.resize(cv2.GaussianBlur(cv2.resize(fr, (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 6), (W, H))
    x0, y0 = (W - w) // 2, H - h; out[y0:, x0:x0 + w] = fg; return out
def ip(x, y, s=INS):
    if s >= 1: return x, y
    return (W - W * s) / 2 + x * s, H - H * s + y * s
HSZ = 84                                                              # headline size: two lines end above y 430

# A — the hook over the whole ДО figure (pulled back); the six zones pop under it; the surgeon in a circle says «сегодня у нас комплексная операция»
HA = top(Drop(["МОЖНО ЛИ ЗА ОДНУ ОПЕРАЦИЮ", "*ИЗМЕНИТЬ СРАЗУ ВСЮ ФИГУРУ?*"], HSZ, LX, 0, maxw=920), 270)
ZA = ["живот", "талия", "спина", "руки", "грудь", "ягодицы"]
def s_a(t):
    fr = inset(full(anon_top(V["IMG_0610"].at(0.2 + t * 0.5), 330), 0.5, 0.55, pull(t, 0, 1.3, 1.0, 3.5)))
    HA.draw(fr, t, 0.05)
    pills_row(fr, t, [(z, 1.3 + 0.22 * k) for k, z in enumerate(ZA)], LX, 1200, maxw=700)            # on the thighs, the breast stays clear
    return fr
def s_a1(t):
    """«сегодня у нас комплексная операция»: the surgeon full frame (the consultation), the hook still on"""
    fr = full(consult(t), 0.5, 0.5, pull(t, T_A1, 1.08, 1.0, 2.0))
    HA.draw(fr, t, 0.05); pills_row(fr, t, [(z, 1.3 + 0.22 * k) for k, z in enumerate(ZA)], LX, 1200, maxw=700)
    return fr

# B — the belly: stretch marks drawn on the lower belly; «спортом … не уберётся» → the dumbbell gets crossed out
HB = lift(Drop(["ЖИВОТ И", "*РАСТЯНУТАЯ КОЖА*"], HSZ, LX, 0, maxw=900), 1290)   # under the belly: the breast stays clear
TB = Track(["это уже не всегда решается спортом"], 32, LX, 0, 400)
STRIAE = [(-170, 0), (-90, 30), (60, 26), (150, -6), (-130, 70), (110, 70)]
def s_b(t):
    lt = t - T_S["B"]; fr = inset(full(anon_top(V["IMG_0611"].at(0.4 + lt * 1.0), 220), 0.5, 0.5, pull(t, T_S["B"], 1.15, 1.0, 2.0)))
    HB.draw(fr, t, T_S["B"] + 0.05); TB.y = HB.y + HB.h + 18 + TB.top; TB.draw(fr, t, R("B", 4.92))
    tr = R("B", 4.12)
    for k, (dx, dy) in enumerate(STRIAE):
        u = t - tr - 0.08 * k
        if u > 0: path_draw(fr, [ip(540 + dx + 70 * i / 16, 1000 + dy + 6 * np.sin(i * 1.3)) for i in range(17)], eo(u / 0.35), PINK, 4)
    ts = R("B", 5.62)
    if t > ts:
        u = t - ts; card(fr, "dumbbell", 910, 1000, 0.62 * eback(u / 0.35), min(1, u / 0.2), eo(u / 0.5))
        xm = R("B", 6.82)
        if t > xm: xmark(fr, 910, 1000, 58, eo((t - xm) / 0.35), PINK, 10)
    return fr

# C — liposuction 360: she turns all the way round (sped up) inside an orbit with a 0→360 counter; the zones pop; then the arm (text off the arm)
HC = top(Drop(["*ЛИПОСАКЦИЯ 360*"], 110, LX, 0, maxw=900), 270)
HC2 = Drop(["*ЛИПОСАКЦИЯ 360*"], 96, LX, 1262, maxw=700)
ZC = [("живот", R("C", 2.0)), ("бока", R("C", 2.4)), ("спина", R("C", 3.38)), ("руки", R("C", 5.86)), ("внутренняя поверхность бёдер", R("C", 6.4))]
def s_c(t):
    lt = t - T_S["C"]
    if t < T_ARM:
        st = min(36.8, 0.2 + max(0, lt - 0.3) * 6.6); fr = inset(full(anon_top(V["IMG_0610"].at(st), 330), 0.5, 0.55, 1.06))
        p = cl((lt - 0.3) / (T_ARM - T_S["C"] - 0.4))
        cx, rx = waist_track(fr, t); cy, ry = 950, 70                    # the orbit follows her waist as she turns
        dashed(fr, ellipse_pts(cx, cy, rx, ry, 0, np.pi, 2 * np.pi), 1, WHITE, 3, 14, 0.6)
        path_draw(fr, ellipse_pts(cx, cy, rx, ry, 0, -np.pi / 2, -np.pi / 2 + 2 * np.pi * eio(p)), 1, PINK, 6)
        a = -np.pi / 2 + 2 * np.pi * eio(p); dot(fr, cx + rx * np.cos(a), cy + ry * np.sin(a), 12)
        HC.draw(fr, t, T_S["C"] + 0.05)
        v = int(round(360 * eio(p))); text_c(fr, f"{v}", 900, 310, bebas(96), PINK); ring(fr, 900, 330, 78, 0.9, 4)
        pills_row(fr, t, ZC[:3], LX, under(HC, 24), maxw=900)
    else:
        u = t - T_ARM; fr, mp = onphoto(ARM, 0.5, 0.45, 1.0, t, T_ARM, 1.12, 1.5)
        cx, cy = mp(1550, 3060); s_ = mp(1, 0)[0] - mp(0, 0)[0]
        dashed(fr, ellipse_pts(cx, cy, 1000 * s_, 270 * s_, -0.06), eo(u / 0.6), PINK, 5, 16)
        HC2.draw(fr, t, T_ARM)                                          # below the arm, on the dark backdrop
        pills_row(fr, t, ZC[3:], LX, 1470, maxw=760)
    return fr

# D — the fat is not thrown away: on the side ДО photo (pulled back) drops travel from the belly and the back to the buttock and the breast
HD = top(Drop(["ЖИР НЕ ПРОСТО УБИРАЕМ", "*ЕГО ИСПОЛЬЗУЕМ*", "*ДЛЯ ГРУДИ И ЯГОДИЦ*"], 78, LX, 0, maxw=920), 270)
D_SRC = [(1480, 1900), (650, 1680)]; D_DST = [(616, 2440), (1460, 1240)]           # belly, back → buttock, breast (photo px)
def flow(fr, t, t0, a, b, bend, k):
    (x0, y0), (x1, y1) = a, b
    path = [(lerp(x0, x1, q) + bend * np.sin(np.pi * q), lerp(y0, y1, q)) for q in np.linspace(0, 1, 40)]
    dashed(fr, path, eo((t - t0) / 0.9), WHITE, 3, 12, 0.7)
    for j in range(4):
        u = (t - t0 - 0.22 * j - 0.1 * k) / 1.0
        if 0 < u < 1.25:
            q = eio(min(1, u)); bx = lerp(x0, x1, q) + bend * np.sin(np.pi * q); by = lerp(y0, y1, q)
            fill_poly(fr, drop_pts(bx, by, 11 * (1 - 0.25 * q)), PINK, 1 - cl((u - 1) / 0.25))
def s_d(t):
    fr, mp = onphoto(PH["do_side"], 0.5, 0.5, 1.0, t, T_S["D"], 1.12, 2.0); m2 = mp
    S_ = [m2(*p) for p in D_SRC]; D_ = [m2(*p) for p in D_DST]
    for (x, y) in S_:
        u = t - T_S["D"] - 0.2
        if u > 0: uu = (u * 1.2) % 1; ring(fr, x, y, 26 + 30 * uu, (1 - uu) * 0.9, 4); dot(fr, x, y, 11 * eback(u / 0.3))
    flow(fr, t, R("D", 1.13), S_[0], D_[0], -130, 0); flow(fr, t, R("D", 1.35), S_[1], D_[0], -65, 1)
    flow(fr, t, R("D", 3.97), S_[0], D_[1], 100, 0); flow(fr, t, R("D", 4.15), S_[1], D_[1], 50, 1)
    for (x, y), tb in zip(D_, (R("D", 3.13), R("D", 4.51))):
        u = t - tb
        if u > 0: ring(fr, x, y, lerp(26, 66, eo(u / 0.5)), 1 - 0.5 * eo(u / 1.0), 5)
    HD.draw(fr, t, T_S["D"] + 0.05); chip(fr, "ДО", LX, under(HD, 22))
    return fr

# E — the breast with her own fat: the text stays above and below the breast, never on it
HE = top(Drop(["ГРУДЬ УВЕЛИЧИВАЕМ", "*СОБСТВЕННЫМ ЖИРОМ*"], HSZ, LX, 0, maxw=900), 270)
BR = [(778, 960), (1428, 960)]
def s_e(t):
    fr, mp = onphoto(PH["do_front"], 0.5, 0.23, 1.6, t, T_S["E"], 1.85, 2.5)
    s_ = mp(1, 0)[0] - mp(0, 0)[0]; tg = R("E", 2.76)
    g = 1 + 0.2 * eio((t - tg) / 1.2)
    for bx, by in BR:
        x, y = mp(bx, by)
        if t > tg - 0.3: path_draw(fr, ellipse_pts(x, y - 20 * s_, 220 * s_ * g, 190 * s_ * g, 0, 0.08 * np.pi, 0.92 * np.pi), eo((t - tg + 0.3) / 0.5), PINK, 6)
    yb = max(mp(0, 960 + 260)[1], 1130)                                # under the breasts
    pills_row(fr, t, [("без подтяжки", R("E", 1.26)), ("без имплантов", R("E", 3.0))], LX, yb, maxw=900)
    chip(fr, "ДО", LX, under(HE, 22))
    if t > R("E", 3.62):
        u = t - R("E", 3.62); text_c(fr, "+1", 790, yb + 120, bebas(int(110 * eback(u / 0.3))), PINK)
        if t > R("E", 4.2): u2 = t - R("E", 4.2); text_c(fr, "1,5", 790, yb + 205, bebas(int(70 * eback(u2 / 0.3))), WHITE)
    HE.draw(fr, t, T_S["E"] + 0.05)
    return fr

# F — more volume where it is missing: the back photo pulled back; the hips widen, a round contour, the projection
HF = top(Drop(["ДОБАВЛЯЕМ ОБЪЁМ", "*ТАМ, ГДЕ ЕГО НЕ ХВАТАЕТ*"], HSZ, LX, 0, maxw=920), 270)
def projection(fr, cx, cy, s, op, p):
    hw = 110 * s; blend(fr, rrect(cx - hw, cy - hw, cx + hw, cy + hw, 28 * s), 0, 0, NAVY, 0.9 * op)
    rrect_line(fr, cx - hw, cy - hw, cx + hw, cy + hw, 28 * s, PINK, 3, op)
    if op < 0.95: return
    k = eio(p); prof = lambda b: [(cx - 40 * s + (30 + b) * s * np.sin(np.pi * i / 20) ** 1.2, cy - 80 * s + 160 * s * i / 20) for i in range(21)]
    path_draw(fr, [(cx - 40 * s, cy - 90 * s), (cx - 40 * s, cy + 90 * s)], 1, WHITE, 3, op=0.5)
    path_draw(fr, prof(0), 1, WHITE, 3, op=0.4); path_draw(fr, prof(40 * k), 1, PINK, 6)
    arrow(fr, (cx + 10 * s, cy), (cx + 80 * s, cy), k, PINK, 5)
def s_f(t):
    fr, mp = onphoto(PH["do_back"], 0.5, 0.52, 1.0, t, T_S["F"], 1.12, 2.5); m2 = mp; s_ = (m2(1, 0)[0] - m2(0, 0)[0])
    th = R("F", 0.58)
    u = t - th
    if u > 0:                                                          # her own hip line, traced on the photo, bows outward
        b = 70 * eio(u / 0.8); n_ = len(HIP_Y)
        for sg, edge in ((-1, HIP_L), (1, HIP_R)):
            pts = [m2(edge[i] + sg * b * np.sin(np.pi * i / (n_ - 1)), HIP_Y[i]) for i in range(n_)]
            path_draw(fr, [m2(edge[i], HIP_Y[i]) for i in range(n_)], 1, WHITE, 3, op=0.45 * min(1, u / 0.3))
            path_draw(fr, pts, eo(u / 0.4), PINK, 6)
            x, y = m2(edge[HIP_W] + sg * b, HIP_Y[HIP_W]); arrow(fr, (x, y), (x + sg * 70, y), eo((u - 0.3) / 0.35), PINK, 5)
    tc = R("F", 4.56)
    for bx in (853, 1296):
        x, y = m2(bx, 2304); u = t - tc
        if u > 0: path_draw(fr, ellipse_pts(x, y, 220 * s_, 230 * s_, 0, -0.1 * np.pi, 1.1 * np.pi), eo(u / 0.6), WHITE, 4)
    tp = R("F", 7.22)
    if t > tp - 0.3: u = t - tp + 0.3; projection(fr, 890, 640, 0.75 * eback(u / 0.35), min(1, u / 0.2), eo((u - 0.3) / 0.7))
    HF.draw(fr, t, T_S["F"] + 0.05); chip(fr, "ДО", LX, under(HF, 22))
    return fr

def _hip_line():
    """left / right body edge of the back ДО photo, row by row over the hips (gaps such as the gluteal cleft closed)"""
    m = (PH["do_back"].astype(np.float32).mean(2) / 255 > 0.22).astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((1, 41), np.uint8)).astype(int)
    ys = np.arange(1760, 2420, 12); L, Rr = [], []
    for y in ys:
        d = np.diff(np.r_[0, m[y], 0]); st, en = np.where(d == 1)[0], np.where(d == -1)[0]; k = (en - st) > 20
        L.append(st[k].min()); Rr.append(en[k].max())
    sm = lambda a: np.convolve(np.pad(np.float32(a), 3, mode="edge"), np.ones(7) / 7, "valid")
    L, Rr = sm(L), sm(Rr); return ys, L, Rr, int(np.argmax(Rr - L))
HIP_Y, HIP_L, HIP_R, HIP_W = _hip_line()
_WT = {"t": -9, "cx": 540, "rx": 330}
def waist_track(fr, t):
    """centre and half-width of the body at waist height in this frame (bright pixels on the dark backdrop), smoothed over time"""
    occ = (fr[900:1000].mean(2) > 0.3).mean(0) > 0.6; d = np.diff(np.r_[0, occ.astype(int), 0]); st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    if len(st):
        k = np.argmax(en - st); cx, rx = (st[k] + en[k]) / 2, (en[k] - st[k]) / 2 * 1.18 + 26
        a = 0.25 if 0 < t - _WT["t"] < 0.2 else 1.0
        _WT["cx"] += a * (cx - _WT["cx"]); _WT["rx"] += a * (rx - _WT["rx"])
    _WT["t"] = t; return _WT["cx"], min(_WT["rx"], 470)

# G — ПОСЛЕ, angle after angle (a flash and a shutter on each), the closing thought
HG = Drop(["ОДНА ОПЕРАЦИЯ", "*И РАБОТА С ПРОПОРЦИЯМИ*", "*ВСЕГО ТЕЛА*"], 88, LX, 0, maxw=900)
TG = Track(["результат ещё ранний,", "ткани продолжают восстанавливаться"], 30, LX, 0, 400)
G_SEQ = ["posle_front", "posle_34", "posle_side", "posle_back"]; G_D = 0.95; G_S = 2.2      # the first ПОСЛЕ is revealed by a slider over the ДО
GP = {n: panel(PH[n], n, W, H, neck_at=300, span=700, key=False) for n in G_SEQ + ["do_front"]}   # all aligned on neck and navel
GP["posle_back"] *= np.clip((1480 - np.arange(H)) / 110, 0, 1)[:, None, None]                    # its floor faded out
def g_start(k): return T_G + (0 if k == 0 else G_S + (k - 1) * G_D)
def s_g(t):
    lt = t - T_G; k = 0 if lt < G_S else min(3, 1 + int((lt - G_S) / G_D)); u = t - g_start(k); n = G_SEQ[k]
    fr = GP[n].copy()
    if k == 0:                                                          # ДО first, then the pink slider sweeps and leaves ПОСЛЕ behind it
        p = eio((lt - 0.35) / 1.3); xs = int(lerp(-6, W + 6, p))
        if xs < W: fr[:, max(0, xs):] = GP["do_front"][:, max(0, xs):]
        if 0 < p < 1:
            fr[:, max(0, xs - 3):min(W, xs + 3)] = PINK
            hx = min(max(xs, SX0 + 40), SX1 - 40); dot(fr, hx, 840, 34, PINK); dot(fr, hx, 840, 28, WHITE)
            for sg in (-1, 1): fill_poly(fr, [(hx + sg * 8, 828), (hx + sg * 20, 840), (hx + sg * 8, 852)], PINK)
        if p < 0.98: chip(fr, "ДО", min(max(xs + 20, LX), 860), 260, 1 - cl((p - 0.8) / 0.18))
        if xs > 300: chip(fr, "ПОСЛЕ", LX, 260, cl((xs - 300) / 120))          # each label stays on its own side of the line
        f = 0
    else:
        chip(fr, "ПОСЛЕ", LX, 260); f = 1 - eo(u / 0.18)
    fr = fr * (1 - f * 0.85) + f * 0.85
    HG.y = 1300 - HG.h; HG.draw(fr, t, T_G + G_S - 0.35); TG.y = HG.y + HG.h + 36 + TG.top; TG.draw(fr, t, T_G + G_S + 0.4)   # after the slider
    return fr

# K — the clinic-style collages, each angle on its own: ПОСЛЕ on top, ДО below (colour graded alike), a shutter on each, then the logo
K_ANG = ("front", "34", "side", "back"); K_D = 1.6; KH = 960; KY = 0             # full screen: ПОСЛЕ top half, ДО bottom half
KNEE = {"do_front": (353, 2975, 1123), "do_34": (479, 3226, 1084), "do_side": (454, 3478, 1000), "do_back": (504, 2974, 1090),
        "posle_front": (97, 2028, 724), "posle_34": (145, 2093, 789), "posle_side": (161, 2125, 660), "posle_back": (0, 1771, 789)}
                                                              # (neck — the shoulder line on the backs —, just below the knees, body centre x) in photo px
def kpanel(n, y_top=40, y_bot=930):
    """neck at y_top, just below the knees at y_bot: every half shows the same stretch of body"""
    a, b, cx = KNEE[n]; s_ = (y_bot - y_top) / (b - a); r = cv2.resize(PH[n], None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    out = np.zeros((KH, W, 3), np.float32); ox, oy = int(cx * s_ - W / 2), int(a * s_ - y_top)
    xs0, ys0 = max(0, ox), max(0, oy); xs1, ys1 = min(r.shape[1], ox + W), min(r.shape[0], oy + KH)
    out[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox] = r[ys0:ys1, xs0:xs1]
    m = np.zeros((KH, W), np.float32); m[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox] = 1
    m = cv2.GaussianBlur(cv2.erode(m, np.ones((31, 31), np.uint8)), (0, 0), 9); return out * m[..., None]
COLLS = [np.vstack([kpanel("posle_" + a), kpanel("do_" + a)]) for a in K_ANG]
def s_k(t):
    lt = t - T_K; k = min(3, int(lt / K_D)); u = lt - k * K_D
    fr = np.zeros((H, W, 3), np.float32); s_ = 1 + 0.02 * (1 - eo(u / 0.35)); c = COLLS[k]
    if s_ > 1.001:
        c = cv2.resize(c, (int(W * s_), int(2 * KH * s_)), interpolation=cv2.INTER_LINEAR); ox, oy = (c.shape[1] - W) // 2, (c.shape[0] - 2 * KH) // 2; c = c[oy:oy + 2 * KH, ox:ox + W]
    fr[KY:KY + 2 * KH] = c; fr[KY + KH - 2:KY + KH + 2] = PINK
    chip(fr, "ПОСЛЕ", LX, 260); chip(fr, "ДО", LX, KH + 30)                   # labels inside the safe zone
    f = 1 - eo(u / 0.18); fr = fr * (1 - f) + f * 0.9
    return fr

SCENES = [(0, s_a), (T_A1, s_a1), (T_S["B"], s_b), (T_S["C"], s_c), (T_S["D"], s_d), (T_S["E"], s_e), (T_S["F"], s_f), (T_G, s_g), (T_K, s_k)]
TRANS = {T_A1: ("zoom", 0.3), T_S["B"]: ("zoom", 0.35), T_S["C"]: ("diag", 0.45), T_ARM: ("flash", 0.18), T_S["D"]: ("zoom", 0.35), T_S["E"]: ("iris", 0.5),
         T_S["F"]: ("zoom", 0.35), T_G: ("iris", 0.5)}
def render(t):
    k = max(i for i, (s0, _) in enumerate(SCENES) if t >= s0); s0, fn = SCENES[k]
    for tt, (kind, d) in TRANS.items():
        if kind == "flash" and tt <= t < tt + d:
            fr = fn(t); f = 1 - eo((t - tt) / d); return (fr * (1 - f) + f).astype(np.float32)
    kind, d = TRANS.get(s0, ("none", 0))
    if kind != "none" and t < s0 + d:
        p = (t - s0) / d; old = SCENES[k - 1][1](t); new = fn(t)
        if kind == "iris": return iris(old, new, p, 540, 900)
        if kind == "zoom": return zoom_through(old, new, p)
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
def tick(): n = int(0.012 * SR); return hp(rng.standard_normal(n), 3000) * env(n, 0.0005, 0.003)
def felt(f=95, dur=0.3):
    n = int(dur * SR); t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t) * env(n, 0.002, 0.07) + 0.5 * lp(rng.standard_normal(n), 600) * env(n, 0.001, 0.018)
def glass(f=1760, dur=1.4):
    n = int(dur * SR); t = np.arange(n) / SR
    return sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / d) for r, a, d in [(1, 1, 0.6), (2.76, 0.45, 0.3), (5.4, 0.2, 0.15)]) * env(n, 0.002, 9) / 1.6
def water(f0=500, f1=1700):
    n = int(0.12 * SR); t = np.arange(n) / SR; ph = 2 * np.pi * np.cumsum(f0 + (f1 - f0) * (1 - np.exp(-t / 0.015))) / SR
    return np.sin(ph) * env(n, 0.001, 0.035)
def swoosh(dur=0.5):
    n = int(dur * SR); t = np.arange(n) / SR; x = lp(hp(rng.standard_normal(n), 800), 5000) * np.sin(np.pi * t / dur) ** 2
    return x / (np.abs(x).max() + 1e-9)
def click():
    n = int(0.09 * SR); t = np.arange(n) / SR; x = hp(rng.standard_normal(n), 2500)
    e = np.exp(-t / 0.004) + 0.7 * np.exp(-np.clip(t - 0.045, 0, None) / 0.005) * (t > 0.045)
    x = x * e / 2; k = felt(160, 0.08); x[:len(k)] += k * 0.4; return x

MUSIC = os.path.join(SRC, "music.wav"); M0 = 0.24                      # the client's track for this reel: its drop (14.0 s) lands on the cut to «липосакция 360»
G_VO, G_GAP = 0.022, 0.075
def track(n):
    if not os.path.exists(MUSIC): return np.zeros((n, 2))
    w = wave.open(MUSIC); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64).reshape(-1, w.getnchannels()) / 32768
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    seg = x[int(M0 * SR):int(M0 * SR) + n]; seg = seg / np.sqrt(np.mean(seg.mean(1) ** 2))
    hop = SR // 10; e = np.sqrt(np.convolve(np.pad(seg.mean(1) ** 2, (0, hop - len(seg) % hop))[:len(seg) // hop * hop + hop].reshape(-1, hop).mean(1), np.ones(25) / 25, "same")) + 1e-6
    g = np.clip(1.15 / e, 1.0, 5.0)                                       # the track's quiet passages (intro, bridge under the ending) lifted, up to +14 dB, slowly
    g = np.interp(np.arange(len(seg)), np.arange(len(g)) * hop + hop / 2, g); seg = seg * g[:, None]
    return np.pad(seg, ((0, n - len(seg)), (0, 0)))

def build_audio(path):
    n = int(DUR * SR); vo = np.zeros(n); act = np.zeros(n); srcs = {}
    ir_t = np.arange(int(0.45 * SR)) / SR; IR = rng.standard_normal(len(ir_t)) * np.exp(-ir_t / 0.11); IR = lp(IR, 3500); IR /= np.sqrt(np.sum(IR ** 2))
    for b, s_, parts, t0, d in SEG:
        if s_ not in srcs:
            w = wave.open(os.path.join(WORK, f"{s_}.wav")); srcs[s_] = hp(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768, 90)
        x = srcs[s_]; seg = x[int(parts[0][0] * SR):int(parts[0][1] * SR)].copy()
        for a, o in parts[1:]:
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
    sfx = np.zeros(n + 2 * SR)
    def put(x, t, g): i = int(max(0, t) * SR); seg = sfx[i:i + len(x)]; seg += x[:len(seg)] * g
    def load(fn):
        w_ = wave.open(fn); x_ = np.frombuffer(w_.readframes(w_.getnframes()), np.int16).astype(np.float64) / 32768
        if w_.getnchannels() == 2: x_ = x_.reshape(-1, 2).mean(1)
        if w_.getframerate() != SR: x_ = np.interp(np.arange(int(len(x_) * SR / w_.getframerate())) * w_.getframerate() / SR, np.arange(len(x_)), x_)
        return x_ / (np.abs(x_).max() + 1e-9)
    SHOT = load(os.path.join(SRC, "shutter_client.wav"))
    kd = os.path.join(SRC, "keyboard.wav")
    KEYS = [load(kd)] if os.path.exists(kd) else [load(os.path.join(SRC, "keys", f"k{i:03d}.wav")) for i in range(1, 9)]
    td = os.path.join(SRC, "tick.wav"); TICK = load(td) if os.path.exists(td) else click()
    n_ch = sum(len(g) for g, _ in HA.rows)
    if len(KEYS) == 1:                                                  # the client's keyboard: as long as the title types, then a short fade
        kl = int((n_ch * HA.stag + 0.35) * SR); kb = KEYS[0][:kl].copy(); fo = int(0.2 * SR); kb[-fo:] *= np.linspace(1, 0, fo); put(kb, 0.05, 0.5)
    else:
        for j in range(n_ch): put(KEYS[j % len(KEYS)], 0.05 + j * HA.stag, 0.32)          # typing under the title
    for k in range(len(ZA)): put(TICK, 1.3 + 0.22 * k, 0.55)                              # the six zones
    for k in range(6): put(tick(), R("B", 4.12) + 0.08 * k, 0.3)                          # stretch marks drawn
    put(TICK, R("B", 5.62), 0.6); put(felt(70, 0.35), R("B", 6.82), 0.5)                   # dumbbell, crossed out
    put(swoosh(1.6), T_S["C"] + 0.3, 0.12)
    for j in range(16): put(tick(), T_S["C"] + 0.4 + 0.3 * j, 0.15)                       # the 360 counter
    for _, t0 in ZC: put(TICK, t0, 0.55)
    for tt in (R("D", 1.13), R("D", 3.97)):
        for j in range(8): put(water(450 + 90 * j, 1500 + 120 * j), tt + 0.13 * j, 0.12)  # the drops travelling
    put(TICK, R("E", 1.26), 0.55); put(TICK, R("E", 3.0), 0.55); put(glass(1174.7), R("E", 3.62), 0.1)
    put(TICK, R("F", 0.58), 0.55); put(TICK, R("F", 4.56), 0.55); put(TICK, R("F", 7.22), 0.55)
    put(swoosh(1.3), T_G + 0.35, 0.18)                                                  # the slider
    for k in range(1, 4): put(SHOT, g_start(k), 0.45)
    for k in range(4): put(SHOT, T_K + 1.6 * k, 0.5)
    put(glass(1318.5), T_K + 3 * 1.6 + 0.5, 0.1)
    sfx[:n] *= 1 - 0.6 * np.clip(dd, 0, 1)
    tt = np.arange(n) / SR; fade = np.clip((DUR - tt) / 1.8, 0, 1) ** 1.5 * np.minimum(1, tt / 0.15)
    mus = track(n) * ((G_GAP - (G_GAP - G_VO) * np.clip(dd, 0, 1)) * fade)[:, None]
    mix = vo[:, None] + mus + (sfx[:n] * np.clip((DUR - tt) / 1.0, 0, 1))[:, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (GAIN / 20); mix /= max(1, np.abs(mix).max() / 0.95)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
GAIN = 1.5 if os.path.exists(MUSIC) else 0.9

def make_cover(path):
    """inside the centre 3:4 (y 240-1680): ДО | ПОСЛЕ side views (fat taken here, added there), the title below"""
    CBG = np.clip(np.float32([0, 8, 28]) / 255 * (1 + 0.3 * np.exp(-(((xx - 540) / 900) ** 2 + ((yy - 520) / 900) ** 2)))[..., None], 0, 1).astype(np.float32)
    fr = CBG.copy()
    for k, n in enumerate(("do_side", "posle_side")):
        c = panel(PH[n], n, 540, 820, neck_at=50, span=380); fr[330:330 + 820, k * 540:(k + 1) * 540] = c
    fr[330:1150, 538:542] = PINK; chip(fr, "ДО", 40, 350); chip(fr, "ПОСЛЕ", 580, 350)
    g = np.clip((yy[:, :1] - 950) / 200, 0, 1)[..., None]; fr = fr * (1 - g) + CBG * g
    blend(fr, logo_alpha(260), 540 - 130, 262, PINK)
    hd = Drop(["ЖИР УБРАЛИ ЗДЕСЬ", "*И ДОБАВИЛИ ТУДА*"], 120, 70, 1180, maxw=900); hd.draw(fr, 99, 0)
    ty = Track(["как меняем пропорции тела за одну операцию"], 32, 72, 0, 400); ty.y = 1180 + hd.h + 46 + ty.top; ty.draw(fr, 99, 0)
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
