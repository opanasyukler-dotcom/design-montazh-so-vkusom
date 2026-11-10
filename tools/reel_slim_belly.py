#!/usr/bin/env python3
"""Reels «Я стройная. Почему живот всё равно не уходит?» — Анна Тваури.

A slim patient whose belly stays: the surgeon's «как спортом это убрать?» (the dumbbell gets crossed out), the skin that hangs and the
muscles that part (diastasis), the breast lift scheme with the T and lipofilling drops, the fat flying from the hips into the breast on
the ДО photo, the next-day «как в молодости» with a clock winding back, the zones on «живот… талия… плоскость», the ПОСЛЕ photo
with the closing thought and the ПОСЛЕ/ДО grid. ADHD-friendly animation, one-word subtitles, whole sentences.

usage:
  REEL_SRC=<folder with IMG_6213/6215/6217/6218, IMG_9958-9960 (ДО), IMG_2448/2450/2453 (ПОСЛЕ), logo.png, shutter_client.wav, song6.wav, keys/> \
  python3 tools/reel_slim_belly.py [--preview t1,t2 | --cover | --audio-only]
Prepared in $REEL_SRC/_work: <name>.mp4 (1080x1920, 30 fps), <name>.wav (48 kHz mono), photos as <name>.png.
"""
import os, sys, json, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_slim_belly"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_slim_belly"
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


# ---------------- voice edit: (block, source, [(in, out), …]) — whole sentences; the pauses between them are tightened
VO = [("A", "IMG_6217", [(3.10, 8.42)]),     # несмотря на то что девушка стройная, вот расскажите мне, как спортом вот это можно убрать
      ("B", "IMG_6217", [(0.00, 2.95)]),     # смотрите, вот как здесь кожа провисает
      ("C", "IMG_6218", [(3.60, 9.70)]),     # мы будем подтягивать грудь, делать липофилинг, то есть подтянем её с Т-образной подтяжкой
      ("D", "IMG_6218", [(26.40, 31.10)]),   # ещё у нас липосакция бёдер, заберём с бёдер жир для того, чтобы добавить в грудь
      ("E", "IMG_6215", [(4.40, 5.30)]),     # (хирург) красота?
      ("E2", "IMG_6215", [(5.30, 9.75)]),    # (пациентка, тихо) да, вообще шикарно. как в молодости — своя громкость
      ("F", "IMG_6215", [(27.20, 30.30)])]   # а живот, совершенно другая талия, у вас хорошая плоскость
GAP = {"B": 0.35, "C": 0.45, "D": 0.45, "E": 0.5, "E2": 0.0, "F": 0.45}
SEG = []; _t = 0.04
for _b, _s, _parts in VO:
    if SEG: _t += GAP[_b]
    _d = sum(o - a for a, o in _parts); SEG.append((_b, _s, _parts, _t, _d)); _t += _d
def vstart(b): return next(s[3] for s in SEG if s[0] == b)
def vend(b): return next(s[3] + s[4] for s in SEG if s[0] == b)
def R(b, rel): return vstart(b) + rel
T_S = {b: vstart(b) - 0.08 for b, *_ in VO}
T_S["A"] = 0.0
T_D2 = R("D", 1.45)                                        # the OR shot gives way to the ДО photo before «заберём»
T_G = vend("F") + 0.6                                      # ПОСЛЕ photo + the closing thought
T_K = T_G + 4.2                                            # ПОСЛЕ / ДО, three angles
DUR = T_K + 3.4

V = {k: Src(os.path.join(WORK, f"{k}.mp4")) for k in ["IMG_6213", "IMG_6215", "IMG_6217", "IMG_6218"]}
def lsrc(b, t):
    """source time of block b's footage at reel time t (the picture runs on before and after the line)"""
    _, _, parts, t0, d = next(x for x in SEG if x[0] == b); return parts[0][0] + (t - t0)

_BURN = {}
def unburn(f):
    """IMG_6217/6218 carry burned-in one-word captions at y≈1140-1195: inpaint the white letters only"""
    y0, y1, x0, x1 = 1120, 1215, 300, 790; reg = f[y0:y1, x0:x1]
    g = cv2.cvtColor(reg, cv2.COLOR_RGB2GRAY); th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, np.ones((11, 11), np.uint8))
    m = ((reg.min(2) > 200) & (th > 22)).astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((7, 7), np.uint8))
    out = f.copy(); out[y0:y1, x0:x1] = cv2.inpaint(np.ascontiguousarray(reg), m, 6, cv2.INPAINT_TELEA); return out
def frame(k, st, top=0):
    f = V[k].at(max(0.0, st))
    if k in ("IMG_6217", "IMG_6218"): f = unburn(f)
    return anon_top(f, top) if top else f

# ---------------- photos: ДО (white stockings) and ПОСЛЕ (colour-matched to the ДО of the same angle)
def photo(n): return cv2.imread(os.path.join(WORK, n + ".png"))[..., ::-1].copy()
ALIGN = {"do_front": (270, 1062, 728), "do_34": (275, 1085, 796), "do_side": (318, 1090, 746),
         "posle_front": (202, 1465, 1243), "posle_34": (205, 1520, 1021), "posle_side": (269, 1480, 1110)}   # (neck, navel row, body centre x)
def skin(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); return (hsv[..., 2] > 90) & (hsv[..., 1] > 25) & (hsv[..., 1] < 170) & ((hsv[..., 0] < 25) | (hsv[..., 0] > 165))
def match_colour(src, ref):
    a = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32); b = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    ms, mr = skin(src), skin(ref)
    for c in range(3):
        s0, s1_ = a[..., c][ms].mean(), a[..., c][ms].std() + 1e-3; r0, r1 = b[..., c][mr].mean(), b[..., c][mr].std() + 1e-3
        k = float(np.clip(r1 / s1_, 0.75, 1.35)); a[..., c] = (a[..., c] - s0) * k + r0
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
def blackdrop(im):
    lum = im.astype(np.float32).mean(2) / 255; k = np.clip((lum - 0.16) / 0.12, 0, 1); k = cv2.GaussianBlur(k * k * (3 - 2 * k), (0, 0), 2)
    k = np.maximum(k, np.clip((im[..., 2].astype(np.float32) - im[..., 0] - 10) / 12, 0, 1)); return k[..., None]
WARM = (2.5, 7.0)                                                       # Lab a/b lift (OpenCV units): a touch pinker, clearly warmer
def skin_stats(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); m = skin(rgb)
    return np.float32([[lab[..., c][m].mean(), lab[..., c][m].std()] for c in range(3)])
def grade(rgb, st, tgt):
    """one grade for all six photos: skin statistics moved to the common target; the black backdrop stays black"""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        k = float(np.clip(tgt[c, 1] / (st[c, 1] + 1e-3), 0.75, 1.35)); lab[..., c] = (lab[..., c] - st[c, 0]) * k + tgt[c, 0]
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    w = blackdrop(rgb); return np.clip(out * w + rgb.astype(np.float32) * (1 - w) * (rgb.max(2, keepdims=True) > 70), 0, 255).astype(np.uint8)
PAIRS = (("front", "IMG_9958", "IMG_2448"), ("34", "IMG_9959", "IMG_2450"), ("side", "IMG_9960", "IMG_2453"))
RAW = {}
for ang, dn, pn in PAIRS: RAW["do_" + ang] = photo(dn); RAW["posle_" + ang] = photo(pn)
STATS = {k: skin_stats(v) for k, v in RAW.items()}
TGT = np.mean([v for v in STATS.values()], 0); TGT[1, 0] += WARM[0]; TGT[2, 0] += WARM[1]
PH = {}
for k, v in RAW.items():
    g_ = grade(v, STATS[k], TGT); st2 = skin_stats(g_); st2[:, 1] = TGT[:, 1]; g_ = grade(g_, st2, TGT)   # second pass, means only: the photos land on the same skin tone
    if k.startswith("posle"): g_ = (g_ * blackdrop(v)).astype(np.uint8)       # the ПОСЛЕ studio backdrop → pure black, judged on the raw photo
    PH[k] = anon_top(g_, ALIGN[k][0] - 40)
def panel(im, n, pw, ph, neck_at=30, span=360):
    neck, navel, cx = ALIGN[n]; s_ = span / (navel - neck); r = cv2.resize(im, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    out = np.zeros((ph, pw, 3), np.float32); ox, oy = int(cx * s_ - pw / 2), int(neck * s_ - neck_at)
    xs0, ys0 = max(0, ox), max(0, oy); xs1, ys1 = min(r.shape[1], ox + pw), min(r.shape[0], oy + ph)
    out[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox] = r[ys0:ys1, xs0:xs1]
    lum = out.mean(2); k = np.clip((lum - 0.2) / 0.14, 0, 1); k = k * k * (3 - 2 * k)
    k = np.maximum(k, np.clip((out[..., 2] - out[..., 0] - 0.04) / 0.05, 0, 1)); return out * k[..., None]

# ---------------- subtitles: hand-checked words, timings from speech recognition (relative to each line's start)
SUBS = {
    'A': [('Несмотря', 0.06, 0.38), ('на', 0.38, 0.54), ('то,', 0.54, 0.6), ('что', 0.6, 0.76), ('девушка', 0.76, 1.22), ('стройная,', 1.22, 2.06),
          ('расскажите', 2.34, 2.84), ('мне,', 2.84, 3.02), ('как', 3.02, 3.66), ('спортом', 3.66, 4.18), ('вот', 4.18, 4.4), ('это', 4.4, 4.56), ('можно', 4.56, 4.9), ('убрать?', 4.9, 5.28)],
    'B': [('Смотрите,', 0.0, 0.58), ('вот', 0.58, 0.84), ('как', 0.84, 1.06), ('здесь', 1.06, 1.32), ('кожа', 1.32, 1.66), ('провисает.', 1.66, 2.86)],
    'C': [('Мы', 0.04, 0.14), ('будем', 0.14, 0.46), ('подтягивать', 0.46, 1.16), ('грудь,', 1.16, 2.04), ('делать', 2.1, 2.38), ('липофилинг,', 2.38, 3.48),
          ('то', 3.56, 3.6), ('есть', 3.6, 3.74), ('подтянем', 3.74, 4.32), ('её', 4.32, 4.74), ('с', 4.74, 4.84), ('Т-образной', 4.84, 5.48), ('подтяжкой.', 5.48, 6.04)],
    'D': [('Ещё', 0.04, 0.3), ('у', 0.3, 0.38), ('нас', 0.38, 0.56), ('липосакция', 0.56, 1.2), ('бёдер.', 1.2, 1.62), ('Заберём', 1.62, 2.04), ('с', 2.04, 2.14), ('бёдер', 2.14, 2.5),
          ('жир', 2.5, 2.8), ('для', 2.8, 2.96), ('того,', 2.96, 3.14), ('чтобы', 3.14, 3.38), ('добавить', 3.38, 3.9), ('в', 3.9, 4.28), ('грудь.', 4.28, 4.6)],
    'E': [('Красота?', 0.1, 0.88)],
    'E2': [('Да,', 0.1, 0.28), ('вообще', 0.28, 0.58), ('шикарно.', 0.58, 1.62), ('Как', 2.78, 3.5), ('в', 3.5, 3.72), ('молодости.', 3.72, 4.28)],
    'F': [('А', 0.08, 0.5), ('живот,', 0.5, 0.74), ('совершенно', 0.74, 1.24), ('другая', 1.24, 1.54), ('талия,', 1.54, 1.86), ('у', 1.94, 1.98), ('вас', 1.98, 2.02),
          ('хорошая', 2.02, 2.48), ('плоскость.', 2.48, 2.98)],
}
def _sub_words():
    out = [(w.strip(",.?"), t0 + a, t0 + e) for b, s_, parts, t0, d in SEG for w, a, e in SUBS[b]]
    return [(w, a, min(e + 0.3, out[i + 1][1] - 0.04) if i + 1 < len(out) else e + 0.3) for i, (w, a, e) in enumerate(out)]
SUB_WORDS = _sub_words()
def sub_y(t): return 1166 if t < vstart("D") else 1420         # on IMG_6217/6218 the plate sits right over the burned-in captions
def subtitles(fr, t):
    for w, a, e in SUB_WORDS:
        t0 = a - 0.04
        if not (t0 <= t < e): continue
        p = eo((t - t0) / 0.12); im = scaled(alpha_img(w, mont(44, 300)), 0.85 + 0.15 * p)
        h = 64; bw = im.shape[1] + 34; op = min(1, (e - t) / 0.06); y = sub_y(t); cx = 540 if y < 1300 else 500
        box = np.zeros((h, int(bw)), np.uint8); cv2.rectangle(box, (0, 0), (box.shape[1] - 1, h - 1), 255, -1)
        blend(fr, cv2.GaussianBlur(box, (0, 0), 1.2).astype(np.float32) / 255, cx - bw / 2, y - h / 2, DEEP, 0.8 * op)
        blend(fr, im, cx - im.shape[1] / 2, y - im.shape[0] / 2 + 3, WHITE, op)

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
def qmarks(fr, t, t0, pts):
    for k, (x, y, sz) in enumerate(pts):
        u = t - t0 - 0.18 * k
        if u < 0: continue
        s_ = eback(u / 0.35); a = alpha_img("?", bebas(max(8, int(sz * s_))))
        blend(fr, a, x - a.shape[1] / 2, y - a.shape[0] / 2 + 8 * np.sin(u * 4 + k), PINK, min(1, u / 0.15))

# A — the hook: a slim woman, yet the belly stays; «спортом?» — the dumbbell gets crossed out
HA = top(Drop(["Я СТРОЙНАЯ.", "*ПОЧЕМУ ЖИВОТ ВСЁ РАВНО*", "*НЕ УХОДИТ?*"], 96, 70, 0, maxw=900), 320)
def s_a(t):
    fr = full(frame("IMG_6217", lsrc("A", t)), 0.5, 0.5, pull(t, 0, 1.12, 1.0, 3.0))
    HA.draw(fr, t, 0.05); y = under(HA, 30); chip(fr, "ДО", 70, y)
    ts = R("A", 3.66)                                                    # «спортом»
    if t > ts:
        u = t - ts; card(fr, "dumbbell", 830, 800, 0.85 * eback(u / 0.35), min(1, u / 0.2), eo(u / 0.5))
        xm = R("A", 4.9)                                                 # «убрать?» — no, not with sport
        if t > xm: xmark(fr, 830, 800, 78, eo((t - xm) / 0.35), PINK, 12)
    qmarks(fr, t, R("A", 4.9), [(200, 820, 120), (300, 700, 80)])
    return fr

# B — the problem is not the weight: skin hangs below the navel, the muscles part along the midline (diastasis)
HB = top(Drop(["ПРОБЛЕМА БЫЛА", "*УЖЕ НЕ В ВЕСЕ*"], 104, 70, 0, maxw=900), 320)
def s_b(t):
    lt = t - T_S["B"]; fr = full(frame("IMG_6217", lsrc("B", t)))
    HB.draw(fr, t, T_S["B"] + 0.05)
    mx = lerp(512, 470, cl(lt / 2.6))                                    # the navel drifts left as the hand comes in
    ts = R("B", 1.32)                                                    # «кожа провисает»: the fold sags, arrows go down
    if t > ts:
        u = t - ts; sag = 26 * eio(u / 0.8)
        pts = [(mx - 230 + 460 * i / 30, 960 + sag * np.sin(np.pi * i / 30) ** 1.5) for i in range(31)]
        path_draw(fr, pts, eo(u / 0.5), PINK, 6)
        for k, x in enumerate((mx - 120, mx, mx + 120)):
            a0 = 990 + sag * 0.9; arrow(fr, (x, a0), (x, a0 + 60), eo((u - 0.2 - 0.12 * k) / 0.35), PINK, 5)
        pillx(fr, "избыток кожи", 70, 1250, eo((u - 0.1) / 0.35), 0.0, 30)
    td = R("B", 2.0)                                                     # diastasis: the two muscle lines move apart
    if t > td:
        u = t - td; sp = lerp(8, 46, eio(u / 0.7))
        for sgn in (-1, 1): dashed(fr, [(mx + sgn * sp, 560), (mx + sgn * sp * 0.8, 790)], eo(u / 0.4), WHITE, 5, 16)
        if u > 0.3:
            q = eo((u - 0.3) / 0.3); arrow(fr, (mx, 675), (mx - sp - 40, 675), q, PINK, 4); arrow(fr, (mx, 675), (mx + sp + 40, 675), q, PINK, 4)
        pillx(fr, "диастаз", 70, 1330, eo((u - 0.15) / 0.35), 0.0, 30)
    y = under(HB, 30); chip(fr, "ДО", 70, y); return fr

# C — and the breast: lift + lipofilling; a little scheme lifts the breast and draws the T
HC = top(Drop(["И ЕЩЁ ХОТЕЛОСЬ", "*ВЕРНУТЬ ФОРМУ ГРУДИ*"], 96, 70, 0, maxw=900), 320)
def breast_scheme(fr, t, cx, cy, s, op, lift_p, t_p, fill_p):
    hw = 150 * s; blend(fr, rrect(cx - hw, cy - hw, cx + hw, cy + hw * 1.05, 34 * s), 0, 0, NAVY, 0.88 * op)
    rrect_line(fr, cx - hw, cy - hw, cx + hw, cy + hw * 1.05, 34 * s, PINK, 3, op)
    if op < 0.95: return
    dy = lerp(42, 0, eio(lift_p)) * s; k = 1 + 0.12 * eio(fill_p)
    pts = [(cx + 70 * s * k * np.cos(a), cy + dy + 10 * s + 60 * s * k * np.sin(a)) for a in np.linspace(-0.15 * np.pi, 1.15 * np.pi, 40)]
    if dy > 0.5 * s: path_draw(fr, [(cx + 70 * s * np.cos(a), cy + 52 * s + 60 * s * np.sin(a)) for a in np.linspace(-0.15 * np.pi, 1.15 * np.pi, 40)], 1, WHITE, 2, op=0.35)
    path_draw(fr, pts, 1, WHITE, 5)
    dot(fr, cx, cy + dy + 22 * s, 9 * s, PINK)
    if lift_p > 0: arrow(fr, (cx - 112 * s, cy + 70 * s), (cx - 112 * s, cy - 40 * s), eo(lift_p), PINK, 5)
    if t_p > 0:                                                          # the T: vertical from the areola down + along the fold
        path_draw(fr, [(cx, cy + dy + 36 * s), (cx, cy + dy + 70 * s * k)], eo(t_p * 2), PINK, 5)
        path_draw(fr, [(cx - 58 * s, cy + dy + 70 * s * k), (cx + 58 * s, cy + dy + 70 * s * k)], eo(t_p * 2 - 1), PINK, 5)
    for j in range(3):                                                    # lipofilling: drops sink into the breast
        u = fill_p * 3 - j
        if 0 < u < 1.4: fill_poly(fr, [(x, y - 70 * s * (1 - eo(min(1, u)))) for x, y in drop_pts(cx - 30 * s + 30 * s * j, cy + dy - 6 * s, 11 * s)], PINK, 1 - cl((u - 1) / 0.4))
def drop_pts(cx, cy, r):
    return [(cx, cy - 1.9 * r)] + [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(-0.2 * np.pi, 1.2 * np.pi, 20)]
def s_c(t):
    lt = t - T_S["C"]; fr = full(frame("IMG_6218", lsrc("C", t), 200), 0.55, 0.3, lerp(1.32, 1.22, eo(lt / 6.0)))
    HC.draw(fr, t, T_S["C"] + 0.1)
    ta = R("C", 0.46); u = t - ta                                       # «подтягивать грудь»
    if u > 0:
        breast_scheme(fr, t, 820, 860, 0.85, min(1, u / 0.25), cl((t - R("C", 1.16)) / 0.8), cl((t - R("C", 4.84)) / 0.9), cl((t - R("C", 2.38)) / 1.1))
    pillx(fr, "Т-образная подтяжка", 70, 1250, eo((t - R("C", 4.84)) / 0.35), 0.0, 30)
    pillx(fr, "+ липофилинг", 70, 1330, eo((t - R("C", 2.38)) / 0.35), 0.0, 30)
    y = under(HC, 30); chip(fr, "ДО", 70, y); return fr

# D — the surgeon in the OR («липосакция бёдер»), then the ДО photo: fat leaves the hips and flies into the breast
HD = top(Drop(["ЖИР С БЁДЕР ИСПОЛЬЗОВАЛИ", "*ДЛЯ ЛИПОФИЛИНГА ГРУДИ*"], 84, 70, 0, maxw=920), 320)
D_IM, D_M = cover(PH["do_front"], W, H, 0.5, 0.42, 1.0)
def pmap(x, y): return D_M[0, 0] * x + D_M[0, 2], D_M[1, 1] * y + D_M[1, 2]
HIPS = [pmap(510, 1330), pmap(1040, 1330)]; BRST = [pmap(500, 740), pmap(935, 750)]
def s_d(t):
    if t < T_D2:
        lt = t - T_S["D"]; fr = full(V["IMG_6213"].at(8.0 + lt), 0.5, 0.35, lerp(1.12, 1.04, eo(lt / 1.5)))
        HD.draw(fr, t, T_S["D"] + 0.1); return fr
    lt = t - T_D2; z = lerp(1.04, 1.0, eo(lt / 1.2)); fr = D_IM.copy()
    if z > 1.001: fr = cv2.warpAffine(fr, np.float32([[z, 0, (1 - z) * 540], [0, z, (1 - z) * 900]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
    HD.draw(fr, t, T_S["D"] + 0.1); chip(fr, "ДО", 70, under(HD, 30))
    th = R("D", 2.14)                                                    # «с бёдер» — the hips light up
    for (x, y) in HIPS:
        u = t - th
        if u < 0: continue
        uu = (u * 1.2) % 1; ring(fr, x, y, 34 + 40 * uu, (1 - uu) * 0.9, 4); dot(fr, x, y, 16 * eback(u / 0.3) * (1 - 0.4 * eo((t - R("D", 3.4)) / 1.0)))
    tf = R("D", 2.7)                                                     # «жир … добавить в грудь» — drops travel up along arcs
    for k, ((x0, y0), (x1, y1)) in enumerate(zip(HIPS, BRST)):
        for j in range(4):
            u = (t - tf - 0.22 * j - 0.1 * k) / 1.0
            if not (0 < u < 1.25): continue
            q = eio(min(1, u)); bx = lerp(x0, x1, q) + (-1 if k == 0 else 1) * 110 * np.sin(np.pi * q); by = lerp(y0, y1, q)
            r = 12 * (1 - 0.25 * q); fill_poly(fr, drop_pts(bx, by, r), PINK, 1 - cl((u - 1) / 0.25))
        if t > tf:
            path = [(lerp(x0, x1, q) + (-1 if k == 0 else 1) * 110 * np.sin(np.pi * q), lerp(y0, y1, q)) for q in np.linspace(0, 1, 40)]
            dashed(fr, path, eo((t - tf) / 0.9), WHITE, 3, 12, 0.7)
    tb = R("D", 4.28)                                                    # «грудь» — the breast fills
    for (x, y) in BRST:
        u = t - tb
        if u > 0: ring(fr, x, y, lerp(30, 70, eo(u / 0.5)), 1 - 0.5 * eo(u / 1.0), 5)
    return fr

# E — the next day, dressing: «как в молодости»; a clock winds back
HE = Drop(["*КАК В МОЛОДОСТИ*"], 150, 70, 0, maxw=920)
TE = Track(["на следующий день после операции"], 34, 72, 0, 400)
def clock(fr, cx, cy, r, t, t0, op):
    ring(fr, cx, cy, r, op, 6)
    for k in range(12):
        a = k * np.pi / 6; path_draw(fr, [(cx + r * 0.82 * np.cos(a), cy + r * 0.82 * np.sin(a)), (cx + r * 0.92 * np.cos(a), cy + r * 0.92 * np.sin(a))], 1, WHITE, 3, op=op)
    sp = -eio(cl((t - t0) / 1.6)) * 4 * np.pi                         # the hands run backwards
    for L, w_, f in ((0.55, 7, 1 / 12), (0.8, 4, 1.0)):
        a = -np.pi / 2 + sp * f + 1.2; path_draw(fr, [(cx, cy), (cx + r * L * np.cos(a), cy + r * L * np.sin(a))], 1, PINK if f == 1 else WHITE, w_, op=op)
    arrow(fr, (cx + r * 1.15 * np.cos(-0.2), cy + r * 1.15 * np.sin(-0.2)), (cx + r * 1.15 * np.cos(-1.3), cy + r * 1.15 * np.sin(-1.3)), op * eo((t - t0) / 0.5), PINK, 5)
def s_e(t):
    lt = t - T_S["E"]; fr = full(V["IMG_6215"].at(lsrc("E", t)), 0.5, 0.45, pull(t, T_S["E"], 1.1, 1.0, 2.0))
    TE.y = 330 + TE.top; TE.draw(fr, t, T_S["E"] + 0.15); chip(fr, "ПОСЛЕ", 70, 400)
    tk = R("E2", 2.78)                                                   # «как в молодости»
    if t > tk - 0.25:
        clock(fr, 780, 700, 110 * eback((t - tk + 0.25) / 0.35), t, tk, min(1, (t - tk + 0.25) / 0.2))
    HE.x = 540 - HE.rows[0][1] / 2; HE.y = 1000; HE.draw(fr, t, tk)
    return fr

# F — «а живот… совершенно другая талия… хорошая плоскость»: the three zones pop, each with its own mini-sign
HF = Track(["первый результат", "после комплексной коррекции"], 32, 72, 0, 400)
ZONES = [("ГРУДЬ", "IMG_6215", R("F", 0.08)), ("ЖИВОТ", "", R("F", 0.5)), ("ТАЛИЯ", "", R("F", 1.54))]
def zone_icon(fr, k, cx, cy, p):
    if k == 0: path_draw(fr, [(cx - 30 + 60 * i / 20, cy - 4 + 22 * np.sin(np.pi * i / 20)) for i in range(21)], p, PINK, 5)                   # a round curve
    if k == 1: path_draw(fr, [(cx - 32, cy + 6), (cx + 32, cy + 6)], p, PINK, 6)                                                                # flat
    if k == 2:                                                                                                                                  # hourglass )(
        for sg in (-1, 1): path_draw(fr, [(cx + sg * (30 - 16 * np.sin(np.pi * i / 20)), cy - 26 + 52 * i / 20) for i in range(21)], p, PINK, 5)
def s_f(t):
    fr = full(V["IMG_6215"].at(lsrc("F", t)), 0.5, 0.45, pull(t, T_S["F"], 1.1, 1.0, 2.0))
    chip(fr, "ПОСЛЕ", 70, 330)
    for k, (lab, _, t0) in enumerate(ZONES):
        lt = t - t0
        if lt < 0: continue
        y = 450 + k * 96; a = alpha_img(lab, bebas(84)); s_ = 0.85 + 0.15 * eback(lt / 0.3); a2 = scaled(a, s_)
        zone_icon(fr, k, 112, y + 40, eo(lt / 0.4)); blend(fr, a2, 170, y + 40 - a2.shape[0] / 2 + 6, WHITE, min(1, lt / 0.12))
    HF.y = 760 + HF.top; HF.draw(fr, t, R("F", 1.9)); return fr

# G — ПОСЛЕ: the closing thought; the flat belly ringed
HG = Drop(["ЕСЛИ ВЕС УЖЕ УШЁЛ,", "А ЖИВОТ ОСТАЛСЯ,", "*ДЕЛО МОЖЕТ БЫТЬ*", "*НЕ В ЛИШНЕМ ВЕСЕ*"], 92, 70, 0, maxw=920, stag=0.02)
TG = Track(["на консультации можно понять причину"], 32, 72, 0, 400)
G_IM, G_M = cover(PH["posle_front"], W, H, 0.5, 0.36, 1.0)
def s_g(t):
    lt = t - T_G; z = lerp(1.08, 1.0, eo(lt / 2.5))
    fr = cv2.warpAffine(G_IM, np.float32([[z, 0, (1 - z) * 540], [0, z, (1 - z) * 700]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
    chip(fr, "ПОСЛЕ", 70, 290)
    nx, ny = G_M[0, 0] * ALIGN["posle_front"][2] + G_M[0, 2], G_M[1, 1] * ALIGN["posle_front"][1] + G_M[1, 2]
    nx, ny = z * nx + (1 - z) * 540, z * ny + (1 - z) * 700
    if lt > 0.4: uu = ((lt - 0.4) * 0.9) % 1; ring(fr, nx, ny, 90 + 40 * uu, (1 - uu) * 0.8, 4)
    HG.y = 1440 - HG.h - 60; HG.draw(fr, t, T_G + 0.3)
    TG.y = HG.y + HG.h + 40 + TG.top; TG.draw(fr, t, T_G + 1.6); return fr

# K — ПОСЛЕ over ДО, front / three-quarter / side, then the logo
KPW, KPH = 360, 690
COLL = np.zeros((2 * KPH, W, 3), np.float32)
for i, ang in enumerate(("front", "34", "side")):
    COLL[:KPH, i * KPW:(i + 1) * KPW] = panel(PH["posle_" + ang], "posle_" + ang, KPW, KPH)
    COLL[KPH:, i * KPW:(i + 1) * KPW] = panel(PH["do_" + ang], "do_" + ang, KPW, KPH)
def s_k(t):
    lt = t - T_K; fr = np.zeros((H, W, 3), np.float32); y0 = 250
    for i in range(3):                                                   # the columns slide in one by one
        p = eo((lt - 0.12 * i) / 0.4)
        if p <= 0: continue
        dy = int((1 - p) * 60); sub = COLL[:, i * KPW:(i + 1) * KPW]
        ya = y0 + dy; fr[ya:ya + 2 * KPH, i * KPW:(i + 1) * KPW] = sub[:min(2 * KPH, H - ya)] * p
    fr[y0 + KPH - 2:y0 + KPH + 2] = PINK
    for i in (1, 2): fr[y0:y0 + 2 * KPH, i * KPW - 1:i * KPW + 1] = DEEP
    chip(fr, "ПОСЛЕ", 20, y0 + 16); chip(fr, "ДО", 20, y0 + KPH + 16)
    f = 1 - eo(lt / 0.18); fr = fr * (1 - f) + f * 0.9
    if lt > 1.6: blend(fr, logo_alpha(260), 540 - 130, y0 + 2 * KPH + 30, PINK, eo((lt - 1.6) / 0.5))
    return fr

SCENES = [(0, s_a), (T_S["B"], s_b), (T_S["C"], s_c), (T_S["D"], s_d), (T_S["E"], s_e), (T_S["F"], s_f), (T_G, s_g), (T_K, s_k)]
TRANS = {T_S["B"]: ("zoom", 0.35), T_S["C"]: ("diag", 0.45), T_S["D"]: ("zoom", 0.35), T_D2: ("flash", 0.18), T_S["E"]: ("iris", 0.5),
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
def felt(f=95, dur=0.3):
    n = int(dur * SR); t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t) * env(n, 0.002, 0.07) + 0.5 * lp(rng.standard_normal(n), 600) * env(n, 0.001, 0.018)
def glass(f=1760, dur=1.4):
    n = int(dur * SR); t = np.arange(n) / SR
    return sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / d) for r, a, d in [(1, 1, 0.6), (2.76, 0.45, 0.3), (5.4, 0.2, 0.15)]) * env(n, 0.002, 9) / 1.6
def water(f0=500, f1=1700):
    n = int(0.12 * SR); t = np.arange(n) / SR; ph = 2 * np.pi * np.cumsum(f0 + (f1 - f0) * (1 - np.exp(-t / 0.015))) / SR
    return np.sin(ph) * env(n, 0.001, 0.035)
def click():
    n = int(0.09 * SR); t = np.arange(n) / SR; x = hp(rng.standard_normal(n), 2500)
    e = np.exp(-t / 0.004) + 0.7 * np.exp(-np.clip(t - 0.045, 0, None) / 0.005) * (t > 0.045)
    x = x * e / 2; k = felt(160, 0.08); x[:len(k)] += k * 0.4; return x

MUSIC = os.path.join(SRC, "song6.wav")                                 # the client's track for this reel (audio of the screen recording)
M0 = 13.0                                                              # the quiet bridge (28-34 s) falls under the lines, the full part under the ending
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
        for a, o in parts[1:]:
            nxt = x[int(a * SR):int(o * SR)]; f = int(0.015 * SR); r = np.linspace(0, 1, f)
            seg = np.concatenate([seg[:-f], seg[-f:] * (1 - r) + nxt[:f] * r, nxt[f:]])
        fi, fo = int(0.03 * SR), int(0.11 * SR)
        if b == "E2": fi = int(0.01 * SR)
        if b == "E": fo = int(0.02 * SR)
        seg[:fi] *= np.sin(np.linspace(0, np.pi / 2, fi)) ** 2; seg[-fo:] *= np.cos(np.linspace(0, np.pi / 2, fo)) ** 2
        fr_rms = np.sqrt(np.mean(seg[:len(seg) // 960 * 960].reshape(-1, 960) ** 2, 1)); g = 1 / (np.sqrt(np.mean(np.sort(fr_rms)[-max(1, len(fr_rms) * 2 // 5):] ** 2)) + 1e-6)
        if b == "E2": seg = hp(seg, 160)                                   # the quiet reaction: lifted to the surgeon's level, rumble cut
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
    if len(KEYS) == 1: put(KEYS[0], 0.05, 0.35)
    else:
        for j in range(n_ch): put(KEYS[j % len(KEYS)], 0.05 + j * HA.stag, 0.32)          # typing under the title
    put(TICK, R("A", 3.66), 0.6); put(felt(70, 0.35), R("A", 4.9), 0.5)                  # dumbbell, crossed out
    put(TICK, R("B", 1.42), 0.6); put(TICK, R("B", 2.15), 0.6)                            # the two pills
    put(TICK, R("C", 0.46), 0.5); put(TICK, R("C", 2.38), 0.6); put(TICK, R("C", 4.84), 0.6)
    put(TICK, R("D", 2.14), 0.6)
    for j in range(8): put(water(450 + 90 * j, 1500 + 120 * j), R("D", 2.7) + 0.13 * j, 0.12)   # the drops travelling
    for j in range(12): put(tick(), R("E2", 2.78) + 0.12 * j, 0.2)                       # the clock winding back
    put(glass(1174.7), R("E2", 2.78), 0.1)
    for _, _, t0 in ZONES: put(TICK, t0, 0.6)
    put(SHOT, T_K, 0.5); put(glass(1318.5), T_K + 1.6, 0.1)
    sfx[:n] *= 1 - 0.6 * np.clip(dd, 0, 1)
    tt = np.arange(n) / SR; fade = np.clip((DUR - tt) / 1.8, 0, 1) ** 1.5 * np.minimum(1, tt / 0.15)
    mus = track(n) * ((G_GAP - (G_GAP - G_VO) * np.clip(dd, 0, 1)) * fade)[:, None]      # the music ducks under every line
    mix = vo[:, None] + mus + (sfx[:n] * np.clip((DUR - tt) / 1.0, 0, 1))[:, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (1.5 / 20); mix /= max(1, np.abs(mix).max() / 0.95)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

def make_cover(path):
    """inside the centre 3:4 (y 240-1680): ДО | ПОСЛЕ fronts, the hook below"""
    CBG = np.clip(np.float32([0, 8, 28]) / 255 * (1 + 0.3 * np.exp(-(((xx - 540) / 900) ** 2 + ((yy - 520) / 900) ** 2)))[..., None], 0, 1).astype(np.float32)
    fr = CBG.copy()
    for k, n in enumerate(("do_front", "posle_front")):
        c = panel(PH[n], n, 540, 820, neck_at=60, span=470); fr[330:330 + 820, k * 540:(k + 1) * 540] = c
    fr[330:1150, 538:542] = PINK; chip(fr, "ДО", 40, 350); chip(fr, "ПОСЛЕ", 580, 350)
    g = np.clip((yy[:, :1] - 950) / 200, 0, 1)[..., None]; fr = fr * (1 - g) + CBG * g
    blend(fr, logo_alpha(260), 540 - 130, 262, PINK)
    hd = Drop(["Я СТРОЙНАЯ.", "*ПОЧЕМУ ЖИВОТ ВСЁ РАВНО*", "*НЕ УХОДИТ?*"], 100, 70, 1180, maxw=900); hd.draw(fr, 99, 0)
    ty = Track(["когда спорт уже не решает проблему"], 34, 72, 0, 400); ty.y = 1180 + hd.h + 46 + ty.top; ty.draw(fr, 99, 0)
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
