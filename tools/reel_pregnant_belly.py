#!/usr/bin/env python3
"""Reels «Меня постоянно принимали за беременную» — история пациентки после абдоминопластики.

The patient's interview voice and the surgeon in the OR, whole sentences; the apron lifted, the comments popping as speech bubbles,
counters (years, births), the belly crossed, «красоты» struck through, the bending video with the surgeon in a circle, the marking and OR,
the client's ПОСЛЕ/ДО collages (one pulls back onto a white page), the interview split under ПОСЛЕ 3/4, hearts, an end card.
Phrase captions in thin Montserrat, centred; no patient face (body framed from the neck down, interview already blurred).

usage: REEL_SRC=<dir: logo.png, tick.wav, keys/, music.wav, tr_IMG_1302.txt, tr_IMG_1233.txt, _work/ with the clips (.mp4 30 fps + .wav) and photos (.jpg)>
       python3 tools/reel_pregnant_belly.py [--preview t1,t2 | --cover | --audio-only]
"""
import os, sys, json, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_pregnant_belly"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_pregnant_belly"
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
    asc, desc = f.getmetrics(); w = int(f.getlength(txt)) + 8; P = _pad(f)          # room above the cap height for Й / Ё, the same below so centring holds
    im = Image.new("L", (w, asc + desc + 4 + 2 * P), 0); ImageDraw.Draw(im).text((4, 2 + P + asc), txt, font=f, fill=255, anchor="ls")
    return np.asarray(im, np.float32) / 255
def _pad(f): return int(f.size * 0.18)
def cap_top(f): asc, _ = f.getmetrics(); return 2 + _pad(f) + asc + f.getbbox("Н", anchor="ls")[1]
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
                o_ = min(1, p * 1.6) * (1 - q)                                # no dark outline on the letters (the client's call)
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
                blend(fr, a, ox + dx + i * sp - 4, s.y + k * s.lh - s.top, s.col, op)

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




# ==== shared helpers (from the whole-figure reel)
def skin(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); return (hsv[..., 2] > 90) & (hsv[..., 1] > 25) & (hsv[..., 1] < 170) & ((hsv[..., 0] < 25) | (hsv[..., 0] > 165))
def onphoto(img, fx, fy, z, t, t0, z0=None, d=2.5):
    """a photo full screen with a slow pull-back; returns the frame and a mapping photo px → screen"""
    zz = z if z0 is None else lerp(z0, z, eo((t - t0) / d)); im, M = cover(img, W, H, fx, fy, zz)
    return im, (lambda x, y: (M[0, 0] * x + M[0, 2], M[1, 1] * y + M[1, 2]))
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


# ---------------- voice: the patient's interview and the surgeon, whole sentences
VO = [("A", "IMG_1302", [(121.08, 123.14), (123.56, 125.60)]),     # а молодая женщина мне говорили, что думали, что я в положении всё время
      ("B1", "IMG_1302", [(15.70, 18.26)]),                         # эта проблема беспокоит меня много лет
      ("B2", "IMG_1302", [(21.78, 22.62), (24.10, 27.42)]),         # четверо родов, постоянные диеты, похудение, а живот не уходил
      ("C", "IMG_1233", [(14.00, 19.70)]),                          # это уже даже не красота, а физиология. человеку трудно нагнуться, ходить и вообще как-то что-то делать
      ("D", "IMG_1233", [(8.78, 10.08), (19.84, 23.55)]),           # вот этот фартук будем убирать, поэтому постараемся облегчить ей жизнь
      ("E", "IMG_1302", [(94.40, 96.70), (96.94, 98.70), (99.04, 100.96)]),   # ещё не могу пока поверить своему счастью. подхожу к зеркалу, смотрюсь, и мне пока даже не верится, что это я
      ("F", "IMG_1302", [(9.32, 10.06), (10.32, 11.92), (13.50, 15.05)]),   # но самое главное, моё эмоциональное состояние, это не передать, оно великолепное («что», «оно просто, ну» cut)
      ("G", "IMG_1302", [(101.14, 104.16)])]                        # если это ваша мечта, если это вас беспокоит
GAP = {"B1": 0.35, "B2": 0.3, "C": 0.4, "D": 0.35, "E": 0.45, "F": 0.4, "G": 0.4}
SEG = []; _t = 0.1
for _b, _s, _parts in VO:
    if SEG: _t += GAP[_b]
    _d = sum(o - a for a, o in _parts); SEG.append((_b, _s, _parts, _t, _d)); _t += _d
def vstart(b): return next(s[3] for s in SEG if s[0] == b)
def vend(b): return next(s[3] + s[4] for s in SEG if s[0] == b)
def Rs(b, src):
    _, _, parts, t0, d = next(x for x in SEG if x[0] == b); acc = 0.0
    for a, o in parts:
        if src <= o: return t0 + acc + max(0.0, src - a)
        acc += o - a
    return t0 + acc
def Sr(b, t):
    _, _, parts, t0, d = next(x for x in SEG if x[0] == b); acc = 0.0; u = t - t0
    if u < 0: return parts[0][0] + u
    for a, o in parts:
        if u <= acc + o - a: return a + u - acc
        acc += o - a
    return parts[-1][1] + u - acc
T_S = {b: vstart(b) - 0.12 for b, *_ in VO}; T_S["A"] = 0.0
T_Z = vend("G") + 1.2; DUR = T_Z + 2.8

V = {k: Src(os.path.join(WORK, f"{k}.mp4")) for k in ("IMG_1240", "IMG_1239", "IMG_1233", "IMG_1238", "IMG_1252", "IMG_1302")}
def photo(n): return cv2.imread(os.path.join(WORK, f"{n}.jpg"))[..., ::-1].copy()
COLS = {"front": photo("IMG_1465"), "34": photo("IMG_1466"), "side": photo("IMG_1467")}     # the client's ПОСЛЕ (top) / ДО (bottom) collages
PO34, POF = photo("IMG_1458"), photo("IMG_1459")
TABLE_Y = {"IMG_1252": 1000}                                                 # only the OR wide shot (the client's call: no blur on the first OR shot)                               # the patient on the table: everything below this row of the frame is blurred
def table_blur(f, y0):
    """a strong, soft blur over the patient's body on the operating table (the surgeon above it stays sharp)"""
    h, w = f.shape[:2]; y0 = int(y0 * h / 1920); reg = f[max(0, y0 - 80):].astype(np.float32)
    bl = cv2.resize(cv2.GaussianBlur(cv2.resize(reg, (max(1, w // 24), max(1, reg.shape[0] // 24))), (0, 0), 2), (w, reg.shape[0]))
    m = np.clip((np.arange(reg.shape[0]) - 0) / 80.0, 0, 1)[:, None, None]; out = f.copy(); out[max(0, y0 - 80):] = (reg * (1 - m) + bl * m).astype(np.uint8); return out
def skin_blur(f, box=(0, 880, 760, 1480)):
    """blur only the patient's exposed skin on the table (the drape, the room and the team stay sharp)"""
    x0, y0, x1, y1 = box; reg = f[y0:y1, x0:x1]; hsv = cv2.cvtColor(reg, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] < 25) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 35) & (hsv[..., 2] > 80)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)); m = cv2.dilate(m, np.ones((25, 25), np.uint8))
    a = np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 12) * 1.6, 0, 1)[..., None]
    bl = cv2.resize(cv2.GaussianBlur(cv2.resize(reg, (max(1, (x1 - x0) // 20), max(1, (y1 - y0) // 20))), (0, 0), 1.5), (x1 - x0, y1 - y0)).astype(np.float32)
    out = f.copy(); out[y0:y1, x0:x1] = (reg * (1 - a) + bl * a).astype(np.uint8); return out
def raw(k, st):
    if k == "IMG_1252": return skin_blur(V[k].at(st))
    f = V[k].at(st); return table_blur(f, TABLE_Y[k]) if k in TABLE_Y else f
def clip(k, st, z=1.0, fx=0.5, fy=0.5): return full(raw(k, st), fx, fy, z)
def body(k, st, z=1.15, fy=0.62): return clip(k, st, z, 0.5, fy)                # her body from the neck down (no face)
def interview(t, b, z=1.0): return full(V["IMG_1302"].at(Sr(b, t)), 0.5, 0.5, z)
def po_photo(img, z=1.1): return full(img, 0.5, 0.42, z)                          # zoomed so the photo's own watermark at the bottom is off the frame

# ---------------- captions: phrases, the spoken word lights up, thin Montserrat, key words pink, centred
FIX = {"мое": "моё", "Еще": "Ещё", "положении": "положении", "похудение,": "похудения,"}
KEY = {"положении", "беременную", "много", "лет.", "четверо", "родов,", "диеты,", "живот", "уходил.", "красота,", "физиология.", "нагнуться,", "ходить", "фартук", "облегчить",
       "счастью.", "верится,", "я.", "эмоциональное", "великолепное.", "передать,", "мечта,", "беспокоит,"}
_W = {k: [(FIX.get(w, w), a, e) for w, a, e in eval(open(os.path.join(SRC, f"tr_{k}.txt")).read().replace("np.float64", ""))] for k in ("IMG_1302", "IMG_1233")}
_W["IMG_1302"] = [x for x in _W["IMG_1302"] if not (9.0 <= x[1] < 15.6)] + [     # her actual words there (checked on a closer listen)
    ("Но", 9.34, 9.54), ("самое", 9.54, 9.68), ("главное,", 9.68, 10.02), ("что", 10.22, 10.34), ("моё", 10.34, 10.52), ("эмоциональное", 10.52, 11.22), ("состояние,", 11.22, 11.90),
    ("оно", 11.90, 12.14), ("просто,", 12.14, 13.06), ("ну,", 13.14, 13.50), ("это", 13.52, 13.66), ("не", 13.66, 13.90), ("передать,", 13.90, 14.20), ("оно", 14.40, 14.54), ("великолепное.", 14.54, 15.0)]
def _words():
    out = []
    for b, s_, parts, t0, d in SEG:
        acc = 0.0
        for a, o in parts:
            for w, ws, we in _W[s_]:
                if a <= (ws + we) / 2 <= o and w: out.append((w, t0 + acc + max(ws, a) - a, t0 + acc + min(we, o) - a, b))
            acc += o - a
    return [(w, a, min(e + 0.3, out[i + 1][1] - 0.04) if i + 1 < len(out) else e + 0.3, b) for i, (w, a, e, b) in enumerate(out)]
WORDS = _words()
def _phr():
    out, cur = [], []
    for w in WORDS:
        if cur and (w[3] != cur[-1][3] or w[1] - cur[-1][2] > 0.45): out.append(cur); cur = []
        cur.append(w)
        if len(cur) >= 3 or w[0][-1:] in ",.?!" or len(" ".join(x[0] for x in cur)) > 18: out.append(cur); cur = []
    if cur: out.append(cur)
    return [(ph, ph[0][1] - 0.06, min(ph[-1][2] + 0.5, (out[k + 1][0][1] - 0.08) if k + 1 < len(out) else 1e9)) for k, ph in enumerate(out)]
PHR = _phr(); CAP_Y = 1460
def subtitles(fr, t):
    for ph, t0, t1 in PHR:
        if not (t0 <= t < t1): continue
        p = eo((t - t0) / 0.18); op = min(1, (t1 - t) / 0.08) * min(1, (t - t0) / 0.1); ims = []
        for w, a, e, _ in ph:
            ww = w.strip(",.?!"); key = w in KEY or ww in KEY
            ims.append((alpha_img(ww, mont(44 if key else 42, 400 if key else 300)), PINK if key else WHITE, a))
        sp = 14; tw = sum(a_.shape[1] for a_, _, _ in ims) + sp * (len(ims) - 1)
        if tw > 640: k_ = 640 / tw; ims = [(scaled(a_, k_), c_, s_) for a_, c_, s_ in ims]; sp = int(sp * k_); tw = sum(a_.shape[1] for a_, _, _ in ims) + sp * (len(ims) - 1)
        x = W / 2 - tw / 2; y = CAP_Y + 16 * (1 - p)
        for a_, col, a in ims:
            lit = 1.0 if t >= a - 0.04 else 0.42
            blend(fr, cv2.GaussianBlur(a_, (0, 0), 3), x + 2, y - a_.shape[0] / 2 + 4, DEEP, 0.6 * op); blend(fr, a_, x, y - a_.shape[0] / 2, col, op * lit); x += a_.shape[1] + sp

# ---------------- look (safe zone: x 72-1018, y 234-1600, not x > 864 below y 1347)
LX = 80; BLK = np.float32([0, 0, 0]); HB_ = 1336
GTOP = np.repeat((np.clip(1 - (yy[:, :1] - 150) / 700, 0, 1) ** 1.3 * 0.35)[..., None], W, 1).astype(np.float32)
GBOT = np.repeat((np.clip((yy[:, :1] - 950) / 560, 0, 1) ** 1.2 * 0.7)[..., None], W, 1).astype(np.float32)
def dark(fr): fr = np.asarray(fr, np.float32).copy(); shade(fr, GTOP, BLK); shade(fr, GBOT, BLK); return fr
def low(hd, bottom=HB_): hd.y = bottom - hd.h; return hd
def navy_card(fr, cx, cy, hw, hh, s, op=1.0):
    if s <= 0.02: return
    blend(fr, rrect(cx - hw * s, cy - hh * s, cx + hw * s, cy + hh * s, 28 * s), 0, 0, NAVY, 0.92 * op)
    rrect_line(fr, cx - hw * s, cy - hh * s, cx + hw * s, cy + hh * s, 28 * s, PINK, 3, op)
def label(fr, txt, x, y, p, size=28):
    if p <= 0: return
    a = alpha_img(txt, mont(size, 400)); w = a.shape[1] + 44; h = int(size * 1.9); s_ = eback(p)
    blend(fr, rrect(x, y - h * s_ / 2, x + w * s_, y + h * s_ / 2, h * s_ / 2), 0, 0, NAVY, 0.9 * min(1, p * 2))
    rrect_line(fr, x, y - h * s_ / 2, x + w * s_, y + h * s_ / 2, h * s_ / 2, PINK, 2, min(1, p * 2))
    if s_ > 0.9: blend(fr, a, x + 22, y - a.shape[0] / 2 + 2, WHITE, cl((p - 0.5) * 2))
def flash(fr, lt, d=0.12):
    if 0 <= lt < d: f_ = 0.3 * (1 - lt / d); fr = fr * (1 - f_) + f_
    return fr
def small(hd, lines, gap=22): tr = Track(lines, 30, LX, 0, 300); tr.y = hd.y + hd.h + gap + tr.top; return tr
def bubble(fr, txt, cx, cy, p, tail=-1):
    """a speech bubble that pops in (somebody else's comment)"""
    if p <= 0: return
    a = alpha_img(txt, mont(34, 500)); w, h = a.shape[1] + 56, 84; s_ = eback(p); op = min(1, p * 2)
    x0, y0, x1, y1 = cx - w * s_ / 2, cy - h * s_ / 2, cx + w * s_ / 2, cy + h * s_ / 2
    blend(fr, rrect(x0, y0, x1, y1, 30 * s_), 0, 0, WHITE, 0.96 * op)
    fill_poly(fr, [(cx + tail * 40 * s_, y1 - 2), (cx + tail * 80 * s_, y1 - 2), (cx + tail * 95 * s_, y1 + 30 * s_)], WHITE, 0.96 * op)
    if s_ > 0.85: blend(fr, a, cx - a.shape[1] / 2, cy - a.shape[0] / 2 + 2, NAVY, cl((p - 0.5) * 2))
def counter_card(fr, t, t0, cx, cy, top_txt, vals, step=0.22, col=PINK, t1=1e9):
    s_ = 0.82 * eback((t - t0) / 0.35) * (1 - eio((t - t1) / 0.3) if t > t1 else 1)
    if t < t0 or s_ <= 0.02: return
    navy_card(fr, cx, cy, 150, 115, s_)
    if s_ > 0.6:
        k = min(len(vals) - 1, int((t - t0 - 0.1) / step)) if t > t0 + 0.1 else 0
        text_c(fr, vals[k], cx, cy + 18, bebas(110), col); text_c(fr, top_txt, cx, cy - 72, mont(24, 500), WHITE)
def white_card(img, t, t0, d=1.2, z1=0.78):
    """the photo pulls back from full screen to a card on a white page (a soft shadow under it)"""
    p = eio((t - t0) / d); z = lerp(1.0, z1, p); fr = np.ones((H, W, 3), np.float32)
    w, h = int(W * z), int(H * z); im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x0, y0 = (W - w) // 2, (H - h) // 2 - int(60 * p)
    sh = np.zeros((H, W), np.float32); sh[y0 + 20:y0 + h + 20, x0 + 10:x0 + w + 10] = 1; sh = cv2.GaussianBlur(sh, (0, 0), 24) * 0.35 * p
    fr *= 1 - sh[..., None]; fr[y0:y0 + h, x0:x0 + w] = im; return fr

# A — 0:00 the apron lifted; the hook; her voice; a flash of the result as a teaser
HA = low(Drop(["МЕНЯ ПОСТОЯННО ПРИНИМАЛИ", "*ЗА БЕРЕМЕННУЮ*"], 92, LX, 0, maxw=920))
T_TEASE = 2.35
def s_a(t):
    fr = dark(body("IMG_1240", 2.6 + t, lerp(1.12, 1.18, eo(t / 4.0)), 0.6)); HA.draw(fr, t, 0.05, T_S["B1"] - 0.3)
    for k, (w0, txt, x, y, tl) in enumerate(((123.82, "«в положении?»", 760, 470, -1), (124.38, "«ты беременная?»", 330, 640, 1))):
        u = t - Rs("A", w0)
        if u > 0: bubble(fr, txt, x, y, eo(u / 0.3) * (1 - cl((t - T_S["B1"] + 0.35) / 0.3)), tl)
    return fr
# B — the story: front, then profile; years; four births; diets; the belly stayed
HB = low(Drop(["Я ХУДЕЛА,", "*НО ЖИВОТ НЕ УХОДИЛ*"], 96, LX, 0, maxw=920))
def s_b(t):
    tp = vstart("B2") - 0.1
    if t < tp: fr = dark(body("IMG_1239", 0.4 + (t - T_S["B1"]), lerp(1.3, 1.24, eo((t - T_S["B1"]) / 3)), 0.66))
    else: fr = dark(flash(body("IMG_1239", 3.8 + (t - tp) * 0.8, 1.3, 0.66), t - tp))
    ty = Rs("B1", 17.40)                                                    # «много лет»: a calendar card, the pages flip
    if ty < t < tp + 0.3: counter_card(fr, t, ty, 820, 470, "МНОГО ЛЕТ", ["1", "3", "5", "7", "10+"], 0.15, PINK, tp)
    tr = Rs("B2", 21.82)                                                    # «четверо родов»: 1→4
    if t > tr: counter_card(fr, t, tr, 820, 470, "РОДОВ", ["1", "2", "3", "4"], 0.16, PINK, Rs("B2", 24.76))
    td = Rs("B2", 24.76)                                                    # «диеты, похудение»: the scale arrow dives
    if t > td:
        s_ = 0.82 * eback((t - td) / 0.35); navy_card(fr, 820, 470, 150, 115, s_)
        if s_ > 0.6: text_c(fr, "ДИЕТЫ", 820, 400, mont(24, 500), WHITE); arrow(fr, (820, 440), (820, 540), eo((t - td - 0.1) / 0.5), PINK, 10)
    tn = Rs("B2", 26.48)                                                    # «не уходил»: the belly crossed
    if t > tn: xmark(fr, 540, 1020, 110, eo((t - tn) / 0.45), PINK, 12)
    if t >= vstart("B2") - 0.1: HB.draw(fr, t, vstart("B2"))
    return fr
# C — the surgeon in the OR: «не красота, а физиология»; on «нагнуться» the bending video, the surgeon in a circle
HC = low(Drop(["ЭТО УЖЕ НЕ ВОПРОС", "*КРАСОТЫ*"], 104, LX, 0, maxw=920)); TC = None
def s_c(t):
    global TC
    tb = Rs("C", 16.54)
    if t < tb: fr = dark(clip("IMG_1233", Sr("C", t), lerp(1.12, 1.04, eo((t - T_S["C"]) / 2.5)), 0.5, 0.42))
    else:
        fr = dark(flash(body("IMG_1239", 7.4 + (t - tb), 1.22, 0.62), t - tb))
        pip(fr, raw("IMG_1233", Sr("C", t)), 820, 470, 150, eo((t - tb - 0.1) / 0.35), 0.5, 0.25, 1.4)
        for k, (w0, lab) in enumerate(((16.70, "нагнуться"), (17.16, "ходить"), (19.50, "делать"))):
            u = t - Rs("C", w0)
            if u > 0: label(fr, lab, LX, 330 + k * 66, eo(u / 0.3), 28)
    HC.draw(fr, t, T_S["C"] + 0.05)
    tk = Rs("C", 15.46)                                                     # «физиология»: «красоты» struck through
    if t > tk:
        x0 = LX; y = HC.y + HC.lh + HC.size * 0.36; wq = bebas(HC.size).getlength("КРАСОТЫ")
        path_draw(fr, [(x0 - 6, y), (x0 + wq + 6, y - 6)], eo((t - tk) / 0.35), PINK, 9)
    TC = TC or small(HC, ["избыток тканей мешал повседневной жизни"]); TC.draw(fr, t, tk + 0.3); return fr
# D — the marking, then the OR wide: АБДОМИНОПЛАСТИКА
HD = low(Drop(["*АБДОМИНОПЛАСТИКА*"], 100, LX, 0, maxw=920)); TD = None
def s_d(t):
    global TD
    to = Rs("D", 19.84)
    if t < to: fr = dark(clip("IMG_1238", 9.0 + (t - T_S["D"]), 1.15, 0.45, 0.45))
    else: fr = dark(flash(clip("IMG_1252", 4.3 + (t - to), lerp(1.1, 1.0, eo((t - to) / 3.0)), 0.5, 0.5), t - to))
    HD.draw(fr, t, T_S["D"] + 0.05); TD = TD or small(HD, ["коррекция большого кожно-жирового фартука"]); TD.draw(fr, t, T_S["D"] + 0.5)
    tf = Rs("D", 8.98)
    if tf < t < to + 0.2: label(fr, "фартук", LX, 330, eo((t - tf) / 0.3) * (1 - cl((t - to) / 0.2)), 30)
    return fr
# E — the first reveal: the side collage, it pulls back onto a white page; then the front collage
HE = low(Drop(["МНЕ ДАЖЕ НЕ ВЕРИТСЯ,", "*ЧТО ЭТО Я*"], 96, LX, 0, maxw=920))
def s_e(t):
    tf = Rs("E", 99.04)
    if t < tf:
        u = t - T_S["E"]; fr = full(COLS["side"], 0.5, 0.5, lerp(1.05, 1.0, eo(u / 1.5))) if u < 2.2 else white_card(COLS["side"], t, T_S["E"] + 2.2, 1.0, 0.8)
        if u < 2.2: chip(fr, "ПОСЛЕ", LX, 262); chip(fr, "ДО", LX, 1000)
        ts = Rs("E", 96.02)                                                 # «счастью»: sparkles
        for k, (x, y) in enumerate(((870, 300), (210, 420), (880, 760))): sparkle(fr, x, y, 36, t, ts + 0.15 * k)
    else:
        fr = flash(full(COLS["front"], 0.5, 0.5, lerp(1.04, 1.0, eo((t - tf) / 2.0))), t - tf); chip(fr, "ПОСЛЕ", LX, 262); chip(fr, "ДО", LX, 1000)
    fr = np.asarray(fr, np.float32); shade(fr, GBOT, BLK); HE.draw(fr, t, Rs("E", 99.68)); return fr
# F — the interview for the first time (blurred face), split with ПОСЛЕ 3/4 on top; hearts on «великолепное»
YD = 960
HF = Drop(["САМОЕ ГЛАВНОЕ,", "*МОЁ ЭМОЦИОНАЛЬНОЕ СОСТОЯНИЕ*"], 72, LX, 0, maxw=920)
GSPL = np.repeat((np.clip((yy[:, :1] - 600) / 350, 0, 1) ** 1.3 * 0.6 * (yy[:, :1] < YD))[..., None], W, 1).astype(np.float32)
GBS = np.repeat((np.clip((yy[:, :1] - 1330) / 300, 0, 1) ** 1.2 * 0.5)[..., None], W, 1).astype(np.float32)
def s_f(t):
    tsw = Rs("F", 13.52)                                                    # «это не передать»: cut to her result in 3/4 (no split — the client's call)
    if t < tsw: fr = dark(interview(t, "F", lerp(1.0, 1.06, eo((t - T_S["F"]) / 2.0))))
    else: fr = dark(flash(po_photo(PO34, lerp(1.16, 1.1, eo((t - tsw) / 2.0))), t - tsw)); chip(fr, "ПОСЛЕ", LX, 262)
    HF.y = HB_ - HF.h
    HF.draw(fr, t, T_S["F"] + 0.05)
    th = Rs("F", 14.54)                                                     # «великолепное»: hearts float up
    if t > th: hearts(fr, t, th, ((860, 820, 0.0), (220, 760, 0.3), (700, 640, 0.6)))
    return fr
# G — two quick comparisons, then the result full; «если вы узнали себя»
HG = low(Drop(["ЕСЛИ ВЫ УЗНАЛИ СЕБЯ", "*В ЭТОЙ ИСТОРИИ...*"], 92, LX, 0, maxw=920))
def s_g(t):
    u = t - T_S["G"]; d = (T_Z - T_S["G"]) / 2; k = min(1, int(u / d)); n = ("34", "side")[k]      # the 3/4 and side ПОСЛЕ/ДО collages
    fr = full(COLS[n], 0.5, 0.5, lerp(1.04, 1.0, eo((u - k * d) / d)))
    if k: v = u - k * d; fr = full(COLS[("34", "side")[k - 1]], 0.5, 0.5, 1.0) * (1 - eio(v / 0.2)) + fr * eio(v / 0.2) if v < 0.2 else fr
    chip(fr, "ПОСЛЕ", LX, 262); chip(fr, "ДО", LX, 1000)
    fr = np.asarray(fr, np.float32); shade(fr, GBOT, BLK); HG.draw(fr, t, T_S["G"] + 0.3); return fr
# Z — the end card
HZ = Drop(["НА КОНСУЛЬТАЦИИ ОБСУДИМ,", "ЧТО МОЖНО СДЕЛАТЬ", "*В ВАШЕМ СЛУЧАЕ*"], 84, LX, 760, maxw=920, stag=0.02)
def s_z(t):
    fr = gbg(t); blend(fr, logo_alpha(300), 540 - 150, 480, PINK, eo((t - T_Z - 0.1) / 0.5)); HZ.draw(fr, t, T_Z + 0.2)
    if t > DUR - 0.8: fr = fr * (1 - eio((t - DUR + 0.8) / 0.8))
    return fr

SCENES = [(0, s_a), (T_S["B1"], s_b), (T_S["C"], s_c), (T_S["D"], s_d), (T_S["E"], s_e), (T_S["F"], s_f), (T_S["G"], s_g), (T_Z, s_z)]
TRANS = {T_S["B1"]: ("zoom", 0.3), T_S["C"]: ("diag", 0.4), T_S["D"]: ("zoom", 0.3), T_S["E"]: ("iris", 0.5), T_S["F"]: ("diag", 0.4), T_S["G"]: ("zoom", 0.3), T_Z: ("iris", 0.5)}
def base(t):
    k = max(i for i, (s0, _) in enumerate(SCENES) if t >= s0); s0, fn = SCENES[k]; kind, d = TRANS.get(s0, ("none", 0))
    if kind != "none" and t < s0 + d:
        p = (t - s0) / d; old = np.asarray(SCENES[k - 1][1](t), np.float32); new = np.asarray(fn(t), np.float32)
        if kind == "iris": return iris(old, new, p, 540, 900)
        return zoom_through(old, new, p) if kind == "zoom" else diag_wipe(old, new, p)
    return fn(t)
LOGO_W = logo_alpha(200)
def render(t):
    fr = np.asarray(base(t), np.float32).copy()
    if t < T_Z: subtitles(fr, t); blend(fr, LOGO_W, W // 2 - LOGO_W.shape[1] // 2, 1540, WHITE, 0.8)
    return fr

# ---------------- audio
MUSIC = os.path.join(SRC, "music.wav"); M0 = 0.6; G_VO, G_GAP = 0.022, 0.075      # the client's track
def track(n):
    if not os.path.exists(MUSIC): return np.zeros((n, 2))
    w = wave.open(MUSIC); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64).reshape(-1, w.getnchannels()) / 32768
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    if w.getframerate() != SR: x = np.stack([np.interp(np.arange(int(len(x) * SR / w.getframerate())) * w.getframerate() / SR, np.arange(len(x)), x[:, c]) for c in range(2)], 1)
    seg = x[int(M0 * SR):int(M0 * SR) + n]; seg = seg / np.sqrt(np.mean(seg.mean(1) ** 2)); return np.pad(seg, ((0, n - len(seg)), (0, 0)))
def pop(f0=1100, f1=380, dur=0.09):
    n = int(dur * SR); t = np.arange(n) / SR; ph = 2 * np.pi * np.cumsum(f1 + (f0 - f1) * np.exp(-t / 0.015)) / SR
    return np.sin(ph) * env(n, 0.002, 0.03)
def build_audio(path):
    n = int(DUR * SR); vo = np.zeros(n); act = np.zeros(n); X = {}
    for b, s_, parts, t0, d in SEG:
        if s_ not in X:
            w = wave.open(os.path.join(WORK, f"{s_}.wav")); X[s_] = hp(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768, 90)
        segs = []
        for a, o in parts:
            sg = X[s_][int(a * SR):int(o * SR)].copy(); fi, fo = int(0.015 * SR), int(0.04 * SR)
            sg[:fi] *= np.sin(np.linspace(0, np.pi / 2, fi)) ** 2; sg[-fo:] *= np.cos(np.linspace(0, np.pi / 2, fo)) ** 2; segs.append(sg)
        sg = np.concatenate(segs); rr = np.sqrt(np.mean(sg[:len(sg) // 960 * 960].reshape(-1, 960) ** 2, 1)); g = 1 / (np.sqrt(np.mean(np.sort(rr)[-max(1, len(rr) * 2 // 5):] ** 2)) + 1e-6)
        i = int(t0 * SR); vo[i:i + len(sg)] += (sg * g)[:n - i]; act[max(0, i - int(0.25 * SR)):i + len(sg) + int(0.1 * SR)] = 1
    vo = np.tanh(vo * 0.16 * 1.6) / 1.6; dd = np.convolve(act, np.ones(int(0.25 * SR)) / int(0.25 * SR), "same")
    sfx = np.zeros(n + 2 * SR)
    def put(x_, t, g_): i = int(max(0, t) * SR); s_ = sfx[i:i + len(x_)]; s_ += x_[:len(s_)] * g_
    def load(fn):
        w_ = wave.open(fn); x_ = np.frombuffer(w_.readframes(w_.getnframes()), np.int16).astype(np.float64) / 32768
        if w_.getnchannels() == 2: x_ = x_.reshape(-1, 2).mean(1)
        if w_.getframerate() != SR: x_ = np.interp(np.arange(int(len(x_) * SR / w_.getframerate())) * w_.getframerate() / SR, np.arange(len(x_)), x_)
        return x_ / (np.abs(x_).max() + 1e-9)
    KEYS = [load(os.path.join(SRC, "keys", f"k{i:03d}.wav")) for i in range(1, 9)]; TICK = load(os.path.join(SRC, "tick.wav"))
    def typing(hd, t0, g_=0.3):
        for j in range(sum(len(q) for q, _ in hd.rows)): put(KEYS[j % 8], t0 + j * hd.stag, g_ * 0.8)
    typing(HA, 0.05, 0.4); typing(HB, vstart("B2")); typing(HC, T_S["C"] + 0.05); typing(HD, T_S["D"] + 0.05); typing(HE, Rs("E", 99.68)); typing(HF, T_S["F"] + 0.05)
    typing(HG, T_S["G"] + 0.3); typing(HZ, T_Z + 0.2, 0.3)
    for w0 in (123.82, 124.38): put(pop(), Rs("A", w0), 0.5)                  # the comments pop
    ty = Rs("B1", 17.40)
    for j in range(5): put(tick(), ty + 0.1 + 0.15 * j, 0.25)
    tr = Rs("B2", 21.82)
    for j in range(4): put(tick(), tr + 0.1 + 0.16 * j, 0.3)
    put(swoosh(0.5), Rs("B2", 24.76), 0.12); put(felt(70, 0.35), Rs("B2", 26.48), 0.5)
    put(TICK, Rs("C", 15.46), 0.5); put(swoosh(0.4), Rs("C", 16.54), 0.12)
    for w0 in (16.70, 17.16, 19.50): put(TICK, Rs("C", w0), 0.45)
    put(TICK, Rs("D", 8.98), 0.5); put(swoosh(0.6), T_S["E"], 0.12)
    for k in range(3): put(glass(1318.5 + 200 * k), Rs("E", 96.02) + 0.15 * k, 0.05)
    put(swoosh(0.5), T_S["E"] + 2.2, 0.1)
    for k in range(3): put(pop(900, 500, 0.08), Rs("F", 14.54) + 0.3 * k, 0.35)
    put(glass(1174.7), T_Z + 0.2, 0.08)
    sfx[:n] *= 1 - 0.55 * np.clip(dd, 0, 1)
    tt = np.arange(n) / SR; fade = np.clip((DUR - tt) / 2.5, 0, 1) ** 1.5 * np.clip(tt / 1.0, 0, 1)
    mus = track(n) * ((G_GAP - (G_GAP - G_VO) * np.clip(dd, 0, 1)) * fade)[:, None]
    mix = vo[:, None] + mus + (sfx[:n] * np.clip((DUR - tt) / 1.0, 0, 1))[:, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (1.5 / 20); mix /= max(1, np.abs(mix).max() / 0.95)
    with wave.open(path, "wb") as w: w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

COVERS = [(["МЕНЯ ВСЁ ВРЕМЯ ПРИНИМАЛИ", "*ЗА БЕРЕМЕННУЮ*"], "история пациентки после абдоминопластики"),
          (["4 РОДОВ.", "*А ЖИВОТ ВСЁ РАВНО НЕ УХОДИЛ*"], "история пациентки после абдоминопластики"),
          (["МНЕ БЫЛО ТЯЖЕЛО", "*ДАЖЕ НАКЛОНЯТЬСЯ*"], "абдоминопластика: не только про красоту"),
          (["ПОСМОТРЕЛА В ЗЕРКАЛО", "*И НЕ ПОВЕРИЛА, ЧТО ЭТО Я*"], "история пациентки после абдоминопластики")]
def make_cover(path, k=0):
    """ДО | ПОСЛЕ fronts side by side (from the client's collage), the title below"""
    fr = np.zeros((H, W, 3), np.float32); c = COLS["front"].astype(np.float32) / 255
    po, do = c[:960, 270:810], c[960:, 270:810]; fr[260:1220, 0:540] = do; fr[260:1220, 540:1080] = po; fr[260:1220, 539:541] = PINK
    chip(fr, "ДО", 40, 280); chip(fr, "ПОСЛЕ", 580, 280)
    g = np.clip((yy[:, :1] - 1000) / 220, 0, 1)[..., None]; fr = fr * (1 - g)
    lines, sub = COVERS[k]; hd = Drop(lines, 100, 70, 1250, maxw=940); hd.draw(fr, 99, 0)
    ty = Track([sub], 32, 72, 0, 300); ty.y = 1250 + hd.h + 44 + ty.top; ty.draw(fr, 99, 0)
    blend(fr, logo_alpha(220), 540 - 110, 1560, WHITE, 0.9)
    cv2.imwrite(path, cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

def to8(fr): return (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)
if __name__ == "__main__":
    print(f"duration {DUR:.2f}s; scenes", [round(s, 2) for s, _ in SCENES])
    if "--preview" in sys.argv:
        for t in [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]:
            cv2.imwrite(os.path.join(WORK, f"prev_{t:05.2f}.jpg"), cv2.cvtColor(to8(render(t)), cv2.COLOR_RGB2BGR))
        sys.exit()
    if "--cover" in sys.argv:
        for k in range(len(COVERS)): make_cover(os.path.join(ROOT, f"{NAME}_cover{k + 1}.jpg"), k)
        sys.exit()
    build_audio(os.path.join(WORK, "mix.wav"))
    if "--audio-only" in sys.argv: sys.exit()
    vid = os.path.join(WORK, "video.mp4")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", vid], stdin=subprocess.PIPE)
    for i in range(int(round(DUR * FPS))): p.stdin.write(to8(render(i / FPS)).tobytes())
    p.stdin.close(); p.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid, "-i", os.path.join(WORK, "mix.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                    os.path.join(ROOT, f"{NAME}.mp4")], check=True)
    for k in range(len(COVERS)): make_cover(os.path.join(ROOT, f"{NAME}_cover{k + 1}.jpg"), k)
    print("done")
