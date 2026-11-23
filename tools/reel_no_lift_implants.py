#!/usr/bin/env python3
"""Reels «Хочу грудь, но не хочу рубцы от подтяжки» — Анна Тваури.

Ptosis grade I, no lift: the surgeon at the consultation and the ДО photo («птоз первой степени» → a sagging-breast scheme), the
question over the three ДО angles («3+ размер» counting), the marking video with the anatomical implant filling the scheme and the
sag lifting, the operating room and «395 мл, высокий профиль», the result angle by angle with a ДО→ПОСЛЕ slider under «в будущем
подтяжка может понадобиться», the question card, and the collages (ДО on top, ПОСЛЕ below). One skin tone for all six photos.

usage:
  REEL_SRC=<folder with _work/IMG_2742, IMG_2743, VS, OR (.mp4 + .wav), the photos (IMG_2725/2731/2734.png, P_front/34/side.jpg),
  tr_IMG_2742.txt, tr_VS.txt, logo.png, tick.wav, keys/ or keyboard.wav, optional shutter_client.wav, music.wav>
  python3 tools/reel_no_lift_implants.py [--preview t1,t2 | --cover | --audio-only]
"""
import os, sys, json, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_no_lift"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_no_lift_implants"
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





# ---------------- voice: the surgeon only (whole sentences)
VO = [("A", "IMG_2742", [(8.76, 16.48)]),     # у нас есть небольшой птоз, птоз первой степени, но мы не хотим делать подтяжку, не хотим, чтобы были рубцы
      ("B", "IMG_2742", [(17.08, 21.78)]),    # и так как объём железы сам по себе не очень большой, а хотим мы 3+ размер
      ("C", "IMG_2742", [(27.94, 34.34)]),    # наша основная задача — заполнить это пространство и вытянуть железу
      ("D", "VS", [(6.02, 11.88)]),           # поставили нашей девочке импланты анатомические 395, высокий профиль
      ("E", "VS", [(23.14, 33.16)])]          # и хочу отметить, что в будущем ей всё равно придётся делать подтяжку… эти же импланты оставим
GAP = {"B": 0.3, "C": 0.35, "D": 0.45, "E": 0.4}
SEG = []; _t = 0.04
for _b, _s, _parts in VO:
    if SEG: _t += GAP[_b]
    _d = sum(o - a for a, o in _parts); SEG.append((_b, _s, _parts, _t, _d)); _t += _d
def vstart(b): return next(s[3] for s in SEG if s[0] == b)
def vend(b): return next(s[3] + s[4] for s in SEG if s[0] == b)
def Rs(b, src):
    _, _, parts, t0, d = next(x for x in SEG if x[0] == b); return t0 + src - parts[0][0]
T_S = {b: vstart(b) - 0.1 for b, *_ in VO}; T_S["A"] = 0.0
T_Q = vend("E") + 0.3; T_K = T_Q + 2.6; K_D = 1.9; DUR = T_K + 3 * K_D + 0.6

V = {k: Src(os.path.join(WORK, f"{k}.mp4")) for k in ("IMG_2742", "IMG_2740", "IMG_2743", "VS", "OR")}
FACE_SPOT = {"IMG_2743": (420, 30, 230, 130), "IMG_2742": (420, 20, 220, 120)}    # the patient's chin at the top edge of the marking shots
def soft_spot(f, cx, cy, rx, ry):
    m = np.zeros(f.shape[:2], np.float32); cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1, -1)
    m = np.clip(cv2.GaussianBlur(m, (0, 0), rx * 0.3) * 1.5, 0, 1)[..., None]
    return (f * (1 - m) + cv2.GaussianBlur(f, (0, 0), 30) * m).astype(np.uint8)
def clip(k, st, z=1.0, fx=0.5, fy=0.5):
    f = V[k].at(st)
    if k in FACE_SPOT and (k != "IMG_2742" or st > 20): f = soft_spot(f, *FACE_SPOT[k])
    return full(f, fx, fy, z)
OR_WIDE = Src(os.path.join(WORK, "OR_wide.mp4"))
def on_white(st, z=1.0):
    """the horizontal operating-room shot kept whole, on a white page"""
    f = OR_WIDE.at(st).astype(np.float32) / 255; h, w = f.shape[:2]; s_ = z; fw, fh = int(W * s_), int(h * W / w * s_)
    fr = np.ones((H, W, 3), np.float32) * np.float32([0.97, 0.96, 0.95]); im = cv2.resize(f, (fw, fh)); x0, y0 = (W - fw) // 2, 720 - fh // 2
    fr[y0:y0 + fh, max(0, x0):max(0, x0) + min(fw, W)] = im[:, max(0, -x0):max(0, -x0) + min(fw, W)]; return fr

# ---------------- photos: ДО and ПОСЛЕ, one skin tone for all six
def photo(n): return cv2.imread(os.path.join(WORK, n))[..., ::-1].copy()
NAMES = {"do_front": "IMG_2725.png", "do_34": "IMG_2731.png", "do_side": "IMG_2734.png", "po_front": "P_front.jpg", "po_34": "P_34.jpg", "po_side": "P_side.jpg"}
ALIGN = {"do_front": (314, 124, 1086), "do_34": (413, 124, 1050), "do_side": (448, 116, 1020),
         "po_front": (93, 83, 747), "po_34": (287, 79, 730), "po_side": (129, 80, 710)}           # (shoulder line, nipple-sticker height, body centre x): the stickers are the same size in every photo → one true scale
def skin(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); return (hsv[..., 2] > 90) & (hsv[..., 1] > 25) & (hsv[..., 1] < 170) & ((hsv[..., 0] < 25) | (hsv[..., 0] > 165))
def skin_stats(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); m = skin(rgb)
    return np.float32([[lab[..., c][m].mean(), lab[..., c][m].std()] for c in range(3)])
def grade(rgb, st, tgt):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3): lab[..., c] = (lab[..., c] - st[c, 0]) * float(np.clip(tgt[c, 1] / (st[c, 1] + 1e-3), 0.8, 1.25)) + tgt[c, 0]
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    k = cv2.GaussianBlur(cv2.dilate(skin(rgb).astype(np.uint8) * 255, np.ones((9, 9), np.uint8)).astype(np.float32) / 255, (0, 0), 8)[..., None]
    return (rgb * (1 - k) + out * k).astype(np.uint8)                     # only the skin is graded: jeans, stockings, backdrop keep their colour
RAW = {k: photo(v) for k, v in NAMES.items()}; ST = {k: skin_stats(v) for k, v in RAW.items()}; TGT = np.mean(list(ST.values()), 0)
PH = {k: grade(v, ST[k], TGT) for k, v in RAW.items()}
CHIN = {"do_front": 0, "do_34": 246, "do_side": 380, "po_front": 0, "po_34": 115, "po_side": 100}          # below the jaw: the start of the neck
KNEE = {"do_front": 3225, "do_34": 3200, "do_side": 3270, "po_front": 1775, "po_34": 1800, "po_side": 1830}
def panel(n, pw=W, ph=960, top=30, bottom=30):
    """from the jaw to the knees: the jaw on `top`, one true scale (nipple stickers) for every photo, nothing above the jaw"""
    sh, sth, cx = ALIGN[n]; s_ = (ph - top - bottom) / 26.0 / sth; ch = CHIN[n]
    if FILL: s_ = ph / ((KNEE[n] - ch) * 1.0)                                  # full screen: the jaw on the top edge, the knees on the bottom edge
    ox, oy = cx * s_ - pw / 2, ch * s_ - (0 if FILL else top)
    out = cv2.warpAffine(PH[n], np.float32([[s_, 0, -ox], [0, s_, -oy]]), (pw, ph), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255
    if FILL: return out                                                   # the studio backdrop kept, the sides continued with it
    m = np.ones((ph, pw), np.float32); yk = KNEE[n] * s_ - oy; rows = np.arange(ph)
    m *= np.clip((rows - top + 2) / 6, 0, 1)[:, None] * np.clip((yk + 30 - rows) / 50, 0, 1)[:, None]
    return out * cv2.GaussianBlur(m, (0, 0), 5)[..., None]
FILL = True
def _unused(n, pw=W, ph=960, sh_at=110, nav_at=560):
    sh, sth, cx = ALIGN[n]; s_ = (nav_at - sh_at) / (1353 / 124 * sth); r = cv2.resize(PH[n], None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    out = np.zeros((ph, pw, 3), np.float32); ox, oy = int(cx * s_ - pw / 2), int(sh * s_ - sh_at)
    xs0, ys0 = max(0, ox), max(0, oy); xs1, ys1 = min(r.shape[1], ox + pw), min(r.shape[0], oy + ph)
    out[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox] = r[ys0:ys1, xs0:xs1]
    m = np.zeros((ph, pw), np.float32); m[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox] = 1
    return out * cv2.GaussianBlur(cv2.erode(m, np.ones((21, 21), np.uint8)), (0, 0), 7)[..., None]
FULL = {n: panel(n, W, H, 300, 300) for n in NAMES}                         # full-screen version, the same framing for ДО and ПОСЛЕ

# ---------------- subtitles
FIX = {"наши": "нашей", "девочки": "девочке", "3": "3+", "плюс": "", "кожаную": "кожную"}
KEY = {"птоз,", "птоз", "подтяжку,", "рубцы.", "3+", "размер,", "заполнить", "вытянуть", "анатомические", "395,", "высокий", "профиль.", "подтяжку", "импланты"}
_W = {k: [(FIX.get(w, w), a, e) for w, a, e in eval(open(os.path.join(SRC, f"tr_{k}.txt")).read().replace("np.float64", ""))] for k in ("IMG_2742", "VS")}
def _sub_words():
    out = []
    for b, s_, parts, t0, d in SEG:
        for a, o in parts:
            for w, ws, we in _W[s_]:
                if a <= (ws + we) / 2 <= o and w: out.append((w, t0 + max(ws, a) - a, t0 + min(we, o) - a))
    return [(w, a, min(e + 0.3, out[i + 1][1] - 0.04) if i + 1 < len(out) else e + 0.3) for i, (w, a, e) in enumerate(out)]
SUB_WORDS = _sub_words(); SUB_Y = 1420
def subtitles(fr, t):
    for w, a, e in SUB_WORDS:
        t0 = a - 0.04
        if not (t0 <= t < e): continue
        key = w in KEY; ww = w.strip(",.?!")
        p = eo((t - t0) / 0.12); im = scaled(alpha_img(ww, mont(48 if key else 44, 500 if key else 300)), 0.85 + 0.15 * p)
        h = 66; bw = im.shape[1] + 34; op = min(1, (e - t) / 0.06)
        box = np.zeros((h, int(bw)), np.uint8); cv2.rectangle(box, (0, 0), (box.shape[1] - 1, h - 1), 255, -1)
        blend(fr, cv2.GaussianBlur(box, (0, 0), 1.2).astype(np.float32) / 255, 500 - bw / 2, SUB_Y - h / 2, DEEP, 0.82 * op)
        blend(fr, im, 500 - im.shape[1] / 2, SUB_Y - im.shape[0] / 2 + 3, PINK if key else WHITE, op)

# ---------------- blocks (safe zone: x 72-1018, y 234-1600, not x > 864 below y 1347)
LX = 80; BLK = np.float32([0, 0, 0])
GTOP = np.repeat((np.clip(1 - (yy[:, :1] - 150) / 760, 0, 1) ** 1.3 * 0.55)[..., None], W, 1).astype(np.float32)
GBOT = np.repeat((np.clip((yy[:, :1] - 1120) / 520, 0, 1) ** 1.4 * 0.4)[..., None], W, 1).astype(np.float32)
def dark(fr): fr = np.asarray(fr, np.float32).copy(); shade(fr, GTOP, BLK); shade(fr, GBOT, BLK); return fr
def pop_s(t, t0, t1=1e9):
    if t < t0 or t > t1 + 0.3: return 0.0
    return eback((t - t0) / 0.35) * (1 - eio((t - t1) / 0.3) if t > t1 else 1)
def navy_card(fr, cx, cy, hw, hh, s, op=1.0):
    if s <= 0.02: return
    blend(fr, rrect(cx - hw * s, cy - hh * s, cx + hw * s, cy + hh * s, 28 * s), 0, 0, NAVY, 0.92 * op)
    rrect_line(fr, cx - hw * s, cy - hh * s, cx + hw * s, cy + hh * s, 28 * s, PINK, 3, op)
def check(fr, cx, cy, r, p, col=PINK): path_draw(fr, [(cx - r, cy), (cx - r * 0.3, cy + r * 0.7), (cx + r, cy - r * 0.7)], p, col, 9)
def breast_icon(fr, cx, cy, s, sag, imp=0.0, col=WHITE):
    """a breast in profile: the lower pole sags by `sag` (0..1); `imp` fills an implant pocket behind it"""
    k = 1 - 0.0 * sag
    wall = [(cx - 60 * s, cy - 110 * s), (cx - 60 * s, cy + 120 * s)]
    pts = [(cx - 60 * s, cy - 90 * s)] + [(cx - 60 * s + 120 * s * np.sin(a) * (1 + 0.15 * imp), cy - 40 * s + 80 * s * (1 - np.cos(a)) * 0.5 + 70 * s * sag * np.sin(a) ** 2) for a in np.linspace(0, np.pi, 30)]
    if imp > 0: fill_poly(fr, [(cx - 60 * s + 95 * s * np.sin(a) * imp, cy - 30 * s + 60 * s * (1 - np.cos(a)) * 0.5) for a in np.linspace(0, np.pi, 30)], PINK, 0.55)
    path_draw(fr, wall, 1, WHITE, 3, op=0.5); path_draw(fr, pts, 1, col, 6)
    nx, ny = pts[16]; dot(fr, nx + 6 * s, ny, 7 * s, PINK)

# A — the hook: the surgeon at the consultation, then the ДО photo; «птоз первой степени» → a sagging breast scheme; «рубцы» → a scar line crossed
HA = top(Drop(["ТОЛЬКО БЕЗ ПОДТЯЖКИ,", "*ПОЛУЧИТСЯ?*"], 104, LX, 0, maxw=900), 270)
def s_a(t):
    tsw = Rs("A", 9.80)                                                   # «птоз» → the ДО photo
    if t < tsw: fr = clip("IMG_2742", 1.3 + t, lerp(1.12, 1.0, eo(t / 2.5)), 0.62, 0.42)
    else: fr = FULL["do_front"].copy()
    fr = dark(fr); HA.draw(fr, t, 0.05)
    tp = Rs("A", 10.48)                                                   # «птоз первой степени»
    if t > tp:
        s_ = 0.75 * pop_s(t, tp); navy_card(fr, 820, 1000, 150, 150, s_)
        if s_ > 0.6: breast_icon(fr, 800, 990, 0.9, eo((t - tp - 0.1) / 0.6)); text_c(fr, "ПТОЗ I", 820, 1110, bebas(46), PINK) if t > tp + 0.5 else None
    tr = Rs("A", 15.86)                                                   # «рубцы» → a T-scar crossed out
    if t > tr:
        u = t - tr; pillx(fr, "без подтяжки и рубцов", LX, 1180, eo(u / 0.3), 0.0, 32)
    return fr

# B — the question over the three ДО angles; «3+ размер» counts up
HB = top(Drop(["А МОЖНО ЛИ", "*ОБОЙТИСЬ БЕЗ ПОДТЯЖКИ?*"], 96, LX, 0, maxw=900), 270)
def s_b(t):
    lt = t - T_S["B"]; d = (vend("B") - T_S["B"]) / 3; k = min(2, int(lt / d)); n = ("do_front", "do_34", "do_side")[k]
    fr = FULL[n].copy(); f = 1 - eo((lt - k * d) / 0.15) if k else 0; fr = fr * (1 - f * 0.7) + f * 0.7
    fr = dark(fr); HB.draw(fr, t, T_S["B"] + 0.05); chip(fr, "ДО", LX, under(HB, 22))
    tz = Rs("B", 20.76)                                                   # «3+ размер»
    if t > tz:
        s_ = 0.75 * pop_s(t, tz); navy_card(fr, 820, 1020, 140, 110, s_)
        if s_ > 0.6: text_c(fr, f"{min(3, 1 + int((t - tz) / 0.25))}+", 820, 1032, bebas(int(150 * min(1, s_ / 0.75))), PINK)
    if t > Rs("B", 18.04): pillx(fr, "своя железа — небольшая", LX, 1180, eo((t - Rs("B", 18.04)) / 0.3), 0.0, 30)
    return fr

# C — the solution: the marking video; «заполнить пространство и вытянуть железу» → the implant fills the scheme, the breast lifts
HC = top(Drop(["ВЫБРАЛИ АНАТОМИЧЕСКИЕ", "*ИМПЛАНТЫ ВЫСОКОГО ПРОФИЛЯ*"], 82, LX, 0, maxw=920), 270)
TC = Track(["при небольшом опущении такой вариант возможен не всегда"], 26, LX, 0, 400)
def s_c(t):
    lt = t - T_S["C"]
    fr = clip("IMG_2743", 0.3 + lt * 0.75, lerp(1.0, 1.12, eio(lt / 6.0)), 0.45, 0.4) if lt < 6.0 else clip("IMG_2742", 30.0 + lt, 1.05)
    fr = dark(fr); HC.draw(fr, t, T_S["C"] + 0.05); TC.y = under(HC, 18) + TC.top; TC.draw(fr, t, T_S["C"] + 0.8)
    tf = Rs("C", 30.72)                                                   # «заполнить» → the pocket fills, «вытянуть» → the sag goes
    s_ = 0.75 * pop_s(t, tf - 0.3); navy_card(fr, 820, 1050, 150, 150, s_)
    if s_ > 0.6: breast_icon(fr, 800, 1040, 0.9, 1 - eio((t - Rs("C", 33.08)) / 0.8), eo((t - tf) / 0.8))
    return fr

# D — the operating room, then the surgeon: «395, высокий профиль»
HD = top(Drop(["*395 МЛ.*", "БЕЗ ПОДТЯЖКИ"], 120, LX, 0, maxw=900), 270)
def s_d(t):
    lt = t - T_S["D"]
    fr = dark(on_white(1.5 + lt, lerp(1.0, 1.08, eo(lt / 5.0)))) if lt < OR_D else dark(clip("VS", Rs_inv_D(t), lerp(1.0, 1.08, eo((lt - OR_D) / 3.0)), 0.5, 0.42))
    if t > Rs("D", 9.28): HD.draw(fr, t, Rs("D", 9.28))
    for k, (lab, w0) in enumerate((("анатомические", 8.40), ("высокий профиль", 10.62))):
        if t > Rs("D", w0): pillx(fr, lab, LX, 1120 + k * 80, eo((t - Rs("D", w0)) / 0.3), 0.0, 32)
    return fr
OR_D = 0.0                                                                # no operating-table shot (the client's call): the surgeon only
def Rs_inv_D(t): _, _, parts, t0, d = next(x for x in SEG if x[0] == "D"); return parts[0][0] + (t - t0)

# E — the result, angle by angle (ДО → ПОСЛЕ with a slider), under the nuance about a future lift
HE = top(Drop(["НО В БУДУЩЕМ", "*ПОДТЯЖКА МОЖЕТ ПОНАДОБИТЬСЯ*"], 84, LX, 0, maxw=920), 270)
E_ANG = ("front", "34", "side")
def s_e(t):
    lt = t - T_S["E"]; d = (vend("E") + 0.3 - T_S["E"]) / 3; k = min(2, int(lt / d)); u = lt - k * d; a = E_ANG[k]
    p = eio((u - 0.5) / 1.2); xs = int(lerp(-6, W + 6, p)); fr = FULL["po_" + a].copy()
    if xs < W: fr[:, max(0, xs):] = FULL["do_" + a][:, max(0, xs):]
    if 0 < p < 1:
        fr[:, max(0, xs - 3):min(W, xs + 3)] = PINK; hx = min(max(xs, 112), 978); dot(fr, hx, 820, 32, PINK); dot(fr, hx, 820, 26, WHITE)
    fr = dark(fr)                                                         # no flash between the angles (the client's call)
    if xs < 820: chip(fr, "ДО", min(max(xs + 20, LX), 860), 1180, 1 - cl((xs - 700) / 120))
    if xs > 300: chip(fr, "ПОСЛЕ", LX, 1180, cl((xs - 300) / 120))
    if t > Rs("E", 24.48): HE.draw(fr, t, Rs("E", 24.48))
    return fr

# Q — the question card
HQ = Drop(["А В ВАШЕМ СЛУЧАЕ", "*МОЖНО ОБОЙТИСЬ*", "*БЕЗ ПОДТЯЖКИ?*"], 96, LX, 720, maxw=920)
def s_q(t):
    lt = t - T_Q; fr = gbg(t); blend(fr, logo_alpha(320), 540 - 160, 470, PINK, eo((lt - 0.1) / 0.5)); HQ.draw(fr, t, T_Q + 0.2); return fr

# K — collages: ДО on top, ПОСЛЕ below, one per angle
COLLS = [np.vstack([panel("do_" + a, W, 960, 40, 30), panel("po_" + a, W, 960, 40, 30)]) for a in E_ANG]
def s_k(t):
    lt = t - T_K; k = min(2, int(lt / K_D)); u = lt - k * K_D
    fr = COLLS[k].copy(); fr[958:962] = PINK; chip(fr, "ДО", LX, 260); chip(fr, "ПОСЛЕ", LX, 1000)
    f = 1 - eo(u / 0.18); return fr * (1 - f) + f * 0.9

SCENES = [(0, s_a), (T_S["B"], s_b), (T_S["C"], s_c), (T_S["D"], s_d), (T_S["E"], s_e), (T_Q, s_q), (T_K, s_k)]
TRANS = {T_S["B"]: ("zoom", 0.35), T_S["C"]: ("diag", 0.45), T_S["D"]: ("zoom", 0.35), T_S["E"]: ("iris", 0.5), T_Q: ("iris", 0.5)}
def base(t):
    k = max(i for i, (s0, _) in enumerate(SCENES) if t >= s0); s0, fn = SCENES[k]
    kind, d = TRANS.get(s0, ("none", 0))
    if kind != "none" and t < s0 + d:
        p = (t - s0) / d; old = np.asarray(SCENES[k - 1][1](t), np.float32); new = np.asarray(fn(t), np.float32)
        if kind == "iris": return iris(old, new, p, 540, 900)
        if kind == "zoom": return zoom_through(old, new, p)
        if kind == "diag": return diag_wipe(old, new, p)
    return fn(t)
LOGO_W = logo_alpha(230)
def render(t):
    fr = np.asarray(base(t), np.float32).copy()
    if t < T_Q:
        subtitles(fr, t)
        white = T_S["D"] <= t < T_S["D"] + OR_D                            # on the white page the logo goes navy
        blend(fr, LOGO_W, 470 - 115, 1535, NAVY if white else WHITE, 0.85)
    return fr

MUSIC = os.path.join(SRC, "music.wav"); M0 = 0.5
G_VO, G_GAP = 0.022, 0.075
def track(n):
    if not os.path.exists(MUSIC): return np.zeros((n, 2))
    w = wave.open(MUSIC); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64).reshape(-1, w.getnchannels()) / 32768
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    seg = x[int(M0 * SR):int(M0 * SR) + n]; seg = seg / np.sqrt(np.mean(seg.mean(1) ** 2))
    return np.pad(seg, ((0, n - len(seg)), (0, 0)))
def build_audio(path):
    n = int(DUR * SR); vo = np.zeros(n); act = np.zeros(n); srcs = {}
    for b, s_, parts, t0, d in SEG:
        if s_ not in srcs: w = wave.open(os.path.join(WORK, f"{s_}.wav")); srcs[s_] = hp(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768, 90)
        x = srcs[s_]; seg = x[int(parts[0][0] * SR):int(parts[0][1] * SR)].copy(); fi, fo = int(0.03 * SR), int(0.1 * SR)
        seg[:fi] *= np.sin(np.linspace(0, np.pi / 2, fi)) ** 2; seg[-fo:] *= np.cos(np.linspace(0, np.pi / 2, fo)) ** 2
        fr_rms = np.sqrt(np.mean(seg[:len(seg) // 960 * 960].reshape(-1, 960) ** 2, 1)); g = 1 / (np.sqrt(np.mean(np.sort(fr_rms)[-max(1, len(fr_rms) * 2 // 5):] ** 2)) + 1e-6)
        i = int(t0 * SR); vo[i:i + len(seg)] += (seg * g)[:n - i]; act[max(0, i - int(0.25 * SR)):i + int(d * SR) + int(0.1 * SR)] = 1
    vo = np.tanh(vo * 0.16 * 1.6) / 1.6
    dd = np.convolve(act, np.ones(int(0.25 * SR)) / int(0.25 * SR), "same")
    sfx = np.zeros(n + 2 * SR)
    def put(x_, t, g_): i = int(max(0, t) * SR); s_ = sfx[i:i + len(x_)]; s_ += x_[:len(s_)] * g_
    def load(fn):
        w_ = wave.open(fn); x_ = np.frombuffer(w_.readframes(w_.getnframes()), np.int16).astype(np.float64) / 32768
        if w_.getnchannels() == 2: x_ = x_.reshape(-1, 2).mean(1)
        if w_.getframerate() != SR: x_ = np.interp(np.arange(int(len(x_) * SR / w_.getframerate())) * w_.getframerate() / SR, np.arange(len(x_)), x_)
        return x_ / (np.abs(x_).max() + 1e-9)
    kd = os.path.join(SRC, "keyboard.wav")
    KEYS = [load(kd)] if os.path.exists(kd) else [load(os.path.join(SRC, "keys", f"k{i:03d}.wav")) for i in range(1, 9)]
    TICK = load(os.path.join(SRC, "tick.wav")); sd = os.path.join(SRC, "shutter_client.wav"); SHOT = load(sd) if os.path.exists(sd) else None
    def typing(hd, t0, g_=0.4):
        n_ch = sum(len(q) for q, _ in hd.rows)
        if len(KEYS) == 1: kl = int((n_ch * hd.stag + 0.35) * SR); kb = KEYS[0][:kl].copy(); f_ = int(0.2 * SR); kb[-f_:] *= np.linspace(1, 0, f_); put(kb, t0, g_)
        else:
            for j in range(n_ch): put(KEYS[j % len(KEYS)], t0 + j * hd.stag, g_ * 0.8)
    typing(HA, 0.05, 0.45); put(TICK, Rs("A", 10.48), 0.6); put(TICK, Rs("A", 15.86), 0.55)
    typing(HB, T_S["B"] + 0.05, 0.3)
    for j in range(3): put(tick(), Rs("B", 20.76) + 0.25 * j, 0.3)
    put(TICK, Rs("B", 18.04), 0.55)
    typing(HC, T_S["C"] + 0.05, 0.3); put(swoosh(0.6), Rs("C", 30.72), 0.15); put(glass(1174.7), Rs("C", 33.08), 0.1)
    put(TICK, Rs("D", 8.40), 0.55); put(TICK, Rs("D", 10.62), 0.55); typing(HD, Rs("D", 9.28), 0.35)
    dE = (vend("E") + 0.3 - T_S["E"]) / 3
    for k in range(3): put(swoosh(1.0), T_S["E"] + k * dE + 0.5, 0.15)
    typing(HE, Rs("E", 24.48), 0.25); typing(HQ, T_Q + 0.2, 0.35); put(glass(1318.5), T_Q + 0.2, 0.1)
    for k in range(3):
        pass                                                              # no sound on the collage cuts (the client's call)
    sfx[:n] *= 1 - 0.55 * np.clip(dd, 0, 1)
    tt = np.arange(n) / SR; fade = np.clip((DUR - tt) / 2.5, 0, 1) ** 1.5 * np.clip(tt / 1.2, 0, 1)
    mus = track(n) * ((G_GAP - (G_GAP - G_VO) * np.clip(dd, 0, 1)) * fade)[:, None]
    mix = vo[:, None] + mus + (sfx[:n] * np.clip((DUR - tt) / 1.0, 0, 1))[:, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (GAIN / 20); mix /= max(1, np.abs(mix).max() / 0.95)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
GAIN = 1.5 if os.path.exists(MUSIC) else 1.0
def pop(f0=1100, f1=380, dur=0.09):
    n = int(dur * SR); t = np.arange(n) / SR; ph = 2 * np.pi * np.cumsum(f1 + (f0 - f1) * np.exp(-t / 0.015)) / SR
    return np.sin(ph) * env(n, 0.002, 0.03)

def make_cover(path):
    """ДО | ПОСЛЕ fronts side by side, the patient's fear as the title"""
    CBG = np.clip(np.float32([0, 8, 28]) / 255 * (1 + 0.3 * np.exp(-(((xx - 540) / 900) ** 2 + ((yy - 520) / 900) ** 2)))[..., None], 0, 1).astype(np.float32)
    fr = CBG.copy()
    for k, n in enumerate(("do_front", "po_front")): fr[300:1140, k * 540:(k + 1) * 540] = panel(n, 540, 840, 30, 20)
    fr[300:1140, 539:541] = DEEP; chip(fr, "ДО", 36, 322); chip(fr, "ПОСЛЕ", 576, 322)
    g = np.clip((yy[:, :1] - 900) / 240, 0, 1)[..., None]; fr = fr * (1 - g) + CBG * g
    blend(fr, logo_alpha(300), 540 - 150, 250, PINK)
    hd = Drop(["ХОЧУ УВЕЛИЧИТЬ ГРУДЬ.", "*НО НЕ ХОЧУ ПОДТЯЖКУ*"], 100, 70, 1140, maxw=940); hd.draw(fr, 99, 0)
    ty = Track(["как мы решили вопрос при птозе 1 степени"], 32, 72, 0, 400); ty.y = 1140 + hd.h + 44 + ty.top; ty.draw(fr, 99, 0)
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
