#!/usr/bin/env python3
"""Reels «Уменьшение груди на 7 размеров» — редукционная маммопластика, Анна Тваури.

Same brand system as tools/reel_postpartum.py (navy #00123A, powder pink, Bebas Neue + Montserrat, notched frame,
logo outro), but a different edit: zoom-out reveals, flash freeze-frames, an iris reveal into ПОСЛЕ, a diagonal
wipe, a picture-in-picture interview card, split-screen cards; letter-drop / marker / slot-roll / tracking text;
line-drawn confirmations (25-year ruler, heartbeat -> check, stars, progress ring, hearts, sparkles, hanger).

usage:
  REEL_SRC=<folder with IMG_*.MOV/JPG, logo.png, music.wav> python3 tools/reel_reduction.py [--preview t1,t2 | --cover | --audio-only]
Sources are first normalised into $REEL_SRC/_work (1080x1920, 30 fps; voice tracks 48 kHz mono wav):
  ffmpeg -i IMG_x.MOV -vf "scale=1080:1920:flags=lanczos,fps=30,format=yuv420p" -an -c:v libx264 -crf 14 _work/IMG_x.mp4
  ffmpeg -i IMG_3434.MOV -vn -ac 1 -ar 48000 _work/IMG_3434.wav   (same for IMG_3424)
"""
import os, sys, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_reduction"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_reduction_25_years"
W, H, FPS, SR = 1080, 1920, 30, 48000
NAVY = np.float32([0, 18, 58]) / 255; PINK = np.float32([233, 197, 193]) / 255; WHITE = np.float32([1, 1, 1])
DEEP = np.float32([0, 8, 30]) / 255

def cl(x): return min(1.0, max(0.0, x))
def eo(x): x = cl(x); return 1 - (1 - x) ** 3
def eio(x): x = cl(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x): x = cl(x); c = 1.7; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p

# ---------------- voice edit: (block, source, src_in, src_out) ----------------
VO = [("P1", "3434", 11.30, 14.55),                                   # я очень мечтала, я к этому шла 25 лет
      ("P2", "3434", 15.58, 18.60),                                   # но очень боялась, но всё равно решилась потом
      ("R", "3434", 5.88, 7.00), ("R", "3434", 7.40, 8.45), ("R", "3434", 9.95, 11.30),   # как вам результат? — отлично. очень даже, слов нет
      ("L", "3434", 37.00, 39.10),                                    # мне нравится в своей груди сейчас
      ("S", "3424", 9.90, 11.95), ("S", "3424", 12.70, 14.10),       # вы прямо изменили стиль. — а мне самой хочется
      ("S", "3424", 21.05, 23.35), ("S", "3424", 24.08, 26.85),      # казалось бы, небольшая деталь, а как многое изменило в самоощущении
      ("D", "3434", 56.88, 59.50)]                                    # мне хочется просто жить красиво и одеваться
GAP_IN = {("R", 1): 0.15, ("R", 2): 0.1, ("S", 1): 0.1, ("S", 2): 0.25, ("S", 3): 0.1}
SEG = []; _t = 0.3; _cnt = {}
for _k, (_b, _s, _a, _o) in enumerate(VO):
    if _k: _t += GAP_IN.get((_b, _cnt.get(_b, 0)), 0.3) if _b == VO[_k - 1][0] else 0.3
    SEG.append((_b, _s, _a, _o, _t)); _cnt[_b] = _cnt.get(_b, 0) + 1; _t += _o - _a
VO_END = _t
def vstart(b, i=0): return [s[4] for s in SEG if s[0] == b][i]
def vend(b): return max(s[4] + s[3] - s[2] for s in SEG if s[0] == b)
T_P2 = vstart("P2") - 0.15; T_R = vstart("R") - 0.2; T_L = vstart("L") - 0.15; T_S = vstart("S") - 0.15
T_FRZ = vstart("S", 2) - 0.05; FRZ = 1.0                               # freeze frame on «казалось бы, небольшая деталь»
T_D = vstart("D") - 0.15; T_F = vend("D") + 0.25; T_CTA = T_F + 3.3; DUR = T_CTA + 3.4
def src_of(t, name):
    best = None
    for _, s, si, so, t0 in SEG:
        if s != name: continue
        if t0 <= t <= t0 + so - si: return si + t - t0
        if t < t0: return si - (t0 - t) if best is None else min(best[0] + t - best[1], si)
        best = (so, t0 + so - si)
    return best[0] + t - best[1]

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
V = {k: Src(os.path.join(WORK, f"IMG_{k}.mp4")) for k in ["3406", "3414", "3423", "3424", "3434"]}
FACE = {"3414": 300, "3423": 230, "3424": 340}                        # the patient stays anonymous: blur everything above this line
def anon(img, key):
    y = FACE.get(key)
    if not y: return img
    top = img[:y + 80].astype(np.float32); bl = cv2.GaussianBlur(cv2.resize(top, None, fx=0.12, fy=0.12), (0, 0), 3)
    bl = cv2.resize(bl, (W, y + 80)); m = np.clip((y + 40 - np.arange(y + 80)) / 80, 0, 1)[:, None, None]
    out = img.copy(); out[:y + 80] = (top * (1 - m) + bl * m).astype(np.uint8); return out
def vid(key, t): return anon(V[key].at(t), key)
def photo(n):
    im = np.asarray(ImageOps.exif_transpose(Image.open(os.path.join(SRC, f"IMG_{n}.JPG"))).convert("RGB").resize((W, H), Image.LANCZOS)).copy()
    im[1500:] = 0                                                       # app watermark lives in the empty black bottom
    return im
PB = {k: photo(n) for k, n in [("f", 8460), ("q", 8461), ("s", 8462)]}
PA = {k: photo(n) for k, n in [("f", 8463), ("q", 8464), ("s", 8465)]}

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

# ---------------- scenes ----------------
# P1 — mirror, ДО. zoom-out reveal from the reflection; letters drop; «7» counts up over a 7-step ruler
H1 = Drop(["УМЕНЬШЕНИЕ ГРУДИ"], 132, 70, 1040)
T1 = Track(["редукционная маммопластика"], 36, 72, 0)
NY = 1215                                                              # cap line of the big «7»
def ruler(fr, t, t0, x0=70, y0=1520, w=900, n=7):
    p = (t - t0) / 1.2
    for i in range(n):
        pi = eo((p * n - i) / 1.5)
        if pi <= 0: break
        x = x0 + w * i / (n - 1); hh = 34 * pi
        bar = np.ones((max(1, int(hh)), 4), np.float32); blend(fr, bar, x - 2, y0 - hh, PINK if i == n - 1 else WHITE, 0.9)
    path_draw(fr, [(x0, y0 + 4), (x0 + w, y0 + 4)], eo((t - t0) / 1.2), WHITE, 2, op=0.5)
def s_p1(t):
    z = lerp(1.55, 1.04, eio(t / 3.2)); fr = full(V["3406"].at(0.6 + t * 0.85), 0.7, 0.32, z); shade(fr, VIG); shade(fr, GBOT)
    H1.draw(fr, t, 0.0)
    if t > 0.45:
        na = alpha_img("НА", bebas(150)); y_small = NY + 330 * 0.72 - 150 * 0.72 - cap_top(bebas(150))
        blend(fr, na, 66, y_small, WHITE, eo((t - 0.45) / 0.3)); x7 = 66 + na.shape[1] + 6
        n = int(round(lerp(1, 7, eio((t - 0.55) / 1.2))))
        a = alpha_img(str(n), bebas(330)); blend(fr, a, x7, NY - cap_top(bebas(330)), PINK, eo((t - 0.55) / 0.3))
        b = alpha_img("РАЗМЕРОВ", bebas(150)); blend(fr, b, x7 + alpha_img("7", bebas(330)).shape[1] + 14, y_small, WHITE, eo((t - 1.5) / 0.4))
    ruler(fr, t, 0.55, y0=NY + 330 * 0.72 + 45)
    T1.y = NY + 330 * 0.72 + 125; T1.draw(fr, t, 1.3)
    return fr

# P2 — full-screen ДО/ПОСЛЕ slider (front photos); heartbeat line turns into a check mark on «решилась»
H2 = Drop(["25 ЛЕТ ХОТЕЛА", "*И ВСЁ РАВНО БОЯЛАСЬ*"], 112, 70, 1270)
T_DECIDE = vstart("P2") + (17.46 - 15.58)                              # «…решилась»
def ecg(fr, t, t0, y=1600):
    xs = np.linspace(70, 1010, 120); base = []
    for x in xs:
        u = (x - 70) / 940; d = 0
        for c in (0.22, 0.5, 0.78):
            v = (u - c) * 60
            d += -70 * np.exp(-((v - 0.4) ** 2) * 3) + 40 * np.exp(-((v + 0.6) ** 2) * 3) + 18 * np.exp(-((v - 2) ** 2) * 0.6)
        base.append((x, y + d))
    p = eo((t - t0) / 1.4); fade = 1 - eio((t - T_DECIDE) / 0.3) if t > T_DECIDE else 1
    path_draw(fr, base, p, WHITE, 3, op=fade)
    if t > T_DECIDE:
        c = (540, y); pc = eio((t - T_DECIDE) / 0.35)
        path_draw(fr, [(c[0] + 48 * np.cos(a), c[1] + 48 * np.sin(a)) for a in np.linspace(-np.pi / 2, 1.5 * np.pi, 50)], pc, PINK, 4)
        path_draw(fr, [(c[0] - 22, c[1] + 2), (c[0] - 6, c[1] + 18), (c[0] + 24, c[1] - 16)], eo((t - T_DECIDE - 0.25) / 0.3), PINK, 6)
def slider_x(lt): return W * (1 - 0.5 * eio((lt - 0.35) / 0.8) - 0.5 * eio((lt - 1.9) / 0.8))   # ДО -> half/half -> ПОСЛЕ
def s_p2(t):
    lt = t - T_P2; z = 1.0 + 0.03 * eio(lt / 3.2)
    b = full(PB["f"], 0.5, 0.38, z); a = full(PA["f"], 0.5, 0.38, z); xd = int(slider_x(lt))
    fr = b.copy(); fr[:, xd:] = a[:, xd:]; shade(fr, GBOT)
    if 0 < xd < W:
        fr[:, max(0, xd - 2):xd + 2] = PINK
        m = np.zeros((H, W), np.uint8); cv2.circle(m, (xd, 760), 32, 255, -1, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, PINK)
        ar = alpha_img("‹ ›", mont(36, 600)); blend(fr, ar, xd - ar.shape[1] / 2, 760 - ar.shape[0] / 2 - 2, NAVY)
    chip(fr, "ДО", 40, 250, eo((lt - 0.15) / 0.3) * cl((xd - 160) / 120))
    chip(fr, "ПОСЛЕ", 875, 250, cl((W - 200 - xd) / 120))
    H2.draw(fr, t, T_P2 + 0.35); ecg(fr, t, T_P2 + 0.8)
    return fr

# R — iris reveal into ПОСЛЕ (music drop); interview PiP; progress ring «1,5 мес»; five stars on «отлично»
H3 = Drop(["ПОСЛЕ ОПЕРАЦИИ"], 92, 300, 400, maxw=700)
T_OK = vstart("R", 1)
PIP = (620, 1180, 960, 1640)
def ring(fr, t, t0, cx=170, cy=430, r=100):
    p = eio((t - t0) / 1.1)
    path_draw(fr, [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 90)], 1, WHITE, 2, op=0.25 * min(1, (t - t0) / 0.3) if t > t0 else 0)
    path_draw(fr, [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(-np.pi / 2, 1.5 * np.pi, 90)], p * 0.75, PINK, 7)
    if t > t0 + 0.2:
        a = alpha_img("1,5", bebas(96)); blend(fr, a, cx - a.shape[1] / 2 + 2, cy - 52 - cap_top(bebas(96)) + 6, WHITE, eo((t - t0 - 0.2) / 0.3))
        b = alpha_img("МЕСЯЦА", mont(24, 600)); blend(fr, b, cx - b.shape[1] / 2, cy + 28, WHITE, eo((t - t0 - 0.3) / 0.3))
def stars(fr, t, t0, x0=90, y=1560, gap=96):
    for i in range(5):
        lt = t - t0 - i * 0.12
        if lt < 0: break
        s = eback(lt / 0.35); pts = star_pts(x0 + i * gap, y, 38 * s)
        fill_poly(fr, pts, PINK, min(1, lt / 0.2)); path_draw(fr, pts, 1, WHITE, 2, True, 0.6)
def pip(fr, t, t0):
    p = eback((t - t0) / 0.45)
    if p <= 0: return
    x0, y0, x1, y1 = PIP; w, h = x1 - x0, y1 - y0; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ww, hh = max(2, int(w * p)), max(2, int(h * p))
    im, _ = cover(V["3434"].at(src_of(t, "3434")), ww, hh, 0.5, 0.42)
    m = np.zeros((hh, ww), np.uint8); cv2.rectangle(m, (0, 0), (ww - 1, hh - 1), 255, -1)
    m = cv2.GaussianBlur(m, (0, 0), 1).astype(np.float32)[..., None] / 255
    xa, ya = int(cx - ww / 2), int(cy - hh / 2); reg = fr[ya:ya + hh, xa:xa + ww]; reg[:] = reg * (1 - m) + im * m
    path_draw(fr, [(xa - 8, ya - 8), (xa + ww + 8, ya - 8), (xa + ww + 8, ya + hh + 8), (xa - 8, ya + hh + 8)], 1, PINK, 3, True)
    lab = alpha_img("пациентка", mont(28, 500)); blend(fr, lab, xa, ya + hh + 18, WHITE, eo((t - t0 - 0.3) / 0.3))
def s_r(t):
    lt = t - T_R; fr = full(vid("3423", 1.0 + lt * 0.9), 0.5, 0.42, 1.0 + 0.05 * eio(lt / 3.6)); shade(fr, GTOP); shade(fr, GBOT * 0.85)
    ring(fr, t, T_R + 0.45); H3.draw(fr, t, T_R + 0.6)
    pip(fr, t, T_R + 0.25); stars(fr, t, T_OK - 0.05)
    a = alpha_img("ОТЛИЧНО", bebas(96))
    if t > T_OK + 0.5: blend(fr, a, 86, 1400 - cap_top(bebas(96)), WHITE, eo((t - T_OK - 0.5) / 0.3))
    return fr
def iris(old, new, p, cx=540, cy=760):
    r = eio(p) * 1500; d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2); m = np.clip((r - d) / 3 + 0.5, 0, 1)[..., None]
    fr = old * (1 - m) + new * m; ring_ = np.clip(1 - np.abs(d - r) / 4, 0, 1)[..., None] * (0 < p < 1)
    return (fr * (1 - ring_) + PINK * ring_).astype(np.float32)

# L — the most beautiful ПОСЛЕ (3/4 photo), push-in; three lines, each with a heart drawn
LL = [Marker("НРАВИТСЯ", "ФОРМА", 104, 150, 1250), Marker("НРАВИТСЯ", "ОТРАЖЕНИЕ", 104, 150, 1380), Marker("НРАВИТСЯ", "СЕБЕ", 104, 150, 1510)]
def s_l(t):
    lt = t - T_L; fr = full(PA["q"], 0.5, 0.36, lerp(1.0, 1.12, eio(lt / 3.0))); shade(fr, GBOT * 1.05)
    for k, m in enumerate(LL):
        t0 = T_L + 0.3 + 0.55 * k; m.draw(fr, t, t0)
        path_draw(fr, heart_pts(96, m.y + 34, 25), eo((t - t0) / 0.5), PINK, 3, True)
    return fr
def zoom_through(old, new, p):
    e = eio(p); o = cv2.resize(old, None, fx=1 + 0.35 * e, fy=1 + 0.35 * e); oy, ox = (o.shape[0] - H) // 2, (o.shape[1] - W) // 2
    o = o[oy:oy + H, ox:ox + W]; s = 0.88 + 0.12 * e; n = cv2.resize(new, None, fx=s, fy=s); nn = np.zeros_like(new) + DEEP
    py, px = (H - n.shape[0]) // 2, (W - n.shape[1]) // 2; nn[py:py + n.shape[0], px:px + n.shape[1]] = n
    return (o * (1 - e) + nn * e).astype(np.float32)

# S — 3424 synced; sparkles on «изменили стиль»; freeze-frame + flash on «небольшая деталь»; ГРУДЬ -> САМООЩУЩЕНИЕ
HS = Drop(["ИЗМЕНИЛАСЬ НЕ ТОЛЬКО ГРУДЬ"], 100, 70, 1250)
RS = Roll("", "ИЗМЕНИЛАСЬ НЕ ТОЛЬКО ГРУДЬ", "ИЗМЕНИЛОСЬ САМООЩУЩЕНИЕ", 100, 70, 1250)
T_SELF = vstart("S", 3) + (25.70 - 24.08)                               # «…самоощущении»
TS = Track(["«вы прямо изменили стиль»"], 36, 72, 1400, 300)
TS2 = Track(["— а мне самой хочется"], 36, 72, 1400, 500, PINK)            # quiet line in the source: subtitled
def s_frame(t):
    return vid("3424", src_of(T_FRZ if T_FRZ <= t < T_FRZ + FRZ else t, "3424"))
def s_s(t):
    lt = t - T_S; z = 1.04
    if T_FRZ <= t < T_FRZ + FRZ: z = 1.04 + 0.1 * eback((t - T_FRZ) / 0.25) * 0.6     # punch-in on the frozen frame
    fr = full(s_frame(t), 0.5, 0.4, z); shade(fr, VIG); shade(fr, GBOT)
    if T_FRZ <= t < T_FRZ + FRZ:
        fl = 1 - eo((t - T_FRZ) / 0.3); fr = fr * (1 - fl) + fl
        a = alpha_img("НЕБОЛЬШАЯ ДЕТАЛЬ", bebas(70)); blend(fr, a, 540 - a.shape[1] / 2, 960, PINK, eo((t - T_FRZ - 0.35) / 0.3))
    if t < T_FRZ: HS.draw(fr, t, T_S + 0.3, T_FRZ - 0.25)
    else: RS.draw(fr, t, T_FRZ + FRZ, T_SELF - 0.5)                    # the line comes back after the freeze, then rolls over
    TS.draw(fr, t, vstart("S") + 0.2, vstart("S", 1) - 0.2); TS2.draw(fr, t, vstart("S", 1) + 0.1, T_FRZ - 0.3)
    for k, (x, y, r) in enumerate([(820, 520, 34), (250, 760, 24), (900, 900, 20), (180, 420, 18), (700, 300, 26)]):
        sparkle(fr, x, y, r, t, vstart("S") + 0.9 + 0.12 * k)
    return fr
def diag_wipe(old, new, p):
    e = eio(p); d = (xx * 0.45 + yy) - lerp(-500, H + 600, e); m = np.clip(-d / 3 + 0.5, 0, 1)[..., None]
    ln = np.clip(1 - np.abs(d) / 4, 0, 1)[..., None] * (0 < p < 1)
    fr = old * (1 - m) + new * m; return (fr * (1 - ln) + PINK * ln).astype(np.float32)

# D — «наряжаться»: zoom-out from the outfit, hanger drawn, sparkles
HD = Drop(["КОГДА СНОВА ХОЧЕТСЯ", "*НАРЯЖАТЬСЯ*"], 120, 70, 1260)
def hanger(fr, t, t0, cx=880, cy=1120, s=1.0):
    p = eo((t - t0) / 0.9)
    hook = [(cx + 22 * s * np.cos(a), cy - 70 * s + 22 * s * np.sin(a)) for a in np.linspace(np.pi, 2.6 * np.pi, 20)]
    body = [(cx, cy - 48 * s), (cx - 110 * s, cy + 30 * s), (cx + 110 * s, cy + 30 * s), (cx, cy - 48 * s)]
    path_draw(fr, hook + body, p, PINK, 4)
def s_d(t):
    lt = t - T_D; fr = full(vid("3424", 18.6 + lt * 0.8), 0.5, 0.55, lerp(1.6, 1.05, eio(lt / 1.6))); shade(fr, GBOT); shade(fr, VIG)
    HD.draw(fr, t, T_D + 0.3); hanger(fr, t, T_D + 0.9)
    for k, (x, y, r) in enumerate([(960, 1020, 22), (790, 1190, 16), (1000, 1210, 14)]): sparkle(fr, x, y, r, t, T_D + 1.4 + 0.15 * k, WHITE)
    return fr

# F — final ДО | ПОСЛЕ split, 25 ЛЕТ ДУМАЛА -> 1 РЕШЕНИЕ
HF1 = Drop(["25 ЛЕТ ДУМАЛА"], 120, 70, 1240); HF2 = Drop(["*1 РЕШЕНИЕ* ИЗМЕНИЛО", "ОЧЕНЬ МНОГОЕ"], 120, 70, 1240 + 120 * 0.93 + 10)
TF = Track(["редукционная маммопластика"], 34, 72, 0)
def s_f(t):
    lt = t - T_F; fr = BG.copy()
    for k, (img, lab) in enumerate([(PB["f"], "ДО"), (PA["f"], "ПОСЛЕ")]):
        p = eio((lt - 0.05 - 0.1 * k) / 0.55); dy = (1 - p) * (-H if k == 0 else H)
        sc = 0.64 + 0.008 * lt                                          # whole body fits the half (photo body: x 150-930, y 300-1440)
        M = np.float32([[sc, 0, 270 - 540 * sc], [0, sc, 330 - 300 * sc]])
        im = cv2.warpAffine(img, M, (540, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)).astype(np.float32) / 255; y0 = int(dy)
        ya, yb = max(0, y0), min(H, y0 + H)
        if yb > ya: fr[ya:yb, k * 540:(k + 1) * 540] = im[ya - y0:yb - y0]
        chip(fr, lab, k * 540 + 24, 240, eo((lt - 0.6 - 0.1 * k) / 0.3))
    shade(fr, GBOT); fr[:, 538:542] = PINK
    HF1.draw(fr, t, T_F + 0.5); HF2.draw(fr, t, T_F + 1.3)
    TF.y = HF2.y + HF2.h + 75; TF.draw(fr, t, T_F + 2.0)
    return fr

# CTA — navy card, line by line, underline arrow, logo
HC = Drop(["ЕСЛИ ДАВНО ДУМАЕТЕ", "ОБ УМЕНЬШЕНИИ ГРУДИ,", "*НАЧНИТЕ С КОНСУЛЬТАЦИИ*"], 104, 70, 520)
def s_cta(t):
    lt = t - T_CTA; fr = BG.copy(); HC.draw(fr, t, T_CTA + 0.2)
    y = HC.y + HC.h + 50; path_draw(fr, [(70, y), (620, y), (600, y - 16), (620, y), (600, y + 16)], eo((t - T_CTA - 1.3) / 0.6), PINK, 4)
    w = 640; lg = logo_alpha(w); h = lg.shape[0]; x0, y0 = 540 - w / 2, 1330 - h / 2; sp = int(w * _split / _lg.shape[1])
    mk, wd = lg[:, :sp].copy(), lg[:, sp:].copy()
    y2, x2 = np.mgrid[0:mk.shape[0], 0:mk.shape[1]]; a2 = (np.arctan2(x2 - mk.shape[1] / 2, -(y2 - mk.shape[0] / 2)) % (2 * np.pi)) / (2 * np.pi)
    mk *= np.clip((eio((lt - 1.0) / 0.9) - a2) * 20, 0, 1); wd[:, int(wd.shape[1] * eio((lt - 1.4) / 0.8)):] = 0
    blend(fr, mk, x0, y0, PINK); blend(fr, wd, x0 + sp, y0, PINK)
    ty = Track(["пластический хирург"], 38, 540, y0 + h + 70, 300, WHITE, True); ty.y = y0 + h + 70 + ty.top; ty.draw(fr, t, T_CTA + 1.9)
    return fr

SCENES = [(0, s_p1), (T_P2, s_p2), (T_R, s_r), (T_L, s_l), (T_S, s_s), (T_D, s_d), (T_F, s_f), (T_CTA, s_cta)]
TRANS = {T_P2: ("flash", 0.3), T_R: ("iris", 0.7), T_L: ("zoom", 0.5), T_S: ("diag", 0.55), T_D: ("iris", 0.5), T_F: ("none", 0), T_CTA: ("zoom", 0.5)}
def render(t):
    k = max(i for i, (s0, _) in enumerate(SCENES) if t >= s0); s0, fn = SCENES[k]
    kind, d = TRANS.get(s0, ("none", 0))
    if kind != "none" and t < s0 + d:
        p = (t - s0) / d
        if kind == "flash":
            fr = fn(t); f = 1 - eo(p); return (fr * (1 - f) + f).astype(np.float32)
        old = SCENES[k - 1][1](t); new = fn(t)
        if kind == "iris": return iris(old, new, p, 540, 700 if s0 == T_R else 1100)
        if kind == "zoom": return zoom_through(old, new, p)
        if kind == "diag": return diag_wipe(old, new, p)
    return fn(t)

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

MUSIC = os.path.join(SRC, "music.wav")
M_DROP = 14.45; M0 = M_DROP - T_R                                     # the track's drop lands on the iris reveal into ПОСЛЕ
G_VO, G_GAP = 0.022, 0.075
def track(n):
    w = wave.open(MUSIC); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64).reshape(-1, w.getnchannels()) / 32768
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    x = x / np.sqrt(np.mean(x[int(M_DROP * SR):].mean(1) ** 2)); x[:int(M_DROP * SR)] *= 4.5
    seg = x[int(M0 * SR):int(M0 * SR) + n]; return np.pad(seg, ((0, n - len(seg)), (0, 0)))

def build_audio(path):
    n = int(DUR * SR); vo = np.zeros(n); act = np.zeros(n); srcs = {}
    for b, s, si, so, t0 in SEG:
        if s not in srcs:
            w = wave.open(os.path.join(WORK, f"IMG_{s}.wav")); srcs[s] = hp(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768, 90)
        seg = srcs[s][int(si * SR):int(so * SR)].copy(); f = int(0.012 * SR); seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
        fr_rms = np.sqrt(np.mean(seg[:len(seg) // 960 * 960].reshape(-1, 960) ** 2, 1))       # 20 ms frames
        seg /= np.sqrt(np.mean(np.sort(fr_rms)[-max(1, len(fr_rms) * 2 // 5):] ** 2)) + 1e-6   # level = the voiced 40 %, quiet lines come up
        i = int(t0 * SR); vo[i:i + len(seg)] += seg[:n - i]
        act[max(0, i - int(0.25 * SR)):i + len(seg) + int(0.1 * SR)] = 1
    vo = np.tanh(vo * 0.16 * 1.6) / 1.6
    d = np.convolve(act, np.ones(int(0.25 * SR)) / int(0.25 * SR), "same")
    tt = np.arange(n) / SR; fade = np.clip((DUR - tt) / 1.8, 0, 1) ** 1.5 * np.minimum(1, tt / 0.15)
    mus = track(n) * ((G_GAP - (G_GAP - G_VO) * np.clip(d, 0, 1)) * fade)[:, None]
    sfx = np.zeros(n + 2 * SR)
    def put(x, t, g): i = int(max(0, t) * SR); seg = sfx[i:i + len(x)]; seg += x[:len(seg)] * g
    WH = whoosh(); WS = whoosh(0.4, 700, 8000)
    put(boom(), 0.0, 0.45)                                               # opening hit
    for k in range(6): put(tick(), 0.55 + 1.2 * eio((k + 1) / 7), 0.16)  # the counter
    put(chime(), 2.1, 0.1)
    put(flash_hit(), T_P2 - 0.02, 0.4)
    for ts in (0.35, 1.9): put(WS, T_P2 + ts, 0.24)                          # slider moves
    put(heartbeat(), T_P2 + 1.0, 0.5); put(heartbeat(), T_P2 + 1.85, 0.4); put(pop(1300, 600), T_DECIDE + 0.25, 0.25)
    put(riser(), T_R - 1.1, 0.12); put(WH, T_R - 0.1, 0.32); put(boom(0.6), T_R - 0.03, 0.22)
    for i in range(5): put(pop(900 + 150 * i, 400), T_OK - 0.05 + i * 0.12, 0.12)
    put(WH, T_L - 0.12, 0.3)
    for k in range(3): put(whoosh(0.3, 1500, 7000), T_L + 0.6 + 0.55 * k, 0.12)
    put(whoosh(0.5, 400, 9000), T_S - 0.1, 0.3)
    for k in range(5): put(sparkle_snd(), vstart("S") + 0.9 + 0.12 * k, 0.12)
    put(shutter(), T_FRZ, 0.35); put(boom(0.5), T_FRZ, 0.2)
    put(whoosh(0.35, 600, 6000), T_SELF - 0.6, 0.18)
    put(WH, T_D - 0.12, 0.3); put(whoosh(0.4, 300, 3000), T_D + 0.9, 0.14)
    for k in range(3): put(sparkle_snd(), T_D + 1.4 + 0.15 * k, 0.1)
    put(WS, T_F, 0.3); put(boom(), T_F + 0.1, 0.25)
    for t0, hd in [(T_F + 0.5, HF1), (T_F + 1.3, HF2)]:
        for k in range(0, hd.n, 3): put(tick(), t0 + k * hd.stag, 0.08)
    put(WH, T_CTA - 0.12, 0.3); put(chime(), T_CTA + 1.05, 0.18); put(chime(1975.5), T_CTA + 1.5, 0.1)
    sfx[:n] *= 1 - 0.8 * np.clip(d, 0, 1)
    mix = vo[:, None] + mus + sfx[:n, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (1.5 / 20); mix /= max(1, np.abs(mix).max() / 0.95)   # ~-16 LUFS
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

def make_cover(path):
    fr = BG.copy(); h2 = 1180
    for k, img in enumerate([PB["f"], PA["f"]]):
        im, _ = cover(img, 540, h2, 0.5, 0.4, 1.35); fr[160:160 + h2, k * 540:(k + 1) * 540] = im
    fr[160:160 + h2, 538:542] = PINK
    m = np.clip((yy[:, :1] - (160 + h2 - 300)) / 300, 0, 1)[..., None]; fr = fr * (1 - m) + BG * m; shade(fr, GBOT)
    chip(fr, "ДО", 40, 200); chip(fr, "ПОСЛЕ", 580, 200)
    hd = Drop(["УМЕНЬШЕНИЕ ГРУДИ", "НА *7 РАЗМЕРОВ*"], 170, 60, 1250, maxw=960); hd.draw(fr, 99, 0)
    ty = Track(["полтора месяца после операции"], 40, 64, 0, 400); ty.y = 1250 + hd.h + 80 + ty.top; ty.draw(fr, 99, 0)
    lg = logo_alpha(380); blend(fr, lg, 540 - 190, 60, PINK)
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
