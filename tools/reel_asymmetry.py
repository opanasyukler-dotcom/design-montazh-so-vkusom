#!/usr/bin/env python3
"""Reels «Почему при асимметрии мы выбрали круглые импланты» — the surgeon in the operating room explains the plan for
breast augmentation with asymmetry correction, Анна Тваури.

One take (C0811) cut sentence by sentence (only the aside to the nurse and fillers removed) over ADHD-friendly
explainers: a push-up profile whose upper pole fills, the ДО photo with level lines and an asymmetry bracket,
a round implant with its only two parameters drawn in (diameter, projection), two implants with volume counters
225/250 → 200/250, «decided in the operating room» chips, and a parameter swarm for the finale. Brand system of the
other reels (deep navy, powder pink, Bebas Neue + thin Montserrat, logo); one-word subtitles; whole sentences.

usage:
  REEL_SRC=<folder with C0811.MP4, DSC01078.JPG, DSC01088.JPG, logo.png, music.wav> python3 tools/reel_asymmetry.py [--preview t1,t2 | --cover | --audio-only]
Prepared in $REEL_SRC/_work: C0811.mp4 (1080x1920, 30 fps), C0811.wav (48 kHz mono).
"""
import os, sys, json, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_asymmetry"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_asymmetry_round_implants"
W, H, FPS, SR = 1080, 1920, 30, 48000
NAVY = np.float32([0, 12, 40]) / 255; PINK = np.float32([233, 197, 193]) / 255; WHITE = np.float32([1, 1, 1])
DEEP = np.float32([0, 5, 20]) / 255

def cl(x): return min(1.0, max(0.0, x))
def eo(x): x = cl(x); return 1 - (1 - x) ** 3
def eio(x): x = cl(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x): x = cl(x); c = 1.7; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p

# ---------------- voice edit: (block, source, [(in, out), …]) — whole sentences from the one take; the pauses between them are tightened
VO = [("S1", "C0811", [(6.90, 10.88)]),                                  # (the trailing «а…» left out)                                  # пациентка пришла с запросом коррекции асимметрии груди и увеличения груди
      ("S2", "C0811", [(13.02, 14.50), (18.16, 21.40)]),                  # «а-а» after «эффект» and the aside to the nurse cut                  # она хотела пуш-ап эффект, (…) при пуш-ап эффекте он лучше добивается именно круглыми имплантами
      ("S3", "C0811", [(22.48, 23.86), (24.70, 26.38), (27.52, 30.80), (32.66, 33.78)]),   # «проблема а-а-а деформации» → «проблема деформации»  # у нас здесь проблема деформации грудной клетки, и всё-таки круглые импланты дают лучшую возможность скорректировать асимметрию
      ("S4", "C0811", [(34.80, 37.62), (38.10, 38.84)]),                  # the drawn-out «и-и» shortened                                  # у круглого импланта только два параметра — проекция и диаметр
      ("S5", "C0811", [(53.50, 53.72), (54.15, 56.72), (57.22, 61.80)]),   # the drawn-out «мы-ы» shortened                  # мы планируем разные объёмы, либо где-то 225 и 250, либо 200 и 250
      ("S6", "C0811", [(63.14, 68.56)]),                                  # к сожалению, такая операция, которую невозможно точно предугадать перед операцией
      ("S7", "C0811", [(71.46, 76.20)]),                                  # здесь разная анатомия абсолютно желёз, разная анатомия грудной клетки
      ("S8", "C0811", [(78.62, 81.60)])]                                  # много параметров, которые нужно одновременно учитывать
GAP = {"S2": 0.3, "S3": 0.45, "S4": 0.3, "S5": 0.45, "S6": 0.45, "S7": 0.3, "S8": 0.3}
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

def T(s_): return at_src("C0811", s_)
T_S = {b: vstart(b) - 0.08 for b, *_ in VO}; T_END = vend("S8") + 0.3; DUR = T_END + 1.6 + 2.2 * 2 + 2.8
T_PUSH = T(13.50); T_ROUND = T(20.26)                                   # «push-up» · «круглыми»
T_DEF = T(24.82); T_ASYM = T(33.14)                                     # «деформации» · «асимметрию»
T_PROJ = T(36.84); T_DIAM = T(38.30)                                    # «проекция» · «диаметр»
T_V1 = T(57.60); T_V2 = T(59.32); T_V3 = T(60.58); T_V4 = T(61.18)      # 225 · 250 · 200 · 250
T_GL = T(72.12); T_CH = T(74.46)                                        # «анатомия желёз» · «грудной клетки»

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
V = {"C0811": Src(os.path.join(WORK, "C0811.mp4"))}

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
    'S1': [('Пациентка', 0.04, 0.68), ('пришла', 0.68, 1.04), ('с', 1.04, 1.08), ('запросом', 1.08, 1.64), ('асимметрии', 1.64, 2.38), ('груди', 2.38, 2.76), ('и', 2.76, 3.02), ('увеличения', 3.02, 3.42), ('груди.', 3.42, 3.9)],
    'S2': [('Она', 0.04, 0.12), ('хотела', 0.12, 0.48), ('пуш-ап', 0.48, 0.94), ('эффект.', 0.94, 1.4), ('Пуш-ап', 1.74, 2.1), ('эффект', 2.1, 2.46), ('лучше', 2.46, 2.74), ('добивается', 2.74, 3.24), ('именно', 3.24, 3.58), ('круглыми', 3.58, 3.98), ('имплантами.', 3.98, 4.52)],
    'S3': [('У', 0.04, 0.08), ('нас', 0.08, 0.4), ('здесь', 0.4, 0.74), ('проблема', 0.74, 1.28), ('деформации', 1.28, 2.12), ('грудной', 2.12, 2.56), ('клетки.', 2.56, 3.1), ('Всё-таки', 3.12, 3.56), ('круглыми', 3.56, 4.04), ('имплантами', 4.04, 4.68), ('они', 4.68, 4.88), ('дают', 4.88, 5.2), ('лучшую', 5.2, 5.58), ('возможность', 5.58, 6.26), ('корректировать', 6.26, 6.8), ('асимметрию.', 6.8, 7.38)],
    'S4': [('У', 0.04, 0.28), ('круглого', 0.28, 0.68), ('импланта', 0.68, 1.14), ('только', 1.14, 1.34), ('два', 1.34, 1.5), ('параметра:', 1.5, 2.06), ('проекция', 2.26, 2.68), ('и', 2.68, 2.98), ('диаметр.', 2.98, 3.52)],
    'S5': [('Мы', 0.04, 0.44), ('планируем', 0.44, 1.0), ('разные', 1.0, 1.46), ('объёмы,', 1.46, 2.22), ('либо', 2.78, 2.8), ('это', 2.8, 3.14), ('225', 3.14, 4.5), ('и', 4.5, 4.88), ('250,', 4.88, 5.5), ('либо', 5.82, 6.16), ('200', 6.16, 6.54), ('и', 6.54, 6.74), ('250.', 6.74, 7.24)],
    'S6': [('К', 0.04, 0.22), ('сожалению,', 0.22, 0.56), ('такая', 0.8, 0.86), ('операция,', 0.86, 1.36), ('которую', 1.6, 1.7), ('невозможно', 1.7, 2.5), ('точно', 2.5, 3.22), ('предугадать', 3.22, 4.02), ('перед', 4.02, 4.38), ('операцией.', 4.38, 4.81)],
    'S7': [('Здесь', 0.04, 0.26), ('разная', 0.26, 0.68), ('анатомия', 0.68, 1.26), ('абсолютно', 1.26, 1.82), ('желёз,', 1.82, 2.28), ('разная', 2.78, 3.02), ('анатомия', 3.02, 3.5), ('грудной', 3.5, 3.96), ('клетки.', 3.96, 4.52)],
    'S8': [('Много', 0.04, 0.38), ('параметров,', 0.38, 1.02), ('которые', 1.26, 1.38), ('нужно', 1.38, 1.68), ('одновременно', 1.68, 2.3), ('учитывать.', 2.3, 2.92)],
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
def SUR(t): return V["C0811"].at(src_of(t, "C0811"))
FACES = np.array(json.load(open(os.path.join(WORK, "C0811_faces.json"))))   # YuNet face box per frame, smoothed
def face_at(t):
    """face centre (a little above the box centre, so the cap is in) and the scale that fits the head in the circle"""
    i = min(max(int(round(src_of(t, "C0811") * FPS)), 0), len(FACES) - 1); x, y, w, h = FACES[i]
    return (x + w / 2, y + h * 0.3), float(np.clip(300 / h, 0.45, 0.7))     # head and shoulders in the circle
FACE = (580, 930); CC, CR, CS = (540, 540), 270, 0.78                 # face in the take; circle centre, radius, scale (a down-scale)
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
    (fx, fy), fs = face_at(t); s = lerp(1.0, fs, p); cx, cy = lerp(fx, CC[0], p), lerp(fy, CC[1], p); r = lerp(1150, CR, p)
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
def shot(t, z):
    """the take full screen, framed low so the surgeon's face sits high and the text has room under the chin (z ≥ 1.12)"""
    return full(SUR(t), 0.53, 0.8, z)
def fit_size(txt, maxw, size):
    while bebas(size).getlength(txt) > maxw and size > 30: size -= 2
    return size
def badge(fr, txt, x, y, p):
    """«ЗАДАЧА №…» label: pink pill with navy text, pops in"""
    if p <= 0: return
    f = bebas(54); a = alpha_img(txt, f); s = eback(p); w, h = (a.shape[1] + 44) * s, 70 * s
    blend(fr, rrect(x, y, x + w, y + h, 35 * s), 0, 0, PINK, min(1, p * 2))
    if s > 0.9: blend(fr, a, x + 22, y + 35 - a.shape[0] / 2 + 5, NAVY, cl((p - 0.5) * 2))
def pillx(fr, txt, x, y, p, tick_p=0.0, size=30):
    if p <= 0: return
    f = mont(size, 600); a = alpha_img(txt, f); s = eback(p); w, h = a.shape[1] + 86, 64
    blend(fr, rrect(x, y, x + w * s, y + h * s, 32 * s), 0, 0, NAVY, 0.9 * min(1, p * 2))
    rrect_line(fr, x, y, x + w * s, y + h * s, 32 * s, PINK, 2, min(1, p * 2))
    if s > 0.9: blend(fr, a, x + 64, y + h / 2 - a.shape[0] / 2 + 2, WHITE, cl((p - 0.5) * 2)); path_draw(fr, [(x + 22, y + 33), (x + 32, y + 43), (x + 48, y + 22)], tick_p, PINK, 5)
def arrow(fr, p0, p1, p, col=PINK, th=6, both=False):
    if p <= 0: return
    x0, y0 = p0; x1, y1 = p1; xe, ye = lerp(x0, x1, p), lerp(y0, y1, p); path_draw(fr, [(x0, y0), (xe, ye)], 1, col, th)
    a = np.arctan2(ye - y0, xe - x0); hs = 22
    for (hx, hy, aa) in ([(xe, ye, a)] + ([(x0, y0, a + np.pi)] if both else [])):
        fill_poly(fr, [(hx, hy), (hx - hs * np.cos(aa - 0.45), hy - hs * np.sin(aa - 0.45)), (hx - hs * np.cos(aa + 0.45), hy - hs * np.sin(aa + 0.45))], col)

# ДО photos (the client's new set, 1974/1932 x 2576): shown through cover() — a down-scale; chin blurred
from PIL import ImageOps
def photo(n): return np.asarray(ImageOps.exif_transpose(Image.open(os.path.join(SRC, "new", n + ".jpg"))).convert("RGB")).copy()
def anon_top(img, y=150):
    top = img[:y + 80].astype(np.float32); bl = cv2.resize(cv2.GaussianBlur(cv2.resize(top, None, fx=0.1, fy=0.1), (0, 0), 3), (img.shape[1], y + 80))
    m = np.clip((y + 40 - np.arange(y + 80)) / 80, 0, 1)[:, None, None]; out = img.copy(); out[:y + 80] = (top * (1 - m) + bl * m).astype(np.uint8); return out
DO_F = anon_top(photo("do_front"), 70); DO_Q = anon_top(photo("do_34"), 130)
def at_photo(M, x, y): return M[0, 0] * x + M[0, 2], M[1, 1] * y + M[1, 2]

# S1 — straight in: the question
H1 = lift(Drop(["ПОЧЕМУ ПРИ АСИММЕТРИИ", "*МЫ ВЫБРАЛИ КРУГЛЫЕ ИМПЛАНТЫ?*"], 84, 70, 0, maxw=860), 1340)
def s1(t):
    fr = shot(t, pull(t, 0, 1.3, 1.14, 2.6)); shade(fr, GBOT)
    for k, (x, y) in enumerate([(860, 430), (210, 560)]):
        lt = t - 0.5 - 0.25 * k
        if lt > 0: text_c(fr, "?", x, y - 8 * np.sin(lt * 4), bebas(int(120 * eback(lt / 0.3))), PINK, min(1, lt * 3))
    H1.draw(fr, t, 0.05); return shake(fr, t, 0.5, 7)

# S2 — task 1: push-up — the profile's upper pole fills, a round implant drops in
HB1 = Drop(["ДОБАВИТЬ НАПОЛНЕННОСТЬ", "*ВЕРХНЕГО ПОЛЮСА*"], 88, 80, 905, maxw=860)
PRO_B = [(330, 1010), (430, 1100), (530, 1200), (600, 1290), (560, 1340), (440, 1350), (330, 1344)]
PRO_A = [(330, 1010), (500, 1050), (630, 1140), (680, 1250), (630, 1335), (480, 1352), (330, 1344)]
def _fit(pts): return [(330 + (x - 330) * 0.85, 1095 + (y - 1010) * 0.8) for x, y in pts]          # under the headline, above the subtitles
PRO_B, PRO_A = _fit(PRO_B), _fit(PRO_A)
def s2(t):
    t0 = T_S["S2"]; fr = circle_layout(t, t0); badge(fr, "ЗАДАЧА 1", 80, 815, eo((t - t0 - 0.2) / 0.3)); HB1.draw(fr, t, t0 + 0.35)
    p = eio((t - T_PUSH) / 0.9); pts = [(lerp(a[0], b[0], p), lerp(a[1], b[1], p)) for a, b in zip(PRO_B, PRO_A)]
    op = eo((t - t0 - 0.6) / 0.4)
    if op > 0:
        path_draw(fr, [(330, 1080), (330, 1380)], 1, WHITE, 3, op=0.5 * op)                       # chest wall
        if p > 0: fill_poly(fr, [PRO_B[0]] + [(lerp(a[0], b[0], p), lerp(a[1], b[1], p)) for a, b in zip(PRO_B[1:4], PRO_A[1:4])] + PRO_B[3:0:-1], PINK, 0.4)
        path_draw(fr, PRO_B, 1, WHITE, 3, op=0.35 * op * (1 - p)); path_draw(fr, pts, eo((t - t0 - 0.6) / 0.6), WHITE, 5, op=op)
        arrow(fr, (780, 1360), (780, 1170), eio((t - T_PUSH - 0.2) / 0.5))
        if t > T_PUSH + 0.4: text_c(fr, "ПУШ-АП", 800, 1125, bebas(56), PINK, eo((t - T_PUSH - 0.4) / 0.3))
    if t > T_ROUND:                                                       # the round implant
        lt = t - T_ROUND; s_ = eback(lt / 0.35); m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (470, 1250), (int(78 * s_), int(78 * s_)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        blend(fr, cv2.GaussianBlur(m, (0, 0), 3).astype(np.float32) / 255, 0, 0, WHITE, 0.4); ring(fr, 470, 1250, 78 * s_, min(1, lt * 3), 4)
    return fr

# S3 — task 2: the ДО photos; level lines show the asymmetry, the chest gets an outline
HB2 = lift(Drop(["СКОРРЕКТИРОВАТЬ", "*АСИММЕТРИЮ*"], 92, 80, 0, maxw=880), 1270)
TB2 = Track(["учитываем и форму груди,", "и анатомию грудной клетки"], 30, 80, 0, 300)
T_Q = T(29.20)                                                            # the 3/4 photo
LV = [((813, 540), (1245, 540), "соски"), ((800, 603), (1250, 618), "складки")]          # do_front px: nipples, inframammary folds
def s3(t):
    t0 = T_S["S3"]; q = t >= T_Q
    if not q:
        z = lerp(1.0, 1.08, eio((t - t0) / 5)); fr, M = cover(DO_F, W, H, 0.5, 0.25, z)
        for k, ((x0, y0), (x1, y1), lab) in enumerate(LV):
            p = eio((t - t0 - 0.5 - 0.5 * k) / 0.6)
            if p <= 0: continue
            a0, a1 = at_photo(M, x0, y0), at_photo(M, x1, y1)
            for xa, ya in (a0, a1):
                n_ = 18; xs = np.linspace(60, 900, 2 * n_)
                for j in range(int(n_ * p)): path_draw(fr, [(xs[2 * j], ya), (xs[2 * j + 1], ya)], 1, PINK if k else WHITE, 3, op=0.85)
            if k == 1 and t > T_DEF:                                      # Δ bracket between the two fold levels
                pb = eio((t - T_DEF) / 0.5); xb = 920; path_draw(fr, [(xb - 20, a0[1]), (xb, a0[1]), (xb, lerp(a0[1], a1[1], pb)), (xb - 20, lerp(a0[1], a1[1], pb))], 1, PINK, 5)
    else:
        lt = t - T_Q; fr, M = cover(DO_Q, W, H, 0.45, 0.25, lerp(1.0, 1.06, eio(lt / 4)))
        fr = fr * (1 - (1 - eo(lt / 0.2))) + (1 - eo(lt / 0.2))
    shade(fr, VIG); shade(fr, GBOT); chip(fr, "ДО", 40, 290)
    badge(fr, "ЗАДАЧА 2", 80, HB2.y - 110, eo((t - t0 - 0.2) / 0.3)); HB2.draw(fr, t, t0 + 0.35)
    TB2.y = HB2.y + HB2.h + 36 + TB2.top; TB2.draw(fr, t, t0 + 1.0)
    if t > T_ASYM: pillx(fr, "асимметрия", 600, 1050, eo((t - T_ASYM) / 0.3), eio((t - T_ASYM - 0.25) / 0.3))
    return fr

# S4 — the round implant has only two parameters: projection and diameter
HB4 = Drop(["ПОЭТОМУ ВЫБОР ПАЛ", "*НА КРУГЛЫЕ ИМПЛАНТЫ*"], 92, 80, 850, maxw=860)
def implant_dome(fr, cx, by, w, h, op=1.0, fill=0.25):
    pts = [(cx + w / 2 * np.cos(a), by - h * np.sin(a)) for a in np.linspace(0, np.pi, 60)]
    fill_poly(fr, pts + [(cx - w / 2, by)], WHITE, fill * op); path_draw(fr, pts + [(cx - w / 2, by), (cx + w / 2, by)], 1, WHITE, 4, op=op)
    sh = [(cx - w * 0.25 + w * 0.12 * np.cos(a), by - h * 0.62 + h * 0.12 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 20)]; fill_poly(fr, sh, WHITE, 0.35 * op)
def s4(t):
    t0 = T_S["S4"]; fr = circle_layout(t, t0, morph=False); HB4.draw(fr, t, t0 + 0.1)
    op = eo((t - t0 - 0.3) / 0.4); s_ = eback((t - t0 - 0.3) / 0.4)
    if op > 0: implant_dome(fr, 470, 1300, 440 * s_, 200 * s_, op)
    arrow(fr, (470, 1296), (470, 1100), eio((t - T_PROJ) / 0.5), both=True)
    if t > T_PROJ + 0.3: text_c(fr, "ПРОЕКЦИЯ", 790, 1150, bebas(56), PINK, eo((t - T_PROJ - 0.3) / 0.3))
    arrow(fr, (250, 1338), (690, 1338), eio((t - T_DIAM) / 0.5), both=True)
    if t > T_DIAM + 0.3: text_c(fr, "ДИАМЕТР", 820, 1330, bebas(56), PINK, eo((t - T_DIAM - 0.3) / 0.3))
    return fr

# S5 — different volumes: two implants with counters, 225 / 250 → 200 / 250
HB5 = Drop(["НА РАЗНЫЕ ГРУДИ МОГУТ", "*ПОНАДОБИТЬСЯ РАЗНЫЕ ОБЪЁМЫ*"], 88, 80, 850, maxw=860)
def vol_left(t):
    if t < T_V1: return 0.0
    if t < T_V3: return 225 * eo((t - T_V1) / 0.7)
    return lerp(225, 200, eio((t - T_V3) / 0.5))
def vol_right(t): return 250 * eo((t - T_V2) / 0.7) if t >= T_V2 else 0.0
def s5(t):
    t0 = T_S["S5"]; fr = circle_layout(t, t0); HB5.draw(fr, t, t0 + 0.2)
    for k, (cx, v, tk) in enumerate([(270, vol_left(t), T_V1), (720, vol_right(t), T_V2)]):
        if t < tk - 0.1: continue
        r = 150 * (max(v, 60) / 250) ** (1 / 3) * eback((t - tk + 0.1) / 0.35)
        r *= 1 + 0.06 * (1 - eo((t - (T_V4 if k else T_V3)) / 0.35)) * (t > (T_V4 if k else T_V3))
        m = np.zeros((H, W), np.uint8); cv2.circle(m, (cx, 1170), int(r), 255, -1, cv2.LINE_AA)
        blend(fr, cv2.GaussianBlur(m, (0, 0), 2).astype(np.float32) / 255, 0, 0, WHITE, 0.22); ring(fr, cx, 1170, r, 1, 4)
        text_c(fr, f"{int(round(v))}", cx, 1155, bebas(96), WHITE); text_c(fr, "мл", cx, 1225, mont(32, 400), PINK)
    if t > T_V2 + 0.3:                                                    # ≠ drawn (Bebas has no glyph for it)
        q = eo((t - T_V2 - 0.3) / 0.3); path_draw(fr, [(470, 1152), (520, 1152)], q, PINK, 7); path_draw(fr, [(470, 1182), (520, 1182)], q, PINK, 7); path_draw(fr, [(512, 1130), (478, 1204)], q, PINK, 6)
    return fr

# S6–S7 — decided in the operating room; the anatomy chips
H6 = lift(Drop(["ОКОНЧАТЕЛЬНОЕ РЕШЕНИЕ ИНОГДА", "ПРИНИМАЕТСЯ *ВО ВРЕМЯ ОПЕРАЦИИ*"], 68, 70, 0, maxw=880), 1335)
def s6(t):
    t0 = T_S["S6"]; t7 = T_S["S7"]
    fr = shot(t, pull(t, t7, 1.28, 1.16, 1.6) if t >= t7 else pull(t, t0, 1.3, 1.14, 1.8)); shade(fr, GBOT)
    if t < t7:
        for k, (x, y) in enumerate([(850, 420), (230, 520), (880, 650)]):
            lt = t - t0 - 1.6 - 0.35 * k
            if lt > 0: text_c(fr, "?", x, y - 10 * np.sin(lt * 4), bebas(int(100 * eback(lt / 0.3))), PINK, min(1, lt * 3) * (1 - cl((t - t7 + 0.4) / 0.4)))
    pillx(fr, "разная анатомия желёз", 80, 1020, eo((t - T_GL) / 0.3), eio((t - T_GL - 0.25) / 0.3))
    pillx(fr, "разная грудная клетка", 80, 1095, eo((t - T_CH) / 0.3), eio((t - T_CH - 0.25) / 0.3))
    H6.draw(fr, t, t0 + 0.15); return fr

# S8 + end — many parameters at once; the one-scheme line
HB8 = Drop(["АСИММЕТРИЮ НЕЛЬЗЯ", "КОРРЕКТИРОВАТЬ *ПО ОДНОЙ СХЕМЕ*"], 88, 80, 850, maxw=860)
PARAMS = [("форма груди", 90, 1050), ("объём", 640, 1050), ("грудная клетка", 90, 1130), ("проекция", 590, 1130), ("диаметр", 90, 1210), ("железа", 400, 1210), ("пуш-ап", 680, 1210)]
def s8(t):
    t0 = T_S["S8"]; fr = circle_layout(t, t0); HB8.draw(fr, t, t0 + 0.2)
    for k, (lab, x, y) in enumerate(PARAMS):
        pillx(fr, lab, x, y, eo((t - t0 - 0.3 - 0.22 * k) / 0.3), eio((t - t0 - 0.5 - 0.22 * k) / 0.3), 28)
    if t > T_END:
        lt = t - T_END; blend(fr, logo_alpha(300), 540 - 150, 1310, PINK, eo(lt / 0.5))
    return fr

# K — collages: ПОСЛЕ on top, ДО below, on black; ПОСЛЕ colour-matched to the ДО of the same angle (skin statistics in Lab)
ALIGN = {"do_front": (135, 926, 990), "posle_front": (144, 948, 651), "do_34": (198, 908, 845), "posle_34": (215, 1060, 656),
         "do_side": (144, 1000, 845), "posle_side": (180, 1010, 656)}   # (neck, navel / waist row, body centre x) in photo px
def skin(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); return (hsv[..., 2] > 90) & (hsv[..., 1] > 25) & (hsv[..., 1] < 170) & ((hsv[..., 0] < 25) | (hsv[..., 0] > 165))
def match_colour(src, ref):
    a = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32); b = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    ms, mr = skin(src), skin(ref)
    for c in range(3):
        s0, s1_ = a[..., c][ms].mean(), a[..., c][ms].std() + 1e-3; r0, r1 = b[..., c][mr].mean(), b[..., c][mr].std() + 1e-3
        k = float(np.clip(r1 / s1_, 0.75, 1.35))
        a[..., c] = (a[..., c] - s0) * k + r0
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
def panel(im, n, ph=960, neck_at=40, span=470):
    neck, navel, cx = ALIGN[n]; s_ = span / (navel - neck); r = cv2.resize(im, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    out = np.zeros((ph, W, 3), np.float32); ox, oy = int(cx * s_ - W / 2), int(neck * s_ - neck_at)
    xs0, ys0 = max(0, ox), max(0, oy); xs1, ys1 = min(r.shape[1], ox + W), min(r.shape[0], oy + ph)
    out[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox] = r[ys0:ys1, xs0:xs1]
    lum = out.mean(2); k = np.clip((lum - 0.2) / 0.14, 0, 1); k = k * k * (3 - 2 * k)
    k = np.maximum(k, np.clip((out[..., 2] - out[..., 0] - 0.04) / 0.05, 0, 1)); out *= k[..., None]   # the navy stickers keep their colour          # studio backdrop → pure black
    return out
COLL = []
for ang in ("front", "34", "side"):
    d = photo("do_" + ang); a = match_colour(photo("posle_" + ang), d)
    COLL.append(np.vstack([panel(anon_top(a, ALIGN["posle_" + ang][0] - 40), "posle_" + ang), panel(anon_top(d, ALIGN["do_" + ang][0] - 40), "do_" + ang)]))
COLL8 = [(np.clip(c, 0, 1) * 255).astype(np.uint8) for c in COLL]
T_K = [T_END + 1.6 + 2.2 * k for k in range(3)]                          # after the finale and the logo
def s_k(t):
    k = max(i for i, tk in enumerate(T_K) if t >= tk - 1e-6); lt = t - T_K[k]
    fr = full(COLL8[k], 0.5, 0.5, 1 + 0.03 * (1 - eo(lt / 0.35))); fr[957:963] = PINK
    chip(fr, "ПОСЛЕ", 40, 290); chip(fr, "ДО", 40, 1000)
    f = 1 - eo(lt / 0.18) if k else 0; return fr * (1 - f) + f

SCENES = [(0, s1), (T_S["S2"], s2), (T_S["S3"], s3), (T_S["S4"], s4), (T_S["S5"], s5), (T_S["S6"], s6), (T_S["S8"], s8), (T_K[0], s_k)]
TRANS = {T_S["S2"]: ("zoom", 0.35), T_S["S3"]: ("iris", 0.5), T_S["S4"]: ("zoom", 0.35), T_S["S5"]: ("diag", 0.45), T_S["S6"]: ("zoom", 0.35), T_S["S8"]: ("zoom", 0.35), T_K[0]: ("iris", 0.5)}
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

MUSIC = os.path.join(SRC, "music.wav")
M_DROP = 14.45; M0 = M_DROP - T_S["S2"]                                # the drop lands on the first explainer
G_VO, G_GAP = 0.022, 0.075
def track(n):
    w = wave.open(MUSIC); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64).reshape(-1, w.getnchannels()) / 32768
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    x = x / np.sqrt(np.mean(x[int(M_DROP * SR):].mean(1) ** 2)); x[:int(M_DROP * SR)] *= 4.5
    bar = 4 * 60 / 137.0; b = int((M_DROP + 20 * bar) * SR); a0 = int((M_DROP + 4 * bar) * SR); w = SR; cf = int(0.3 * SR)
    ref = x[b:b + w].mean(1); a = max(range(a0 - int(0.08 * SR), a0 + int(0.08 * SR), 12), key=lambda k: float(np.dot(ref, x[k:k + w].mean(1))))
    fade = np.sin(np.linspace(0, np.pi / 2, cf))[:, None] ** 2; seam = x[b:b + cf] * (1 - fade) + x[a:a + cf] * fade
    parts = [x[:b]]; total = b
    while total < int(M0 * SR) + n:
        parts += [seam, x[a + cf:b]]; total += b - a
    x = np.concatenate(parts)
    seg = x[int(M0 * SR):int(M0 * SR) + n]; return np.pad(seg, ((0, n - len(seg)), (0, 0)))

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
    for k in range(2): put(pop(1300 - 200 * k, 500, 0.1), 0.5 + 0.25 * k, 0.12)                         # question marks
    put(marimba(PENTA[1], 0.4), T_S["S2"] + 0.2, 0.16); put(swell(0.9), T_PUSH - 0.2, 0.06); put(water(500, 1500), T_ROUND, 0.25)
    put(marimba(PENTA[2], 0.4), T_S["S3"] + 0.2, 0.16)
    for k in range(2): put(paper(0.2, 4000), T_S["S3"] + 0.5 + 0.5 * k, 0.1)                          # level lines draw
    put(felt(110, 0.25), T_DEF, 0.3); put(click(), T_Q, 0.5); put(marimba(PENTA[4], 0.45), T_ASYM, 0.2)
    put(marimba(PENTA[2], 0.45), T_PROJ, 0.2); put(marimba(PENTA[4], 0.45), T_DIAM, 0.2)
    for tk, dur in ((T_V1, 0.7), (T_V2, 0.7)):
        for j in range(8): put(tick(), tk + dur * j / 8, 0.22)                                             # counters
    put(marimba(PENTA[3], 0.5), T_V2 + 0.3, 0.18); put(felt(130, 0.2), T_V3, 0.3); put(pop(900, 500, 0.08), T_V4, 0.12)
    for k in range(3): put(pop(1200, 500, 0.08), T_S["S6"] + 1.6 + 0.35 * k, 0.08)
    put(water(600, 1500), T_GL, 0.22); put(water(700, 1600), T_CH, 0.22)
    for k in range(len(PARAMS)): put(pop(1000 + 60 * k, 450, 0.08), T_S["S8"] + 0.3 + 0.22 * k, 0.09)
    put(glass(1174.7), T_END, 0.14)
    for tk in T_K: put(click(), tk, 0.9)                                                              # shutter on every collage
    sfx[:n] *= 1 - 0.8 * np.clip(dd, 0, 1)
    mix = vo[:, None] + mus + sfx[:n, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (-1.5 / 20); mix /= max(1, np.abs(mix).max() / 0.95)                           # ≈ -16.5 LUFS like the other reels
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

def make_cover(path):
    """everything inside the centre 3:4 (y 240-1680) that the profile grid shows; a deeper navy than the video"""
    CBG = np.clip(np.float32([0, 8, 28]) / 255 * (1 + 0.3 * np.exp(-(((xx - 540) / 900) ** 2 + ((yy - 520) / 900) ** 2)))[..., None], 0, 1).astype(np.float32)
    fr = CBG.copy(); im, M = cover(DO_F, W, 820, 0.5, 0.18, 1.0); y0 = 330
    g = np.maximum(np.clip((np.arange(820) - 600) / 220, 0, 1), np.clip((120 - np.arange(820)) / 120, 0, 1))[:, None, None]; fr[y0:y0 + 820] = im * (1 - g) + CBG[y0:y0 + 820] * g
    for k, ((x0, y0_), (x1, y1_), _) in enumerate(LV):
        for xa, ya in (at_photo(M, x0, y0_), at_photo(M, x1, y1_)):
            xs = np.linspace(60, 1020, 36)
            for j in range(18): path_draw(fr, [(xs[2 * j], ya + y0), (xs[2 * j + 1], ya + y0)], 1, PINK if k else WHITE, 3, op=0.85)
    blend(fr, logo_alpha(300), 540 - 150, 262, PINK)
    hd = Drop(["ПРИ АСИММЕТРИИ НУЖНЫ", "*РАЗНЫЕ ИМПЛАНТЫ?*"], 104, 70, 1190, maxw=880); hd.draw(fr, 99, 0)
    ty = Track(["почему мы выбрали круглую форму"], 34, 72, 0, 400); ty.y = 1190 + hd.h + 50 + ty.top; ty.draw(fr, 99, 0)
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
