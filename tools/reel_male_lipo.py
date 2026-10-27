#!/usr/bin/env python3
"""Reels «Лучше раньше, чем позже» — мужчина, комплексная коррекция (липосакция боков/живота, абдоминопластика,
липосакция груди), Анна Тваури.

Brand system of the other reels (navy, powder pink, Bebas Neue + Montserrat, logo). The patient's own words carry it:
his quote first (no ДО until 0:04), then ДО with a flipping calendar («долго откладывал»), the surgeon's plan with a
checklist ticked on every named zone, a ДО/ПОСЛЕ slider on «всё стоило того», three «2 недели» collages with shutter
clicks, logo outro. Whole sentences only; the patient stays anonymous (face blurred).

usage:
  REEL_SRC=<folder with IMG_4316/4342/4355.MOV, IMG_8746/8749/8752/8755-8757.JPG, logo.png, music.wav> python3 tools/reel_male_lipo.py [--preview t1,t2 | --cover | --audio-only]
Sources are first normalised into $REEL_SRC/_work (1080x1920, 30 fps; voices 48 kHz mono wav):
  ffmpeg -i IMG_x.MOV -vf "scale=1080:1920:flags=lanczos,fps=30,format=yuv420p" -an -c:v libx264 -crf 14 _work/IMG_x.mp4
  ffmpeg -i IMG_4355.MOV -vn -ac 1 -ar 48000 _work/IMG_4355.wav   (same for IMG_4316)
"""
import os, sys, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_male_lipo"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_male_better_sooner"
W, H, FPS, SR = 1080, 1920, 30, 48000
NAVY = np.float32([0, 18, 58]) / 255; PINK = np.float32([233, 197, 193]) / 255; WHITE = np.float32([1, 1, 1])
DEEP = np.float32([0, 8, 30]) / 255

def cl(x): return min(1.0, max(0.0, x))
def eo(x): x = cl(x); return 1 - (1 - x) ** 3
def eio(x): x = cl(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x): x = cl(x); c = 1.7; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p

# ---------------- voice edit: (block, source, [(in, out), …]) — whole sentences; parts inside one entry are joined with a 15 ms crossfade
VO = [("A", "4355", [(8.70, 11.16), (11.40, 12.80)]),               # подумал-подумал и согласился. лучше раньше, чем позже. (stumble «рано» removed)
      ("B", "4355", [(7.30, 8.40)]),                                 # давно надо было делать
      ("C", "4316", [(10.66, 16.52)]),                               # будем делать липосакцию боков, липосакцию живота, сделаем абдоминопластику
      ("C2", "4316", [(20.50, 23.30)]),                              # и будем делать ещё липосакцию молочных желёз мужчине
      ("D", "4355", [(45.08, 47.92)]),                               # всё стоило того, я знал, на что шёл
      ("E", "4355", [(48.06, 52.75)]),                               # результат… я начинаю видеть его, и в целом я доволен
      ("E2", "4355", [(52.88, 55.35)])]                              # а это пока с отёком ещё, ещё нужно всё уменьшиться
GAP = {"B": 0.35, "C": 1.9, "C2": 0.35, "D": 0.5, "E": 0.9, "E2": 0.25}   # the ДО hold after «давно надо было делать», slider hold after D
SEG = []; _t = 0.25
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
T_B = vstart("B") - 0.15; T_C = vstart("C") - 0.2; T_D = vstart("D") - 0.2; T_E = vstart("E") - 0.15
T_END = vend("E2") + 0.35; DUR = T_END + 3.2
T_FLANK = at_src("4316", 12.88); T_BELLY = at_src("4316", 14.60); T_ABDO = at_src("4316", 16.02); T_CHEST = at_src("4316", 22.48)

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
V = {k: Src(os.path.join(WORK, f"IMG_{k}.mp4")) for k in ["4355", "4316", "4342"]}
def anon(img, y=180):
    """blur everything above `y` (face)"""
    top = img[:y + 80].astype(np.float32); bl = cv2.resize(cv2.GaussianBlur(cv2.resize(top, None, fx=0.12, fy=0.12), (0, 0), 3), (W, y + 80))
    m = np.clip((y + 40 - np.arange(y + 80)) / 80, 0, 1)[:, None, None]; out = img.copy(); out[:y + 80] = (top * (1 - m) + bl * m).astype(np.uint8); return out
def photo(n):
    return np.asarray(ImageOps.exif_transpose(Image.open(os.path.join(SRC, f"IMG_{n}.JPG"))).convert("RGB").resize((W, H), Image.LANCZOS)).copy()
def cutout(n, dy=0):
    """Bazaart cut-outs: crop to the torso (never enlarged: the source is 2048 px wide), app watermark dropped"""
    im = np.asarray(ImageOps.exif_transpose(Image.open(os.path.join(SRC, f"IMG_{n}.JPG"))).convert("RGB"))
    s = im.shape[1] / W; x0, y0, x1, y1 = [int(v * s) for v in (140, 380, 940, 1450)]; y0 -= dy; y1 -= dy
    c = cv2.resize(im[y0:y1, x0:x1], (W, int(W * (y1 - y0) / (x1 - x0))), interpolation=cv2.INTER_AREA)
    fr = np.zeros((H, W, 3), np.uint8); fr[230:230 + c.shape[0]] = c[:H - 230]; return fr
BEF, AFT = cutout(8749), cutout(8746, 92)                             # ПОСЛЕ sits 92 px higher in its photo: nipple stickers line up
COLLS = [photo(n) for n in (8755, 8756, 8757)]                          # «2 недели 3 дня после операции», ПОСЛЕ on top, ДО below

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

# ---------------- scenes ----------------
# A — the patient's face (blurred at the source), his quote; nothing of ДО yet
HA = Drop(["ЛУЧШЕ РАНЬШЕ,", "*ЧЕМ ПОЗЖЕ*"], 150, 70, 1180)
TA = Track(["история пациента после комплексной коррекции"], 34, 72, 0, 300)
def s_a(t):
    fr = full(V["4355"].at(src_of(t, "4355")), 0.5, 0.36, lerp(1.06, 1.16, eio(t / 4.3))); shade(fr, VIG); shade(fr, GBOT)
    lg = logo_alpha(360); blend(fr, lg, 540 - 180, 210, PINK, eo((t + 0.1) / 0.5) * 0.9)
    HA.draw(fr, t, -0.2); TA.y = HA.y + HA.h + 70 + TA.top; TA.draw(fr, t, 0.9)
    return fr

# B — ДО for the first time; a calendar flips its pages, then a pink tick: «долго откладывал, но решился»
HB = Drop(["ДОЛГО ОТКЛАДЫВАЛ,", "*НО В ИТОГЕ РЕШИЛСЯ*"], 120, 70, 1250)
def calendar(fr, t, t0, cx=880, cy=1020, w=150):
    lt = t - t0
    if lt < 0: return
    s = eback(lt / 0.35); hw, hh = w / 2 * s, w * 0.55 * s
    box = [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]
    fill_poly(fr, box, NAVY, 0.85); path_draw(fr, box, 1, WHITE, 3, True)
    fill_poly(fr, [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy - hh + 34 * s), (cx - hw, cy - hh + 34 * s)], PINK)
    for dx in (-0.45, 0.45): path_draw(fr, [(cx + dx * hw, cy - hh - 14 * s), (cx + dx * hw, cy - hh + 10 * s)], 1, WHITE, 5)
    n_flip = 6; fl = (lt - 0.4) / 0.22                                 # pages flip one after another
    if 0 < fl < n_flip:
        u = fl % 1; y_top = cy - hh + 34 * s; y = lerp(cy + hh, y_top, eio(u))
        fill_poly(fr, [(cx - hw + 4, y_top), (cx + hw - 4, y_top), (cx + hw - 4, y), (cx - hw + 4, y)], WHITE, 0.25 * (1 - u))
    for i in range(3):                                                  # day dots
        for j in range(4):
            x = cx - hw + (26 + j * 33) * s; y = cy - hh + (60 + i * 30) * s
            if s > 0.5: fill_poly(fr, [(x + 4 * np.cos(a), y + 4 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8)], WHITE, 0.6)
    if fl >= n_flip:
        q = eio((fl - n_flip) * 0.22 / 0.35)
        path_draw(fr, [(cx - 34, cy + 14), (cx - 8, cy + 40), (cx + 40, cy - 14)], q, PINK, 10)
def s_b(t):
    lt = t - T_B; fr = full(anon(V["4342"].at(22.0 + lt * 0.9)), 0.5, 0.4, lerp(1.12, 1.02, eio(lt / 1.5))); shade(fr, VIG); shade(fr, GBOT)
    chip(fr, "ДО", 40, 250, eo((lt - 0.2) / 0.3)); HB.draw(fr, t, T_B + 0.25); calendar(fr, t, T_B + 0.6)
    return fr

# C — the surgeon's plan; a checklist ticks every zone she names
HC = Drop(["КОМПЛЕКСНАЯ *КОРРЕКЦИЯ*"], 110, 70, 250)
ITEMS = [("ЛИПОСАКЦИЯ БОКОВ", T_FLANK), ("ЛИПОСАКЦИЯ ЖИВОТА", T_BELLY), ("АБДОМИНОПЛАСТИКА", T_ABDO), ("ЛИПОСАКЦИЯ ГРУДИ", T_CHEST)]
def checklist(fr, t, x=80, y0=1170, dy=96):
    for k, (lab, t0) in enumerate(ITEMS):
        lt = t - t0 + 0.15
        if lt < 0: continue
        y = y0 + k * dy; s = eback(lt / 0.3); b = 26 * s
        box = [(x + 26 - b, y + 26 - b), (x + 26 + b, y + 26 - b), (x + 26 + b, y + 26 + b), (x + 26 - b, y + 26 + b)]
        path_draw(fr, box, 1, PINK, 4, True)
        path_draw(fr, [(x + 12, y + 28), (x + 23, y + 40), (x + 44, y + 12)], eio((lt - 0.2) / 0.3), PINK, 6)
        a = alpha_img(lab, bebas(70)); p = eo((lt - 0.1) / 0.4); a2 = a.copy(); a2[:, int(a.shape[1] * p):] = 0
        blend(fr, a2, x + 80, y + 26 - a.shape[0] / 2 + 4, WHITE)
def s_c(t):
    lt = t - T_C; fr = full(V["4316"].at(src_of(t, "4316")), 0.5, 0.4, lerp(1.0, 1.06, eio(lt / 10)))
    shade(fr, GTOP); shade(fr, GBOT)
    band = np.clip(1 - np.abs(yy[:, :1] - 1610) / 90, 0, 1)[..., None] * 0.9; shade(fr, band)   # the burned-in subtitles of the source
    HC.draw(fr, t, T_C + 0.3); checklist(fr, t)
    return fr

# D — ДО/ПОСЛЕ slider on «всё стоило того»
HD = Drop(["ВСЁ СТОИЛО *ТОГО*"], 132, 70, 1520)
TD = Track(["«я знал, на что шёл»"], 36, 72, 0, 300)
def slider_x(lt): return W * (1 - 0.5 * eio((lt - 0.5) / 0.8) - 0.5 * eio((lt - 2.0) / 0.8))
def s_d(t):
    lt = t - T_D; xd = int(slider_x(lt)); fr = BEF.astype(np.float32) / 255; a = AFT.astype(np.float32) / 255
    fr[:, xd:] = a[:, xd:]; shade(fr, GBOT * 0.9)
    if 0 < xd < W:
        fr[230:1680, max(0, xd - 2):xd + 2] = PINK
        m = np.zeros((H, W), np.uint8); cv2.circle(m, (xd, 900), 32, 255, -1, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, PINK)
        ar = alpha_img("‹ ›", mont(36, 600)); blend(fr, ar, xd - ar.shape[1] / 2, 900 - ar.shape[0] / 2 - 2, NAVY)
    chip(fr, "ДО", 40, 250, eo((lt - 0.1) / 0.3) * cl((xd - 160) / 120)); chip(fr, "ПОСЛЕ", 875, 250, cl((W - 200 - xd) / 120))
    HD.draw(fr, t, T_D + 0.2); TD.y = HD.y + HD.h + 60 + TD.top; TD.draw(fr, t, T_D + 0.9)
    return fr

# E — three «2 недели» collages, flash + shutter on each; short pills for «я доволен» / «пока ещё с отёком»
T_K = [T_E, T_E + (T_END - T_E) / 3, T_E + 2 * (T_END - T_E) / 3]
T_EDEMA = vstart("E2")
def pill(fr, txt, t, t0, t1=1e9, y=1560):
    if t < t0 or t > t1 + 0.3: return
    op = eo((t - t0) / 0.3) * ((1 - eio((t - t1) / 0.3)) if t > t1 else 1)
    a = alpha_img(txt, bebas(78)); w = a.shape[1] + 60; h = 104; x = 540 - w / 2
    box = np.zeros((h, w), np.uint8); cv2.rectangle(box, (0, 0), (w - 1, h - 1), 255, -1); blend(fr, box.astype(np.float32) / 255, x, y, NAVY, 0.88 * op)
    ln = np.zeros((h, w), np.uint8); cv2.rectangle(ln, (1, 1), (w - 2, h - 2), 255, 2, cv2.LINE_AA); blend(fr, ln.astype(np.float32) / 255, x, y, PINK, op)
    blend(fr, a, x + 30 - 4, y + (h - a.shape[0]) / 2 + 4, WHITE, op)
def s_e(t):
    k = max(i for i, tk in enumerate(T_K) if t >= tk); lt = t - T_K[k]
    fr = COLLS[k].astype(np.float32) / 255
    chip(fr, "ПОСЛЕ", 30, 240); chip(fr, "ДО", 30, 1070)
    pill(fr, "Я ДОВОЛЕН", t, at_src("4355", 51.52) - 0.1, T_EDEMA - 0.2); pill(fr, "ПОКА ЕЩЁ С ОТЁКОМ", t, T_EDEMA + 0.1)
    f = 1 - eo(lt / 0.18); return fr * (1 - f) + f

# END — the quote again + logo
HZ = Drop(["ЛУЧШЕ РАНЬШЕ,", "*ЧЕМ ПОЗЖЕ*"], 150, 70, 560)
def s_end(t):
    lt = t - T_END; fr = BG.copy(); HZ.draw(fr, t, T_END + 0.2)
    w = 640; lg = logo_alpha(w); h = lg.shape[0]; x0, y0 = 540 - w / 2, 1180 - h / 2; sp = int(w * _split / _lg.shape[1])
    mk, wd = lg[:, :sp].copy(), lg[:, sp:].copy()
    y2, x2 = np.mgrid[0:mk.shape[0], 0:mk.shape[1]]; a2 = (np.arctan2(x2 - mk.shape[1] / 2, -(y2 - mk.shape[0] / 2)) % (2 * np.pi)) / (2 * np.pi)
    mk *= np.clip((eio((lt - 0.9) / 0.9) - a2) * 20, 0, 1); wd[:, int(wd.shape[1] * eio((lt - 1.3) / 0.8)):] = 0
    blend(fr, mk, x0, y0, PINK); blend(fr, wd, x0 + sp, y0, PINK)
    ty = Track(["пластический хирург"], 40, 540, 0, 300, WHITE, True); ty.y = y0 + h + 70 + ty.top; ty.draw(fr, t, T_END + 1.9)
    return fr

SCENES = [(0, s_a), (T_B, s_b), (T_C, s_c), (T_D, s_d), (T_E, s_e), (T_END, s_end)]
TRANS = {T_B: ("flash", 0.3), T_C: ("zoom", 0.5), T_D: ("iris", 0.6), T_E: ("push", 0.45), T_END: ("zoom", 0.5)}
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
M_DROP = 14.45; M0 = M_DROP - T_B                                      # the track's drop lands on the first ДО frame
G_VO, G_GAP = 0.022, 0.075
def track(n):
    w = wave.open(MUSIC); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64).reshape(-1, w.getnchannels()) / 32768
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    x = x / np.sqrt(np.mean(x[int(M_DROP * SR):].mean(1) ** 2)); x[:int(M_DROP * SR)] *= 4.5
    # the reel is longer than the track: loop bars 4..20 after the drop (137 bpm), seam matched by waveform, 0.3 s crossfade
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
    put(felt(70, 0.5), 0.0, 0.45); put(glass(1174.7), 0.05, 0.1)
    for k in range(6): put(paper(0.12, 4500), T_B + 0.6 + 0.4 + k * 0.22, 0.16)     # calendar pages
    put(marimba(PENTA[4], 0.5), T_B + 0.6 + 0.4 + 6 * 0.22, 0.22)                    # tick
    for k, (_, t0) in enumerate(ITEMS): put(marimba(PENTA[k + 1], 0.45), t0 + 0.05, 0.22)   # checklist ticks rise in pitch
    put(swell(0.9), T_D + 0.1, 0.06)
    for tk in T_K: put(click(), tk - 0.01, 1.4)                                       # «щёлк» on every collage
    put(glass(1174.7), T_END + 0.9, 0.2); put(glass(1760), T_END + 1.25, 0.12)
    for i, f in enumerate([PENTA[0], PENTA[2], PENTA[3], PENTA[5]]): put(marimba(f, 0.6), T_END + 1.9 + i * 0.12, 0.12)
    sfx[:n] *= 1 - 0.8 * np.clip(dd, 0, 1)
    mix = vo[:, None] + mus + sfx[:n, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (1.0 / 20); mix /= max(1, np.abs(mix).max() / 0.95)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

def make_cover(path):
    fr = BG.copy()
    for k, img in enumerate([BEF, AFT]):
        im = cv2.resize(img[200:1700], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x0 = k * 540; fr[180:180 + im.shape[0], x0:x0 + 540] = im[:, :540]
    fr[180:930, 538:542] = PINK; chip(fr, "ДО", 30, 210); chip(fr, "ПОСЛЕ", 570, 210)
    m = np.clip((yy[:, :1] - 780) / 200, 0, 1)[..., None]; fr = fr * (1 - m) + BG * m
    lg = logo_alpha(380); blend(fr, lg, 540 - 190, 60, PINK)
    hd = Drop(["ЛУЧШЕ РАНЬШЕ,", "*ЧЕМ ПОЗЖЕ*"], 190, 60, 1060, maxw=960); hd.draw(fr, 99, 0)
    ty = Track(["история пациента", "после комплексной коррекции"], 42, 64, 0, 400); ty.y = 1060 + hd.h + 90 + ty.top; ty.draw(fr, 99, 0)
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
