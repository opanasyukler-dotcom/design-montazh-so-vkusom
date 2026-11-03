#!/usr/bin/env python3
"""Reels «Наступил период любования собой» — the patient's path from the first interview to «прекрасный период любования
собой» (abdominoplasty + breast lipofilling), Анна Тваури.

Story, not a list of procedures: it opens on her line after recovery, goes back to ДО («хочется быть красивой»), the plan,
why she chose the surgeon, day 1, three weeks («результат ещё формируется»), then ПОСЛЕ and three ПОСЛЕ (top) / ДО (bottom)
collages from the neck down. A recovery timeline (ДО · 1 день · 3 недели · 1+ месяц) runs across the top. Faces are
blurred (YuNet boxes per frame), nipples carry small brand stickers (tracked in the ДО video), one-word subtitles,
synthesised sound effects of this reel's own palette (bubbles, shimmer, soft steps, rising progress tone, shutter).

usage:
  REEL_SRC=<folder with v32/v0301/v0319/v0410/v19 .mov, p2/p3/p12/p0319 .jpeg, IMG_9145-9147.JPG, logo.png, music.wav>
      python3 tools/reel_admire.py [--preview t1,t2 | --cover | --audio-only]
Prepared in $REEL_SRC/_work: <v>.mp4 (1080x1920, 30 fps), <v>.wav (48 kHz mono), <v>_faces.json (face box per frame),
v19_pasties.json (two tracked pasty centres per frame), nipples.json (sticker centres/radii on the photos);
person masks from rembg (isnet-general-use) in $REEL_SRC/_cut/m_<photo>.png.
"""
import os, sys, json, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_retouch import retouch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_admire"))
WORK = os.path.join(SRC, "_work"); CUT = os.path.join(SRC, "_cut")
NAME = "Reels_admire_yourself"
W, H, FPS, SR = 1080, 1920, 30, 48000
NAVY = np.float32([0, 12, 40]) / 255; PINK = np.float32([233, 197, 193]) / 255; WHITE = np.float32([1, 1, 1])
DEEP = np.float32([0, 5, 20]) / 255

def cl(x): return min(1.0, max(0.0, x))
def eo(x): x = cl(x); return 1 - (1 - x) ** 3
def eio(x): x = cl(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x): x = cl(x); c = 1.7; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p

# ---------------- voice edit: (block, source, [(in, out), …]) — whole sentences; inside a sentence only long silences are shortened
VO = [("A", "v0410", [(9.06, 10.06), (10.62, 12.68)]),                     # наступил прекрасный период любования собой (teaser)
      ("B", "v32", [(21.84, 23.50), (23.96, 26.10)]),                     # состояние, кожа, тело. хочется быть красивой
      ("C", "v32", [(4.76, 7.36)]),                                       # абдоминопластику и липофилинг груди
      ("D", "v32", [(37.48, 40.08), (40.74, 42.26), (42.64, 44.70)]),     # понравилось, как разговаривает, как объясняет. результат понравился, поэтому выбрала её
      ("E", "v0301", [(36.76, 37.62), (38.74, 40.14), (40.98, 42.00)]),   # уже хожу. ходила вот на перевязку. так, всё хорошо
      ("F", "v0319", [(21.98, 28.70)]),                                   # ну, пока результат оценить не могу в полном объёме
      ("G", "v0319", [(41.90, 43.10), (43.45, 48.86)]),                   # наверное, не так быстро, как хотелось бы, но идёт на поправку, уже намного лучше
      ("H", "v0410", [(7.00, 8.50), (9.06, 10.06), (10.62, 12.70)])]      # всё хорошо, всё отлично. наступил прекрасный период любования собой
GAP = {"B": 0.45, "C": 0.3, "D": 0.9, "E": 0.5, "F": 0.5, "G": 0.35, "H": 0.6}
SEG = []; _t = 0.04                                                        # the reel starts on the first word
for _b, _s, _parts in VO:
    if SEG: _t += GAP[_b]
    _d = sum(o - a for a, o in _parts); SEG.append((_b, _s, _parts, _t, _d)); _t += _d
def vstart(b): return next(s[3] for s in SEG if s[0] == b)
def vend(b): return next(s[3] + s[4] for s in SEG if s[0] == b)
def seg(b): return next(s for s in SEG if s[0] == b)
def TB(b, src_t):
    """reel time of source time `src_t` inside block b"""
    _, _, parts, t0, _ = seg(b); acc = 0
    for a, o in parts:
        if a <= src_t <= o: return t0 + acc + src_t - a
        acc += o - a
    raise ValueError((b, src_t))
def vsrc(b, t):
    """source time of block b's footage at reel time t (the picture keeps running before and after the line)"""
    _, _, parts, t0, d = seg(b); lt = t - t0
    if lt <= 0: return parts[0][0] + lt
    acc = 0
    for a, o in parts:
        if lt <= acc + o - a: return a + lt - acc
        acc += o - a
    return parts[-1][1] + lt - acc
T_S = {b: vstart(b) - 0.1 for b, *_ in VO}
T_H2 = TB("H", 9.06) - 0.05                                                # «наступил…»: ПОСЛЕ
T_K = [vend("H") + 0.4 + 1.8 * k for k in range(3)]                        # three collages, 1.8 s each, shutter on every cut
T_Z = T_K[-1] + 2.6; DUR = T_Z

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


V = {k: Src(os.path.join(WORK, f"{k}.mp4")) for k in ["v32", "v0301", "v0319", "v0410", "v19"]}
FACES = {k: np.array(json.load(open(os.path.join(WORK, f"{k}_faces.json")))) for k in ["v32", "v0301", "v0319", "v0410"]}
PASTIES = np.array(json.load(open(os.path.join(WORK, "v19_pasties.json"))))

# ---------- brand sticker over the nipples (small: just over the nipple / the pasty) ----------
_STK = None
def sticker_rgba(r):
    """the brand sticker (the client's round «ANNA TVAURI · пластический хирург» artwork), scaled DOWN to radius r; float RGBA"""
    global _STK
    if _STK is None:
        im = cv2.imread(os.path.join(SRC, "sticker.png"), cv2.IMREAD_UNCHANGED); _STK = np.dstack([im[..., 2::-1], im[..., 3]]).astype(np.float32) / 255
    sz = 2 * r + 2; return cv2.resize(_STK, (sz, sz), interpolation=cv2.INTER_AREA)
_SC = {}
def put_sticker(img, cx, cy, r):
    """composite a sticker onto a float or uint8 RGB image in place (with a soft contact shadow)"""
    r = int(round(r))
    if r not in _SC: _SC[r] = sticker_rgba(r)
    s = _SC[r]; h, w = s.shape[:2]; x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
    if xb <= xa or yb <= ya: return
    a = s[ya - y0:yb - y0, xa - x0:xb - x0, 3:4]; c = s[ya - y0:yb - y0, xa - x0:xb - x0, :3]
    sh = cv2.GaussianBlur(s[..., 3], (0, 0), max(1, r * 0.08))[ya - y0:yb - y0, xa - x0:xb - x0, None] * 0.25
    reg = img[ya:yb, xa:xb]
    if img.dtype == np.uint8:
        f = reg.astype(np.float32) / 255; f = f * (1 - sh); f = f * (1 - a) + c * a; reg[:] = (f * 255 + 0.5).astype(np.uint8)
    else: reg *= 1 - sh; reg *= 1 - a; reg += c * a

# ---------- faces: soft, heavy blur on the tracked face box ----------
def blur_face(img, box, grow=1.3):
    x, y, w, h = box; cx, cy = x + w / 2, y + h / 2 - 0.05 * h; rx, ry = w * grow / 2, h * grow / 2
    x0, y0, x1, y1 = int(max(0, cx - rx * 1.2)), int(max(0, cy - ry * 1.2)), int(min(W, cx + rx * 1.2)), int(min(H, cy + ry * 1.2))
    reg = img[y0:y1, x0:x1]
    small = cv2.resize(reg, (max(2, reg.shape[1] // 16), max(2, reg.shape[0] // 16)), interpolation=cv2.INTER_AREA)
    bl = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2), (reg.shape[1], reg.shape[0]), interpolation=cv2.INTER_LINEAR)
    yy_, xx_ = np.mgrid[y0:y1, x0:x1]; d = ((xx_ - cx) / rx) ** 2 + ((yy_ - cy) / ry) ** 2
    m = (np.clip((1.3 - d) / 0.75, 0, 1) ** 1.5)[..., None]                                 # wide soft edge
    out = img.copy(); out[y0:y1, x0:x1] = (reg * (1 - m) + bl * m).astype(img.dtype); return out
def interview(k, src_t):
    i = int(round(src_t * FPS)); f = V[k].at(src_t); return blur_face(f, FACES[k][min(max(i, 0), len(FACES[k]) - 1)])
def anon_top(img, y=150):
    top = img[:y + 80].astype(np.float32); bl = cv2.resize(cv2.GaussianBlur(cv2.resize(top, None, fx=0.1, fy=0.1), (0, 0), 3), (W, y + 80))
    m = np.clip((y + 40 - np.arange(y + 80)) / 80, 0, 1)[:, None, None]; out = img.copy(); out[:y + 80] = (top * (1 - m) + bl * m).astype(np.uint8); return out
def v19_frame(src_t):
    i = min(max(int(round(src_t * FPS)), 0), len(PASTIES) - 1); f = anon_top(V["v19"].at(src_t), 140).copy()
    for k in range(2): put_sticker(f, PASTIES[i][2 * k], PASTIES[i][2 * k + 1], 60)
    return f

# ---------- photos: stickers at source resolution ----------
NIP = {"p2": [(1030, 1633, 100), (1894, 1588, 100)], "p3": [(1726, 1519, 96), (2376, 1472, 76)], "p12": [(2230, 1452, 64)],   # centred on the pasties / nipples
       "p0319": [(1024, 1260, 100), (2016, 1276, 100)],
       "IMG_9145": [(392, 673, 58), (1114, 665, 58)], "IMG_9146": [(822, 648, 56), (1306, 660, 44)], "IMG_9147": [(1105, 696, 44)]}
def photo(n):
    p = os.path.join(SRC, n + (".JPG" if n.startswith("IMG") else ".jpeg"))
    im = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB")).copy()
    im = retouch(im, 0.3, 0.45, 0.06).copy()                              # light skin retouch: softer underwear marks and unevenness, texture kept
    for x, y, r in NIP.get(n, []): put_sticker(im, x, y, r)
    return im
PH = {n: photo(n) for n in NIP}
class Still:
    """a photo shown full frame: scaled DOWN once (src rows from `top`), then gently zoomed around (cx, cy) of the frame"""
    def __init__(s, n, top, cx_src, zmax=1.1):
        im = PH[n]; s.sc = H / (im.shape[0] - top) * zmax; s.zmax = zmax
        s.img = cv2.resize(im[top:], None, fx=s.sc, fy=s.sc, interpolation=cv2.INTER_AREA); s.cx = cx_src * s.sc; s.top = top
    def at(s, z=1.0, fy=0.5):
        k = z / s.zmax; ox = s.cx - W / 2 / k; oy = (s.img.shape[0] - H / k) * fy
        ox = min(max(0, ox), s.img.shape[1] - W / k)
        M = np.float32([[k, 0, -ox * k], [0, k, -oy * k]])
        return cv2.warpAffine(s.img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255
    def pt(s, x, y, z=1.0, fy=0.5):
        """frame position of a source point (for zones / stickers drawn on top)"""
        k = z / s.zmax; ox = min(max(0, s.cx - W / 2 / k), s.img.shape[1] - W / k); oy = (s.img.shape[0] - H / k) * fy
        return ((x * s.sc - ox) * k, ((y - s.top) * s.sc - oy) * k)
P_A, P_C, P_G, P_H = Still("IMG_9146", 0, 760), Still("p2", 250, 1443), Still("p0319", 200, 1511), Still("IMG_9145", 0, 753)

# ---------- collages: ПОСЛЕ on top, ДО below; person cut out (rembg) onto navy; aligned by neck and navel ----------
BODY = {"p2": (430, 2530), "p3": (430, 2545), "p12": (450, 2550), "IMG_9145": (0, 1330), "IMG_9146": (0, 1240), "IMG_9147": (0, 1330)}
LEGS = {"p2": 3150, "p3": 3540, "p12": 3540}                                 # ДО: the stockings start below these rows
PANEL_BG = {}
def panel(n, pw=W, ph=960, neck_at=70, span=600):
    if (ph, pw) not in PANEL_BG:
        g = np.linspace(0, 1, ph)[:, None, None]; PANEL_BG[ph, pw] = (NAVY * (1 - g * 0.6) + DEEP * g * 0.6 + np.zeros((ph, pw, 3))).astype(np.float32)
    im = PH[n].astype(np.float32) / 255; a = cv2.imread(os.path.join(CUT, f"m_{n}.png"), 0).astype(np.float32) / 255
    neck, navel = BODY[n]; s = span / (navel - neck)
    rows = np.arange(a.shape[0])[:, None]; a = a * np.clip((rows - (neck - 60)) / 60, 0, 1)            # nothing above the neck
    if n in LEGS: a = a * np.clip((LEGS[n] - rows) / 90, 0, 1)                                    # nor the stockings
    for x, y, r in NIP.get(n, []): cv2.circle(a, (int(x), int(y)), int(r) + 1, 1.0, -1, cv2.LINE_AA)       # stickers stay whole even past the body edge
    band = a[neck:navel]; cols = np.arange(a.shape[1]); cx = float((band.sum(0) * cols).sum() / max(1, band.sum()))
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)[..., None]
    ox, oy = int(cx * s - pw / 2), int(neck * s - neck_at); out = PANEL_BG[ph, pw].copy()
    xs0, ys0 = max(0, ox), max(0, oy); xs1, ys1 = min(im.shape[1], ox + pw), min(im.shape[0], oy + ph)
    sub = out[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]; aa = a[ys0:ys1, xs0:xs1]; sub[:] = sub * (1 - aa) + im[ys0:ys1, xs0:xs1] * aa
    return out
COLL = [np.vstack([panel(a), panel(b)]) for a, b in (("IMG_9145", "p2"), ("IMG_9146", "p3"), ("IMG_9147", "p12"))]
COLL8 = [(np.clip(c, 0, 1) * 255).astype(np.uint8) for c in COLL]


# ---------------- subtitles: hand-checked words, timings from speech recognition on the cut lines (relative to each line's start)
SUBS = {
    'A': [('Наступил', 0.08, 1.08), ('прекрасный', 1.08, 1.7), ('период', 1.7, 2.06), ('любования', 2.06, 2.58), ('собой.', 2.58, 2.94)],
    'B': [('Состояние,', 0.04, 1.04), ('кожа,', 1.26, 1.66), ('тело.', 1.78, 2.5), ('Хочется', 2.5, 2.9), ('быть', 2.9, 3.08), ('красивой.', 3.08, 3.64)],
    'C': [('Абдоминопластику', 0.04, 1.4), ('и', 1.4, 1.64), ('липофилинг', 1.64, 2.14), ('груди.', 2.14, 2.48)],
    'D': [('Понравилось,', 0.04, 0.64), ('как', 0.84, 0.94), ('разговаривает,', 0.94, 1.68), ('как', 1.88, 1.92), ('объясняет.', 1.92, 2.66), ('Результат', 2.8, 3.22), ('понравился,', 3.22, 4.14), ('поэтому', 4.54, 5.08), ('выбрала', 5.08, 5.76), ('её.', 5.76, 5.92)],
    'E': [('Уже', 0.16, 0.52), ('хожу.', 0.52, 0.88), ('Ходила', 0.92, 1.26), ('вот', 1.26, 1.42), ('на', 1.42, 1.62), ('перевязку.', 1.62, 2.22), ('Так,', 2.4, 2.52), ('всё', 2.6, 2.74), ('хорошо.', 2.74, 3.12)],
    'F': [('Ну,', 0.04, 0.44), ('пока', 1.2, 1.62), ('результат', 1.62, 2.62), ('оценить', 2.62, 3.5), ('не', 3.5, 5.02), ('могу', 5.02, 5.44), ('в', 5.44, 5.8), ('полном', 5.8, 6.1), ('объёме.', 6.1, 6.64)],
    'G': [('Наверное,', 0.04, 0.9), ('не', 0.9, 1.94), ('так', 1.94, 2.16), ('быстро,', 2.16, 2.44), ('как', 2.44, 2.64), ('хотелось', 2.64, 3.14), ('бы,', 3.14, 3.44), ('но', 3.44, 3.7), ('идёт', 3.7, 4.4), ('на', 4.4, 4.88), ('поправку,', 4.88, 5.34), ('уже', 5.34, 5.68), ('намного', 5.68, 6.06), ('лучше.', 6.06, 6.34)],
    'H': [('Всё', 0.04, 0.24), ('хорошо,', 0.24, 0.64), ('всё', 0.78, 0.96), ('отлично.', 0.96, 1.56), ('Наступил', 1.82, 2.58), ('прекрасный', 2.58, 3.18), ('период', 3.18, 3.54), ('любования', 3.54, 4.08), ('собой.', 4.08, 4.4)],
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


# ---------------- reel-specific pieces ----------------
def fit_size(txt, maxw, size):
    while bebas(size).getlength(txt) > maxw and size > 30: size -= 2
    return size
def pillx(fr, txt, x, y, p, tick_p=0.0, size=30):
    """navy pill with a pink tick, pops in (p 0→1)"""
    if p <= 0: return
    f = mont(size, 600); a = alpha_img(txt, f); s = eback(p); w, h = a.shape[1] + 86, 64
    blend(fr, rrect(x, y, x + w * s, y + h * s, 32 * s), 0, 0, NAVY, 0.9 * min(1, p * 2))
    rrect_line(fr, x, y, x + w * s, y + h * s, 32 * s, PINK, 2, min(1, p * 2))
    if s > 0.9: blend(fr, a, x + 64, y + h / 2 - a.shape[0] / 2 + 2, WHITE, cl((p - 0.5) * 2)); path_draw(fr, [(x + 22, y + 33), (x + 32, y + 43), (x + 48, y + 22)], tick_p, PINK, 5)
STOPS = [(170, "ДО"), (400, "1 ДЕНЬ"), (630, "3 НЕДЕЛИ"), (860, "1+ МЕСЯЦ")]
def timeline(fr, t, stage, t0, y=292):
    """recovery path across the top: the pink line runs to the current stop, which pulses"""
    op = eo((t - (T_S["B"] if stage == 0 else t0 - 0.4)) / 0.4) if stage == 0 else 1.0
    shade(fr, ((1 - ramp(250, 470)) * 0.55).astype(np.float32))
    x0, x1 = STOPS[0][0], STOPS[-1][0]
    path_draw(fr, [(x0, y), (x1, y)], 1, WHITE, 3, op=0.35 * op)
    xprev = STOPS[max(0, stage - 1)][0]; xcur = lerp(xprev, STOPS[stage][0], eio((t - t0) / 0.7)) if stage else x0
    if xcur > x0: path_draw(fr, [(x0, y), (xcur, y)], 1, PINK, 5, op=op)
    for k, (x, lab) in enumerate(STOPS):
        m = np.zeros((H, W), np.uint8)
        if k < stage or (k == stage and xcur >= x - 1):
            r = 11 if k < stage else int(13 + 3 * np.sin(t * 6)); cv2.circle(m, (x, y), r, 255, -1, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, PINK, op)
            if k == stage: u = (t * 1.2) % 1; ring(fr, x, y, 16 + 18 * u, op * (1 - u), 2)
        else:
            cv2.circle(m, (x, y), 9, 255, 2, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, WHITE, 0.6 * op)
        f = mont(27, 700 if k == stage else 500); a = alpha_img(lab, f)
        blend(fr, a, x - a.shape[1] / 2, y + 22, PINK if k == stage else WHITE, op * (1 if k == stage else 0.6))
def sweep(fr, t, t0, d=1.0, x_from=-400, x_to=1500, width=160, k=0.22, mask=None):
    """diagonal light sweep across the picture"""
    if not (t0 <= t <= t0 + d): return
    x = lerp(x_from, x_to, eio((t - t0) / d)); dd = (xx + yy * 0.35 - x - 336) / width
    g = np.exp(-dd ** 2)[..., None] * k * np.sin(np.pi * (t - t0) / d)
    if mask is not None: g = g * mask
    fr += g * (1 - fr)
def footprints(fr, t, t0, n=4):
    for i in range(n):
        lt = t - t0 - 0.2 * i
        if lt < 0: continue
        x = 150 + (i % 2) * 70; y = 1060 - i * 95; s = eback(lt / 0.25)
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (int(x), int(y)), (int(17 * s), int(30 * s)), -12 if i % 2 else 12, 0, 360, 255, -1, cv2.LINE_AA)
        for j, (dx, dy, r) in enumerate([(-12, -40, 6), (-2, -46, 6), (8, -44, 5), (16, -38, 5)]): cv2.circle(m, (int(x + dx * s), int(y + dy * s)), int(r * s), 255, -1, cv2.LINE_AA)
        blend(fr, m.astype(np.float32) / 255, 0, 0, PINK, 0.85 * (1 - cl((lt - 2.2) / 0.5)))
def progress(fr, t, p, x0=80, y0=1262, x1=760, h=28, op=1.0):
    blend(fr, rrect(x0, y0, x1, y0 + h, h // 2), 0, 0, WHITE, 0.18 * op)
    if p > 0.01:
        xe = x0 + (x1 - x0) * p; blend(fr, rrect(x0, y0, max(x0 + h, xe), y0 + h, h // 2), 0, 0, PINK, 0.95 * op)
        sx = x0 + ((t * 260) % max(1, xe - x0)); sh = np.exp(-((xx[y0:y0 + h] - sx) / 30) ** 2)[..., None] * 0.5
        reg = fr[y0:y0 + h, int(x0):int(xe)]; reg += sh[:, int(x0):int(xe)] * (1 - reg) * op
def up_arrow(fr, t, t0, x=860, y=1080):
    lt = t - t0
    if lt < 0: return
    p = eio(lt / 0.5); yy0 = y + 60 - 120 * p
    path_draw(fr, [(x, y + 60), (x, yy0)], 1, PINK, 9); fill_poly(fr, [(x - 26, yy0 + 18), (x + 26, yy0 + 18), (x, yy0 - 18)], PINK, min(1, p * 2))

# ---------------- scenes ----------------
# A — opening: ДО → ПОСЛЕ slider on the full photos (their own backdrop, nothing cut), aligned by neck and navel
HA = Drop(["НАСТУПИЛ ПЕРИОД", "*ЛЮБОВАНИЯ СОБОЙ*"], 104, 70, 300, maxw=860)
TA = Track(["больше месяца после операции"], 32, 72, 0, 300)
SL_B = SL_A = None; T_SW = (0.55, 1.75)
def full_aligned(n, neck_at=225, span=1100):
    """the whole photo scaled DOWN so neck→navel = `span`, neck at `neck_at`, body centred; any uncovered strip takes the photo's backdrop"""
    im = PH[n]; neck, navel = BODY[n]; s = span / (navel - neck)
    a = cv2.imread(os.path.join(CUT, f"m_{n}.png"), 0).astype(np.float32) / 255; band = a[neck:navel]
    cx = float((band.sum(0) * np.arange(a.shape[1])).sum() / max(1, band.sum()))
    r = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); bgc = np.median(np.concatenate([r[:40, :40].reshape(-1, 3), r[:40, -40:].reshape(-1, 3)]), 0)
    out = np.zeros((H, W, 3), np.float32) + bgc; ox, oy = int(cx * s - W / 2), int(neck * s - neck_at)
    xs0, ys0 = max(0, ox), max(0, oy); xs1, ys1 = min(r.shape[1], ox + W), min(r.shape[0], oy + H)
    out[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox] = r[ys0:ys1, xs0:xs1]
    if ys0 - oy > 0:                                                        # a strip above the photo: soft hand-over to its backdrop
        k = ys0 - oy; g = np.clip((np.arange(k + 60) - k) / 60, 0, 1)[:, None, None]; out[:k + 60] = out[:k + 60] * g + bgc * (1 - g)
    return out / 255
def s_a(t):
    global SL_B, SL_A
    if SL_B is None: SL_B = anon_top((full_aligned("p2") * 255).astype(np.uint8), 200).astype(np.float32) / 255; SL_A = full_aligned("IMG_9145")
    z = lerp(1.04, 1.0, eo(t / 3.0)); xd = int(W * (1 - eio((t - T_SW[0]) / (T_SW[1] - T_SW[0]))))
    fr = SL_B.copy(); fr[:, xd:] = SL_A[:, xd:]
    if z != 1: fr = cv2.warpAffine(fr, np.float32([[z, 0, 540 * (1 - z)], [0, z, 900 * (1 - z)]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
    shade(fr, ((1 - ramp(250, 640)) * 0.6).astype(np.float32)); shade(fr, (ramp(1330, 1560) * 0.6).astype(np.float32))   # light, only where the text sits
    if 0 < xd < W:
        fr[:, max(0, xd - 3):xd + 3] = PINK
        m = np.zeros((H, W), np.uint8); cv2.circle(m, (xd, 900), 32, 255, -1, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, PINK)
        ar = alpha_img("‹ ›", mont(34, 600)); blend(fr, ar, xd - ar.shape[1] / 2, 900 - ar.shape[0] / 2 - 2, NAVY)
    chip(fr, "ДО", 40, 1300, cl((xd - 200) / 100)); chip(fr, "ПОСЛЕ", 40, 1300, cl((160 - xd) / 100))
    if t > T_SW[1]:
        for k, (x, y) in enumerate([(330, 760), (780, 800), (560, 640)]): sparkle(fr, x, y, 26, t, T_SW[1] + 0.2 + 0.18 * k)
    HA.draw(fr, t, 0.1); TA.y = HA.y + HA.h + 36 + TA.top; TA.draw(fr, t, 0.8)
    return fr

# B — back to ДО: «состояние, кожа, тело. хочется быть красивой»
HB = lift(Drop(["С ЭТОГО ВСЁ", "*НАЧИНАЛОСЬ*"], 124, 70, 0, maxw=860), 1300)
def s_b(t):
    t0 = T_S["B"]; lt = t - t0; fr = full(v19_frame(0.3 + lt), 0.5, 0.5, lerp(1.06, 1.0, eo(lt / 1.4)) + bump(t, vstart("B") + 2.5, 0.03))
    shade(fr, VIG); shade(fr, GBOT); timeline(fr, t, 0, t0); chip(fr, "ДО", 40, 380)
    pillx(fr, "кожа", 600, 640, eo((t - vstart("B") - 1.22) / 0.3) * (1 - cl((t - vstart("B") - 3.9) / 0.3)), eio((t - vstart("B") - 1.4) / 0.3))
    pillx(fr, "тело", 250, 760, eo((t - vstart("B") - 1.74) / 0.3) * (1 - cl((t - vstart("B") - 3.9) / 0.3)), eio((t - vstart("B") - 1.9) / 0.3))
    HB.draw(fr, t, t0 + 0.15); return fr

# C — the plan on the ДО photo: belly zone and both breasts light up as she names them
HC = Drop(["ПЛАН *ОПЕРАЦИИ*"], 104, 70, 300, maxw=860)
ITEMS = [("АБДОМИНОПЛАСТИКА", vstart("C") + 0.05), ("ЛИПОФИЛИНГ ГРУДИ", vstart("C") + 1.6)]
def checklist(fr, t, x=80, y0=420, dy=80, size=62):
    for k, (lab, t0) in enumerate(ITEMS):
        lt = t - t0
        if lt < 0: continue
        y = y0 + k * dy; s = eback(lt / 0.3); b = 22 * s
        path_draw(fr, [(x + 22 - b, y + 22 - b), (x + 22 + b, y + 22 - b), (x + 22 + b, y + 22 + b), (x + 22 - b, y + 22 + b)], 1, PINK, 4, True)
        path_draw(fr, [(x + 10, y + 23), (x + 19, y + 33), (x + 36, y + 10)], eio((lt - 0.15) / 0.3), PINK, 5)
        a = alpha_img(lab, bebas(size)); p = eo((lt - 0.1) / 0.35); a2 = a.copy(); a2[:, int(a.shape[1] * p):] = 0; blend(fr, a2, x + 64, y + 22 - a.shape[0] / 2 + 5, WHITE)
def s_c(t):
    t0 = T_S["C"]; lt = t - t0; z = lerp(1.0, 1.05, eio(lt / 4)); fr = P_C.at(z, 0.0)
    shade(fr, ((1 - ramp(240, 640)) * 0.85).astype(np.float32)); shade(fr, VIG)
    q1 = eio((t - ITEMS[0][1] - 0.1) / 0.6); q2 = eio((t - ITEMS[1][1] - 0.1) / 0.6)
    if q1 > 0:                                                            # the belly: a dashed rounded frame that pulses
        bx, by = P_C.pt(1443, 2560, z, 0.0); w_, h_ = 300 * (1 + 0.03 * np.sin(t * 6)), 170 * (1 + 0.03 * np.sin(t * 6))
        blend(fr, cv2.GaussianBlur(rrect(bx - w_, by - h_, bx + w_, by + h_, 60), (0, 0), 8), 0, 0, PINK, 0.18 * q1)
        perim = 2 * (2 * w_ + 2 * h_); n = 40
        pts = [(bx - w_, by - h_), (bx + w_, by - h_), (bx + w_, by + h_), (bx - w_, by + h_), (bx - w_, by - h_)]
        Ls = np.r_[0, np.cumsum([np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(4)])]
        for j in range(int(n * q1)):
            if j % 2: continue
            dd = np.linspace(j * Ls[-1] / n, (j + 1) * Ls[-1] / n, 3); px = np.interp(dd, Ls, [p[0] for p in pts]); py = np.interp(dd, Ls, [p[1] for p in pts])
            path_draw(fr, list(zip(px, py)), 1, PINK, 5)
    for k, (x, y, r) in enumerate(NIP["p2"]):                             # the breasts: glowing rings
        if q2 <= 0: break
        cx, cy = P_C.pt(x, y, z, 0.0); rr = 150 * (1 + 0.04 * np.sin(t * 6 + k))
        ring(fr, cx, cy, rr, q2, 5); sparkle(fr, cx + rr * 0.7, cy - rr * 0.7, 22, t, ITEMS[1][1] + 0.3 + 0.15 * k)
    HC.draw(fr, t, t0 + 0.1); checklist(fr, t); return fr

# D — choosing the surgeon: reasons tick in as she names them
HD = lift(Drop(["Я ВЫБРАЛА ВРАЧА", "*ИНТУИТИВНО*"], 96, 70, 0, maxw=860), 1255)
TD = Track(["первая пластическая операция"], 32, 72, 0, 300)
REASONS = [("как разговаривает", 0.94), ("как объясняет", 1.92), ("результат", 2.8)]
def s_d(t):
    t0 = T_S["D"]; fr = full(interview("v32", vsrc("D", t)), 0.5, 0.45, pull(t, t0, 1.12, 1.0, 1.6)); shade(fr, GBOT); shade(fr, GBOT * 0.4)
    for k, (lab, rel) in enumerate(REASONS):
        tk = vstart("D") + rel - 0.1; pillx(fr, lab, 520, 760 + k * 80, eo((t - tk) / 0.3), eio((t - tk - 0.25) / 0.3))
    hk = vstart("D") + 4.54
    if t > hk:
        u = t - hk; s = eback(u / 0.35) * 3.4; fill_poly(fr, heart_pts(880, 740 - 20 * cl(u / 2), s), PINK, 0.95)
    HD.draw(fr, t, t0 + 0.15); TD.y = HD.y + HD.h + 40 + TD.top; TD.draw(fr, t, t0 + 0.8)
    return fr

# E — day 1: «уже хожу» (footsteps), «на перевязку» (plaster), «всё хорошо»
HE = lift(Drop(["1 ДЕНЬ", "*ПОСЛЕ ОПЕРАЦИИ*"], 100, 70, 0, maxw=860), 1255)
TE = Track(["восстановление только начинается"], 32, 72, 0, 300)
def s_e(t):
    t0 = T_S["E"]; fr = full(interview("v0301", vsrc("E", t)), 0.5, 0.45, pull(t, t0, 1.12, 1.0, 1.6)); shade(fr, GBOT); shade(fr, GBOT * 0.4)
    timeline(fr, t, 1, t0 + 0.2); footprints(fr, t, vstart("E") + 0.1)
    tp = vstart("E") + 1.6
    if t > tp: icon(fr, "plus", 860, 760, 0.55 * eback((t - tp) / 0.3), eo((t - tp) / 0.4))
    HE.draw(fr, t, t0 + 0.15); TE.y = HE.y + HE.h + 40 + TE.top; TE.draw(fr, t, t0 + 0.8)
    return fr

# F — three weeks: «пока результат оценить не могу» — the bar is still loading
HF = Drop(["3 *НЕДЕЛИ*"], 140, 70, 1010, maxw=860); HF2 = Drop(["РЕЗУЛЬТАТ ЕЩЁ ФОРМИРУЕТСЯ"], 66, 70, 1150, maxw=860)
def s_f(t):
    t0 = T_S["F"]; fr = full(interview("v0319", vsrc("F", t)), 0.5, 0.45, pull(t, t0, 1.12, 1.0, 1.6)); shade(fr, GBOT); shade(fr, GBOT * 0.5)
    timeline(fr, t, 2, t0 + 0.2); HF.draw(fr, t, t0 + 0.15); HF2.draw(fr, t, t0 + 0.6)
    progress(fr, t, 0.55 * eio((t - t0 - 0.9) / 5.5), op=eo((t - t0 - 0.8) / 0.3)); return fr
# G — «не так быстро, как хотелось бы, но идёт на поправку»: the 3-week photo, the bar moves on, an arrow up
def s_g(t):
    t0 = T_S["G"]; lt = t - t0; z = lerp(1.0, 1.06, eio(lt / 6)); fr = P_G.at(z, 0.0); shade(fr, GBOT); shade(fr, GBOT * 0.5)
    timeline(fr, t, 2, -10); HF.draw(fr, t, -10); HF2.draw(fr, t, -10)
    progress(fr, t, 0.55 + 0.3 * eio((t - vstart("G") - 3.4) / 2.6)); up_arrow(fr, t, vstart("G") + 5.6)
    return fr

# H — a month on: «всё хорошо, всё отлично» (interview) → «наступил…» on ПОСЛЕ with the big line
HH = lift(Drop(["И ВОТ ТЕПЕРЬ УЖЕ", "*МОЖНО ЛЮБОВАТЬСЯ*"], 116, 70, 0, maxw=860), 1290)
def s_h(t):
    t0 = T_S["H"]
    if t < T_H2:
        fr = full(interview("v0410", vsrc("H", t)), 0.5, 0.45, pull(t, t0, 1.1, 1.0, 1.4)); shade(fr, GBOT); timeline(fr, t, 3, t0 + 0.2); return fr
    lt = t - T_H2; z = lerp(1.0, 1.07, eio(lt / 4)); fr = P_H.at(z, 0.5); shade(fr, GBOT); shade(fr, GBOT * 0.4)
    sweep(fr, t, T_H2 + 0.6, 1.1); timeline(fr, t, 3, -10)
    for k, (x, y) in enumerate([(300, 640), (800, 600), (540, 900)]): sparkle(fr, x, y, 28, t, T_H2 + 1.0 + 0.2 * k)
    HH.draw(fr, t, T_H2 + 0.05)
    f = 1 - eo(lt / 0.18); return fr * (1 - f) + f

# K — three collages, ПОСЛЕ on top / ДО below, shutter on every cut
def s_k(t):
    k = max(i for i, tk in enumerate(T_K) if t >= tk - 1e-6); lt = t - T_K[k]
    fr = full(COLL8[k], 0.5, 0.5, 1 + 0.03 * (1 - eo(lt / 0.35))); fr[957:963] = PINK
    chip(fr, "ПОСЛЕ", 40, 280); chip(fr, "ДО", 40, 1000)
    f = 1 - eo(lt / 0.18) if k else 0; return fr * (1 - f) + f

# Z — the closing frame: ПОСЛЕ cut out on navy with a soft glow
FIN = None
def fin_img():
    global FIN
    if FIN is None:
        im = PH["IMG_9145"].astype(np.float32) / 255; a = cv2.imread(os.path.join(CUT, "m_IMG_9145.png"), 0).astype(np.float32) / 255
        s = 1120 / im.shape[0]; im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)[..., None]
        FIN = (im, a, int(753 * s))
    return FIN
HZ = lift(Drop(["АБДОМИНОПЛАСТИКА +", "*ЛИПОФИЛИНГ ГРУДИ*"], 104, 70, 0, maxw=860), 1330)
TZ = Track(["показываем результат больше чем через месяц"], 30, 72, 0, 300)
def s_z(t):
    lt = t - T_Z; fr = gbg(t); im, a, cx = fin_img(); y0 = 380 - int(30 * (1 - eo(lt / 1.2))); x0 = 540 - cx
    glow = np.exp(-(((xx - 540) / 380) ** 2 + ((yy - 820) / 520) ** 2))[..., None] * 0.35 * eo(lt / 0.8); fr += glow * (PINK - fr)
    h_, w_ = im.shape[:2]; xa, xb = max(0, x0), min(W, x0 + w_); ya, yb = max(0, y0), min(H, y0 + h_)
    sub = fr[ya:yb, xa:xb]; aa = a[ya - y0:yb - y0, xa - x0:xb - x0] * eo(lt / 0.5); sub[:] = sub * (1 - aa) + im[ya - y0:yb - y0, xa - x0:xb - x0] * aa
    shade(fr, (ramp(1080, 1400) * 0.8).astype(np.float32))
    blend(fr, logo_alpha(280), 540 - 140, 290, PINK, eo((lt - 0.2) / 0.5))
    HZ.draw(fr, t, T_Z + 0.3); TZ.y = HZ.y + HZ.h + 40 + TZ.top; TZ.draw(fr, t, T_Z + 1.0)
    return fr

SCENES = [(0, s_a), (T_S["B"], s_b), (T_S["C"], s_c), (T_S["D"], s_d), (T_S["E"], s_e), (T_S["F"], s_f), (T_S["G"], s_g), (T_S["H"], s_h), (T_K[0], s_k)]
TRANS = {T_S["B"]: ("zoom", 0.35), T_S["C"]: ("flash", 0.2), T_S["D"]: ("zoom", 0.35), T_S["E"]: ("diag", 0.45), T_S["F"]: ("zoom", 0.35),
         T_S["G"]: ("flash", 0.2), T_S["H"]: ("push", 0.4), T_K[0]: ("iris", 0.5)}
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
M_DROP = 14.45; M0 = M_DROP - T_S["B"]                                  # the drop lands on the first ДО frame
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


# this reel's own palette: water bubbles, glitter, soft steps, kalimba, a rising tone for the progress bar, a camera shutter
def bubble(f0=380, f1=900, dur=0.14):
    n = int(dur * SR); t = np.arange(n) / SR; ph = 2 * np.pi * np.cumsum(f0 + (f1 - f0) * (t / dur) ** 0.6) / SR
    return np.sin(ph) * env(n, 0.003, 0.045)
def glitter(dur=0.9, k=14):
    n = int(dur * SR); out = np.zeros(n); r = np.random.default_rng(7)
    for _ in range(k):
        f = r.uniform(3000, 7000); i = int(r.uniform(0, dur * 0.75) * SR); m = int(0.12 * SR); t = np.arange(m) / SR
        g = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.03) * r.uniform(0.4, 1); out[i:i + m] += g[:n - i]
    return out / (np.abs(out).max() + 1e-9)
def step():
    n = int(0.16 * SR); t = np.arange(n) / SR
    return lp(rng.standard_normal(n), 500) * env(n, 0.004, 0.035) * 1.6 + np.sin(2 * np.pi * 70 * t) * env(n, 0.002, 0.05) * 0.6
def kalimba(f=880, dur=0.7):
    n = int(dur * SR); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.35) + 0.25 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t / 0.04)) * env(n, 0.002, 9)
def rise(dur=2.0, f0=330, f1=660):
    n = int(dur * SR); t = np.arange(n) / SR; ph = 2 * np.pi * np.cumsum(f0 * (f1 / f0) ** (t / dur)) / SR
    return (np.sin(ph) + 0.3 * np.sin(2 * ph)) * np.sin(np.pi * t / dur) ** 2 * 0.5
def shutter_cam():
    n = int(0.2 * SR); x = np.zeros(n)
    for t0, g in ((0.0, 1.0), (0.075, 0.75)):
        i = int(t0 * SR); m = int(0.03 * SR); x[i:i + m] += hp(rng.standard_normal(m), 2000) * np.exp(-np.arange(m) / SR / 0.006) * g
        k = felt(180, 0.05); x[i:i + len(k)] += k * 0.35 * g
    return x
KAL = [523.3, 587.3, 659.3, 784.0, 880.0, 1046.5]

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
    put(kalimba(KAL[4], 0.7), T_SW[1], 0.14)                                                           # ПОСЛЕ revealed
    for k in range(3): put(bubble(700 + 120 * k, 1500 + 120 * k, 0.1), T_SW[1] + 0.2 + 0.18 * k, 0.06)  # sparkles
    put(kalimba(KAL[0], 0.5), T_S["B"] + 0.1, 0.12)                                                    # timeline appears
    put(bubble(), vstart("B") + 1.22, 0.16); put(bubble(420, 980), vstart("B") + 1.74, 0.16)             # «кожа», «тело»
    for k, (_, t0) in enumerate(ITEMS): put(kalimba(KAL[2 + k * 2], 0.6), t0 + 0.1, 0.2)                # the plan ticks
    for k, (_, rel) in enumerate(REASONS): put(bubble(450 + 80 * k, 1100 + 80 * k), vstart("D") + rel - 0.1, 0.15)
    put(kalimba(KAL[3], 0.6), vstart("D") + 4.54, 0.14); put(kalimba(KAL[5], 0.7), vstart("D") + 4.66, 0.12)   # heart
    put(kalimba(KAL[1], 0.5), T_S["E"] + 0.2, 0.12)                                                    # timeline → 1 день
    for k in range(4): put(step(), vstart("E") + 0.1 + 0.2 * k, 0.35)                                  # footsteps
    put(bubble(500, 1300), vstart("E") + 1.6, 0.16)                                                    # plaster
    put(kalimba(KAL[2], 0.5), T_S["F"] + 0.2, 0.12)                                                    # timeline → 3 недели
    put(rise(5.5, 300, 520), T_S["F"] + 0.9, 0.035); put(rise(2.6, 520, 700), vstart("G") + 3.4, 0.04)  # the bar loading
    put(kalimba(KAL[4], 0.6), vstart("G") + 5.6, 0.18)                                                 # arrow up
    put(kalimba(KAL[3], 0.5), T_S["H"] + 0.2, 0.12)                                                    # timeline → 1+ месяц
    put(glitter(1.2), T_H2 + 0.6, 0.12); [put(kalimba(f, 0.8), T_H2 + 0.05 + 0.1 * i, 0.1) for i, f in enumerate([KAL[0], KAL[2], KAL[4]])]
    for tk in T_K: put(shutter_cam(), tk, 0.22)                                                        # collages
    sfx[:n] *= 1 - 0.75 * np.clip(dd, 0, 1)
    mix = vo[:, None] + mus + sfx[:n, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (1.0 / 20); mix /= max(1, np.abs(mix).max() / 0.95)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

def make_cover(path):
    """everything inside the centre 3:4 (y 240-1680) that the profile grid shows; a deeper navy than the video"""
    CBG = np.clip(np.float32([0, 10, 36]) / 255 * (1 + 0.3 * np.exp(-(((xx - 540) / 900) ** 2 + ((yy - 520) / 900) ** 2)))[..., None], 0, 1).astype(np.float32)
    fr = CBG.copy(); im, a, cx = fin_img(); s = 0.78; im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)[..., None]
    x0, y0 = 540 - int(cx * s), 340; h_, w_ = im.shape[:2]
    glow = np.exp(-(((xx - 540) / 330) ** 2 + ((yy - 700) / 420) ** 2))[..., None] * 0.3; fr += glow * (PINK - fr)
    sub = fr[y0:y0 + h_, x0:x0 + w_]; sub[:] = sub * (1 - a) + im * a
    shade(fr, (ramp(820, 1080) * 0.97).astype(np.float32))                   # the waistband melts into the navy before the text
    blend(fr, logo_alpha(300), 540 - 150, 262, PINK)
    hd = Drop(["НАСТУПИЛ ПЕРИОД", "*ЛЮБОВАНИЯ СОБОЙ*"], 112, 70, 1150, maxw=860); hd.draw(fr, 99, 0)
    ty = Track(["после абдоминопластики", "и липофилинга груди"], 34, 72, 0, 400); ty.y = 1150 + hd.h + 56 + ty.top; ty.draw(fr, 99, 0)
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
