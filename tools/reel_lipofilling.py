#!/usr/bin/env python3
"""Reels «Можно ли увеличить грудь без имплантов?» — липофилинг груди, Анна Тваури.

Brand system as in tools/reel_postpartum.py / tools/reel_reduction.py (navy, powder pink, Bebas Neue + Montserrat,
notched frame, logo), built around the surgeon's talking head, so the explaining is done by line animation:
a body schematic where fat droplets travel from the donor zones into the breast, waist slims / breast fills
(«два эффекта»), droplets that drop away («30–50 %»), an incision crossed out for punctures, a soft ripple,
repeat arrows, a «stories highlight» bubble for «Смотрите актуальное».

usage:
  REEL_SRC=<folder with IMG_3858.MP4, C0818.MP4, IMG_8523/8524.PNG, logo.png, music.wav> python3 tools/reel_lipofilling.py [--preview t1,t2 | --cover | --audio-only]
Sources are first normalised into $REEL_SRC/_work (1080x1920, 30 fps; voice 48 kHz mono wav):
  ffmpeg -i IMG_3858.MP4 -vf "scale=1080:1920:flags=lanczos,fps=30,format=yuv420p" -an -c:v libx264 -crf 14 _work/IMG_3858.mp4   (same for C0818)
  ffmpeg -i IMG_3858.MP4 -vn -ac 1 -ar 48000 _work/IMG_3858.wav
"""
import os, sys, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_lipofilling"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_lipofilling_no_implants"
W, H, FPS, SR = 1080, 1920, 30, 48000
NAVY = np.float32([0, 18, 58]) / 255; PINK = np.float32([233, 197, 193]) / 255; WHITE = np.float32([1, 1, 1])
DEEP = np.float32([0, 8, 30]) / 255

def cl(x): return min(1.0, max(0.0, x))
def eo(x): x = cl(x); return 1 - (1 - x) ** 3
def eio(x): x = cl(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x): x = cl(x); c = 1.7; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p

# ---------------- voice edit: (block, src_in, src_out) — everything is the surgeon, IMG_3858 ----------------
VO = [("H", 1.05, 4.95),                                               # можно ли увеличить грудь без имплантов? да, конечно, можно.
      ("M", 5.04, 14.58),                                              # это методика липофилинга… и пересаживаем в область груди.
      ("T", 14.62, 23.40),                                             # тем самым мы добиваемся сразу двух результатов… увеличиваем грудь.
      ("W", 26.86, 35.70),                                             # минусы состоят в том, что часть жира… от 30 до 50 процентов жира, который может уходить.
      ("N", 41.15, 50.42),                                             # у нас нет разрезов… на ощупь никто не поймёт, что эта грудь увеличена чем-либо.
      ("R", 50.44, 55.05)]                                             # и плюс, конечно же, эту процедуру при желании можно повторить.
# whole sentences only: every cut sits in a real pause, so no line is chopped mid-thought
HOLD = {"R": 4.35}                                                     # the second case slide stays ~4 s before the last line
SEG = []; _t = 0.25
for _k, (_b, _a, _o) in enumerate(VO):
    if _k: _t += HOLD.get(_b, 0.45)
    SEG.append((_b, "3858", _a, _o, _t)); _t += _o - _a
def vstart(b, i=0): return [s[4] for s in SEG if s[0] == b][i]
def vend(b): return max(s[4] + s[3] - s[2] for s in SEG if s[0] == b)
def at_src(src_t):                                                     # reel time where a source moment is heard
    for _, _, si, so, t0 in SEG:
        if si <= src_t <= so: return t0 + src_t - si
    raise ValueError(src_t)
T_M = vstart("M") - 0.15; T_X = T_M + 1.8                              # OR b-roll, then the body schematic
T_T = vstart("T") - 0.15; T_W = vstart("W") - 0.15; T_N = vstart("N") - 0.15
T_S = at_src(44.62) - 0.1; T_SL1 = at_src(47.30); T_SL2 = vend("N") + 0.4    # «натуральная грудь» card, then the two case slides
T_R = vstart("R") - 0.15; T_END = vend("R") + 0.25; DUR = T_END + 3.0
T_YES = at_src(3.76); T_PCT = at_src(31.96); T_PUNCT = at_src(43.20)
def src_of(t, name="3858"):
    best = None
    for _, s, si, so, t0 in SEG:
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
V = {k: Src(os.path.join(WORK, f"{k}.mp4")) for k in ["IMG_3858", "C0818"]}
def talk(t): return V["IMG_3858"].at(src_of(t))
def case_crops():
    """six ДО/ПОСЛЕ pairs cut out of the two presentation slides"""
    out = []
    for n, rows in [("8523", [(208, 611), (957, 1360), (1707, 2109)]), ("8524", [(204, 606), (953, 1356), (1703, 2105)])]:
        im = np.asarray(Image.open(os.path.join(SRC, f"IMG_{n}.PNG")).convert("RGB"))
        for r0, r1 in rows: out.append((im[r0 + 3:r1 - 3, 99:641].copy(), im[r0 + 3:r1 - 3, 679:1225].copy()))
    return out
CASES = case_crops()

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

# ---------------- line-art body schematic ----------------
def spline(pts, n=12):
    """Catmull-Rom through the points"""
    P = [pts[0]] + list(pts) + [pts[-1]]; out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = map(np.float32, P[i - 1:i + 3])
        for u in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(np.float32(pts[-1])); return out
BX, BY, BS = 540, 560, 0.70                                            # schematic placement (leaves room for the circle)
def bp(x, y): return (BX + x * BS, BY + y * BS)
def body(fr, t, t0, grow=0.0, slim=0.0, op=1.0):
    """front torso: outline draws itself; grow = breast fullness 0..1, slim = waist 0..1"""
    p = eio((t - t0) / 1.0)
    for sx in (-1, 1):
        side = [(55, 0), (60, 50), (130, 70), (200, 95), (235, 135), (215, 200), (205, 300), (195, 380),
                (165 - 22 * slim, 470), (178 - 10 * slim, 545), (225, 640), (235, 720), (215, 850)]
        path_draw(fr, [bp(sx * x, y) for x, y in spline(side)], p, WHITE, 3, op=op)
        g = grow; br = [(45, 225), (40 - 4 * g, 295 + 10 * g), (75, 348 + 22 * g), (140 + 6 * g, 352 + 26 * g), (188 + 10 * g, 310 + 14 * g), (196 + 6 * g, 250)]
        path_draw(fr, [bp(sx * x, y) for x, y in spline(br)], eio((t - t0 - 0.4) / 0.8), WHITE, 3, op=op)
    path_draw(fr, [bp(6 * np.cos(a), 520 + 6 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 16)], eio((t - t0 - 0.8) / 0.3), WHITE, 2, op=op)
DONORS = [(-185, 455), (185, 455), (-70, 570), (70, 570), (-200, 780), (200, 780)]
def drop_pts(cx, cy, r):
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(-np.pi / 6, np.pi * 7 / 6, 18)] + [(cx, cy - 2.1 * r)]
def droplets(fr, t, t0, t1):
    """fat droplets ride a curve from every donor zone into the breast on the same side"""
    if t < t0 or t > t1 + 0.6: return
    for k, (dx, dy) in enumerate(DONORS):
        tx, ty = (115 if dx > 0 else -115), 300
        cx_, cy_ = dx * 1.55, (dy + ty) / 2
        for j in range(3):
            u = ((t - t0) * 0.75 - j * 0.33 - k * 0.07) % 1.0
            if t > t1 and u < (t - t1) * 1.6: continue                 # stream stops
            if (t - t0) * 0.75 - j * 0.33 - k * 0.07 < 0: continue
            x = (1 - u) ** 2 * dx + 2 * (1 - u) * u * cx_ + u ** 2 * tx; y = (1 - u) ** 2 * dy + 2 * (1 - u) * u * cy_ + u ** 2 * ty
            r = 9 * np.sin(np.pi * min(1, u * 1.15)) + 2
            fill_poly(fr, [bp(*q) for q in drop_pts(x, y, r)], PINK)
    for dx, dy in DONORS:                                              # donor zones pulse
        rr = 16 + 10 * ((t - t0) * 1.3 % 1); a = 1 - (t - t0) * 1.3 % 1
        path_draw(fr, [bp(dx + rr * np.cos(q), dy + rr * np.sin(q)) for q in np.linspace(0, 2 * np.pi, 24)], 1, PINK, 2, op=a * (1 - eio((t - t1) / 0.4) if t > t1 else 1))
def badge(fr, x, y, sym, t, t0):
    s = eback((t - t0) / 0.35)
    if s <= 0: return
    m = np.zeros((H, W), np.uint8); cv2.circle(m, (int(x), int(y)), int(28 * s), 255, -1, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, PINK)
    bar = np.ones((max(1, int(6 * s)), max(1, int(26 * s))), np.float32); blend(fr, bar, x - bar.shape[1] / 2, y - bar.shape[0] / 2, NAVY)
    if sym == "+": blend(fr, bar.T.copy(), x - bar.shape[0] / 2, y - bar.shape[1] / 2, NAVY)
def pip(fr, t, t0, cx=540, cy=1430, r=255):
    """the surgeon keeps talking in a round window"""
    p = eback((t - t0) / 0.45)
    if p <= 0: return
    rr = max(4, int(r * p)); im, _ = cover(talk(t), 2 * rr, 2 * rr, 0.5, 0.42, 1.0)     # face and shoulders centred
    yy2, xx2 = np.mgrid[0:2 * rr, 0:2 * rr]; m = np.clip(rr - np.sqrt((xx2 - rr + 0.5) ** 2 + (yy2 - rr + 0.5) ** 2), 0, 1)[..., None].astype(np.float32)
    reg = fr[cy - rr:cy + rr, cx - rr:cx + rr]; reg[:] = reg * (1 - m) + im * m
    path_draw(fr, [(cx + (rr + 8) * np.cos(a), cy + (rr + 8) * np.sin(a)) for a in np.linspace(-np.pi / 2, 1.5 * np.pi, 80)], 1, PINK, 3)

# ---------------- scenes ----------------
# hook: surgeon full frame, slow push-in, letters drop; «ДА» stamp
HH = Drop(["МОЖНО ЛИ УВЕЛИЧИТЬ", "ГРУДЬ *БЕЗ ИМПЛАНТОВ?*"], 124, 70, 1260)
def stamp(fr, t, t0, cx=830, cy=1010):
    lt = t - t0
    if lt < 0: return
    s = 1 + 0.9 * (1 - eback(lt / 0.35)); op = min(1, lt / 0.12)
    m = np.zeros((H, W), np.uint8); cv2.circle(m, (cx, cy), int(100 * s), 255, -1, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, PINK, op)
    path_draw(fr, [(cx + 116 * s * np.cos(a), cy + 116 * s * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 70)], 1, PINK, 4, op=op)
    a = alpha_img("ДА", bebas(int(130 * s))); M = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), 8, 1.0)
    a = cv2.warpAffine(a, M, (a.shape[1], a.shape[0])); blend(fr, a, cx - a.shape[1] / 2, cy - a.shape[0] / 2 - 8, NAVY, op)
def s_hook(t):
    fr = full(talk(t), 0.5, 0.3, lerp(1.0, 1.14, eio(t / 4.2))); shade(fr, VIG); shade(fr, GBOT)
    lg = logo_alpha(360); blend(fr, lg, 540 - 180, 210, PINK, eo((t + 0.1) / 0.5) * 0.9)
    HH.draw(fr, t, -0.25); stamp(fr, t, T_YES)
    return fr

# method: OR b-roll -> body schematic, droplets flow, implant crossed out; surgeon in a round window
HM = Drop(["СВОЯ ЖИРОВАЯ ТКАНЬ", "*ВМЕСТО ИМПЛАНТОВ*"], 112, 70, 250)
TM = Track(["увеличение груди собственным жиром"], 38, 72, 0, 500, PINK)
def implant_x(fr, t, t0, cx=880, cy=830):
    p = eio((t - t0) / 0.6)
    pts = [(cx + 62 * np.cos(a), cy + 46 * np.sin(a) + 12 * np.sin(a) ** 2 * (np.sin(a) > 0)) for a in np.linspace(0, 2 * np.pi, 50)]
    path_draw(fr, pts, p, WHITE, 3, True)
    lab = alpha_img("импланты", mont(28, 400)); blend(fr, lab, cx - lab.shape[1] / 2, cy + 64, WHITE, eo((t - t0 - 0.3) / 0.3))
    q = eio((t - t0 - 0.7) / 0.35)
    path_draw(fr, [(cx - 80, cy - 62), (cx + 80, cy + 62)], q, PINK, 7); path_draw(fr, [(cx + 80, cy - 62), (cx - 80, cy + 62)], eio((t - t0 - 0.9) / 0.35), PINK, 7)
def s_broll(t):
    lt = t - T_M; fr = full(V["C0818"].at(1.2 + lt), 0.5, 0.35, lerp(1.18, 1.02, eio(lt / 2.0))); shade(fr, VIG); shade(fr, GBOT)
    a = alpha_img("ЛИПОФИЛИНГ", bebas(170)); blend(fr, a, 66, 1400 - cap_top(bebas(170)), WHITE, eo((lt - 0.2) / 0.4))
    TM.y = 1400 + 125 + TM.top; TM.draw(fr, t, T_M + 0.5)
    return fr
def s_schema(t):
    fr = BG.copy(); grow = eio((t - T_X - 1.0) / (T_T - T_X)) * 0.6 + 0.4 * eio((t - T_T - 0.3) / 1.6)
    slim = eio((t - T_T - 0.3) / 1.6)
    body(fr, t, T_X + 0.1, grow, slim)
    droplets(fr, t, T_X + 0.9, T_T + 2.4)
    for k, (lab, (dx, dy)) in enumerate([("бока", DONORS[0]), ("живот", DONORS[2]), ("бёдра", DONORS[4])]):
        op = eo((t - T_X - 1.2 - 0.15 * k) / 0.3) * (1 - eio((t - T_T) / 0.3)); lx, ly = bp(-290, dy)
        if op > 0:
            path_draw(fr, [bp(dx - 18, dy), (lx + 8, ly)], 1, WHITE, 1, op=op * 0.7)
            a = alpha_img(lab, mont(28, 400)); blend(fr, a, lx - a.shape[1], ly - 20, WHITE, op)
    pip(fr, t, T_X + 0.2)
    if t < T_T:
        HM.draw(fr, t, T_X + 0.2); implant_x(fr, t, T_X + 1.6)
    else:
        HT.draw(fr, t, T_T + 0.15); TT1.draw(fr, t, T_T + 0.7); TT2.draw(fr, t, T_T + 1.3)
        px, py = bp(250, 470); badge(fr, px, py, "−", t, T_T + 0.6)
        px, py = bp(250, 300); badge(fr, px, py, "+", t, T_T + 1.2)
    return fr
HT = Drop(["ДВА ЭФФЕКТА *СРАЗУ*"], 112, 70, 250)
TT1 = Track(["− меньше объёма в донорских зонах"], 36, 72, 420, 400)
TT2 = Track(["+ больше наполненности груди"], 36, 72, 472, 500, PINK)

# warning: surgeon, zoom-out; 10 droplets, part of them falls away; «30–50 %»
HW = Drop(["ВАЖНО:", "*ЧАСТЬ ЖИРА МОЖЕТ УЙТИ*"], 112, 70, 1150)
TW = Track(["объём приживления индивидуален"], 36, 72, 0, 300)
def drops_row(fr, t, t0, y=1580):
    for i in range(10):
        x = 95 + i * 62; lt = t - t0 - i * 0.05
        if lt < 0: continue
        s = eback(lt / 0.3); pts = drop_pts(x, y, 20 * s)
        if i >= 7:                                                      # 30 %: gone for sure
            q = eio((t - T_PCT - 0.1 - (i - 7) * 0.1) / 0.6); pts = [(px, py + 140 * q ** 2) for px, py in pts]
            fill_poly(fr, pts, PINK, 1 - q)
        elif i >= 5:                                                    # up to 50 %: may go
            q = eio((t - T_PCT - 0.6 - (i - 5) * 0.1) / 0.4); fill_poly(fr, pts, PINK, 1 - 0.85 * q); path_draw(fr, pts + pts[:1], 1, PINK, 2, op=q)
        else: fill_poly(fr, pts, PINK)
    if t > T_PCT + 0.3:
        a = alpha_img("30-50%", bebas(104)); blend(fr, a, 712, y - 30 - cap_top(bebas(104)) + 4, PINK, eo((t - T_PCT - 0.3) / 0.3))
def s_warn(t):
    lt = t - T_W; fr = full(talk(t), 0.5, 0.3, lerp(1.3, 1.04, eio(lt / 1.4))); shade(fr, VIG); shade(fr, GBOT)
    HW.draw(fr, t, T_W + 0.25); TW.y = HW.y + HW.h + 60 + TW.top; TW.draw(fr, t, T_W + 1.0)
    drops_row(fr, t, T_W + 0.6)
    return fr

# no incisions: text card + incision crossed out / punctures (surgeon in the circle), then slide 1 full screen
HN = Drop(["БЕЗ ИМПЛАНТОВ", "*БЕЗ БОЛЬШИХ РАЗРЕЗОВ*"], 116, 70, 300)
SLIDES = [np.asarray(Image.open(os.path.join(SRC, f"IMG_{n}.PNG")).convert("RGB")) for n in ("8523", "8524")]
def slide_frame(i):
    """the whole presentation slide, scaled down to the frame width — never enlarged"""
    im = SLIDES[i]; s = W / im.shape[1]; im = cv2.resize(im, (W, int(im.shape[0] * s)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    fr = np.zeros((H, W, 3), np.float32) + im[2:6].mean((0, 1)); y = (H - im.shape[0]) // 2; fr[y:y + im.shape[0]] = im; return fr
SLIDE_FR = [slide_frame(0), slide_frame(1)]
def incision(fr, t, t0, x0=90, y=1000):
    path_draw(fr, [(x0, y), (x0 + 380, y - 10)], eio((t - t0) / 0.5), WHITE, 5)
    a = alpha_img("разрез", mont(32, 400)); blend(fr, a, x0, y + 26, WHITE, eo((t - t0 - 0.2) / 0.3))
    path_draw(fr, [(x0 - 10, y - 46), (x0 + 390, y + 38)], eio((t - t0 - 0.6) / 0.3), PINK, 8)
    for k in range(3):
        lt = t - T_PUNCT - k * 0.15
        if lt > 0:
            r = 11 * eback(lt / 0.3); m = np.zeros((H, W), np.uint8); cv2.circle(m, (640 + k * 90, y), max(1, int(r)), 255, -1, cv2.LINE_AA)
            blend(fr, m.astype(np.float32) / 255, 0, 0, PINK)
    b = alpha_img("только проколы", mont(32, 500)); blend(fr, b, 615, y + 26, PINK, eo((t - T_PUNCT - 0.4) / 0.3))
def s_noinc(t):
    fr = BG.copy(); HN.draw(fr, t, T_N + 0.3); incision(fr, t, T_N + 1.0); pip(fr, t, T_N + 0.2, 540, 1420, 275)
    return fr

# soft & natural: text card + ripple (surgeon in the circle), then slide 2 full screen
HS = Drop(["МЯГКИЙ И ЕСТЕСТВЕННЫЙ", "*РЕЗУЛЬТАТ*"], 116, 70, 300)
def ripple(fr, t, t0, cx=540, cy=900):
    if t < t0: return
    for k in range(3):
        u = ((t - t0) * 0.8 - k / 3) % 1
        if (t - t0) * 0.8 - k / 3 < 0: continue
        path_draw(fr, [(cx + (20 + 150 * u) * np.cos(a), cy + (20 + 150 * u) * 0.6 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 60)], 1, PINK, 3, op=(1 - u) ** 1.5)
    m = np.zeros((H, W), np.uint8); cv2.circle(m, (cx, cy), 14, 255, -1, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, PINK)
    a = alpha_img("на ощупь", mont(32, 500)); blend(fr, a, cx - a.shape[1] / 2, cy + 110, WHITE, eo((t - t0) / 0.3))
def s_soft(t):
    if t >= T_SL2: return SLIDE_FR[1].copy()
    if t >= T_SL1: return SLIDE_FR[0].copy()
    fr = BG.copy(); HS.draw(fr, t, T_S + 0.2); ripple(fr, t, T_S + 0.5); pip(fr, t, T_S + 0.1, 540, 1420, 275)
    return fr

# repeat: surgeon full frame, repeat arrows
HR = Drop(["ПРИ ЖЕЛАНИИ", "*МОЖНО ПОВТОРИТЬ*"], 124, 70, 1300)
def repeat_icon(fr, t, t0, cx=880, cy=1390, r=70):
    p = eio((t - t0) / 0.8); rot = (t - t0) * 1.6
    for h0 in (0, np.pi):
        arc = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(h0 + rot, h0 + rot + 2.4, 30)]
        path_draw(fr, arc, p, PINK, 6)
        if p > 0.95:
            e = h0 + rot + 2.4; tip = (cx + r * np.cos(e), cy + r * np.sin(e)); d = np.array([-np.sin(e), np.cos(e)])
            n = np.array([np.cos(e), np.sin(e)]); fill_poly(fr, [tip + d * 20, tip + n * 16 - d * 6, tip - n * 16 - d * 6], PINK)
def s_rep(t):
    lt = t - T_R; fr = full(talk(t), 0.5, 0.3, lerp(1.0, 1.08, eio(lt / 3.5))); shade(fr, VIG); shade(fr, GBOT)
    HR.draw(fr, t, T_R + 0.3); repeat_icon(fr, t, T_R + 0.8)
    return fr

# end card: question + the real «Липофилинг» highlight cover + logo
HE = Drop(["ХОТИТЕ ПОНЯТЬ,", "ПОДОЙДЁТ ЛИ ВАМ", "*ЛИПОФИЛИНГ?*"], 128, 70, 330)
_hl = np.asarray(Image.open(os.path.join(SRC, "highlight.jpg")).convert("RGB")).astype(np.float32) / 255
HL = cv2.resize(_hl[12:212, 24:224], (300, 300), interpolation=cv2.INTER_CUBIC)                   # circle of the highlight cover
def s_end(t):
    lt = t - T_END; fr = BG.copy(); HE.draw(fr, t, T_END + 0.2)
    cx, cy = 540, 1130; p = eback((lt - 0.9) / 0.5)
    if p > 0:
        d = max(4, int(300 * p)); im = cv2.resize(HL, (d, d)); yy2, xx2 = np.mgrid[0:d, 0:d]
        m = np.clip(d / 2 - np.sqrt((xx2 - d / 2 + 0.5) ** 2 + (yy2 - d / 2 + 0.5) ** 2), 0, 1)[..., None].astype(np.float32)
        reg = fr[cy - d // 2:cy - d // 2 + d, cx - d // 2:cx - d // 2 + d]; reg[:] = reg * (1 - m) + im * m
        rot = (lt - 0.9) * 1.2
        path_draw(fr, [(cx + (160 * p) * np.cos(a), cy + (160 * p) * np.sin(a)) for a in np.linspace(-np.pi / 2 + rot, 1.5 * np.pi + rot, 90)], eio((lt - 1.0) / 0.8), PINK, 5)
        tap = (lt - 1.9) % 1.2
        if lt > 1.9: path_draw(fr, [(cx + 80 + (10 + 45 * tap) * np.cos(a), cy + 80 + (10 + 45 * tap) * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 40)], 1, WHITE, 2, op=1 - tap / 1.2)
        lab = alpha_img("Липофилинг", mont(40, 500)); blend(fr, lab, cx - lab.shape[1] / 2, cy + 185, WHITE, eo((lt - 1.2) / 0.3))
    ty = Track(["смотрите актуальное «Липофилинг»"], 38, 540, 0, 300, WHITE, True); ty.y = 1440 + ty.top; ty.draw(fr, t, T_END + 1.5)
    lg = logo_alpha(420); blend(fr, lg, 540 - 210, 1580, PINK, eo((lt - 2.0) / 0.5))
    return fr

SCENES = [(0, s_hook), (T_M, s_broll), (T_X, s_schema), (T_W, s_warn), (T_N, s_noinc), (T_S, s_soft), (T_R, s_rep), (T_END, s_end)]
TRANS = {T_M: ("zoom", 0.5), T_X: ("iris", 0.6), T_W: ("diag", 0.55), T_N: ("flash", 0.3), T_S: ("push", 0.5), T_R: ("zoom", 0.5), T_END: ("iris", 0.6)}
INNER = {T_SL1: 0.45, T_SL2: 0.45}                                     # slides push in inside their scenes
def push(old, new, p):
    e = eio(p); x = int(W * e); fr = np.empty_like(old); fr[:, :W - x] = old[:, x:]; fr[:, W - x:] = new[:, :x]
    if 0 < x < W: fr[:, max(0, W - x - 2):W - x + 2] = PINK
    return fr
def render(t):
    k = max(i for i, (s0, _) in enumerate(SCENES) if t >= s0); s0, fn = SCENES[k]
    for ti, d in INNER.items():
        if s0 < ti <= t < ti + d: return push(fn(ti - 0.01), fn(t), (t - ti) / d)
    kind, d = TRANS.get(s0, ("none", 0))
    if kind != "none" and t < s0 + d:
        p = (t - s0) / d
        if kind == "flash":
            fr = fn(t); f = 1 - eo(p); return (fr * (1 - f) + f).astype(np.float32)
        old = SCENES[k - 1][1](t); new = fn(t)
        if kind == "iris": return iris(old, new, p, 540, 900)
        if kind == "zoom": return zoom_through(old, new, p)
        if kind == "diag": return diag_wipe(old, new, p)
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
PENTA = [587.3, 659.3, 740.0, 880.0, 987.8, 1174.7]                    # D major pentatonic, matches the music
MUSIC = os.path.join(SRC, "music.wav")
M_DROP = 14.45; M0 = M_DROP - T_X                                      # the track's drop lands on the body schematic
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
    n = int(DUR * SR); vo = np.zeros(n); act = np.zeros(n)
    w = wave.open(os.path.join(WORK, "IMG_3858.wav")); src = hp(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768, 90)
    ir_t = np.arange(int(0.45 * SR)) / SR; IR = rng.standard_normal(len(ir_t)) * np.exp(-ir_t / 0.11); IR = lp(IR, 3500); IR /= np.sqrt(np.sum(IR ** 2))
    room = src[int(0.15 * SR):int(0.95 * SR)].copy()                   # the room before she speaks
    for b, s_, si, so, t0 in SEG:
        seg = src[int(si * SR):int(so * SR)].copy()
        fi, fo = int(0.03 * SR), int(0.11 * SR)
        seg[:fi] *= np.sin(np.linspace(0, np.pi / 2, fi)) ** 2; seg[-fo:] *= np.cos(np.linspace(0, np.pi / 2, fo)) ** 2     # soft edges
        fr_rms = np.sqrt(np.mean(seg[:len(seg) // 960 * 960].reshape(-1, 960) ** 2, 1)); g = 1 / (np.sqrt(np.mean(np.sort(fr_rms)[-max(1, len(fr_rms) * 2 // 5):] ** 2)) + 1e-6)
        tail = np.convolve(seg[-int(0.25 * SR):], IR)[int(0.25 * SR):] * 0.22                  # short room tail: the last word decays instead of stopping
        seg = np.concatenate([seg, tail]) * g
        i = int(t0 * SR); vo[i:i + len(seg)] += seg[:n - i]; act[max(0, i - int(0.25 * SR)):i + int((so - si) * SR) + int(0.1 * SR)] = 1
    rt = np.resize(room, n) * (1 / (np.sqrt(np.mean(room ** 2)) + 1e-9)) * 0.004                     # constant quiet room tone under everything
    vo = np.tanh((vo + rt) * 0.16 * 1.6) / 1.6
    d = np.convolve(act, np.ones(int(0.25 * SR)) / int(0.25 * SR), "same")
    tt = np.arange(n) / SR; fade = np.clip((DUR - tt) / 1.8, 0, 1) ** 1.5 * np.minimum(1, tt / 0.15)
    mus = track(n) * ((G_GAP - (G_GAP - G_VO) * np.clip(d, 0, 1)) * fade)[:, None]
    sfx = np.zeros(n + 2 * SR)
    def put(x, t, g): i = int(max(0, t) * SR); seg = sfx[i:i + len(x)]; seg += x[:len(seg)] * g
    put(felt(70, 0.5), 0.0, 0.5); put(glass(1174.7), 0.05, 0.1)
    put(felt(80, 0.4), T_YES, 0.6); put(paper(0.18, 4000), T_YES - 0.04, 0.3); put(glass(1760), T_YES + 0.05, 0.12)   # «ДА» stamp
    put(marimba(PENTA[0]), T_X + 0.3, 0.2)                                         # transitions themselves stay silent
    for k, hd_t in enumerate([T_X + 0.2, T_T + 0.15, T_W + 0.25, T_N + 0.3, T_S + 0.2, T_R + 0.3, T_END + 0.2]):
        put(marimba(PENTA[(k * 2) % 6], 0.4), hd_t + 0.05, 0.12)                       # each headline lands on a note
    k = 0; t = T_X + 0.9
    while t < T_T + 2.4:                                               # droplets landing in the breast
        put(water(450 + 60 * (k % 6), 1500 + 90 * (k % 4)), t, 0.12); t += 0.25; k += 1
    put(paper(0.22, 5000), T_X + 2.3, 0.22); put(felt(150, 0.2), T_X + 2.5, 0.2)       # implant crossed out
    put(marimba(PENTA[1] / 2, 0.5), T_T + 0.6, 0.25); put(marimba(PENTA[4], 0.5), T_T + 1.2, 0.25)   # − low / + high
    for i in range(10): put(water(700 + 40 * i, 1300 + 40 * i), T_W + 0.6 + i * 0.05, 0.05)
    for i in range(3): put(water(1600, 400), T_PCT + 0.3 + i * 0.1, 0.25)               # droplets fall away (falling pitch)
    put(paper(0.5, 6000), T_N + 1.0, 0.2); put(paper(0.15, 5000), T_N + 1.6, 0.25)     # incision drawn / crossed
    for i in range(3): put(marimba(PENTA[3 + i] * 2, 0.15), T_PUNCT + i * 0.15, 0.12)  # punctures
    put(swell(0.8), T_S + 0.3, 0.07)
    put(paper(0.6, 2000), T_R + 0.8, 0.18)
    put(glass(1174.7), T_END + 0.95, 0.2); put(glass(1760), T_END + 1.25, 0.12)
    for i, f in enumerate([PENTA[0], PENTA[2], PENTA[3], PENTA[5]]): put(marimba(f, 0.6), T_END + 1.9 + i * 0.12, 0.12)
    sfx[:n] *= 1 - 0.8 * np.clip(d, 0, 1)
    mix = vo[:, None] + mus + sfx[:n, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (1.0 / 20); mix /= max(1, np.abs(mix).max() / 0.95)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

def make_cover(path):
    fr = BG.copy()
    for r, ci in enumerate([2, 4]):
        for k, (img, lab) in enumerate(zip(CASES[ci], ("ДО", "ПОСЛЕ"))):
            x0 = 40 + k * 510; y0 = 200 + r * 420; card(fr, img, (x0, y0, x0 + 490, y0 + 380), 0, 0, 1, 0, 0, 0, 0.5, 0.5, 1.02)
            chip(fr, lab, x0 + 14, y0 + 380 - 72)
    lg = logo_alpha(380); blend(fr, lg, 540 - 190, 70, PINK)
    hd = Drop(["ХОЧУ УВЕЛИЧИТЬ ГРУДЬ", "*БЕЗ ИМПЛАНТОВ*"], 160, 60, 1150, maxw=960); hd.draw(fr, 99, 0)
    ty = Track(["что может дать липофилинг"], 44, 64, 0, 400); ty.y = 1150 + hd.h + 80 + ty.top; ty.draw(fr, 99, 0)
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
