#!/usr/bin/env python3
"""Reels «После родов: грудь + живот за один этап» — Анна Тваури.

Brand: navy #001D55, powder pink #E9C5C1 (logo), white. Fonts: Bebas Neue (bold, headlines) + Montserrat Light.
Layout idea from the reference: thin white frame with a stepped notch in the bottom-right corner; the reel
alternates full-screen moments with the footage living inside that frame (FUROR-style: wave transitions,
words rising out of a mask, typed small lines, line-drawn metaphors).

usage:
  REEL_SRC=<folder with IMG_*.MOV/MP4/JPG + logo.png> python3 tools/reel_postpartum.py            # full render
  REEL_SRC=... python3 tools/reel_postpartum.py --preview 1.0,5.0,9.0                            # still frames
  REEL_SRC=... python3 tools/reel_postpartum.py --cover
  REEL_SRC=... python3 tools/reel_postpartum.py --audio-only                                     # new mix, same picture
Sources are first normalised into $REEL_SRC/_work (1080x1920, 30 fps; voice tracks as 48 kHz wav):
  ffmpeg -i IMG_x.MOV -vf "scale=1080:1920:flags=lanczos,fps=30,format=yuv420p" -an -c:v libx264 -crf 14 _work/IMG_x.mp4
  ffmpeg -i IMG_2994.MP4 -vn -ac 1 -ar 48000 _work/IMG_2994.wav   (same for IMG_0331)
"""
import os, sys, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_postpartum"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_postpartum_breast_belly"
W, H, FPS, SR = 1080, 1920, 30, 48000
NAVY = np.float32([0, 18, 58]) / 255; PINK = np.float32([233, 197, 193]) / 255; WHITE = np.float32([1, 1, 1])
DEEP = np.float32([0, 8, 30]) / 255

# ---------------- easing ----------------
def cl(x): return min(1.0, max(0.0, x))
def eo(x): x = cl(x); return 1 - (1 - x) ** 3
def eio(x): x = cl(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x): x = cl(x); c = 1.6; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p

# ---------------- voice edit: (block, source, src_in, src_out) -> reel start computed with short gaps ----------------
VO = [("A", "2994", 16.30, 22.95),                                   # после родов… стенка ослабла и грудь немного провисла
      ("B", "2994", 31.30, 32.98), ("B", "2994", 36.00, 38.76),      # поэтому спортом, к сожалению, этот живот не выправить
      ("C", "2994", 44.86, 46.35), ("C", "2994", 46.70, 50.00),      # достаточно большой диастаз… только ушиванием мышц
      ("D", "2994", 59.68, 61.97), ("D", "2994", 69.40, 71.45),      # убрали кожу ниже пупка. перенесли пупок, сузили талию
      ("E", "2994", 84.54, 86.40), ("E", "2994", 87.45, 89.25),      # она категорически была против разрезов на груди
      ("E3", "2994", 91.45, 93.64),                                  # мы поставили анатомические импланты
      ("F", "0331", 1.60, 5.95),                                     # сегодня первые сутки после большой комплексной операции
      ("G", "0331", 28.20, 32.05)]                                   # получили замечательный результат, всё в штатном режиме
SEG = []; _t = 0.25
for _k, (_b, _s, _a, _o) in enumerate(VO):
    if _k: _t += 0.08 if _b[0] == VO[_k - 1][0][0] else 0.22
    SEG.append((_b, _s, _a, _o, _t)); _t += _o - _a
VO_END = _t
def bstart(b): return next(s[4] for s in SEG if s[0] == b)
T_A = 3.95; T_B = bstart("B") - 0.15; T_C = bstart("C") - 0.15; T_D = bstart("D") - 0.15; T_E = bstart("E") - 0.15
T_E3 = bstart("E3") - 0.12; T_F = bstart("F") - 0.15; T_G = bstart("G") - 0.15; T_OUT = T_G + 4.9; DUR = T_OUT + 3.6
WAVE = 0.55
def src_of(t, name):
    """source time for synced footage; in the gaps the picture keeps running (no freeze)"""
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
def alpha_img(txt, f, stroke=0):
    asc, desc = f.getmetrics(); w = int(f.getlength(txt)) + 8 + 2 * stroke
    im = Image.new("L", (w, asc + desc + 4 + 2 * stroke), 0)
    ImageDraw.Draw(im).text((4 + stroke, 2 + stroke + asc), txt, font=f, fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255)
    return np.asarray(im, np.float32) / 255
def outline_img(txt, f, sw=3):
    a = alpha_img(txt, f, sw); asc, _ = f.getmetrics(); b = np.zeros_like(a)
    im = Image.new("L", (a.shape[1], a.shape[0]), 0); ImageDraw.Draw(im).text((4 + sw, 2 + sw + asc), txt, font=f, fill=255, anchor="ls")
    b = np.asarray(im, np.float32) / 255
    return np.clip(a - b, 0, 1)
def cap_top(f): asc, _ = f.getmetrics(); return 2 + asc + f.getbbox("Н", anchor="ls")[1]   # y of the cap line inside alpha_img

def blend(fr, a, x, y, col, op=1.0):
    x, y = int(round(x)), int(round(y)); h, w = a.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0 or op <= 0.003: return
    m = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * op
    reg = fr[y0:y1, x0:x1]; reg *= 1 - m; reg += m * col

class Head:
    """Bebas lines; every word rises out of its own mask, one after another. *word* … word* = pink."""
    def __init__(s, lines, size, x, y, maxw=940, lh=0.93, gap=0.075):
        while True:
            f = bebas(size); rows = []; pink = False; sp = f.getlength(" ")
            for ln in lines:
                items, cx = [], 0
                for wd in ln.split():
                    if wd.startswith("*"): pink = True
                    a = alpha_img(wd.strip("*"), f); items.append((a, PINK if pink else WHITE, cx)); cx += a.shape[1] - 8 + sp
                    if wd.endswith("*"): pink = False
                rows.append((items, cx - sp))
            if max(r[1] for r in rows) <= maxw or size < 40: break
            size -= 4
        s.rows, s.size, s.x, s.y, s.gap, s.lh = rows, size, x, y, gap, int(size * lh)
        s.top = cap_top(f); s.h = s.lh * (len(rows) - 1) + (size * 0.72)
        s.words = [(r, it) for r, (its, _) in enumerate(rows) for it in its]
    def word_x(s, row, idx): return s.x + s.rows[row][0][idx][2]
    def draw(s, fr, t, t0, t1=1e9, dur=0.5):
        if t < t0 or t > t1 + 0.4: return
        q = eio((t - t1) / 0.35) if t > t1 else 0.0
        for k, (r, (a, col, dx)) in enumerate(s.words):
            p = eo((t - t0 - k * s.gap) / dur)
            if p <= 0: continue
            hh = a.shape[0]; dy = int((1 - p) * hh * 0.85); b = np.zeros_like(a)
            if dy < hh: b[dy:] = a[:hh - dy]
            blend(fr, b, s.x + dx - 4, s.y + r * s.lh - s.top - q * 40, col, (1 - q) * min(1, p * 2.5))

class Type:
    """small Montserrat lines typed letter by letter"""
    def __init__(s, lines, size=38, x=72, y=0, wght=300, cps=34, col=WHITE):
        f = mont(size, wght); s.lines = []
        for ln in lines:
            a = alpha_img(ln, f); xs = [4 + f.getlength(ln[:i]) for i in range(len(ln) + 1)]; s.lines.append((a, xs, len(ln)))
        s.x, s.y, s.lh, s.cps, s.col, s.top = x, y, int(size * 1.45), cps, col, cap_top(f)
        s.n = sum(n for _, _, n in s.lines)
    def draw(s, fr, t, t0, t1=1e9):
        if t < t0 or t > t1 + 0.4: return
        op = 1 - eio((t - t1) / 0.3) if t > t1 else 1.0
        c = (t - t0) * s.cps
        for k, (a, xs, n) in enumerate(s.lines):
            if c <= 0: break
            i = int(min(c, n)); fx = c - i if i < n else 0; b = a.copy(); cut = int(xs[i])
            if i < n:
                nxt = int(xs[i + 1]); b[:, cut:nxt] *= fx; b[:, nxt:] = 0
            blend(fr, b, s.x - 4, s.y + k * s.lh - s.top, s.col, op)
            c -= n

# ---------------- geometry: the notched brand frame ----------------
def notch_poly(x0, y0, x1, y1, nw, nh, o=0):
    nx, ny = x1 - nw, y1 - nh
    return np.float32([(x0 - o, y0 - o), (x1 + o, y0 - o), (x1 + o, ny + o), (nx + o, ny + o), (nx + o, y1 + o), (x0 - o, y1 + o)])
CARD = (60, 230, 1020, 1240); NW, NH = 200, 240
def poly_mask(pts):
    m = np.zeros((H * 2, W * 2), np.uint8)                              # 2x supersampled for clean edges
    cv2.fillPoly(m, [np.round(pts * 2).astype(np.int32)], 255, cv2.LINE_AA)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
def path_draw(fr, pts, p, col, th=3, closed=True, op=1.0):
    """draw a polyline progressively (p = 0..1 of its length)"""
    if p <= 0: return
    P = list(pts) + ([pts[0]] if closed else []); L = [np.hypot(*(np.subtract(P[i + 1], P[i]))) for i in range(len(P) - 1)]
    left = p * sum(L); m = np.zeros((H, W), np.uint8)
    for i in range(len(L)):
        if left <= 0: break
        a = np.float32(P[i]); b = np.float32(P[i + 1]); f = min(1, left / max(L[i], 1e-6)); e = a + (b - a) * f
        cv2.line(m, (int(a[0] * 16), int(a[1] * 16)), (int(e[0] * 16), int(e[1] * 16)), 255, th, cv2.LINE_AA, shift=4); left -= L[i]
    al = m.astype(np.float32)[..., None] / 255 * op; fr *= 1 - al; fr += al * col

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
V = {k: Src(os.path.join(WORK, f"IMG_{k}.mp4")) for k in ["0336", "0337", "0338", "0332", "2994", "0331"]}
def photo(n): return np.asarray(ImageOps.exif_transpose(Image.open(os.path.join(SRC, f"IMG_{n}.JPG"))).convert("RGB").resize((W, H), Image.LANCZOS))
BEF = photo(8392)
AFT = cv2.warpAffine(photo(8391), np.float32([[0.865, 0, 540 * 0.135], [0, 0.865, 100 * 0.135]]), (W, H), borderValue=(0, 0, 0))  # shoulders + navel line up with ДО

def cover(img, w, h, fx=0.5, fy=0.5, z=1.0):
    """cover-fit (img -> w x h) keeping (fx, fy) in the middle when possible; returns float image and the matrix"""
    ih, iw = img.shape[:2]; s = max(w / iw, h / ih) * z
    tx = min(0, max(w - iw * s, w / 2 - fx * iw * s)); ty = min(0, max(h - ih * s, h / 2 - fy * ih * s))
    M = np.float32([[s, 0, tx], [0, s, ty]])
    return cv2.warpAffine(img, M, (int(w), int(h)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255, M

# ---------------- static layers ----------------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
BG = NAVY * (1 + 0.35 * np.exp(-(((xx - 540) / 900) ** 2 + ((yy - 520) / 900) ** 2)))[..., None] * 1.0
BG = np.clip(BG, 0, 1).astype(np.float32)
def ramp(y0, y1): return np.clip((yy[:, :1] - y0) / (y1 - y0), 0, 1)[..., None]
GBOT = (ramp(950, 1650) ** 1.3 * 0.92).astype(np.float32)
GTOP = ((1 - ramp(260, 820)) ** 1.4 * 0.9).astype(np.float32)
VIG = (np.clip(((xx - 540) / 760) ** 2 + ((yy - 960) / 1250) ** 2, 0, 1) ** 1.5 * 0.35)[..., None].astype(np.float32)
def shade(fr, g, col=DEEP): fr *= 1 - g; fr += g * col

_lg = np.asarray(Image.open(os.path.join(SRC, "logo.png")).convert("RGBA"))[..., 3].astype(np.float32) / 255
_ys, _xs = np.where(_lg > 0.05); _lg = _lg[_ys.min():_ys.max() + 1, _xs.min():_xs.max() + 1]
_cols = np.where(_lg.max(0) > 0.05)[0]; _gap = np.argmax(np.diff(_cols)); _split = (_cols[_gap] + _cols[_gap + 1]) // 2
LOGO_MARK, LOGO_WORD = _lg[:, :_split], _lg[:, _split:]
def logo_alpha(width):
    return cv2.resize(_lg, (width, int(_lg.shape[0] * width / _lg.shape[1])), interpolation=cv2.INTER_AREA)

def chip(fr, txt, x, y, op=1.0):
    a = alpha_img(txt, mont(30, 600)); w, h = a.shape[1] + 34, 58
    box = np.zeros((h + 4, w + 4), np.uint8); cv2.rectangle(box, (2, 2), (w + 1, h + 1), 255, -1)
    bb = cv2.GaussianBlur(box, (0, 0), 0.8).astype(np.float32) / 255
    blend(fr, bb, x, y, NAVY, 0.8 * op)
    ln = np.zeros_like(box); cv2.rectangle(ln, (2, 2), (w + 1, h + 1), 255, 2, cv2.LINE_AA); blend(fr, ln.astype(np.float32) / 255, x, y, PINK, op)
    blend(fr, a, x + 19 - 4 + 2, y + 2 + (h - a.shape[0]) / 2 + 2, WHITE, op)

def callout(fr, pt, lab, txt, t, t0, t1=1e9):
    """pulsing dot -> thin line -> typed label with a pink underline"""
    if t < t0 or t > t1 + 0.3: return
    op = 1 - eio((t - t1) / 0.3) if t > t1 else 1.0
    m = np.zeros((H, W), np.uint8); p = eback((t - t0) / 0.35); r = 9 * p
    cv2.circle(m, (int(pt[0] * 16), int(pt[1] * 16)), int(r * 16), 255, -1, cv2.LINE_AA, shift=4)
    rr = 14 + 22 * (((t - t0) * 1.4) % 1)
    cv2.circle(m, (int(pt[0] * 16), int(pt[1] * 16)), int(rr * 16), int(255 * (1 - ((t - t0) * 1.4) % 1)), 2, cv2.LINE_AA, shift=4)
    al = m.astype(np.float32)[..., None] / 255 * op; fr *= 1 - al; fr += al * PINK
    path_draw(fr, [pt, lab], eo((t - t0 - 0.15) / 0.35), WHITE, 2, False, op)
    ty = Type([txt], 36, 0, 0, 500, 40); a = ty.lines[0][0]; right = lab[0] > pt[0]
    lx = lab[0] + 14 if right else lab[0] - a.shape[1] - 6
    ty.x = lx + 4; ty.y = lab[1] - 26 + ty.top - 2; ty.draw(fr, t, t0 + 0.45, t1)
    ul = np.ones((3, int((a.shape[1] - 8) * eo((t - t0 - 0.5) / 0.5)) + 1), np.float32)
    blend(fr, ul, lx + 4 if right else lab[0] - ul.shape[1] - 10, lab[1] + 22, PINK, op)

def notch_num(fr, n, t, t0, x1=CARD[2], y1=CARD[3]):
    if t < t0: return
    a = outline_img(n, bebas(200), 2); p = eo((t - t0) / 0.6)
    cx, cy = x1 - NW / 2, y1 - NH / 2; b = a.copy(); b[:, int(b.shape[1] * p):] = 0
    blend(fr, b, cx - a.shape[1] / 2, cy - a.shape[0] / 2 + 6, PINK, 1.0)

# ---------------- card helpers ----------------
def card_frame(content, t, p, t_line, fx=0.5, fy=0.5, z=1.0, bg=None):
    """content (1080x1920 uint8) shown in the morphing frame: p=0 full screen -> p=1 notched card"""
    e = eio(p); x0, y0, x1, y1 = [lerp(a, b, e) for a, b in zip((0, 0, W, H), CARD)]
    fr = (BG if bg is None else bg).copy()
    img, _ = cover(content, x1 - x0 + 1, y1 - y0 + 1, fx, fy, z)
    pts = notch_poly(x0, y0, x1, y1, NW * e, NH * e)
    m = poly_mask(pts)[..., None]; ix0, iy0 = int(x0), int(y0)
    sub = np.zeros_like(fr); sub[iy0:iy0 + img.shape[0], ix0:ix0 + img.shape[1]] = img[:H - iy0, :W - ix0]
    fr = fr * (1 - m) + sub * m
    if t >= t_line: path_draw(fr, notch_poly(*CARD, NW, NH, 16), eio((t - t_line) / 0.9) * e, WHITE, 2)
    return fr.astype(np.float32)

def full(content, fx=0.5, fy=0.5, z=1.0):
    img, M = cover(content, W, H, fx, fy, z); return img, M

# ---------------- scenes ----------------
def h_src(t): return 0.2 + t * 0.9                                     # skip the first frames: the sticker slips off there
HOOK = Head(["ПОСЛЕ РОДОВ ТЕЛО", "УЖЕ НЕ ВОЗВРАЩАЕТСЯ", "*НАЗАД*"], 150, 70, 1170)
def s_hook(t):
    fr, _ = full(V["0336"].at(h_src(t)), 0.5, 0.42, 1.04 + 0.02 * t); shade(fr, VIG); shade(fr, GBOT)
    lg = logo_alpha(400); blend(fr, lg, 540 - 200, 215, PINK, eo((t + 0.1) / 0.5))
    HOOK.draw(fr, t, -0.3)
    blend(fr, np.ones((4, int(260 * eo((t - 0.9) / 0.6)) + 1), np.float32), 72, HOOK.y + HOOK.h + 60, PINK)
    return fr

HA = Head(["ПАЦИЕНТКА РОЖАЛА", "И *КОРМИЛА*"], 112, 70, 1300)
TA = Type(["после родов ослабла передняя брюшная стенка,", "а грудь немного опустилась"], 36, 72, 1300 + HA.h + 70)
def s_a(t):
    lt = t - T_A
    if lt < 0.65: c = V["0336"].at(h_src(t))
    else: c = V["2994"].at(src_of(t, "2994"))
    fr = card_frame(c, t, lt / 0.6, T_A + 0.3, 0.5, 0.45 if lt < 0.65 else 0.36, 1.0 + 0.012 * lt)
    if 0.5 < lt < 0.8:                                                   # soft dissolve from ДО-video into the consultation
        k = (lt - 0.5) / 0.3; f2 = card_frame(V["0336"].at(h_src(t)), t, 1, T_A + 0.3, 0.5, 0.45); fr = f2 * (1 - k) + fr * k
    HA.draw(fr, t, T_A + 0.45); TA.draw(fr, t, T_A + 1.1); notch_num(fr, "01", t, T_A + 0.7)
    return fr

HB = Head(["ИНОГДА СПОРТ", "*УЖЕ НЕ РЕШАЕТ ПРОБЛЕМУ*"], 120, 70, 270)
TB = Type(["если есть избыток кожи и диастаз"], 38, 72, 270 + HB.h + 62)
def b_src(t): return 0.1 + (t - T_B) * 0.7                       # close pinch shot, slowed down
def s_b(t):
    lt = t - T_B; fr, M = full(V["0337"].at(b_src(t)), 0.5, 0.55, 1.03 + 0.012 * lt); shade(fr, VIG); shade(fr, GTOP)
    HB.draw(fr, t, T_B + 0.3); TB.draw(fr, t, T_B + 1.0)
    P = lambda x, y: (M[0, 0] * x + M[0, 2], M[1, 1] * y + M[1, 2])
    callout(fr, P(430, 960), (250, 1180), "растяжки", t, T_B + 1.7)
    callout(fr, P(620, 880), (720, 660), "избыток кожи", t, T_B + 2.4)
    return fr

HC = Head(["УШИЛИ *ДИАСТАЗ*"], 130, 70, 1300, maxw=700)
TC = Type(["сформировали более собранную", "переднюю брюшную стенку"], 36, 72, 1300 + HC.h + 70)
T_OP = bstart("C") + (46.35 - 44.86) + 0.08 + 1.1                     # «…убрать ушиванием» -> operating-room footage
def stitch(fr, t, t0, ox=880, oy=1285):
    """two muscle bands drawn apart, a pink lacing pulls them together"""
    if t < t0: return
    a = eo((t - t0) / 0.5); d = lerp(58, 16, eio((t - t0 - 0.9) / 0.8)); h = 190
    for sx in (-1, 1):
        path_draw(fr, [(ox + sx * d, oy), (ox + sx * (d + 8), oy + h / 2), (ox + sx * d, oy + h)], a, WHITE, 3, False)
    zz = []; n = 7
    for i in range(n + 1): zz.append((ox + (-d if i % 2 == 0 else d), oy + 10 + i * (h - 20) / n))
    path_draw(fr, zz, eo((t - t0 - 0.35) / 0.9), PINK, 3, False)
def s_c(t):
    lt = t - T_C
    if lt < 0.65: c = V["0337"].at(b_src(t)); fy = 0.55
    elif t < T_OP: c = V["2994"].at(src_of(t, "2994")); fy = 0.36
    else: c = V["0338"].at(0.4 + (t - T_OP) * 0.9); fy = 0.42
    fr = card_frame(c, t, lt / 0.6, T_C + 0.3, 0.5, fy, 1.0 + 0.012 * lt)
    for tb, a_get in [(T_C + 0.5, lambda: card_frame(V["0337"].at(b_src(t)), t, 1, T_C + 0.3, 0.5, 0.55)),
                      (T_OP - 0.15, lambda: card_frame(V["2994"].at(src_of(t, "2994")), t, 1, T_C + 0.3, 0.5, 0.36))]:
        if tb < t < tb + 0.3: k = (t - tb) / 0.3; fr = a_get() * (1 - k) + fr * k
    HC.draw(fr, t, T_C + 0.45); TC.draw(fr, t, T_C + 1.2); stitch(fr, t, T_C + 0.9); notch_num(fr, "02", t, T_C + 0.7)
    return fr

HD = Head(["*АБДОМИНОПЛАСТИКА*"], 124, 70, 265, maxw=940)
LD = [Head(["+ НОВЫЙ КОНТУР ЖИВОТА"], 70, 70, 265 + HD.h + 50), Head(["+ БОЛЕЕ ВЫРАЖЕННАЯ ТАЛИЯ"], 70, 70, 265 + HD.h + 125)]
for _h in LD: _h.words[0] = (0, (_h.words[0][1][0], PINK, _h.words[0][1][2]))
T_WIPE = T_D + 1.25
def s_d(t):
    lt = t - T_D; z = 1.0 + 0.02 * lt
    b, _ = full(BEF, 0.5, 0.45, z); a, _ = full(AFT, 0.5, 0.45, z)
    p = eio((t - T_WIPE) / 0.95); xd = int(W * (1 - p)); fr = b.copy(); fr[:, xd:] = a[:, xd:]
    shade(fr, GTOP); shade(fr, GBOT * 0.6)
    if 0 < p < 1:
        fr[:, max(0, xd - 2):xd + 2] = PINK
        m = np.zeros((H, W), np.uint8); cv2.circle(m, (xd, 1060), 30, 255, -1, cv2.LINE_AA); blend(fr, m.astype(np.float32) / 255, 0, 0, PINK)
        ar = alpha_img("‹ ›", mont(34, 600)); blend(fr, ar, xd - ar.shape[1] / 2, 1060 - ar.shape[0] / 2 - 2, NAVY)
    chip(fr, "ДО", 70, 1500, 1 - eo((t - T_WIPE - 0.3) / 0.3) if t > T_WIPE else eo((t - T_D - 0.4) / 0.4))
    if t > T_WIPE + 0.5: chip(fr, "ПОСЛЕ", 70, 1500, eo((t - T_WIPE - 0.5) / 0.3))
    HD.draw(fr, t, T_D + 0.3)
    LD[0].draw(fr, t, bstart("D") + 1.1); LD[1].draw(fr, t, bstart("D") + (61.97 - 59.68) + 0.08 + 0.9)
    return fr

HE = Head(["ПОДТЯЖКУ ГРУДИ", "*НЕ ДЕЛАЛИ*"], 120, 70, 1300)
TE = Type(["пациентка была против дополнительных", "разрезов на груди"], 36, 72, 1300 + HE.h + 70)
HE3 = Head(["АНАТОМИЧЕСКИЕ", "ИМПЛАНТЫ"], 130, 70, 1190)
def s_e(t):
    lt = t - T_E
    if t < T_E3:
        fr = card_frame(V["2994"].at(src_of(t, "2994")), t, 1, T_E + 0.15, 0.5, 0.33, 1.02 + 0.01 * lt)
        HE.draw(fr, t, T_E + 0.35, T_E3 - 0.05); TE.draw(fr, t, T_E + 1.4, T_E3 - 0.05); notch_num(fr, "03", t, T_E + 0.5)
        x0 = HE.x - 6; w = HE.rows[0][0][0][0].shape[1] - 2; ps = eio((t - T_E - 1.1) / 0.45)   # strike through «ПОДТЯЖКУ»
        if ps > 0 and t < T_E3: blend(fr, np.ones((10, int(w * ps) + 1), np.float32), x0, HE.y + HE.size * 0.3, PINK, 1 - eo((t - T_E3 + 0.1) / 0.2))
        if t > T_E3 - 0.3:                                               # the implant video slides into the card
            k = eo((t - T_E3 + 0.3) / 0.3); f2 = card_frame(V["0332"].at(0.0), t, 1, -1, 0.5, 0.4); fr = fr * (1 - k) + f2 * k
        return fr
    le = t - T_E3; c = V["0332"].at(min(2.95, le * 0.95))
    fr = card_frame(c, t, 1 - le / 0.6, 1e9, 0.5, 0.4 if le < 0.6 else 0.4, 1.0 + 0.015 * le)
    if le > 0.45:
        sh = GBOT * eo((le - 0.45) / 0.3); shade(fr, sh)
        HE3.draw(fr, t, T_E3 + 0.5)
        n = int(round(375 * eo((t - T_E3 - 0.9) / 0.9))); y = HE3.y + HE3.h + 40
        if t > T_E3 + 0.9:
            a = alpha_img(str(n), bebas(210)); blend(fr, a, 66, y - cap_top(bebas(210)), PINK)
            m = alpha_img("МЛ", bebas(96)); blend(fr, m, 66 + alpha_img("375", bebas(210)).shape[1] + 4, y + 210 * 0.72 - 96 * 0.72 - cap_top(bebas(96)), PINK, eo((t - T_E3 - 1.6) / 0.3))
    return fr

HF = Head(["ПЕРВЫЕ СУТКИ", "*ПОСЛЕ ОПЕРАЦИИ*"], 130, 70, 1240)
TF = Type(["большая комплексная операция:", "абдоминопластика + импланты"], 36, 72, 1240 + HF.h + 70)
def s_f(t):
    lt = t - T_F; fr, _ = full(V["0331"].at(src_of(t, "0331")), 0.5, 0.3, 1.05 + 0.01 * lt); shade(fr, VIG); shade(fr, GBOT)
    HF.draw(fr, t, T_F + 0.35); TF.draw(fr, t, T_F + 1.1)
    return fr

COLL = [8393, 8395, 8396]                                             # surgeon's own collages (front, 3/4, side): ПОСЛЕ on top, ДО below
COLL_IMG = [photo(n).astype(np.float32) / 255 for n in COLL]
T_COLL = [T_G, T_G + 1.7, T_G + 3.3]; PUSH = 0.4
def s_g(t):
    """the three collages full frame, uncropped; each next one pushes in from the right"""
    k = max(i for i, tc in enumerate(T_COLL) if t >= tc)
    if k > 0 and t < T_COLL[k] + PUSH:
        p = eio((t - T_COLL[k]) / PUSH); x = int(W * (1 - p)); fr = np.empty_like(BG)
        fr[:, :x] = COLL_IMG[k - 1][:, W - x:]; fr[:, x:] = COLL_IMG[k][:, :W - x]; fr[:, max(0, x - 2):x + 2] = PINK
    else: fr = COLL_IMG[k].copy()
    lt = t - T_G
    chip(fr, "ПОСЛЕ", 30, 240, eo((lt - 0.35) / 0.3)); chip(fr, "ДО", 30, 1010, eo((lt - 0.5) / 0.3))
    return fr

HG = Head(["ГРУДЬ + ЖИВОТ", "*ЗА ОДИН ЭТАП*"], 150, 70, 470)
TG = Type(["дальше будем показывать", "восстановление и результат"], 36, 72, 470 + HG.h + 70)
def s_out(t):
    lt = t - T_OUT; fr = BG.copy(); w = 760
    HG.draw(fr, t, T_OUT + 0.3); TG.draw(fr, t, T_OUT + 1.0)
    lg = logo_alpha(w); h = lg.shape[0]; x, y = 540 - w / 2, 1250 - h / 2
    sp = int(w * _split / _lg.shape[1]); mk = lg[:, :sp].copy(); wd = lg[:, sp:].copy()
    yy2, xx2 = np.mgrid[0:mk.shape[0], 0:mk.shape[1]]; ang = (np.arctan2(xx2 - mk.shape[1] / 2, -(yy2 - mk.shape[0] / 2)) % (2 * np.pi)) / (2 * np.pi)
    mk *= np.clip((eio((lt - 1.0) / 0.9) - ang) * 20, 0, 1)
    wd[:, int(wd.shape[1] * eio((lt - 1.45) / 0.8)):] = 0
    blend(fr, mk, x, y, PINK); blend(fr, wd, x + sp, y, PINK)
    ty = Type(["пластический хирург"], 40, 0, 0, 300, 30); ty.x = 540 - ty.lines[0][0].shape[1] / 2 + 4; ty.y = y + h + 70 + ty.top
    ty.draw(fr, t, T_OUT + 2.0)
    return fr

SCENES = [(0, s_hook), (T_A, s_a), (T_B, s_b), (T_C, s_c), (T_D, s_d), (T_E, s_e), (T_F, s_f), (T_G, s_g), (T_OUT, s_out)]
WAVED = {T_B, T_D, T_E, T_F, T_G, T_OUT}
def wave_mix(old, new, p, seed=0.0):
    """new scene rises from the bottom behind a soft wave edge with a pink line on it"""
    e = eio(p); amp = 70 * np.sin(np.pi * e)
    yb = (H + 160) * (1 - e) - 80 + amp * np.sin(np.arange(W, dtype=np.float32) / W * 2 * np.pi * 1.1 + seed + e * 2.5)
    d = yy - yb[None, :]; m = np.clip(d / 3 + 0.5, 0, 1)[..., None]
    fr = old * (1 - m) + new * m
    ln = np.clip(1 - np.abs(d - 3) / 3.5, 0, 1)[..., None] * (0 < e < 1); fr = fr * (1 - ln) + PINK * ln
    return fr.astype(np.float32)
def render(t):
    k = max(i for i, (s0, _) in enumerate(SCENES) if t >= s0); s0, fn = SCENES[k]
    if s0 in WAVED and t < s0 + WAVE: return wave_mix(SCENES[k - 1][1](t), fn(t), (t - s0) / WAVE, k * 1.7)
    return fn(t)

# ---------------- audio ----------------
def lp(x, fc):
    try:
        from scipy.signal import lfilter; a = np.exp(-2 * np.pi * fc / SR); return lfilter([1 - a], [1, -a], x)
    except ImportError:
        X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); return np.fft.irfft(X / np.sqrt(1 + (f / fc) ** 2), len(x))
def hp(x, fc): return x - lp(x, fc)
rng = np.random.default_rng(7)
def env(n, a, d): t = np.arange(n) / SR; return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / d)
def whoosh(dur=0.6, f0=300, f1=5000):
    n = int(dur * SR); x = rng.standard_normal(n); t = np.arange(n) / SR
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1 / SR); out = np.zeros(n)
    for k in range(8):                                                  # stepped band sweep
        c = f0 * (f1 / f0) ** (k / 7); bp = np.exp(-((np.log(f + 1) - np.log(c)) ** 2) / 0.25)
        seg = np.fft.irfft(X * bp, n); w = np.exp(-((t / dur - k / 7) ** 2) / 0.02); out += seg * w
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
def riser(dur=1.2):
    n = int(dur * SR); t = np.arange(n) / SR; x = hp(rng.standard_normal(n), 1500) * (t / dur) ** 2
    return x / np.abs(x).max() + 0.3 * np.sin(2 * np.pi * np.cumsum(300 + 900 * (t / dur) ** 2) / SR) * (t / dur) ** 2
def scribble(dur=0.5):
    n = int(dur * SR); x = hp(rng.standard_normal(n), 2500); t = np.arange(n) / SR
    return x * (0.6 + 0.4 * np.sin(2 * np.pi * 14 * t)) * np.sin(np.pi * t / dur) * 0.6

def music(n):
    """soft ambient bed: vi-IV-I-V pads (D major), sub, gentle kick + shaker after the hook, bell arpeggio"""
    t = np.arange(n) / SR; bar = 60 / 92 * 4
    nm = lambda m: 440 * 2 ** ((m - 69) / 12)
    chords = [[47, 59, 62, 66, 69], [43, 59, 62, 66, 71], [38, 57, 62, 66, 69], [45, 57, 61, 64, 69]]  # Bm7 Gmaj7 D A
    pad = np.zeros(n); sub = np.zeros(n); arp = np.zeros(n)
    for ci in range(int(np.ceil(t[-1] / (2 * bar))) + 1):
        ch = chords[ci % 4]; s0 = ci * 2 * bar; i0 = int(s0 * SR); i1 = min(n, int((s0 + 2 * bar + 0.6) * SR))
        if i0 >= n: break
        tt = t[i0:i1] - s0; e = np.minimum(1, tt / 0.6) * np.clip((2 * bar + 0.6 - tt) / 0.6, 0, 1)
        for m in ch[1:]:
            for dt in (-0.07, 0.0, 0.08):
                f = nm(m) * 2 ** (dt / 12)
                pad[i0:i1] += e * sum(np.sin(2 * np.pi * f * h * tt + h) / h ** 1.7 for h in (1, 2, 3, 4))
        sub[i0:i1] += e * np.sin(2 * np.pi * nm(ch[0]) * tt)
        notes = [ch[1] + 12, ch[2] + 12, ch[3] + 12, ch[4] + 12, ch[3] + 12, ch[2] + 12, ch[4] + 12, ch[3] + 12]
        for k in range(16):
            a0 = int((s0 + k * bar / 8) * SR); m = notes[k % 8]
            if a0 >= n: break
            ln = min(n - a0, int(0.7 * SR)); ta = np.arange(ln) / SR
            arp[a0:a0 + ln] += np.sin(2 * np.pi * nm(m) * ta) * np.exp(-ta / 0.22) * (0.8 if k % 2 == 0 else 0.5)
    pad = lp(pad, 1800); beat = 60 / 92
    kick = np.zeros(n); hat = np.zeros(n); pump = np.ones(n)
    for k in range(int(t[-1] / beat) + 1):
        tb = k * beat
        if tb < T_A or tb > T_OUT + 0.2: continue
        i = int(tb * SR); kk = boom(0.4) * 0.8; ln = min(n - i, len(kk)); kick[i:i + ln] += kk[:ln]
        pl = min(n - i, int(0.35 * SR)); pump[i:i + pl] = np.minimum(pump[i:i + pl], 0.55 + 0.45 * np.linspace(0, 1, pl) ** 0.7)
        j = int((tb + beat / 2) * SR); hh = tick() * 2.2; ln = min(n - j, len(hh))
        if ln > 0: hat[j:j + ln] += hh[:ln]
    mix = pad / np.abs(pad).max() * 0.55 * pump + sub * 0.22 * pump + arp / max(1e-6, np.abs(arp).max()) * 0.16 + kick * 0.42 + hat * 0.12
    return mix / np.abs(mix).max()

MUSIC = os.path.join(SRC, "music.wav")                               # ffmpeg -i track.mp4 -vn -ac 2 -ar 48000 music.wav
M0, M_DROP = 10.50, 14.45                                              # the track's drop lands on the first frame morph (T_A)
G_VO, G_GAP = 0.022, 0.075                                             # music RMS under the voice / between phrases (~-33 / -22 dBFS): voice sits ~13 dB above
def track(n):
    w = wave.open(MUSIC); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64).reshape(-1, w.getnchannels()) / 32768
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    x = x / np.sqrt(np.mean(x[int(M_DROP * SR):].mean(1) ** 2))         # loud part -> RMS 1
    x[:int(M_DROP * SR)] *= 4.5                                         # lift the soft intro under the hook
    seg = x[int(M0 * SR):int(M0 * SR) + n]
    return np.pad(seg, ((0, n - len(seg)), (0, 0)))

def build_audio(path):
    n = int(DUR * SR); vo = np.zeros(n); act = np.zeros(n); srcs = {}
    for b, s, si, so, t0 in SEG:
        if s not in srcs:
            w = wave.open(os.path.join(WORK, f"IMG_{s}.wav")); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
            srcs[s] = hp(x, 90)
        seg = srcs[s][int(si * SR):int(so * SR)].copy(); f = int(0.012 * SR); seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
        seg /= np.sqrt(np.mean(seg[np.abs(seg) > 0.01] ** 2)) + 1e-6; i = int(t0 * SR); vo[i:i + len(seg)] += seg[:n - i]
        act[max(0, i - int(0.25 * SR)):i + len(seg) + int(0.1 * SR)] = 1          # music dips before the first word
    vo = np.tanh(vo * 0.16 * 1.6) / 1.6                                  # gentle level + soft limiting
    d = np.convolve(act, np.ones(int(0.25 * SR)) / int(0.25 * SR), "same")
    tt = np.arange(n) / SR; fade = np.clip((DUR - tt) / 1.6, 0, 1) ** 1.5 * np.minimum(1, tt / 0.15)
    if os.path.exists(MUSIC): mus = track(n) * ((G_GAP - (G_GAP - G_VO) * np.clip(d, 0, 1)) * fade)[:, None]
    else: mus = (music(n) * (0.26 - 0.19 * d) * fade)[:, None] * np.float64([[1.0, 0.92]])   # fallback: synth bed
    sfx = np.zeros(n + 2 * SR)
    def put(x, t, g):
        i = int(max(0, t) * SR); seg = sfx[i:i + len(x)]; seg += x[:len(seg)] * g
    WH = whoosh(); WS = whoosh(0.4, 700, 8000); BM = boom(); PP = pop(); CH = chime(); RS = riser(); SC = scribble()
    put(BM, 0.0, 0.5); put(CH, 0.05, 0.12)
    for T in sorted(WAVED): put(WH, T - 0.12, 0.32); put(boom(0.5), T - 0.04, 0.2)   # thump in the gap, not on the first word
    for T in (T_A, T_C): put(WS, T - 0.05, 0.3); put(PP, T + 0.6, 0.25)         # frame morphs
    put(WS, T_E3 - 0.05, 0.3)
    for hd, t0 in [(HOOK, -0.3), (HA, T_A + 0.45), (HB, T_B + 0.3), (HC, T_C + 0.45), (HD, T_D + 0.3), (HE, T_E + 0.35),
                   (HE3, T_E3 + 0.5), (HF, T_F + 0.35), (HG, T_OUT + 0.3)]:
        for k in range(len(hd.words)): put(pop(900 + 60 * (k % 3), 320), t0 + k * hd.gap + 0.08, 0.10)
    for ty, t0 in [(TA, T_A + 1.1), (TB, T_B + 1.0), (TC, T_C + 1.2), (TE, T_E + 1.4), (TF, T_F + 1.1), (TG, T_OUT + 1.0)]:
        for k in range(ty.n):
            if k % 2 == 0: put(tick(), t0 + k / ty.cps, 0.06)
    for t0 in (T_B + 1.7, T_B + 2.4): put(pop(1500, 700), t0, 0.22)
    put(SC, T_C + 1.25, 0.18)                                                    # lacing
    put(WS, T_WIPE - 0.05, 0.28)   # list items appear silently: they land on the voice
    put(whoosh(0.35, 1500, 9000), T_E + 1.05, 0.3)                              # strike-through
    for k in range(14): put(tick(), T_E3 + 0.9 + 0.9 * (k / 14) ** 0.5, 0.18)
    put(CH, T_E3 + 1.8, 0.14)
    put(RS, T_OUT - 1.15, 0.12); put(CH, T_OUT + 1.05, 0.2); put(chime(1975.5), T_OUT + 1.5, 0.1)
    for tc in T_COLL[1:]: put(WS, tc - 0.1, 0.25)                                 # collage push
    sfx[:n] *= 1 - 0.8 * np.clip(d, 0, 1)                                     # effects step back under the voice
    mix = vo[:, None] + mus + sfx[:n, None] * np.float64([[0.92, 1.0]])
    mix /= max(1, np.abs(mix).max() / 0.95)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

# ---------------- cover ----------------
def make_cover(path):
    fr = BG.copy(); h2 = 1180
    for k, img in enumerate([BEF, AFT]):
        im, _ = cover(img, 540, h2, 0.5, 0.4, 1.22); fr[160:160 + h2, k * 540:(k + 1) * 540] = im   # zoomed: stays above AFT's empty bottom
    fr[160:160 + h2, 538:542] = PINK
    m = np.clip((yy[:, :1] - (160 + h2 - 300)) / 300, 0, 1)[..., None]; fr = fr * (1 - m) + BG * m; shade(fr, GBOT)
    chip(fr, "ДО", 40, 200); chip(fr, "ПОСЛЕ", 580, 200)
    hd = Head(["ПОСЛЕ РОДОВ СПОРТ", "*УЖЕ НЕ ПОМОГАЛ*"], 170, 60, 1250, maxw=960); hd.draw(fr, 9, 0)
    ty = Type(["грудь + живот за одну операцию"], 44, 64, 1250 + hd.h + 80, 400); ty.draw(fr, 9, 0)
    lg = logo_alpha(380); blend(fr, lg, 540 - 190, 60, PINK)
    cv2.imwrite(path, cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

def to8(fr): return (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)
if __name__ == "__main__":
    print(f"duration {DUR:.2f}s; scenes", [round(s, 2) for s, _ in SCENES])
    if "--preview" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]
        for t in ts: cv2.imwrite(os.path.join(WORK, f"prev_{t:05.2f}.jpg"), cv2.cvtColor(to8(render(t)), cv2.COLOR_RGB2BGR))
        sys.exit()
    if "--cover" in sys.argv: make_cover(os.path.join(ROOT, f"{NAME}_cover.jpg")); sys.exit()
    build_audio(os.path.join(WORK, "mix.wav"))
    vid = os.path.join(WORK, "video.mp4")
    if "--audio-only" not in sys.argv:                                    # --audio-only: remix sound onto the last render
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                              "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", vid], stdin=subprocess.PIPE)
        N = int(round(DUR * FPS))
        for i in range(N):
            p.stdin.write(to8(render(i / FPS)).tobytes())
            if i % 60 == 0: print(f"{i}/{N}", flush=True)
        p.stdin.close(); p.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid, "-i", os.path.join(WORK, "mix.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", os.path.join(ROOT, f"{NAME}.mp4")], check=True)
    make_cover(os.path.join(ROOT, f"{NAME}_cover.jpg"))
    print("done")
