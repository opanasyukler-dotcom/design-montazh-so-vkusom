#!/usr/bin/env python3
"""Reels «Сняли стрипы» — комплексная операция (Т-образная подтяжка груди с липофилингом, абдоминопластика, липосакция),
Анна Тваури.

Brand system of the other reels (navy, powder pink, Bebas Neue + Montserrat, logo), cut fast for short attention spans:
a teaser of ПОСЛЕ under tape strips that peel off, ДО on the surgeon's phone and in the studio, a checklist that ticks
on every named procedure over changing ДО angles, donor zones lighting up on the body, the scar close-up traced in pink,
the silhouette, three ДО (top) / ПОСЛЕ (bottom) collages cut neck to fingertips with shutter clicks, the live «мы очень
довольны» and the closing card. Whole sentences only; faces are blurred; ПОСЛЕ photos get a light skin retouch that keeps
the scars visible.

usage:
  REEL_SRC=<folder with IMG_0331/0355.MOV, IMG_1635/1636/1654.MP4, IMG_0305/0310/0313.heic, IMG_1637/1643/1645/1651.JPG, logo.png, music.wav> python3 tools/reel_stripes.py [--preview t1,t2 | --cover | --audio-only]
Sources are first normalised into $REEL_SRC/_work (1080x1920, 30 fps; the voice 48 kHz mono wav):
  ffmpeg -i IMG_x.MOV -vf "scale=1080:1920:flags=lanczos,fps=30,format=yuv420p" -an -c:v libx264 -crf 14 _work/IMG_x.mp4
  ffmpeg -i IMG_1635.MP4 -vn -ac 1 -ar 48000 _work/IMG_1635.wav
"""
import os, sys, subprocess, wave, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_retouch import retouch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("REEL_SRC", os.path.join(ROOT, "dl_stripes"))
WORK = os.path.join(SRC, "_work")
NAME = "Reels_stripes_removed"
W, H, FPS, SR = 1080, 1920, 30, 48000
NAVY = np.float32([0, 18, 58]) / 255; PINK = np.float32([233, 197, 193]) / 255; WHITE = np.float32([1, 1, 1])
DEEP = np.float32([0, 8, 30]) / 255

def cl(x): return min(1.0, max(0.0, x))
def eo(x): x = cl(x); return 1 - (1 - x) ** 3
def eio(x): x = cl(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x): x = cl(x); c = 1.7; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def lerp(a, b, p): return a + (b - a) * p

# ---------------- voice edit: (block, source, [(in, out), …]) — whole sentences, all from the surgeon's video IMG_1635
VO = [("A", "1635", [(2.14, 5.22)]),                                   # сегодня хочу показать нашу замечательную пациентку
      ("B", "1635", [(5.20, 7.36)]),                                   # вот такой у нас был исходник
      ("C", "1635", [(7.48, 17.18)]),                                  # мы выполнили комплексную операцию. это была подтяжка груди… абдоминопластика и липосакция
      ("D", "1635", [(17.36, 20.90)]),                                 # забор жира у нас был по бокам, внутренние бёдра, колени
      ("E", "1635", [(24.90, 30.06)]),                                 # сегодня сняли стрипы уже, они нам уже не нужны. шовчики абсолютно ровненькие
      ("F", "1635", [(35.82, 38.70)]),                                 # на данном этапе, я считаю, очень прекрасный результат
      ("G", "1635", [(41.74, 44.96)]),                                 # красивый, хороший силуэт, очень симпатичная грудь
      ("Q", "1635", [(38.70, 39.52)]),                                 # вам всё нравится?
      ("R", "1635", [(39.50, 40.68)]),                                 # мы очень довольны (the patient, quieter — levelled on its own)
      ("S", "1635", [(40.70, 41.72)])]                                 # и мы довольны
GAP = {"B": 0.3, "C": 2.0, "D": 0.35, "E": 0.45, "F": 1.1, "G": 0.35, "Q": 3.2, "R": 0.0, "S": 0.12}
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

T_B = vstart("B") - 0.12; T_B2 = vend("B") + 0.12; T_C = vstart("C") - 0.15; T_C2 = at_src("1635", 10.00)
T_D = vstart("D") - 0.15; T_E = vstart("E") - 0.15; T_E2 = at_src("1635", 28.36); T_F = vstart("F") - 0.15
T_G = vstart("G") - 0.15; T_Q = vstart("Q") - 0.15; T_Z = vend("S") + 0.45; DUR = T_Z + 3.6
T_ITEMS = [at_src("1635", s_) for s_ in (10.46, 12.34, 14.66, 16.26)]  # подтяжка · липофилингом · абдоминопластика · липосакция
T_ZONES = [at_src("1635", s_) for s_ in (18.80, 19.50, 20.40)]         # бокам · внутренние бёдра · колени
T_SWAP = at_src("1635", 37.70)                                         # «прекрасный результат»
T_K = [T_G, at_src("1635", 43.45), vend("G") + 0.6]                    # collage cuts: in the pauses, the last one holds

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
V = {k: Src(os.path.join(WORK, f"IMG_{k}.mp4")) for k in ["1635", "1636", "1654", "0331", "0355"]}
def anon(img, y=180):
    """blur everything above `y` (face)"""
    if y <= 0: return img
    top = img[:y + 80].astype(np.float32); bl = cv2.resize(cv2.GaussianBlur(cv2.resize(top, None, fx=0.12, fy=0.12), (0, 0), 3), (W, y + 80))
    m = np.clip((y + 40 - np.arange(y + 80)) / 80, 0, 1)[:, None, None]; out = img.copy(); out[:y + 80] = (top * (1 - m) + bl * m).astype(np.uint8); return out
def photo(n):
    ext = "heic" if n in ("0305", "0310", "0313") else "JPG"
    if ext == "heic":
        import pillow_heif; pillow_heif.register_heif_opener()
    return np.asarray(ImageOps.exif_transpose(Image.open(os.path.join(SRC, f"IMG_{n}.{ext}"))).convert("RGB"))
def soft(rgb, strong=True):
    """light skin retouch on ПОСЛЕ; the scar frame gets almost none so the seams stay readable"""
    return retouch(rgb, 0.3, 0.2, 0.05) if strong else retouch(rgb, 0.15, 0.0, 0.03)
def box_frame(im, box):
    """crop a 9:16 `box` (source px) and scale it DOWN to the frame"""
    x0, y0, x1, y1 = box; return cv2.resize(im[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_AREA)

# neck → fingertips for the collages: (neck y, fingertip y, body centre x) in source px
NECK = {"0305": (220, 2620, 1142), "0310": (240, 2900, 1160), "0313": (280, 3080, 1156),
        "1637": (140, 1950, 788), "1643": (300, 2120, 788), "1645": (300, 2160, 788)}
def panel(im, n, bh=930, pw=W, ph=960):
    """one collage half: body scaled so neck→fingertips is `bh` px (a down-scale), centred; edges fade into the backdrop"""
    y0, y1, cx = NECK[n]; s = bh / (y1 - y0); hw = int(pw / 2 / s); x0 = max(0, cx - hw); x1 = min(im.shape[1], cx + hw)
    c = cv2.resize(im[y0:y1, x0:x1], (int((x1 - x0) * s), bh), interpolation=cv2.INTER_AREA).astype(np.float32)
    bgc = np.float32(np.median(im[y0:y0 + 200, max(0, x0):max(0, x0) + 60].reshape(-1, 3), 0))
    out = np.zeros((ph, pw, 3), np.float32) + bgc; ox = int((x0 - (cx - hw)) * s); oy = (ph - bh) // 2
    fx = np.minimum(1, np.minimum(np.arange(c.shape[1]), c.shape[1] - 1 - np.arange(c.shape[1])) / 40)[None, :, None]
    out[oy:oy + bh, ox:ox + c.shape[1]] = c * fx + bgc * (1 - fx)
    return out / 255
PH = {n: photo(n) for n in ["0305", "0310", "0313", "1637", "1643", "1645", "1651"]}
for n in ("1637", "1643", "1645"): PH[n] = soft(PH[n])
SCAR_IM = soft(PH.pop("1651"), False)
COLLS = []
for b, a in (("0305", "1637"), ("0310", "1643"), ("0313", "1645")):
    COLLS.append(np.vstack([panel(PH[b], b), panel(PH[a], a)]).astype(np.float32))
CHK = [box_frame(PH["0305"], (342, 300, 1942, 3144)), box_frame(PH["0310"], (300, 300, 1900, 3144)),
       box_frame(PH["0305"], (342, 900, 1942, 3744)), box_frame(PH["0313"], (300, 900, 1900, 3744))]   # ДО: chest · ¾ chest · belly · flank
OPEN = PH["1637"]; FINAL = anon(cv2.resize(PH["1637"], (W, H), interpolation=cv2.INTER_AREA), 60)

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
    'A': [('Сегодня', 0.04, 0.82), ('хочу', 0.82, 1.1), ('показать', 1.1, 1.58), ('нашу', 1.58, 1.82), ('замечательную', 1.82, 2.38), ('пациентку.', 2.38, 3.06)],
    'B': [('Вот', 0.06, 0.2), ('такой', 0.2, 0.34), ('у', 0.34, 0.36), ('нас', 0.36, 0.72), ('был', 0.72, 0.9), ('исходник.', 0.9, 1.62)],
    'C': [('Мы', 0.08, 0.44), ('выполнили', 0.44, 1.02), ('комплексную', 1.02, 1.62), ('операцию.', 1.62, 2.26), ('Это', 2.58, 2.76), ('была', 2.76, 2.98),
          ('подтяжка', 2.98, 4.38), ('груди', 4.38, 4.76), ('с', 4.76, 4.86), ('липофилингом', 4.86, 5.66), ('Т-образная.', 5.66, 6.44),
          ('У', 6.76, 6.78), ('нас', 6.78, 7.0), ('была', 7.0, 7.18), ('абдоминопластика', 7.18, 8.54), ('и', 8.54, 8.78), ('липосакция.', 8.78, 9.54)],
    'D': [('Забор', 0.08, 0.4), ('жира', 0.4, 0.76), ('у', 0.76, 0.78), ('нас', 0.78, 1.06), ('был', 1.06, 1.28), ('по', 1.28, 1.44), ('бокам,', 1.44, 1.9),
          ('внутренние', 2.0, 2.48), ('бёдра,', 2.48, 2.92), ('колени.', 3.04, 3.4)],
    'E': [('Сегодня', 0.08, 0.5), ('сняли', 0.5, 1.36), ('стрипы', 1.36, 1.9), ('уже,', 1.9, 2.2), ('они', 2.2, 2.38), ('нам', 2.38, 2.56), ('уже', 2.56, 2.76),
          ('не', 2.76, 2.92), ('нужны.', 2.92, 3.42), ('Шовчики', 3.5, 4.0), ('абсолютно', 4.0, 4.48), ('ровненькие.', 4.48, 5.06)],
    'F': [('На', 0.08, 0.32), ('данном', 0.32, 0.68), ('этапе,', 0.68, 1.36), ('я', 1.36, 1.44), ('считаю,', 1.44, 1.76), ('очень', 1.76, 1.92),
          ('прекрасный', 1.92, 2.44), ('результат.', 2.44, 2.8)],
    'G': [('Красивый,', 0.08, 0.56), ('хороший', 0.78, 0.96), ('силуэт,', 0.96, 1.54), ('очень', 1.7, 1.92), ('симпатичная', 1.92, 2.68), ('грудь.', 2.68, 3.12)],
    'Q': [('Вам', 0.06, 0.16), ('всё', 0.16, 0.3), ('нравится?', 0.3, 0.74)],
    'R': [('Мы', 0.06, 0.32), ('очень', 0.32, 0.5), ('довольны.', 0.5, 0.96)],
    'S': [('И', 0.06, 0.26), ('мы', 0.26, 0.36), ('довольны.', 0.36, 0.96)],
}
SUB_Y = 1420                                                           # Instagram Reels safe zone: x <= 940, y 250-1500
def _sub_lines():
    out = []
    for b, s_, parts, t0, d in SEG:
        line = []
        for k, (w, a, e) in enumerate(SUBS[b]):
            line.append((w, t0 + a, t0 + e))
            txt = " ".join(x[0] for x in line)
            if len(txt) >= 22 or w[-1] in ".,?!" and len(txt) >= 12 or k == len(SUBS[b]) - 1:
                out.append(line); line = []
    return out
SUB_LINES = _sub_lines()
def subtitles(fr, t):
    for i, line in enumerate(SUB_LINES):
        t0 = line[0][1] - 0.08; t1 = min(line[-1][2] + 0.35, SUB_LINES[i + 1][0][1] - 0.08 if i + 1 < len(SUB_LINES) else 1e9)
        if not (t0 <= t < t1): continue
        f = mont(48, 300); sp = f.getlength(" "); ims = [alpha_img(w, f) for w, _, _ in line]   # thin Montserrat, like the small lines
        tw = sum(a.shape[1] - 8 for a in ims) + sp * (len(ims) - 1); x = 500 - tw / 2; h = 74
        op = min(1, (t - t0) / 0.12, (t1 - t) / 0.12)
        box = np.zeros((h, int(tw + 44)), np.uint8); cv2.rectangle(box, (0, 0), (box.shape[1] - 1, h - 1), 255, -1)
        blend(fr, cv2.GaussianBlur(box, (0, 0), 1.2).astype(np.float32) / 255, x - 22, SUB_Y - h / 2, DEEP, 0.8 * op)
        for (w, a, e), im in zip(line, ims):
            col = PINK if a <= t < e + 0.05 else WHITE
            blend(fr, im, x - 4, SUB_Y - im.shape[0] / 2 + 3, col, op); x += im.shape[1] - 8 + sp
def lift(hd, bottom=1230):
    """bottom-anchor a headline inside the safe zone"""
    hd.y = bottom - hd.h; return hd

# ---------------- scenes ----------------
def chip_c(fr, txt, cx, y, op=1.0): chip(fr, txt, cx - (alpha_img(txt, mont(30, 600)).shape[1] + 34) / 2, y, op)
def flash_in(fr, t, t0, d=0.2): f = 1 - eo((t - t0) / d) if t >= t0 else 0.0; return fr * (1 - f) + f
def pulse_dot(fr, x, y, t, t0, r=16):
    lt = t - t0
    if lt < 0: return
    s = eback(lt / 0.3); m = np.zeros((H, W), np.uint8); cv2.circle(m, (int(x), int(y)), int(r * s), 255, -1, cv2.LINE_AA)
    blend(fr, m.astype(np.float32) / 255, 0, 0, PINK)
    u = (lt * 1.1) % 1; rr = r + 6 + 26 * u
    path_draw(fr, [(x + rr * np.cos(a), y + rr * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 40)], 1, PINK, 3, op=1 - u)
def mapM(M, x, y): return M[0, 0] * x + M[0, 2], M[1, 1] * y + M[1, 2]

# A — a teaser of ПОСЛЕ: a soft-focus window opens a little; the picture stays half-hidden
HA = lift(Drop(["СЕГОДНЯ СНЯЛИ", "*СТРИПЫ*"], 150, 70, 0, maxw=840), 1240)
TA = Track(["показываю результат комплексной операции"], 32, 72, 0, 300)
STRIPS = [(258 + k * 94, 812 + (k % 2) * 10, (-1) ** k * 4 + 2) for k in range(7)]
T_PEEL = [0.95 + 0.2 * k for k in range(7)]
def tape(fr, x, y, ang, p, w=34, h=200):
    """one Steri-Strip across the incision: lies flat, then peels up, spins and flies off (p 0→1)"""
    if p >= 1: return
    e = eio(p); cx = x + (x - 540) * 0.6 * e; cy = y - 900 * e ** 1.4; a = ang + 260 * e * (1 if x > 540 else -1); sc = 1 + 0.3 * e
    pts = cv2.boxPoints(((cx, cy), (w * sc, h * sc), a)); sh = cv2.boxPoints(((cx + 6, cy + 8), (w * sc, h * sc), a))
    fill_poly(fr, sh, DEEP, 0.25 * (1 - e)); fill_poly(fr, pts, np.float32([0.97, 0.95, 0.92]), 0.95 * (1 - e ** 3))
    for k in range(1, 6):                                                # the fibre lines on the tape
        q = k / 6; p0 = pts[0] + (pts[3] - pts[0]) * q; p1 = pts[1] + (pts[2] - pts[1]) * q
        path_draw(fr, [p0, p1], 1, np.float32([0.85, 0.82, 0.8]), 1, op=0.6 * (1 - e ** 3))
def s_a(t):
    z = lerp(1.32, 1.22, eio(t / 3.3)); sharp = full(OPEN, 0.5, 0.42, z)
    bl = cv2.GaussianBlur(cv2.resize(sharp, (W // 4, H // 4)), (0, 0), 6); bl = cv2.resize(bl, (W, H))
    r = 0.55 + 0.25 * eo((t - T_PEEL[-1] - 0.2) / 0.8)                # the window opens a little — never the full result
    d = np.sqrt(((xx - 540) / (380 * r)) ** 2 + ((yy - 820) / (300 * r)) ** 2); m = np.clip(1.5 - d, 0, 1)[..., None] * min(1, 0.35 + t / 1.2)
    fr = bl * (1 - m) + sharp * m; fr *= 1 - 0.45 * (1 - m); fr += 0.45 * (1 - m) * DEEP; shade(fr, VIG * 2); shade(fr, GBOT)
    lg = logo_alpha(320); blend(fr, lg, 500 - 160, 290, PINK, eo((t + 0.1) / 0.5) * 0.9)
    HA.draw(fr, t, 0.15); TA.y = HA.y + HA.h + 46 + TA.top; TA.draw(fr, t, 0.9)
    return fr

# B — ДО: first on the surgeon's phone, then in the studio with the three problem zones pinned
HB = lift(Drop(["НАШЕ *ДО*"], 170, 70, 0, maxw=840), 1230)
PINS = [("ГРУДЬ", 720, 600, 60), ("ЖИВОТ", 560, 930, 60), ("ТАЛИЯ", 250, 860, 40)]   # label, x, y (0331 frame), label dx
def s_b(t):
    if t < T_B2:
        fr = full(V["1635"].at(src_of(t, "1635")), 0.5, 0.45, 1.0)
    else:
        lt = t - T_B2; fr, M = cover(anon(V["0331"].at(18.6 + lt * 0.7), 150), W, H, 0.5, 0.45, lerp(1.08, 1.0, eio(lt / 2.0)))
    shade(fr, VIG); shade(fr, GBOT); chip(fr, "ДО", 40, 280)
    if t >= T_B2:
        for k, (lab, x, y, dx) in enumerate(PINS):
            t0 = T_B2 + 0.15 + 0.3 * k; px, py = mapM(M, x, y); pulse_dot(fr, px, py, t, t0)
            if t >= t0 + 0.1: chip(fr, lab, px + dx, py - 29, eo((t - t0 - 0.1) / 0.25))
        fr = flash_in(fr, t, T_B2, 0.22)
    HB.draw(fr, t, T_B + 0.1)
    return fr

# C — «комплексная операция»: the marking, then a checklist ticks on every procedure she names over changing ДО angles
HC = lift(Drop(["КОМПЛЕКСНАЯ", "*ОПЕРАЦИЯ*"], 150, 70, 0, maxw=840), 1230)
def s_c(t):
    lt = t - T_C; fr = full(anon(V["0355"].at(0.5 + lt), 200), 0.5, 0.45, lerp(1.0, 1.06, eio(lt / 3))); shade(fr, VIG); shade(fr, GBOT)
    chip(fr, "РАЗМЕТКА", 40, 280); HC.draw(fr, t, T_C + 0.15)
    return fr
ITEMS = ["Т-ОБРАЗНАЯ ПОДТЯЖКА ГРУДИ", "+ ЛИПОФИЛИНГ", "+ АБДОМИНОПЛАСТИКА", "+ ЛИПОСАКЦИЯ"]
def fit_size(txt, maxw, size):
    while bebas(size).getlength(txt) > maxw and size > 40: size -= 2
    return size
ITEM_SZ = min(fit_size(s_, 940 - 164, 84) for s_ in ITEMS)
TC = Track(["всё за одну операцию"], 34, 72, 0, 300)
GCHK = (ramp(560, 900) * 0.9).astype(np.float32)
def checklist(fr, t, items, times, x=80, y0=860, dy=112, size=84):
    for k, (lab, t0) in enumerate(zip(items, times)):
        lt = t - t0 + 0.15
        if lt < 0: continue
        y = y0 + k * dy; s = eback(lt / 0.3); b = 28 * s
        box = [(x + 28 - b, y + 28 - b), (x + 28 + b, y + 28 - b), (x + 28 + b, y + 28 + b), (x + 28 - b, y + 28 + b)]
        path_draw(fr, box, 1, PINK, 4, True)
        path_draw(fr, [(x + 13, y + 30), (x + 25, y + 43), (x + 47, y + 13)], eio((lt - 0.2) / 0.3), PINK, 6)
        a = alpha_img(lab, bebas(size)); p = eo((lt - 0.1) / 0.4); a2 = a.copy(); a2[:, int(a.shape[1] * p):] = 0
        blend(fr, a2, x + 84, y + 28 - a.shape[0] / 2 + 6, WHITE)
def s_c2(t):
    k = max([0] + [i for i, t0 in enumerate(T_ITEMS) if t >= t0 - 0.12]); t0 = max(T_C2, T_ITEMS[k] - 0.12) if k else T_C2
    z = 1.0 + 0.05 * (1 - eo((t - t0) / 0.35)) + 0.03 * eio((t - t0) / 2.5)          # punch-in on every new angle
    fr = full(CHK[k], 0.5, 0.35, z); shade(fr, VIG); shade(fr, GCHK)
    if k and t - t0 < 0.15: fr = flash_in(fr, t, t0, 0.15)
    chip(fr, "ДО", 40, 280); TC.y = 790 + TC.top; TC.draw(fr, t, T_C2 + 0.1)
    checklist(fr, t, ITEMS, T_ITEMS, size=ITEM_SZ)
    return fr

# D — donor zones light up on the ДО photo as she names them
HD = Drop(["ЖИР ВЗЯЛИ ИЗ НЕСКОЛЬКИХ", "*ДОНОРСКИХ ЗОН*"], 96, 70, 290, maxw=900)
TD = Track(["бока, внутренние бёдра, колени"], 34, 72, 0, 300)
_MD = np.float32([[0.5, 0, -27], [0, 0.5, -280]])
DONOR = cv2.warpAffine(PH["0305"], _MD, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255
GDT = ((1 - ramp(380, 640)) ** 1.2 * 0.93).astype(np.float32)
ZONES = [("БОКА", [(293, 645, 62, 120), (813, 645, 62, 120)], (553, 600)),
         ("ВНУТРЕННИЕ БЁДРА", [(455, 1095, 52, 110), (623, 1095, 52, 110)], (540, 945)),
         ("КОЛЕНИ", [(398, 1288, 62, 62), (667, 1288, 62, 62)], (540, 1200))]
def dashed_ellipse(fr, cx, cy, rx, ry, p, col=PINK, rot=0.0):
    n = 36; m = int(n * p)
    for i in range(m):
        if i % 2: continue
        a0, a1 = rot - np.pi / 2 + 2 * np.pi * i / n, rot - np.pi / 2 + 2 * np.pi * (i + 1) / n
        path_draw(fr, [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in np.linspace(a0, a1, 5)], 1, col, 5)
def s_d(t):
    lt = t - T_D; z = lerp(1.0, 1.04, eio(lt / 4)); fr = DONOR.copy()
    if z != 1: fr = cv2.warpAffine(fr, np.float32([[z, 0, 540 * (1 - z)], [0, z, 960 * (1 - z)]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
    shade(fr, GDT); shade(fr, GBOT * 0.7)
    for k, (lab, ells, (lx, ly)) in enumerate(ZONES):
        t0 = T_ZONES[k] - 0.1
        if t < t0: continue
        q = eio((t - t0) / 0.6); br = 1 + 0.06 * np.sin((t - t0) * 7)
        for (cx, cy, rx, ry) in ells:
            cx, cy = 540 + (cx - 540) * z, 960 + (cy - 960) * z
            m = np.zeros((H, W), np.uint8); cv2.ellipse(m, (int(cx), int(cy)), (int(rx * br), int(ry * br)), 0, 0, 360, 255, -1, cv2.LINE_AA)
            blend(fr, cv2.GaussianBlur(m, (0, 0), 6).astype(np.float32) / 255, 0, 0, PINK, 0.28 * q)
            dashed_ellipse(fr, cx, cy, rx * br, ry * br, q, rot=(t - t0) * 0.8)
        chip_c(fr, lab, 540 + (lx - 540) * z, 960 + (ly - 960) * z, eo((t - t0 - 0.2) / 0.3))
    HD.draw(fr, t, T_D + 0.1); TD.y = HD.y + HD.h + 40 + TD.top; TD.draw(fr, t, T_D + 0.8)
    return fr

# E — «сняли стрипы» (the surgeon's hand on the belly), then the scar close-up: the seams are traced in pink
HE1 = Marker("СТРИПЫ УЖЕ", "СНЯЛИ", 130, 70, 1110)
HE = lift(Drop(["ШВЫ УЖЕ МОЖНО *ОЦЕНИТЬ*"], 116, 70, 0, maxw=880), 1250)
TE = Track(["реабилитация продолжается"], 34, 72, 0, 300)
SCARS = [[(215, 935), (300, 995), (400, 1060), (500, 1140), (600, 1185), (700, 1210), (800, 1210), (900, 1165), (1000, 1090), (1080, 1030), (1160, 980), (1205, 955)],
         [(320, 300), (380, 360), (470, 385), (560, 375), (640, 340)], [(880, 330), (950, 370), (1040, 385), (1130, 370), (1190, 320)],
         [(420, 300), (440, 340), (465, 380)], [(1062, 318), (1056, 350), (1048, 382)]]          # IMG_1651 px: belly scar, breast folds, vertical limbs of the T
def s_e(t):
    if t < T_E2:
        lt = t - T_E; fr = full(anon(V["1636"].at(1.0 + lt), 60), 0.5, 0.45, 1.0); shade(fr, VIG); shade(fr, GBOT)
        chip(fr, "ПОСЛЕ", 40, 280); HE1.draw(fr, t, T_E + 0.2)
        return fr
    lt = t - T_E2; fr, M = cover(SCAR_IM, W, H, 0.5, 0.37, lerp(1.0, 1.2, eio(lt / 3.5))); shade(fr, GBOT)
    for k, pts in enumerate(SCARS):
        q = eio((lt - 0.25 - (0 if k == 0 else 0.5 + 0.12 * k)) / (0.9 if k == 0 else 0.45))
        ox, oy = (20, 0) if k >= 3 else (0, 20)                         # drawn just beside the seam, never over it
        path_draw(fr, [(mapM(M, x, y)[0] + ox, mapM(M, x, y)[1] + oy) for x, y in pts], q, PINK, 4, op=0.95)
    for k, (x, y) in enumerate([(215, 935), (1205, 955), (750, 1212)]):
        px, py = mapM(M, x, y); sparkle(fr, px, py + 20, 26, t, T_E2 + 1.2 + 0.15 * k)
    chip(fr, "ШВЫ", 40, 280); HE.draw(fr, t, T_E2 + 0.15); TE.y = HE.y + HE.h + 40 + TE.top; TE.draw(fr, t, T_E2 + 0.8)
    return flash_in(fr, t, T_E2, 0.22)

# F — the whole silhouette turning; «не одна зона» rolls over to «весь силуэт»
HF = Drop(["МЕНЯЕТСЯ"], 130, 70, 1000, maxw=840)
RF = Roll("", "НЕ ОДНА ЗОНА", "ВЕСЬ СИЛУЭТ", 130, 70, 1000 + int(130 * 0.93), maxw=840)
def s_f(t):
    lt = t - T_F; z = 1.0 + 0.04 * (1 - eo((t - T_SWAP) / 0.4)) * (t >= T_SWAP)
    fr = full(anon(V["1654"].at(12.6 + lt * 0.9), 110), 0.5, 0.45, z); shade(fr, VIG); shade(fr, GBOT)
    chip(fr, "ПОСЛЕ", 40, 280); HF.draw(fr, t, T_F + 0.15); RF.draw(fr, t, T_F + 0.45, T_SWAP)
    for k, (x, y) in enumerate([(215, 640), (865, 640), (540, 380)]): sparkle(fr, x, y, 30, t, T_SWAP + 0.2 + 0.15 * k)
    return fr

# G — three ДО (top) / ПОСЛЕ (bottom) collages, neck to fingertips, with a shutter click on every cut
ANG = ["ФАС", "ВПОЛОБОРОТА", "ПРОФИЛЬ"]
COLL8 = [(np.clip(c, 0, 1) * 255).astype(np.uint8) for c in COLLS]
def s_g(t):
    k = max(i for i, tk in enumerate(T_K) if t >= tk - 1e-6); lt = t - T_K[k]
    fr = full(COLL8[k], 0.5, 0.5, 1 + 0.03 * (1 - eo(lt / 0.35)))
    fr[957:963] = PINK; chip(fr, "ДО", 40, 280); chip(fr, "ПОСЛЕ", 40, 1000)
    a = alpha_img(ANG[k], mont(30, 600)); chip(fr, ANG[k], 940 - a.shape[1] - 34, 280)
    return flash_in(fr, t, T_K[k], 0.18) if k else fr

# Q — live: «вам всё нравится?» — «мы очень довольны»
HQ = lift(Drop(["МЫ ОЧЕНЬ *ДОВОЛЬНЫ*"], 140, 70, 0, maxw=840), 1230)
HEARTS = [(820, 560, 0.3), (250, 640, 0.6), (700, 400, 0.9)]
def s_q(t):
    lt = t - T_Q; fr = full(V["1635"].at(src_of(t, "1635")), 0.5, 0.45, lerp(1.0, 1.04, eio(lt / 3))); shade(fr, VIG); shade(fr, GBOT)
    chip(fr, "ПОСЛЕ", 40, 280); HQ.draw(fr, t, vstart("R") + 0.05)
    for x, y, d in HEARTS:
        u = t - vstart("R") - d
        if 0 <= u < 1.8: s = eback(u / 0.3) * 3.2; fill_poly(fr, heart_pts(x, y - 120 * u, s), PINK, 0.9 * (1 - eio((u - 1.2) / 0.6)))
    return fr

# Z — closing card on the retouched ПОСЛЕ
HZ = lift(Drop(["ДАЛЬШЕ ПОКАЖЕМ,", "КАК БУДЕТ МЕНЯТЬСЯ", "*РЕЗУЛЬТАТ*"], 116, 70, 0, maxw=840), 1300)
def s_z(t):
    lt = t - T_Z; fr = full(FINAL, 0.5, 0.42, lerp(1.0, 1.06, eio(lt / 3.6))); shade(fr, GBOT); shade(fr, GBOT * 0.5)
    lg = logo_alpha(320); blend(fr, lg, 500 - 160, 290, PINK, eo((lt - 0.2) / 0.5))
    chip(fr, "ПОСЛЕ", 40, 400, eo((lt - 0.3) / 0.3)); HZ.draw(fr, t, T_Z + 0.3)
    return fr

SCENES = [(0, s_a), (T_B, s_b), (T_C, s_c), (T_C2, s_c2), (T_D, s_d), (T_E, s_e), (T_F, s_f), (T_G, s_g), (T_Q, s_q), (T_Z, s_z)]
TRANS = {T_B: ("zoom", 0.4), T_C: ("diag", 0.5), T_C2: ("push", 0.4), T_D: ("iris", 0.55), T_E: ("zoom", 0.45), T_F: ("diag", 0.5),
         T_G: ("iris", 0.55), T_Q: ("zoom", 0.45), T_Z: ("iris", 0.6)}
def push(old, new, p):
    e = eio(p); x = int(W * e); fr = np.empty_like(old); fr[:, :W - x] = old[:, x:]; fr[:, W - x:] = new[:, :x]
    if 0 < x < W: fr[:, max(0, W - x - 2):W - x + 2] = PINK
    return fr
def render(t):
    k = max(i for i, (s0, _) in enumerate(SCENES) if t >= s0); s0, fn = SCENES[k]
    kind, d = TRANS.get(s0, ("none", 0))
    if kind != "none" and t < s0 + d:
        p = (t - s0) / d; old = SCENES[k - 1][1](t); new = fn(t)
        if kind == "iris": return iris(old, new, p, 540, 900)
        if kind == "zoom": return zoom_through(old, new, p)
        if kind == "push": return push(old, new, p)
        if kind == "diag": return diag_wipe(old, new, p)
    return fn(t)
def render(t, _base=render):
    fr = np.asarray(_base(t), np.float32).copy(); subtitles(fr, t)
    return fr

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
    for k in range(3): put(marimba(PENTA[k + 1], 0.4), T_B2 + 0.15 + 0.3 * k, 0.18)                   # ДО pins
    for k, t0 in enumerate(T_ITEMS): put(marimba(PENTA[k + 2], 0.45), t0 + 0.05, 0.22)               # checklist
    for k, t0 in enumerate(T_ZONES): put(water(480 + 140 * k, 1700 + 200 * k), t0 - 0.05, 0.28)      # donor zones
    put(felt(120, 0.25), T_E + 0.55, 0.25)                                                            # «сняли» marker
    put(swell(0.9), T_E2 - 0.3, 0.06); put(sparkle_snd(), T_E2 + 1.2, 0.05)                          # scar trace
    put(marimba(PENTA[3], 0.5), T_SWAP, 0.16); put(marimba(PENTA[5], 0.6), T_SWAP + 0.1, 0.14); put(sparkle_snd(), T_SWAP + 0.2, 0.05)
    for tk in T_K: put(click(), tk, 1.0)                                                              # «щёлк» on every collage
    for x, y, d in HEARTS: put(pop(1300, 600, 0.08), vstart("R") + d, 0.06)
    put(glass(1174.7), T_Z + 0.3, 0.16)
    for i, f in enumerate([PENTA[0], PENTA[2], PENTA[3], PENTA[5]]): put(marimba(f, 0.6), T_Z + 1.2 + i * 0.12, 0.12)
    sfx[:n] *= 1 - 0.8 * np.clip(dd, 0, 1)
    mix = vo[:, None] + mus + sfx[:n, None] * np.float64([[0.92, 1.0]])
    mix *= 10 ** (1.0 / 20); mix /= max(1, np.abs(mix).max() / 0.95)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

def make_cover(path):
    """everything inside the centre 3:4 (y 240-1680) that the profile grid shows; a deeper navy than the video"""
    CBG = np.clip(np.float32([0, 10, 36]) / 255 * (1 + 0.3 * np.exp(-(((xx - 540) / 900) ** 2 + ((yy - 520) / 900) ** 2)))[..., None], 0, 1).astype(np.float32)
    fr = CBG.copy()
    for k, (n, im) in enumerate([("0305", PH["0305"]), ("1637", PH["1637"])]):
        fr[340:1140, k * 540:(k + 1) * 540] = panel(im, n, 770, 540, 800)
    fr[340:1140, 538:542] = PINK; chip(fr, "ДО", 40, 370); chip(fr, "ПОСЛЕ", 580, 370)
    m = np.clip((yy[:, :1] - 1115) / 50, 0, 1)[..., None]; fr = fr * (1 - m) + CBG * m
    lg = logo_alpha(300); blend(fr, lg, 540 - 150, 262, PINK)
    hd = Drop(["СНЯЛИ СТРИПЫ.", "*ПОКАЗЫВАЮ РЕЗУЛЬТАТ*"], 118, 70, 1190, maxw=900); hd.draw(fr, 99, 0)
    ty = Track(["после комплексной коррекции", "груди и живота"], 34, 72, 0, 400); ty.y = 1190 + hd.h + 56 + ty.top; ty.draw(fr, 99, 0)
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
