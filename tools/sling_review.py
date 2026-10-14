"""«Обзор на слинг» — vertical product review cut to the voice-over (2160x3840, 50 fps, Sony 4K 25 fps sources, rotated).
Voice: sling_cut.py (false start + pauses removed) → DeepFilterNet → EQ/comp/loudnorm = voice_final.wav.
Look: minimal Manrope typography (Light kickers, Bold titles, hairlines), mask-reveal word animation, numbered benefit cards
01–04, pop-in chips, 0→100 «/10» counter, ticked recap; slow push-ins, zoom-blur on section changes; synthesized SFX.
Work dir: dl/C08xx.MP4, voice_final.wav, cwords.json, Manrope-*.ttf.
  python3 -I sling_review.py test 1.0,17,33.5,48.6,52   (PNG previews, 1080 wide)
  python3 -I sling_review.py render f0 f1 out.mp4       (frame range, video only)
  python3 -I sling_review.py audio mix.wav
"""
import json, subprocess, sys, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, K, FPS = 1080, 1920, 2, 50
OW, OH = W * K, H * K
VOICE_DUR, TAIL = 53.32, 0.55
DUR = VOICE_DUR + TAIL
NF = int(round(DUR * FPS))
FONTS = {"L": "Manrope-Light.ttf", "R": "Manrope-Regular.ttf", "M": "Manrope-Medium.ttf", "B": "Manrope-Bold.ttf", "XB": "Manrope-ExtraBold.ttf"}
WHITE, BLUSH = (255, 255, 255), (247, 223, 221)
GRADE = "eq=contrast=1.04:saturation=1.07:gamma=0.98,colorbalance=rm=0.02:gm=0.0:bm=-0.025:rh=0.01:bh=-0.01,unsharp=5:5:0.35:5:5:0"

# ---------------------------------------------------------------- shots, cut to the voice
# (t0, clip, src_start, z0, z1, cx, cy, section)  — a shot lasts until the next t0; section=True → zoom-blur transition in
SHOTS = [
    (0.00, "C0898", 82.0, 1.00, 1.06, .50, .40, False),   # «Девочки, я не понимаю…» — face
    (1.40, "C0898", 6.0, 1.00, 1.06, .50, .45, False),    # «…почему не купила эту штуку раньше» — her with the sling
    (3.38, "C0900", 3.0, 1.00, 1.06, .45, .45, True),     # «одна из лучших покупок» — shoulder bow detail
    (5.20, "C0901", 0.2, 1.00, 1.06, .45, .55, False),    #   buckle detail
    (7.25, "C0903", 1.0, 1.00, 1.06, .45, .62, False),    # «ручной ребёнок» — baby asleep on mum
    (10.00, "C0905", 8.0, 1.00, 1.06, .50, .45, False),   # «отказывается существовать отдельно от мамы» — baby in the sling
    (13.85, "C0898", 66.0, 1.00, 1.06, .52, .42, False),  # «настоящим спасением»
    (16.45, "C0899", 20.0, 1.00, 1.06, .55, .55, True),   # 01 — hands free at the counter
    (18.20, "C0904", 8.5, 1.00, 1.06, .50, .45, True),    # 02 — back view, strap across the back
    (22.60, "C0899", 6.5, 1.00, 1.06, .48, .48, True),    # 03 — snack with the baby in the sling
    (25.78, "C0899", 58.0, 1.00, 1.06, .55, .45, False),  #   coffee
    (28.50, "C0899", 31.0, 1.00, 1.06, .60, .50, False),  #   at the sink
    (30.68, "C0905", 11.2, 1.00, 1.06, .45, .55, False),  #   «пока она рядом со мной»
    (32.10, "C0898", 32.0, 1.00, 1.06, .50, .42, True),   # 04 — tired, hand on the head
    (35.70, "C0899", 48.2, 1.00, 1.06, .45, .52, False),  #   looks at the baby, then facepalm
    (38.33, "C0898", 26.5, 1.00, 1.06, .52, .42, False),  #   hand in the hair
    (40.10, "C0903", 2.0, 1.00, 1.06, .45, .60, False),   #   «ни на секунду» — baby holds on
    (42.00, "C0903", 14.5, 1.00, 1.06, .50, .45, True),   # «в полном восторге» — dad comes in
    (44.15, "C0903", 17.5, 1.00, 1.06, .50, .40, False),  #   kiss
    (47.00, "C0902", 3.0, 1.00, 1.06, .50, .45, True),    # «100 из 10» — close-up smile
    (49.60, "C0903", 22.6, 1.00, 1.06, .50, .40, True),   # recap — family
]
SECTION_CUTS = [s[0] for s in SHOTS if s[7]]

# ---------------------------------------------------------------- easing
def cl(x): return min(1.0, max(0.0, x))
def oc(x): x = cl(x); return 1 - (1 - x) ** 3
def oexp(x): x = cl(x); return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)
def oback(x, s=1.9): x = cl(x) - 1; return 1 + (s + 1) * x ** 3 + s * x ** 2

# ---------------------------------------------------------------- sprites
_fonts, _spr = {}, {}
def font(w, size):
    k = (w, size)
    if k not in _fonts: _fonts[k] = ImageFont.truetype(FONTS[w], int(round(size * K)))
    return _fonts[k]

class Spr:
    def __init__(s, rgbp, a, pad, w, h): s.rgbp, s.a, s.pad, s.w, s.h = rgbp, a, pad, w, h

def text_spr(text, w, size, col=WHITE, track=0.0, shadow=0.5, glow=0.0):
    key = (text, w, size, col, track, shadow, glow)
    if key in _spr: return _spr[key]
    f = font(w, size); tr = track * size * K
    wd = [f.getlength(c) for c in text]
    tw = int(np.ceil(sum(wd) + tr * max(0, len(text) - 1))) + 2
    asc, desc = f.getmetrics(); th = asc + desc
    pad = int(size * K * (0.9 if glow else 0.35))
    im = Image.new("L", (tw + 2 * pad, th + 2 * pad), 0); d = ImageDraw.Draw(im); x = pad
    for c, cw in zip(text, wd): d.text((x, pad), c, font=f, fill=255); x += cw + tr
    a = np.asarray(im, np.float32) / 255
    sh = cv2.GaussianBlur(a, (0, 0), max(1.0, size * K * 0.09)) * shadow if shadow else np.zeros_like(a)
    if shadow: sh = np.roll(sh, int(size * K * 0.03), axis=0)
    if glow:   # soft dark halo so the line reads over white fabric
        g = cv2.GaussianBlur(cv2.dilate(a, np.ones((int(size * K * .25),) * 2, np.uint8)), (0, 0), size * K * 0.3); g = g / max(1e-3, g.max()) * glow
        sh = 1 - (1 - sh) * (1 - np.clip(g, 0, 1))
    out_a = a + sh * (1 - a)
    rgbp = a[..., None] * (np.float32(col) / 255)
    s = Spr(rgbp.astype(np.float32), out_a.astype(np.float32), pad, tw, th); _spr[key] = s; return s

def put(fr, s, x, y, alpha=1.0, clip=None, scale=1.0):
    """place sprite s with its text box top-left at (x, y) in 4K px; clip=(y0, y1) visible rows; scale about box centre"""
    if alpha <= 0.003: return
    rgbp, a, pad = s.rgbp, s.a, s.pad
    if abs(scale - 1) > 1e-3:
        hh, ww = a.shape; nw, nh = max(1, int(ww * scale)), max(1, int(hh * scale))
        rgbp = cv2.resize(rgbp, (nw, nh), interpolation=cv2.INTER_LINEAR); a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x += (s.w - s.w * scale) / 2; y += (s.h - s.h * scale) / 2; pad = pad * scale
    X0, Y0 = int(round(x - pad)), int(round(y - pad))
    hh, ww = a.shape
    x0, y0, x1, y1 = max(0, X0), max(0, Y0), min(OW, X0 + ww), min(OH, Y0 + hh)
    if clip: y0, y1 = max(y0, int(clip[0])), min(y1, int(clip[1]))
    if x1 <= x0 or y1 <= y0: return
    sa = a[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] * alpha; sc = rgbp[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] * alpha
    reg = fr[y0:y1, x0:x1].astype(np.float32) / 255
    fr[y0:y1, x0:x1] = ((reg * (1 - sa[..., None]) + sc) * 255 + 0.5).astype(np.uint8)

def rect(fr, x, y, w, h, alpha, col=WHITE):
    x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0: return
    reg = fr[y0:y1, x0:x1].astype(np.float32)
    fr[y0:y1, x0:x1] = (reg * (1 - alpha) + np.float32(col) * alpha + 0.5).astype(np.uint8)

# ---------------------------------------------------------------- overlay elements (1080-space coordinates)
REV, OUT = 0.55, 0.28
class Txt:
    """words reveal from behind a mask (slide up), staggered; fade/lift out"""
    def __init__(s, t0, t1, text, w, size, x, y, col=WHITE, track=0.0, stagger=0.07, align="l", alpha=1.0, sfx=None):
        s.t0, s.t1, s.words, s.w, s.size, s.x, s.y, s.col, s.track, s.st, s.align, s.alpha, s.sfx = \
            t0, t1, text.split(" "), w, size, x, y, col, track, stagger, align, alpha, sfx
        s.sp = [text_spr(t, w, size, col, track) for t in s.words]
        s.gap = font(w, size).getlength(" ") * 1.15 + 2 * track * size * K
        s.width = sum(p.w for p in s.sp) + s.gap * (len(s.sp) - 1)
    def draw(s, fr, t):
        if t < s.t0 or t > s.t1 + OUT: return
        out = oc((t - s.t1) / OUT) if t > s.t1 else 0
        X = s.x * K - (s.width / 2 if s.align == "c" else 0); Y = s.y * K
        for i, p in enumerate(s.sp):
            e = oexp((t - s.t0 - i * s.st) / REV)
            if e > 0:
                dy = (1 - e) * p.h * 1.05 - out * 14 * K
                put(fr, p, X, Y + dy, s.alpha * min(1, e * 1.6) * (1 - out), clip=(Y - p.h * 0.25 - out * 14 * K, Y + p.h * 1.12))
            X += p.w + s.gap

class Line:
    def __init__(s, t0, t1, x, y, L, th=1.6, alpha=0.85, align="l"):
        s.t0, s.t1, s.x, s.y, s.L, s.th, s.alpha, s.align = t0, t1, x, y, L, th, alpha, align
    def draw(s, fr, t):
        if t < s.t0 or t > s.t1 + OUT: return
        e = oexp((t - s.t0) / 0.7); out = oc((t - s.t1) / OUT) if t > s.t1 else 0
        L = s.L * K * e; x = s.x * K - (L / 2 if s.align == "c" else 0)
        if out: x += s.L * K * out; L *= 1 - out
        rect(fr, x, s.y * K, L, s.th * K, s.alpha)

def pill_spr(text, size=34):
    key = ("pill", text, size)
    if key in _spr: return _spr[key]
    f = font("M", size); tw = f.getlength(text); asc, desc = f.getmetrics()
    px, py, lw = int(size * K * 0.75), int(size * K * 0.42), max(2, int(1.4 * K))
    bw, bh = int(tw + 2 * px), int(asc + desc * 0.4 + 2 * py); pad = int(size * K * 0.4)
    im = Image.new("L", (bw + 2 * pad, bh + 2 * pad), 0); d = ImageDraw.Draw(im)
    d.rounded_rectangle((pad, pad, pad + bw - 1, pad + bh - 1), radius=bh // 2, outline=255, width=lw)
    d.text((pad + px, pad + py - desc * 0.15), text, font=f, fill=255)
    fill = Image.new("L", im.size, 0); ImageDraw.Draw(fill).rounded_rectangle((pad, pad, pad + bw - 1, pad + bh - 1), radius=bh // 2, fill=255)
    a = np.asarray(im, np.float32) / 255; fa = np.asarray(fill, np.float32) / 255 * 0.16   # faint dark glass inside the pill
    sh = cv2.GaussianBlur(a, (0, 0), size * K * 0.08) * 0.45
    out_a = a + (fa + sh * (1 - fa)) * (1 - a)
    s = Spr((a[..., None] * np.float32([1, 1, 1])).astype(np.float32), out_a.astype(np.float32), pad, bw, bh); _spr[key] = s; return s

class Pill:
    def __init__(s, t0, t1, text, x, y, size=34):
        s.t0, s.t1, s.x, s.y, s.sp = t0, t1, x, y, pill_spr(text, size); s.sfx = "pop"
    def draw(s, fr, t):
        if t < s.t0 or t > s.t1 + OUT: return
        e = (t - s.t0) / 0.42; out = oc((t - s.t1) / OUT) if t > s.t1 else 0
        put(fr, s.sp, s.x * K, s.y * K - out * 12 * K, min(1, e * 3) * (1 - out), scale=0.6 + 0.4 * oback(e))

def tick_spr(size):
    """circle outline for the check rows"""
    key = ("ring", size)
    if key in _spr: return _spr[key]
    r = int(size * K / 2); pad = int(size * K * 0.3); n = 2 * (r + pad)
    im = Image.new("L", (n, n), 0); ImageDraw.Draw(im).ellipse((pad, pad, pad + 2 * r, pad + 2 * r), outline=255, width=max(2, int(1.6 * K)))
    a = np.asarray(im, np.float32) / 255; sh = cv2.GaussianBlur(a, (0, 0), size * K * 0.08) * 0.45
    s = Spr((a[..., None] * np.float32([1, 1, 1])).astype(np.float32), (a + sh * (1 - a)).astype(np.float32), pad, 2 * r, 2 * r); _spr[key] = s; return s

class Check:
    """ring pops, the tick draws itself, the label slides in"""
    def __init__(s, t0, t1, text, x, y, size=46):
        s.t0, s.t1, s.text, s.x, s.y, s.size = t0, t1, text, x, y, size; s.sfx = "tick"
        s.ring = tick_spr(size * 1.05); s.lab = Txt(t0 + 0.12, t1, text, "M", size * 0.92, x + size * 1.55, y - size * 0.12, stagger=0.04)
    def draw(s, fr, t):
        if t < s.t0 or t > s.t1 + OUT: return
        out = oc((t - s.t1) / OUT) if t > s.t1 else 0; al = 1 - out
        e = (t - s.t0) / 0.4
        put(fr, s.ring, s.x * K, s.y * K, min(1, e * 3) * al, scale=0.5 + 0.5 * oback(e))
        p = oc((t - s.t0 - 0.12) / 0.35)
        if p > 0:
            d = s.size * 1.05 * K; ox, oy = s.x * K, s.y * K
            pts = np.float32([[ox + d * .28, oy + d * .52], [ox + d * .45, oy + d * .68], [ox + d * .74, oy + d * .34]])
            seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1); L = seg.sum() * p
            poly = [pts[0]]
            for i in range(2):
                if L <= 0: break
                f = min(1, L / seg[i]); poly.append(pts[i] + (pts[i + 1] - pts[i]) * f); L -= seg[i]
            x0, y0 = int(ox), int(oy); w_, h_ = int(d) + 4, int(d) + 4
            m = np.zeros((h_, w_), np.uint8)
            cv2.polylines(m, [((np.float32(poly) - [x0, y0]) * 16).astype(np.int32)], False, 255, int(2.6 * K), cv2.LINE_AA, shift=4)
            a = m.astype(np.float32) / 255 * al
            reg = fr[y0:y0 + h_, x0:x0 + w_].astype(np.float32)
            fr[y0:y0 + h_, x0:x0 + w_] = (reg * (1 - a[..., None]) + np.float32(BLUSH) * a[..., None] + .5).astype(np.uint8)
        s.lab.draw(fr, t)

class Counter:
    def __init__(s, t0, tdone, t1, x, y, size=300):
        s.t0, s.td, s.t1, s.x, s.y, s.size = t0, tdone, t1, x, y, size; s.sfx = "count"
    def val(s, t): return int(round(100 * oc((t - s.t0) / (s.td - s.t0))))
    def draw(s, fr, t):
        if t < s.t0 or t > s.t1 + OUT: return
        out = oc((t - s.t1) / OUT) if t > s.t1 else 0; e = oexp((t - s.t0) / REV)
        sp = text_spr(str(s.val(t)), "XB", s.size, WHITE, -0.03)
        Y = s.y * K + (1 - e) * sp.h * 0.5 - out * 14 * K
        put(fr, sp, s.x * K, Y, min(1, e * 1.5) * (1 - out))
        # «/10» parked right of the final «100» so it does not jump while counting
        w100 = text_spr("100", "XB", s.size, WHITE, -0.03).w
        sl = text_spr("/10", "L", s.size * 0.42, BLUSH, 0.0)
        e2 = oexp((t - s.t0 - 0.15) / REV)
        put(fr, sl, s.x * K + w100 + 14 * K, s.y * K + sp.h * 0.5 + (1 - e2) * 40 * K - out * 14 * K, min(1, e2 * 1.5) * (1 - out))
        pk = cl((t - s.td) / 0.25)
        if 0 < pk < 1:  # soft flash ring of the final value
            put(fr, text_spr("100", "XB", s.size, BLUSH, -0.03), s.x * K, Y, 0.5 * (1 - pk), scale=1 + 0.12 * pk)

# ---------------------------------------------------------------- graphics timeline
X0 = 80
TOP = 330          # Instagram safe zone: nothing above ~y 300 (status bar / Reels header / 3:4 grid crop)
def card(t0, t1, num, title_lines, sub=None, tsfx=True):
    els = [Txt(t0, t1, num, "L", 80, X0 - 4, TOP - 10, BLUSH, -0.02, sfx="whoosh" if tsfx else None),
           Txt(t0 + 0.1, t1, "/ 04", "L", 28, X0 + 108, TOP + 38, WHITE, 0.08, alpha=0.75),
           Line(t0 + 0.12, t1, X0, TOP + 100, 210)]
    y = TOP + 116
    for i, l in enumerate(title_lines):
        els.append(Txt(t0 + 0.2 + i * 0.12, t1, l, "B", 60, X0, y)); y += 72
    if sub: els.append(Txt(t0 + 0.45, t1, sub, "R", 32, X0, y + 10, track=0.01, stagger=0.05))
    return els

def kicker(t0, t1, text, big, y=TOP, sub=None):
    els = [Txt(t0, t1, text, "L", 32, X0, y, track=0.12, stagger=0.05, sfx="swish"),
           Line(t0 + 0.08, t1, X0, y + 50, 90)]
    if big:
        yy = y + 66
        for i, l in enumerate(big):
            els.append(Txt(t0 + 0.15 + i * 0.12, t1, l, "B", 64, X0, yy)); yy += 76
        if sub: els.append(Txt(t0 + 0.4, t1, sub, "R", 32, X0, yy + 8, stagger=0.05))
    return els

EL = []
# title
EL += [Txt(0.15, 3.15, "обзор на", "L", 36, X0, TOP, track=0.14, stagger=0.06, sfx="whoosh"),
       Txt(0.30, 3.15, "слинг", "XB", 140, X0 - 6, TOP + 36, track=-0.02, stagger=0.0),
       Line(0.45, 3.15, X0, TOP + 226, 270)]
EL += kicker(3.45, 7.05, "лучшая покупка", ["одна из лучших", "за последнее время"])
EL += kicker(7.32, 13.75, "для кого", ["для «ручных»", "малышей"])
EL += kicker(13.92, 16.30, "для меня это", ["настоящее", "спасение"])
EL += card(16.50, 18.10, "01", ["свободные", "руки"])
EL += card(18.27, 22.50, "02", ["спине", "легче"], "не нужно носить на одной руке")
EL += card(22.68, 32.00, "03", ["дела по дому"], "а малыш рядом и на ручках")
EL += card(32.17, 41.85, "04", ["спасает", "в тяжёлые дни"])
PY = TOP + 290
EL += [Pill(34.63, 41.85, "регрессы", X0, PY, 26), Pill(35.77, 41.85, "капризы", X0 + 190, PY, 26),
       Pill(38.40, 41.85, "«только на ручках»", X0, PY + 66, 26), Pill(40.89, 41.85, "ни на секунду", X0, PY + 132, 26)]
EL += kicker(42.05, 46.85, "итого", ["в полном", "восторге"], sub="жалею только, что не купила раньше")
EL += [Txt(47.05, 49.50, "моя оценка", "L", 34, X0, TOP, track=0.12, stagger=0.05, sfx="swish"),
       Line(47.12, 49.50, X0, TOP + 52, 90), Counter(47.10, 48.55, 49.50, X0 - 8, TOP + 70, 200)]
EL += [Txt(49.68, 53.32, "однозначно", "L", 32, X0, TOP, track=0.12, stagger=0.05, sfx="whoosh"),
       Line(49.75, 53.32, X0, TOP + 50, 90),
       Txt(49.82, 53.32, "самая полезная", "B", 56, X0, TOP + 68), Txt(49.94, 53.32, "покупка для мамы", "B", 56, X0, TOP + 136)]
for i, (txt, tt) in enumerate([("свободные руки", 50.45), ("спине легче", 50.75), ("дела по дому", 51.05), ("спокойнее в тяжёлые дни", 51.35)]):
    EL.append(Check(tt, 53.32, txt, X0, TOP + 232 + i * 60, 36))

# ---------------------------------------------------------------- subtitles (phrase chunks from the cut word times)
FIX = {"ребенок,": "ребёнок,", "ребенка": "ребёнка"}
PHRASES = """Девочки, | я не понимаю, | почему я не купила | эту штуку раньше. | Это просто одна | из лучших моих покупок | за последнее время. |
Особенно, | если у вас тоже | ручной ребёнок, | который периодически | отказывается существовать | отдельно от мамы, |
слинг оказался для меня | настоящим спасением. | Во-первых, | освобождаются руки, | во-вторых, | спине гораздо легче, |
чем постоянно таскать | ребёнка на одной руке. | И самое главное, | когда малышка хочет быть | только на руках, | я теперь могу |
хотя бы спокойно | походить по дому | и поделать какие-то | несложные дела, | пока она рядом со мной. | Особенно спасает |
в периоды этих | всяких регрессов, | капризов | и вот этих | прекрасных дней, | когда ребёнка | невозможно от себя |
оторвать ни на секунду. | Короче, | я в полном восторге, | жалею только об одном, | что не заказала | его раньше. |
Моя оценка просто | 100 из 10, | однозначно | одна из самых | полезных покупок | для материнства."""
def chunks():
    Wd = json.load(open("cwords.json")); ws = []
    for a, b, w in Wd:
        w = FIX.get(w, w)
        if w.startswith("-") and ws: ws[-1][2] += w; ws[-1][1] = b; continue
        ws.append([a, b, w])
    ph = [p.split() for p in PHRASES.replace("\n", " ").split("|")]
    assert sum(map(len, ph)) == len(ws), (sum(map(len, ph)), len(ws))
    out, k = [], 0
    for p in ph: out.append(ws[k:k + len(p)]); k += len(p)
    res = []
    for i, c in enumerate(out):
        t0 = c[0][0] - 0.04; t1 = out[i + 1][0][0] - 0.04 if i + 1 < len(out) else VOICE_DUR
        t1 = min(t1, c[-1][1] + 0.45)
        res.append((t0, t1, " ".join(x[2] for x in c).rstrip(".,")))
    return res
SUBS = chunks()
SUB_Y, SUB_SIZE = 1340, 44          # above the Reels caption / buttons zone
def draw_subs(fr, t):
    for t0, t1, txt in SUBS:
        if t0 <= t < t1:
            sp = text_spr(txt, "M", SUB_SIZE, WHITE, 0.0, shadow=0.5)
            e = oc((t - t0) / 0.16)
            put(fr, sp, (W * K - sp.w) / 2, SUB_Y * K + (1 - e) * 10 * K, min(1, e * 1.4))

# ---------------------------------------------------------------- gradients
_g = np.linspace(0, 1, OH, dtype=np.float32)
G_TOP = (np.clip(1 - _g / 0.42, 0, 1) ** 1.6).astype(np.float32)          # top 42 %
G_BOT = (np.clip((_g - 0.50) / 0.50, 0, 1) ** 1.8).astype(np.float32)     # long black spray-in over the bottom half, densest at the very bottom
def top_strength(t):
    k = 0.0
    for a, b, s in [(0.1, 3.2, .42), (3.4, 7.1, .40), (7.3, 13.8, .40), (13.9, 16.35, .40), (16.45, 41.9, .48),
                    (42.0, 46.9, .44), (47.0, 49.55, .46), (49.6, 99, .58)]:
        k = max(k, s * oc((t - a) / 0.3) * (1 - oc((t - b) / 0.3)))
    return k
def shade(fr, t):
    ts = 0.0          # no dark haze behind the typography
    if ts > 0.01:
        n = int(OH * 0.42); m = (1 - G_TOP[:n] * ts)[:, None, None]
        fr[:n] = (fr[:n].astype(np.float32) * m).astype(np.uint8)
    n0 = int(OH * 0.50); m = (1 - G_BOT[n0:] * 0.62)[:, None, None]
    fr[n0:] = (fr[n0:].astype(np.float32) * m).astype(np.uint8)

# ---------------------------------------------------------------- footage
def shot_at(t):
    i = max(j for j, s in enumerate(SHOTS) if s[0] <= t + 1e-6); return i
def shot_end(i): return SHOTS[i + 1][0] if i + 1 < len(SHOTS) else DUR

class Dec:
    def __init__(s, i, u0):
        t0, clip, src = SHOTS[i][:3]; dur = shot_end(i) - t0 - u0 + 0.2
        s.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{src + u0:.3f}", "-i", f"dl/{clip}.MP4", "-t", f"{dur:.3f}",
                                "-vf", f"fps={FPS},scale={OW}:{OH}:flags=bicubic,{GRADE},format=rgb24", "-f", "rawvideo", "-"],
                               stdout=subprocess.PIPE, bufsize=OW * OH * 3 * 2)
        s.last = None
    def get(s):
        b = s.p.stdout.read(OW * OH * 3)
        if len(b) == OW * OH * 3: s.last = np.frombuffer(b, np.uint8).reshape(OH, OW, 3)
        return s.last
    def close(s): s.p.kill()

def zoom_of(i, u):
    t0, clip, src, z0, z1, cx, cy, sec = SHOTS[i]; d = shot_end(i) - t0
    z = z0 + (z1 - z0) * cl(u / d)
    if sec: z += 0.05 * (1 - oc(u / 0.45))                       # punch-in settle after a section change
    return z, cx, cy
def warp(img, z, cx, cy):
    c = np.float32([cx * OW, cy * OH])
    M = np.float32([[z, 0, c[0] * (1 - z)], [0, z, c[1] * (1 - z)]])
    return cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
def zoomblur(img, amt, cx, cy, n=7):
    acc = np.zeros(img.shape, np.float32)
    for k in range(n): acc += warp(img, 1 + amt * k / (n - 1), cx, cy)
    return (acc / n).astype(np.uint8)
def blur_amt(t):
    for c in SECTION_CUTS:
        if c == 0: continue
        d = t - c
        if -0.12 <= d < 0: return 0.10 * oc((d + 0.12) / 0.12)
        if 0 <= d < 0.16: return 0.10 * (1 - oc(d / 0.16))
    return 0.0

def compose(fr, t):
    shade(fr, t)
    for e in EL: e.draw(fr, t)
    draw_subs(fr, t)
    if t > VOICE_DUR - 0.15:                                      # fade to black at the end
        k = cl((t - (VOICE_DUR - 0.15)) / (TAIL + 0.1)); fr[:] = (fr.astype(np.float32) * (1 - k)).astype(np.uint8)
    if t < 0.12: fr[:] = (fr.astype(np.float32) * oc(t / 0.12)).astype(np.uint8)
    return fr

def frame_from(raw, i, t):
    u = t - SHOTS[i][0]; z, cx, cy = zoom_of(i, u)
    f = warp(raw, z, cx, cy); b = blur_amt(t)
    if b > 0.004: f = zoomblur(f, b, cx, cy)
    return np.ascontiguousarray(f)

def render(f0, f1, out):
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
                            "-color_trc", "bt709", "-colorspace", "bt709", out], stdin=subprocess.PIPE)
    dec, di = None, -1
    for n in range(f0, f1):
        t = n / FPS; i = shot_at(t)
        if i != di:
            if dec: dec.close()
            dec, di = Dec(i, t - SHOTS[i][0]), i
        fr = compose(frame_from(dec.get(), i, t), t)
        enc.stdin.write(fr.tobytes())
        if n % 100 == 0: print(n, flush=True)
    dec.close(); enc.stdin.close(); enc.wait()

def test(ts):
    for t in ts:
        i = shot_at(t); d = Dec(i, t - SHOTS[i][0]); raw = d.get(); d.close()
        fr = compose(frame_from(raw, i, t), t)
        cv2.imwrite(f"test_{t:05.2f}.jpg", cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])

# ---------------------------------------------------------------- sound effects
SR = 48000
rng = np.random.default_rng(11)
def lp(x, cut):
    y = np.zeros_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * cut / SR)
    for i in range(len(x)): s += a[i] * (x[i] - s); y[i] = s
    return y
def whoosh(d=0.5, lo=300, hi=5000, peak=0.6):
    n = int(d * SR); t = np.linspace(0, 1, n)
    env = np.where(t < peak, (t / peak) ** 2, ((1 - t) / (1 - peak)) ** 1.5)
    cut = lo + (hi - lo) * np.sin(np.pi * np.clip(t / (peak * 2), 0, 1)) ** 2
    x = rng.standard_normal(n); y = (lp(x, cut) - lp(x, cut * 0.25)) * env; return y / np.abs(y).max()
def pop(f0=260, f1=700):
    n = int(0.09 * SR); t = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(f0 + f1 * np.exp(-t * 60)) / SR; y = np.sin(ph) * np.exp(-t * 45); return y / np.abs(y).max()
def click():
    n = int(0.012 * SR); x = np.diff(rng.standard_normal(n), prepend=0); y = x * np.exp(-np.arange(n) / SR * 500); return y / np.abs(y).max()
def bell(fs=((1318.5, 1, 3), (1975.5, .5, 4), (2637, .3, 6), (659.25, .4, 2.5)), d=1.4):
    n = int(d * SR); t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * k) for f, a, k in fs); y *= np.minimum(1, t / 0.004); return y / np.abs(y).max()
def tick():
    n = int(0.06 * SR); t = np.arange(n) / SR
    y = np.sin(2 * np.pi * 1760 * t) * np.exp(-t * 70) + 0.5 * np.sin(2 * np.pi * 2640 * t) * np.exp(-t * 90); return y / np.abs(y).max()

def audio(out):
    trk = np.zeros(int((DUR + 2) * SR))
    def add(sig, t, g):
        i = int(max(0, t) * SR); seg = trk[i:i + len(sig)]; seg += sig[:len(seg)] * g
    WB, WS, SW, PO, CK, DI, TK = whoosh(0.6), whoosh(0.32, 600, 7000, 0.5), whoosh(0.24, 1500, 9000, 0.4), pop(), click(), bell(), tick()
    for c in SECTION_CUTS:
        if c > 0: add(WB, c - 0.36, 0.22)
    for e in EL:
        sfx = getattr(e, "sfx", None)
        if sfx == "whoosh": add(WS, e.t0 - 0.12, 0.16)
        elif sfx == "swish": add(SW, e.t0 - 0.05, 0.12)
        elif sfx == "pop": add(PO, e.t0, 0.30)
        elif sfx == "tick": add(PO, e.t0, 0.18); add(TK, e.t0 + 0.12, 0.12)
        elif sfx == "count":
            last = -1
            for k in range(int((e.td - e.t0) * 200)):
                tt = e.t0 + k / 200; v = e.val(tt)
                if v // 5 != last // 5: add(CK, tt, 0.07); last = v
            add(DI, e.td, 0.20); add(PO, e.td, 0.22)
    add(WB[::-1] * np.linspace(1, 0, len(WB)), VOICE_DUR - 0.3, 0.10)   # soft reverse swell into the fade
    trk = trk[:int(DUR * SR)]
    v = wave.open("voice_final.wav"); vo = np.frombuffer(v.readframes(v.getnframes()), np.int16).astype(np.float32) / 32768
    mix = np.zeros_like(trk); mix[:min(len(vo), len(mix))] += vo[:len(mix)]
    mix += trk
    pk = np.abs(mix).max(); mix = mix / max(1.0, pk / 0.95)
    st = np.stack([mix, mix], 1)
    with wave.open(out, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(st, -1, 1) * 32767).astype(np.int16).tobytes())
    print("audio", round(len(mix) / SR, 2), "peak", round(float(pk), 2))

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test": test([float(x) for x in sys.argv[2].split(",")])
    elif cmd == "render": render(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif cmd == "audio": audio(sys.argv[2])
    elif cmd == "info":
        print("frames", NF); [print(round(a, 2), round(b, 2), s) for a, b, s in SUBS]
