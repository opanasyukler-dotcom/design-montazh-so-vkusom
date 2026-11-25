"""Carousel 'Ложная кривизна ног' (1080x1350): slide 1 static JPG, slides 2-12 animated MP4.

Brand post style (navy, powder Bebas, thin Montserrat, ANNA TVAURI mark). From slide 2 on, text
appears in sequence and the explanation is shown on the pictures: the inner contour of the shins is
traced, the gap is highlighted, the target contour is drawn, the result is revealed with markers.

usage: python3 post_false_curvature.py MATERIALS_DIR OUT_DIR [slide numbers...]
"""
import os, sys, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter

SRC, OUT = sys.argv[1:3]
ONLY = {int(a) for a in sys.argv[3:]}
SURGEON = "DSC00898.JPG"; AI_LEGS = "16282750-2831-4390-9648-FAF201E06D32.png"; HANDS = "5223F1B9-E4B1-4ABB-8DB1-EDB2B7955A04.png"
MAIN = "front_retouch.jpg"   # retouched front before/after of the main patient
sys.argv = [sys.argv[0], os.path.join(SRC, MAIN), OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surgeon_style as ss  # noqa: E402
from surgeon_style import Slide, bebas, mont, logo, vshade, icon_circle, POWDER, NAVY, DEEP, WHITE, S  # noqa: E402

W, H, FPS, TOTAL = 1080, 1350, 30, 12
HOLD = 3.0  # seconds of fully static final frame at the end of every animated slide
_cache = {}


# ------------------------------------------------------------------ assets
def photo(name, max_side=2600):
    if name not in _cache:
        im = ImageOps.exif_transpose(Image.open(os.path.join(SRC, name))).convert("RGB")
        if max(im.size) > max_side: im.thumbnail((max_side, max_side), Image.LANCZOS)
        _cache[name] = im
    return _cache[name]


def fit(im, w, h, cx=0.5, cy=0.5):
    s = max(w / im.width, h / im.height)
    im = im.resize((max(w, round(im.width * s)), max(h, round(im.height * s))), Image.LANCZOS)
    x, y = int((im.width - w) * cx), int((im.height - h) * cy)
    return im.crop((x, y, x + w, y + h))


def half(side, frac):
    """crop of the main before/after composite: side 'b' (left half) or 'a' (right half), frac=(x0,y0,x1,y1)"""
    im = photo(MAIN); hw = im.width // 2; ox = 0 if side == "b" else hw
    x0, y0, x1, y1 = frac
    return im.crop((ox + int(x0 * hw), int(y0 * im.height), ox + int(x1 * hw), int(y1 * im.height)))


CROP = (0.15, 0.22, 0.85, 0.86)   # knees to ankles


# ------------------------------------------------------------------ text helpers
def wrap(text, f, width):
    out, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if f.getlength(t) / S <= width or not line: line = t
        else: out.append(line); line = w
    return out + [line]


def para(sl, x, y, text, size=30, w=300, col=WHITE, width=920, lh=1.32, anchor="la"):
    f = mont(size, w); rows = wrap(text, f, width)
    for i, r in enumerate(rows): sl.text((x, y + i * size * lh), r, f, col, anchor)
    return y + len(rows) * size * lh


def para_h(text, size=30, w=300, width=920, lh=1.32):
    return len(wrap(text, mont(size, w), width)) * size * lh


def heart(sl, cx, cy, r, col=POWDER):
    t = np.linspace(0, 2 * np.pi, 120)
    xs = 16 * np.sin(t) ** 3; ys = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    sl.d.polygon([((cx + x * r / 16) * S, (cy + y * r / 16) * S) for x, y in zip(xs, ys)], fill=col + (255,))


def pill(sl, x, y, text, bg, fg, size=24, w=500, anchor="lt"):
    f = mont(size, w); tw = f.getlength(text) / S; ph = size + 22
    if anchor == "mm": x, y = x - (tw + 32) / 2, y - ph / 2
    sl.rrect((x, y, x + tw + 32, y + ph), ph / 2, bg, 235)
    sl.text((x + 16, y + ph / 2), text, f, fg, "lm")
    return (x, y, x + tw + 32, y + ph)


def smooth(pts, n=80):
    """Catmull-Rom through the points"""
    p = np.float32(pts); p = np.vstack([p[0], p, p[-1]]); out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n // (len(p) - 3), endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-2]); return np.float32(out)


def mapper(box, frac):
    """half-image fractions -> slide coordinates inside a card showing crop `frac`"""
    x0, y0, x1, y1 = box; cx0, cy0, cx1, cy1 = frac
    return lambda fx, fy: (x0 + (fx - cx0) / (cx1 - cx0) * (x1 - x0), y0 + (fy - cy0) / (cy1 - cy0) * (y1 - y0))


# ------------------------------------------------------------------ animation
class El:
    def __init__(self, rgba, t0=0.0, kind="rise", d=0.6, pulse=False):
        bb = rgba.getbbox() or (0, 0, 1, 1)
        self.x, self.y = bb[0], bb[1]
        a = np.asarray(rgba.crop(bb)).astype(np.float32) / 255
        self.rgb = a[..., :3] * a[..., 3:]; self.a = a[..., 3:]
        self.t0, self.kind, self.d, self.pulse = t0, kind, d, pulse

    def draw(self, fr, t):
        p = (t - self.t0) / self.d
        if p <= 0: return
        q = min(p, 1); e = 1 - (1 - q) ** 3
        rgb, a, x, y, k = self.rgb, self.a, self.x, self.y, 1.0
        if self.kind == "rise": y += int(40 * (1 - e)); k = e
        elif self.kind == "fade": k = e
        elif self.kind == "pop":
            k = min(1, p * 2.2); s = 0.9 + 0.1 * e
            if s < 0.995:
                h0, w0 = a.shape[:2]; nw, nh = max(1, int(w0 * s)), max(1, int(h0 * s))
                rgb = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize((nw, nh))).astype(np.float32) / 255
                a = (np.asarray(Image.fromarray((a[..., 0] * 255).astype(np.uint8)).resize((nw, nh))).astype(np.float32) / 255)[..., None]
                x += (w0 - nw) // 2; y += (h0 - nh) // 2
        elif self.kind == "wipe":
            cut = max(1, int(a.shape[1] * e)); rgb, a = rgb[:, :cut], a[:, :cut]
        elif self.kind == "down":
            cut = max(1, int(a.shape[0] * e)); rgb, a = rgb[:cut], a[:cut]
        if self.pulse and 0 < t - self.t0 - self.d < 3.2:   # two soft pulses, then it settles and stays
            k *= 0.72 + 0.28 * math.cos((t - self.t0 - self.d) * 2 * math.pi / 1.6)
        h, w = a.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x1 <= x0 or y1 <= y0: return
        sl_ = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x)); d = fr[y0:y1, x0:x1]
        d *= 1 - a[sl_] * k; d += rgb[sl_] * k


def layer(fn):
    sl = Slide(W, H, Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))); fn(sl)
    return sl.im.resize((W, H), Image.LANCZOS)


def E(fn, t0=0.0, kind="rise", d=0.6, pulse=False):
    return El(layer(fn), t0, kind, d, pulse)


def navy_bg():
    sl = Slide(W, H); sl.rrect((0, 0, W, H), 0, NAVY); vshade(sl, 0, H, 0.0, 0.5, DEEP)
    return sl


def photo_bg(name, cx=0.5, cy=0.5, k=0.5, dark=0.45, blur=0, gray=0.2):
    im = fit(photo(name), W * S, H * S, cx, cy)
    if blur: im = im.filter(ImageFilter.GaussianBlur(blur * S))
    a = np.asarray(im).astype(np.float32) / 255
    a = a * (1 - gray) + a.mean(2, keepdims=True) * gray
    a = a * (1 - k) + np.float32(NAVY) / 255 * k * (0.4 + 0.6 * a.mean(2, keepdims=True) * 1.5)
    return Slide(W, H, Image.fromarray(np.clip(a * (1 - dark) * 255, 0, 255).astype(np.uint8)))


def chrome(sl, n):
    logo(sl, 84)
    sl.text((W - 80, 1306), f"{n:02d} / {TOTAL:02d}", mont(24, 400), WHITE, "rs", a=170, spacing=2)


def render(n, bg_sl, els, dur):
    chrome(bg_sl, n)
    base = np.asarray(bg_sl.im.resize((W, H), Image.LANCZOS).convert("RGB")).astype(np.float32) / 255
    out = os.path.join(OUT, f"{n:02d}.mp4")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    last = None
    dur = max(dur, max((e.t0 + e.d + (3.2 if e.pulse else 0)) for e in els)) + HOLD  # everything stays on screen, static
    for i in range(int(dur * FPS)):
        fr = base.copy(); t = i / FPS
        for e in els: e.draw(fr, t)
        last = (np.clip(fr, 0, 1) * 255).astype(np.uint8); p.stdin.write(last.tobytes())
    p.stdin.close(); p.wait()
    Image.fromarray(last).save(os.path.join(OUT, f"{n:02d}_last.jpg"), quality=92)
    print("done", n, flush=True)


def card_el(im, box, t0, kind="rise", r=24, cx=0.5, cy=0.5, shape="rect"):
    def f(sl):
        x0, y0, x1, y1 = box; w, h = int((x1 - x0) * S), int((y1 - y0) * S)
        m = Image.new("L", (w, h), 0); dm = ImageDraw.Draw(m)
        if shape == "circle": dm.ellipse((0, 0, w - 1, h - 1), fill=255)
        else: dm.rounded_rectangle((0, 0, w - 1, h - 1), radius=r * S, fill=255)
        sl.shadow(box, r if shape == "rect" else (x1 - x0) // 2, 20, 120)
        sl.im.paste(fit(im, w, h, cx, cy), (int(x0 * S), int(y0 * S)), m)
    return E(f, t0, kind, 0.7)


def swipe(sl, x, y, col=POWDER):
    for i in range(3):
        cx = x + i * 26; sl.poly([(cx, y - 13), (cx + 13, y), (cx, y + 13)], col, 2.4, 255 - i * 60)


def arrow_to(sl, a, b, col, bend=0.2, w=2.6):
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    ctrl = (mx - (b[1] - a[1]) * bend, my + (b[0] - a[0]) * bend)
    path = ss.bez(a, ctrl, b, n=40)[:-3]
    sl.poly(path, DEEP, w + 3.4, 110); sl.head(path, DEEP, w + 3.4, 16, 110)
    sl.poly(path, col, w); sl.head(path, col, w, 15)


# contours on the main patient (fractions of a half image)
B_L = [(0.46, 0.36), (0.43, 0.44), (0.43, 0.52), (0.43, 0.60), (0.44, 0.68), (0.45, 0.76), (0.455, 0.82)]
B_R = [(0.47, 0.37), (0.50, 0.44), (0.505, 0.52), (0.50, 0.60), (0.495, 0.68), (0.49, 0.76), (0.485, 0.82)]
T_L = [(0.46, 0.36), (0.45, 0.44), (0.455, 0.52), (0.457, 0.60), (0.46, 0.68), (0.462, 0.76), (0.462, 0.82)]
T_R = [(0.47, 0.37), (0.484, 0.44), (0.482, 0.52), (0.48, 0.60), (0.478, 0.68), (0.476, 0.76), (0.476, 0.82)]
A_L = [(0.52, 0.38), (0.515, 0.45), (0.515, 0.55), (0.51, 0.65), (0.505, 0.75)]
A_R = [(0.535, 0.38), (0.545, 0.45), (0.55, 0.55), (0.555, 0.65), (0.56, 0.75)]


# ------------------------------------------------------------------ slides
def s1():
    sl = photo_bg(SURGEON, cx=0.5, cy=0.22, k=0.4, dark=0.2)
    vshade(sl, 0, 240, 0.55, 0.0, DEEP); vshade(sl, 520, 1000, 0.0, 0.95, DEEP)
    chrome(sl, 1)
    y = sl.lines((80, 720), ["ВСЮ ЖИЗНЬ ДУМАЛИ, ЧТО У ВАС", "КРИВЫЕ НОГИ И С ЭТИМ", "НИЧЕГО НЕЛЬЗЯ СДЕЛАТЬ?"], bebas(88), POWDER, lh=0.92)
    sl.poly([(82, y + 30), (200, y + 30)], POWDER, 2)
    y = para(sl, 80, y + 58, "А что, если дело вовсе не в костях?", 38, 400, WHITE, 900)
    swipe(sl, 86, y + 46)
    sl.save("01.jpg")
    print("done 1", flush=True)


def s2():
    bg = navy_bg(); els = []
    t = "Если вы думаете, что у вас ортопедические проблемы, возможно, на самом деле"
    els.append(E(lambda sl: para(sl, 80, 150, t, 32, 300, WHITE, 900), 0.2))
    y0 = 150 + para_h(t, 32, 300, 900) + 16
    els.append(E(lambda sl: sl.lines((80, y0), ["У ВАС ДЕФИЦИТ", "МЯГКИХ ТКАНЕЙ"], bebas(112), POWDER, lh=0.9), 1.3, "pop", 0.7))
    # the patient right away: before with the gap where soft tissue is missing, after with the contour filled
    top = int(y0 + 235); gap_ = 22; cw = (W - 120 - gap_) // 2
    ch = int(cw * ((CROP[3] - CROP[1]) * photo(MAIN).height) / ((CROP[2] - CROP[0]) * (photo(MAIN).width // 2)))
    ch = min(ch, 1272 - top); bb = (60, top, 60 + cw, top + ch); ab = (60 + cw + gap_, top, 60 + 2 * cw + gap_, top + ch)
    mb, ma = mapper(bb, CROP), mapper(ab, CROP)
    els.append(card_el(half("b", CROP), bb, 0.6))
    els.append(E(lambda sl: pill(sl, bb[0] + 16, bb[1] + 16, "ДО", NAVY, WHITE, 24, 600), 0.9, "fade"))
    els.append(E(lambda sl: contours(sl, mb, [B_L, B_R], WHITE), 2.3, "down", 1.0))
    def gapfill(sl):
        poly = [mb(*q) for q in smooth(B_L, 60)] + [mb(*q) for q in smooth(B_R, 60)[::-1]]
        g = Image.new("RGBA", sl.im.size, (0, 0, 0, 0)); ImageDraw.Draw(g).polygon([(x * S, y * S) for x, y in poly], fill=(247, 196, 208, 175))
        sl.im.alpha_composite(g.filter(ImageFilter.GaussianBlur(3 * S)))
    els.append(E(gapfill, 3.2, "fade", 0.8, pulse=True))
    gx, gy = mb(0.465, 0.62)
    els.append(E(lambda sl: (arrow_to(sl, (bb[0] + cw / 2, bb[3] - 70), (gx, gy + 30), POWDER, 0.2),
                             pill(sl, bb[0] + cw / 2, bb[3] - 50, "не хватает объёма", POWDER, NAVY, 22, 500, "mm")), 3.6, "fade"))
    els.append(card_el(half("a", CROP), ab, 4.6, "wipe"))
    els.append(E(lambda sl: pill(sl, ab[0] + 16, ab[1] + 16, "ПОСЛЕ", POWDER, NAVY, 24, 600), 5.2, "fade"))
    els.append(E(lambda sl: contours(sl, ma, [A_L, A_R], POWDER, dashed=False, w=3), 5.4, "down", 1.0))
    els.append(E(lambda sl: pill(sl, ab[0] + cw / 2, ab[3] - 50, "объём восполнен", POWDER, NAVY, 22, 500, "mm"), 6.0, "fade"))
    render(2, bg, els, 8.0)


def icon_dress(sl, c):
    x, y = c; sl.circle(c, 48, POWDER, 2)
    sl.d.polygon([((x + dx) * S, (y + dy) * S) for dx, dy in [(-9, -22), (9, -22), (12, -8), (24, 24), (-24, 24), (-12, -8)]],
                 outline=POWDER + (255,), width=int(2.2 * S))
    sl.poly([(-9 + x, y - 22), (-6 + x, y - 30)], POWDER, 2); sl.poly([(9 + x, y - 22), (6 + x, y - 30)], POWDER, 2)
    sl.poly([(x - 12, y - 8), (x + 12, y - 8)], POWDER, 2)


def icon_mirror(sl, c):
    x, y = c; sl.circle(c, 48, POWDER, 2)
    sl.d.ellipse(((x - 15) * S, (y - 28) * S, (x + 15) * S, (y + 14) * S), outline=POWDER + (255,), width=int(2.2 * S))
    sl.poly([(x, y + 14), (x, y + 24)], POWDER, 2.2); sl.poly([(x - 12, y + 26), (x + 12, y + 26)], POWDER, 2.2)
    sl.poly([(x - 6, y - 14), (x + 2, y - 22)], POWDER, 1.6, 200)


def icon_gap(sl, c):
    x, y = c; sl.circle(c, 48, POWDER, 2)
    sl.poly(smooth([(x - 6, y - 28), (x - 12, y - 6), (x - 11, y + 12), (x - 6, y + 28)], 30), POWDER, 2.4)
    sl.poly(smooth([(x + 6, y - 28), (x + 12, y - 6), (x + 11, y + 12), (x + 6, y + 28)], 30), POWDER, 2.4)
    sl.poly([(x - 5, y), (x + 5, y)], POWDER, 1.6); sl.head([(x + 1, y), (x + 5, y)], POWDER, 1.6, 4)
    sl.head([(x - 1, y), (x - 5, y)], POWDER, 1.6, 4)


def s3():
    bg = navy_bg(); els = []
    items = [(icon_dress, "Знаю, как хочется надеть короткое платье, но в итоге выбираешь брюки."),
             (icon_mirror, "Когда перед зеркалом пытаешься поставить ноги так, чтобы они выглядели ровнее."),
             (icon_gap, "Когда даже при сомкнутых коленях между голенями остаётся большое расстояние.")]
    y = 190
    for i, (ic, t) in enumerate(items):
        yy = y
        els.append(E(lambda sl, ic=ic, yy=yy: ic(sl, (132, yy + 46)), 0.3 + i * 1.6, "pop", 0.6))
        els.append(E(lambda sl, t=t, yy=yy: para(sl, 210, yy + 4, t, 34, 300, WHITE, 800), 0.5 + i * 1.6))
        y += max(150, para_h(t, 34, 300, 800) + 70)
        if i < 2: els.append(E(lambda sl, yy=y: sl.poly([(210, yy - 34), (330, yy - 34)], POWDER, 1.4, 160), 0.9 + i * 1.6, "wipe"))
    yb = y + 40
    els.append(E(lambda sl: (sl.poly([(82, yb), (200, yb)], POWDER, 2), sl.lines((80, yb + 30), ["БЛАГО, ЕСТЬ", "ЛИПОФИЛИНГ"], bebas(118), POWDER, lh=0.9),
                             swipe(sl, 90, yb + 260)), 5.2, "pop", 0.7))
    render(3, bg, els, 8.5)


def contours(sl, mp, sets, col, dashed=True, w=2.6):
    for pts in sets:
        p = smooth([mp(*q) for q in pts], 90)
        if dashed: sl.dashed(p, DEEP, w + 3, 10, 8, 110); sl.dashed(p, col, w, 10, 8, 255)
        else: sl.poly(p, DEEP, w + 3, 110); sl.poly(p, col, w)


def s4():
    bg = navy_bg(); els = []
    els.append(E(lambda sl: sl.lines((80, 150), ["ПОСМОТРИТЕ НА МОЮ ПАЦИЕНТКУ", "ДО ОПЕРАЦИИ"], bebas(80), POWDER, lh=0.92), 0.2))
    box = (60, 330, 510, 942); mp = mapper(box, CROP)
    els.append(card_el(half("b", CROP), box, 0.6))
    els.append(E(lambda sl: pill(sl, box[0] + 16, box[1] + 16, "ДО", NAVY, WHITE, 24, 600), 0.9, "fade"))
    t1 = "Обратите внимание на внутренний контур голеней и выраженное расстояние между ними."
    t2 = "В её случае причина заключалась не в искривлении костей, а в недостаточном объёме мягких тканей."
    els.append(E(lambda sl: para(sl, 550, 340, t1, 29, 300, WHITE, 470), 1.2))
    els.append(E(lambda sl: contours(sl, mp, [B_L, B_R], WHITE), 1.8, "down", 1.2))
    def gapfill(sl):
        poly = [mp(*q) for q in smooth(B_L, 60)] + [mp(*q) for q in smooth(B_R, 60)[::-1]]
        g = Image.new("RGBA", sl.im.size, (0, 0, 0, 0)); ImageDraw.Draw(g).polygon([(x * S, y * S) for x, y in poly], fill=(247, 196, 208, 175))
        sl.im.alpha_composite(g.filter(ImageFilter.GaussianBlur(3 * S)))
    els.append(E(gapfill, 3.0, "fade", 0.8, pulse=True))
    gx, gy = mp(0.47, 0.60)
    els.append(E(lambda sl: (arrow_to(sl, (gx + 120, gy + 120), (gx + 6, gy + 4), POWDER, 0.25),
                             pill(sl, gx + 120, gy + 140, "расстояние", POWDER, NAVY, 22, 500, "mm")), 3.4, "fade"))
    y2 = 340 + para_h(t1, 29, 300, 470) + 30
    els.append(E(lambda sl: para(sl, 550, y2, t2, 29, 300, WHITE, 470), 4.4))
    els.append(E(lambda sl: (para(sl, 80, 985, "Это называется", 30, 300, WHITE, 900),
                             sl.lines((80, 1030), ["ЛОЖНОЙ КРИВИЗНОЙ НОГ"], bebas(96), POWDER)), 6.2, "pop", 0.7))
    els.append(E(lambda sl: para(sl, 80, 1150, "И с этим мы можем работать.", 34, 500, WHITE, 900), 7.6))
    render(4, bg, els, 10.0)


def s5():
    bg = navy_bg(); els = []
    els.append(E(lambda sl: sl.lines((80, 150), ["В ДАННОМ СЛУЧАЕ ПЕРЕДО МНОЙ", "СТОЯЛА ЗАДАЧА:"], bebas(80), POWDER, lh=0.92), 0.2))
    t = "Добавить объём именно там, где его не хватало, чтобы сформировать более плавный и гармоничный контур ног."
    els.append(E(lambda sl: para(sl, 80, 330, t, 31, 300, WHITE, 920), 0.9))
    top = 330 + para_h(t, 31, 300, 920) + 30; hgt = 1240 - top; wid = int(hgt * (CROP[2] - CROP[0]) * 512 / ((CROP[3] - CROP[1]) * 761))
    x0 = 80; box = (x0, int(top), x0 + wid, int(top + hgt)); mp = mapper(box, CROP)
    els.append(card_el(half("b", CROP), box, 0.5))
    els.append(E(lambda sl: contours(sl, mp, [B_L, B_R], WHITE), 1.6, "fade"))
    els.append(E(lambda sl: contours(sl, mp, [T_L, T_R], POWDER, dashed=False, w=3.2), 2.6, "down", 1.2))
    def fill(sl):
        g = Image.new("RGBA", sl.im.size, (0, 0, 0, 0)); dg = ImageDraw.Draw(g)
        for a, b in ((B_L, T_L), (B_R, T_R)):
            poly = [mp(*q) for q in smooth(a, 60)] + [mp(*q) for q in smooth(b, 60)[::-1]]
            dg.polygon([(x * S, y * S) for x, y in poly], fill=(247, 196, 208, 225))
        sl.im.alpha_composite(g.filter(ImageFilter.GaussianBlur(4 * S))); sl.im.alpha_composite(g)
    els.append(E(fill, 4.0, "fade", 0.8, pulse=True))
    # legend on the right of the card
    lx = box[2] + 50; ly = box[1] + 40
    els.append(E(lambda sl: (sl.dashed([(lx, ly), (lx + 60, ly)], WHITE, 2.6, 10, 8), para(sl, lx, ly + 24, "контур сейчас", 28, 300, WHITE, 360)), 1.6, "fade"))
    els.append(E(lambda sl: (sl.poly([(lx, ly + 130), (lx + 60, ly + 130)], POWDER, 3.2), para(sl, lx, ly + 154, "контур, который хотим получить", 28, 300, POWDER, 360)), 2.6, "fade"))
    els.append(E(lambda sl: (sl.rrect((lx, ly + 290, lx + 60, ly + 316), 8, POWDER, 200),
                             para(sl, lx, ly + 334, "здесь добавляем объём", 28, 500, POWDER, 360)), 4.0, "fade"))
    gx, gy = mp(0.45, 0.56)
    els.append(E(lambda sl: arrow_to(sl, (lx - 10, ly + 360), (gx - 4, gy), POWDER, -0.2), 4.4, "fade"))
    render(5, bg, els, 8.5)


def step_icon(sl, c, kind, r=56):
    x, y = c; sl.circle(c, r, POWDER, 2)
    if kind == "take":
        sl.d.polygon([(px * S, py * S) for px, py in [(x, y - 26), (x + 16, y + 2), (x + 16, y + 8)] +
                      [(x + 16 * math.cos(a), y + 8 + 16 * math.sin(a)) for a in np.linspace(0, math.pi, 30)] +
                      [(x - 16, y + 8), (x - 16, y + 2)]], outline=POWDER + (255,), width=int(2.4 * S))
    elif kind == "prep":
        sl.poly([(x - 30, y + 14), (x + 14, y - 30)], POWDER, 2.4)
        sl.d.polygon([((x - 22 + dx) * S, (y + 22 + dy) * S) for dx, dy in [(0, 0), (-8, -8), (28, -44), (36, -36)]],
                     outline=POWDER + (255,), width=int(2.2 * S))
        sl.poly([(x + 10, y - 34), (x + 26, y - 18)], POWDER, 2.4); sl.poly([(x - 26, y + 26), (x - 36, y + 36)], POWDER, 2.4)
    else:
        sl.circle(c, 26, POWDER, 2.2); sl.circle(c, 12, POWDER, 2.2)
        sl.poly([(x, y - 34), (x, y + 34)], POWDER, 1.6, 140); sl.poly([(x - 34, y), (x + 34, y)], POWDER, 1.6, 140)


def s6():
    bg = photo_bg(HANDS, cx=0.5, cy=0.4, k=0.55, dark=0.62, blur=3); els = []
    els.append(E(lambda sl: sl.lines((80, 150), ["БЕЗ УСТАНОВКИ ИМПЛАНТОВ"], bebas(96), POWDER), 0.2, "pop"))
    els.append(E(lambda sl: para(sl, 80, 260, "Так работает липофилинг голеней.", 36, 400, WHITE, 900), 0.9))
    els.append(E(lambda sl: para(sl, 80, 320, "Мы забрали жировую ткань, подготовили и пересадили в те зоны, где не хватало объёма.",
                                 30, 300, WHITE, 920), 1.5))
    cy = 560; xs = (220, 540, 860)
    labels = [("забрали", "жировую ткань"), ("подготовили", "к пересадке"), ("пересадили", "в зоны дефицита")]
    for i, (x, kind) in enumerate(zip(xs, ("take", "prep", "place"))):
        els.append(E(lambda sl, x=x, kind=kind, i=i: (sl.circle((x, cy), 56, NAVY, 0, 255, fill=NAVY + (215,)), step_icon(sl, (x, cy), kind),
                                                     sl.text((x, cy + 88), labels[i][0], mont(30, 600), WHITE, "ma"),
                                                     sl.text((x, cy + 128), labels[i][1], mont(26, 300), WHITE, "ma")), 2.3 + i * 0.9, "pop"))
        if i < 2:
            a, b = xs[i], xs[i + 1]
            els.append(E(lambda sl, a=a, b=b: (sl.dashed(ss.bez((a + 70, cy), ((a + b) / 2, cy - 28), (b - 70, cy), n=30), POWDER, 2, 9, 8, 230),
                                               sl.head(ss.bez((a + 70, cy), ((a + b) / 2, cy - 28), (b - 70, cy), n=30), POWDER, 2, 13)),
                         2.8 + i * 0.9, "wipe", 0.5))
    # where: inner contour on a round crop of the legs
    box = (640, 800, 1000, 1160); fr = (0.10, 0.30, 0.85, 0.80); mp = mapper(box, fr)
    els.append(card_el(half("b", fr), box, 5.0, "pop", shape="circle"))
    els.append(E(lambda sl: contours(sl, mp, [B_L[1:], B_R[1:]], POWDER, dashed=False, w=3.4), 5.6, "down", 1.0, pulse=True))
    els.append(E(lambda sl: para(sl, 80, 860, "Работали преимущественно с внутренним контуром голеней.", 34, 400, WHITE, 520), 5.3))
    els.append(E(lambda sl: arrow_to(sl, (470, 1060), (650, 990), POWDER, -0.25), 5.9, "fade"))
    render(6, bg, els, 8.5)


def s7():
    bg = navy_bg(); els = []
    def title(sl):
        f = bebas(76); sl.text((80, 150), "ПОСМОТРИТЕ, КАК ИЗМЕНИЛИСЬ НОГИ", f, POWDER)
        heart(sl, 80 + f.getlength("ПОСМОТРИТЕ, КАК ИЗМЕНИЛИСЬ НОГИ") / S + 30, 182, 17)
    els.append(E(title, 0.2))
    bul = ["Более наполненный внутренний контур.", "Плавный переход от колена к голени.",
           "Менее выраженное расстояние между голенями.", "Визуально более ровная линия ног."]
    by = 252
    cw, ch = 450, 612; bx = (60, 1080 - 60 - cw); cy0 = 440
    bb = (bx[0], cy0, bx[0] + cw, cy0 + ch); ab = (bx[1], cy0, bx[1] + cw, cy0 + ch)
    ma = mapper(ab, CROP)
    els.append(card_el(half("b", CROP), bb, 0.5))
    els.append(E(lambda sl: pill(sl, bb[0] + 16, bb[1] + 16, "ДО", NAVY, WHITE, 24, 600), 0.8, "fade"))
    els.append(card_el(half("a", CROP), ab, 1.2, "wipe"))
    els.append(E(lambda sl: pill(sl, ab[0] + 16, ab[1] + 16, "ПОСЛЕ", POWDER, NAVY, 24, 600), 1.8, "fade"))
    marks = [lambda sl: contours(sl, ma, [A_L, A_R], POWDER, dashed=False, w=3),
             lambda sl: [sl.circle(ma(*q), 15, POWDER, 2.6) for q in ((0.49, 0.44), (0.575, 0.44))],
             lambda sl: (arrow_to(sl, ma(0.36, 0.62), ma(0.505, 0.62), POWDER, 0), arrow_to(sl, ma(0.71, 0.62), ma(0.56, 0.62), POWDER, 0)),
             lambda sl: sl.dashed([ma(0.53, 0.36), ma(0.53, 0.84)], WHITE, 1.8, 8, 8, 200)]
    for i, b in enumerate(bul):
        t0 = 2.4 + i * 1.1
        els.append(E(lambda sl, b=b, i=i: (sl.rrect((82, by + i * 44 + 10, 96, by + i * 44 + 24), 2, POWDER),
                                         sl.text((112, by + i * 44), b, mont(29, 300), WHITE)), t0))
        els.append(E(marks[i], t0 + 0.2, "down" if i == 0 else "fade", 0.8))
    els.append(E(lambda sl: (para(sl, 80, 1080, "Мы не меняли положение костей.", 30, 300, WHITE, 920),
                             para(sl, 80, 1122, "Мы изменили объём и распределение мягких тканей.", 30, 600, POWDER, 920)), 7.2))
    render(7, bg, els, 10.0)


def example(n, title, front, back, title_size=80):
    """one patient per slide: front and back before/after, stacked (backgrounds removed to black, retouched)"""
    bg = navy_bg(); els = []
    y = 150 + len(title) * title_size * 0.92
    els.append(E(lambda sl: sl.lines((80, 150), title, bebas(title_size), POWDER, lh=0.92), 0.2))
    cw = 760; fr_ = (0.0, 0.30, 1.0, 0.90); ch = int(cw * (fr_[3] - fr_[1]))
    top = int(y + 30); gap_ = 22; x0 = (W - cw) // 2
    for i, (f, lab) in enumerate(((front, "спереди"), (back, "сзади"))):
        box = (x0, top + i * (ch + gap_), x0 + cw, top + i * (ch + gap_) + ch)
        im = photo(f); im = im.crop((0, int(im.height * fr_[1]), im.width, int(im.height * fr_[3])))
        els.append(card_el(im, box, 0.7 + i * 0.9, "rise"))
        els.append(E(lambda sl, box=box, lab=lab: (pill(sl, box[0] + 14, box[1] + 14, lab, WHITE, NAVY, 22, 600),
                                                  pill(sl, box[0] + 14, box[3] - 54, "до", NAVY, WHITE, 20, 600),
                                                  pill(sl, box[0] + cw / 2 + 14, box[3] - 54, "после", POWDER, NAVY, 20, 600)),
                     1.1 + i * 0.9, "fade"))
    render(n, bg, els, 5.0)


def s8():
    example(8, ["ТАКЖЕ ПОКАЗЫВАЮ ДРУГИЕ", "ПРИМЕРЫ МОИХ ПАЦИЕНТОК"], "IMG_1150_black.JPG", "IMG_1151_black.JPG")


def s8b():
    example(9, ["ЕЩЁ ОДИН ПРИМЕР"], "IMG_1153_black.JPG", "IMG_1154_black.JPG")


def s9():
    bg = navy_bg(); els = []
    els.append(E(lambda sl: sl.lines((80, 160), ["А ЕСЛИ У ВАС", "ПОХОЖАЯ СИТУАЦИЯ?"], bebas(110), POWDER, lh=0.9), 0.2, "pop"))
    els.append(E(lambda sl: para(sl, 80, 400, "Не спешите думать, что вам обязательно нужны импланты или сложная операция на костях.",
                                 34, 300, WHITE, 920), 1.0))
    chips = [("импланты", 80, 620), ("операция на костях", 80, 720)]
    for i, (t, x, y) in enumerate(chips):
        def chip(sl, t=t, x=x, y=y):
            f = mont(34, 400); tw = f.getlength(t) / S
            sl.rrect((x, y, x + tw + 60, y + 70), 35, NAVY, 0, outline=POWDER, w=1.6)
            sl.text((x + 30, y + 35), t, f, WHITE, "lm")
        els.append(E(chip, 2.0 + i * 0.5, "pop"))
        tw = mont(34, 400).getlength(t) / S
        els.append(E(lambda sl, x=x, y=y, tw=tw: sl.poly([(x + 16, y + 37), (x + tw + 44, y + 33)], POWDER, 3.4), 3.0 + i * 0.4, "wipe", 0.5))
    els.append(E(lambda sl: (sl.poly([(82, 900), (200, 900)], POWDER, 2),
                             para(sl, 80, 930, "Но и липофилинг подходит не всем.", 40, 600, POWDER, 920)), 4.4))
    render(10, bg, els, 7.0)


def s10():
    bg = photo_bg(AI_LEGS, cx=0.85, cy=0.5, k=0.5, dark=0.7, blur=2); els = []
    els.append(E(lambda sl: sl.lines((80, 160), ["ВАЖНО", "РАЗОБРАТЬСЯ:"], bebas(120), POWDER, lh=0.9), 0.2, "pop"))
    items = ["с чем связана именно ваша особенность,", "оценить состояние тканей", "и наличие донорского жира."]
    y = 470
    for i, t in enumerate(items):
        yy = y + i * 150
        els.append(E(lambda sl, yy=yy: icon_circle(sl, (124, yy + 28), "check", 36), 1.0 + i * 1.0, "pop"))
        els.append(E(lambda sl, t=t, yy=yy: para(sl, 190, yy, t, 36, 300, WHITE, 800), 1.2 + i * 1.0))
        if i < 2: els.append(E(lambda sl, yy=yy: sl.dashed([(124, yy + 70), (124, yy + 140)], POWDER, 1.6, 5, 6, 180), 1.6 + i * 1.0, "down", 0.4))
    render(11, bg, els, 6.5)


def s11():
    bg = photo_bg(SURGEON, cx=0.5, cy=0.5, k=0.5, dark=0.55, blur=4); els = []
    vshade(bg, 0, H, 0.3, 0.75, DEEP)
    t1 = "Если вы годами обращаете внимание на форму своих ног и хотели бы её изменить,"
    els.append(E(lambda sl: para(sl, 80, 200, t1, 36, 300, WHITE, 920), 0.3))
    y = 200 + para_h(t1, 36, 300, 920) + 24
    els.append(E(lambda sl: sl.lines((80, y), ["Я ЖДУ ВАС", "НА КОНСУЛЬТАЦИИ"], bebas(120), POWDER, lh=0.9), 1.3, "pop"))
    y2 = y + 250
    els.append(E(lambda sl: sl.poly([(82, y2), (200, y2)], POWDER, 2), 2.2, "wipe", 0.5))
    els.append(E(lambda sl: para(sl, 80, y2 + 34, "Я помогу разобраться в причине и расскажу, какой результат мы можем получить в вашем случае.",
                                 34, 300, WHITE, 920), 2.5))
    render(12, bg, els, 7.0)


SLIDES = [s1, s2, s3, s4, s5, s6, s7, s8, s8b, s9, s10, s11]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i, fn in enumerate(SLIDES, 1):
        if not ONLY or i in ONLY: fn()
