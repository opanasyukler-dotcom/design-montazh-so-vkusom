"""Шаблоны постов и сторис для хирурга в стиле референсов (сторис «Что меняется после операции…»
и посты «Перестаньте убеждать себя…»), перенесённые на палитру и шрифты бренда.

Палитра бренда:  MAUVE #686068 · POWDER #E2D2D3 · NAVY #102E46
Посты  (1080x1350): заголовки Bebas Neue, текст тонкий Montserrat Light.
Сторис (1080x1920): классический Montserrat — Bold для акцентов, Light для текста.

usage: python3 surgeon_style.py PHOTO.jpg OUT_DIR
Всё рисуется в 2x и уменьшается до 1080 — линии, стрелки и иконки получаются гладкими.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

PHOTO, OUT = sys.argv[1:3]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEBAS = os.path.join(ROOT, "skills/montage-svg-typography/Шрифты/Коллекция шрифтов/BebasNeueCyrillic.ttf")
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")

# brand palette (sampled from the brand swatches)
MAUVE = (104, 96, 104)
POWDER = (226, 210, 211)
NAVY = (16, 46, 70)
INK = (40, 36, 42)          # near-black with a mauve cast, for text on powder
DEEP = (10, 22, 34)         # darkest navy for cinematic backgrounds
WHITE = (255, 255, 255)
S = 2                       # supersampling


def mont(size, w=300):
    f = ImageFont.truetype(MONT, size * S); f.set_variation_by_axes([w]); return f


def bebas(size): return ImageFont.truetype(BEBAS, size * S)


class Slide:
    def __init__(self, w, h, bg=None):
        self.w, self.h = w, h
        self.im = bg.convert("RGBA") if bg is not None else Image.new("RGBA", (w * S, h * S), NAVY + (255,))
        self.d = ImageDraw.Draw(self.im)

    # ---------- text
    def text(self, xy, s, f, col=WHITE, anchor="la", a=255, spacing=0):
        x, y = xy
        if spacing:  # letter-spacing for thin caps
            tw = sum(f.getlength(c) for c in s) + spacing * S * (len(s) - 1)
            if anchor[0] == "m": x -= tw / S / 2
            elif anchor[0] == "r": x -= tw / S
            cx = x * S
            for c in s:
                self.d.text((cx, y * S), c, font=f, fill=col + (a,), anchor="l" + anchor[1]); cx += f.getlength(c) + spacing * S
            return
        self.d.text((x * S, y * S), s, font=f, fill=col + (a,), anchor=anchor)

    def lines(self, xy, rows, f, col=WHITE, lh=1.15, anchor="la", a=255):
        x, y = xy
        for i, r in enumerate(rows):
            self.text((x, y + i * f.size / S * lh), r, f, col, anchor, a)
        return y + len(rows) * f.size / S * lh

    def glow_text(self, xy, s, f, col=WHITE, anchor="la", blur=10, a=140):
        sh = Image.new("RGBA", self.im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((xy[0] * S, xy[1] * S + 3), s, font=f, fill=DEEP + (a,), anchor=anchor)
        self.im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(blur * S)))
        self.text(xy, s, f, col, anchor)

    # ---------- line art
    def poly(self, pts, col=WHITE, w=2, a=255):
        self.d.line([(x * S, y * S) for x, y in pts], fill=col + (a,), width=int(w * S), joint="curve")

    def dashed(self, pts, col=WHITE, w=1.6, dash=9, gap=8, a=200):
        pts = np.float32(pts); seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); acc = np.concatenate([[0], np.cumsum(seg)])
        on, s0 = True, 0.0
        while s0 < acc[-1]:
            s1 = min(s0 + (dash if on else gap), acc[-1])
            if on:
                ss = np.linspace(s0, s1, 6)
                p = np.stack([np.interp(ss, acc, pts[:, 0]), np.interp(ss, acc, pts[:, 1])], 1)
                self.poly(p, col, w, a)
            s0, on = s1, not on

    def head(self, pts, col=WHITE, w=2, size=13, a=255, filled=False):
        b = np.float32(pts[-1]); p = np.float32(pts[-4] if len(pts) > 4 else pts[0])
        v = (b - p) / (np.linalg.norm(b - p) + 1e-6); n = np.float32([-v[1], v[0]])
        l, r = b - v * size + n * size * 0.55, b - v * size - n * size * 0.55
        if filled: self.d.polygon([tuple(b * S), tuple(l * S), tuple(r * S)], fill=col + (a,))
        else: self.poly([l, b, r], col, w, a)

    def dot(self, xy, r=5, col=WHITE, a=255):
        x, y = xy; self.d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S), fill=col + (a,))

    def circle(self, xy, r, col=WHITE, w=1.6, a=255, fill=None):
        x, y = xy
        self.d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S), outline=col + (a,), width=int(w * S),
                       fill=None if fill is None else fill)

    def rrect(self, box, r, fill, a=255, outline=None, w=1.5):
        # draw on its own layer and composite, so translucent plates blend instead of punching holes
        ly = Image.new("RGBA", self.im.size, (0, 0, 0, 0))
        ImageDraw.Draw(ly).rounded_rectangle(tuple(v * S for v in box), radius=r * S, fill=fill + (a,),
                                             outline=None if outline is None else outline + (255,), width=int(w * S))
        self.im.alpha_composite(ly)

    def shadow(self, box, r, blur=18, a=90):
        sh = Image.new("RGBA", self.im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle(tuple(v * S for v in (box[0], box[1] + 8, box[2], box[3] + 8)), radius=r * S, fill=DEEP + (a,))
        self.im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(blur * S)))

    def save(self, name):
        out = self.im.resize((self.w, self.h), Image.LANCZOS).convert("RGB")
        out.save(os.path.join(OUT, name), quality=95, subsampling=0); return out


# ---------- geometry
def bez(*p, n=80):
    p = [np.float32(q) for q in p]; t = np.linspace(0, 1, n)[:, None]
    if len(p) == 3: return (1 - t) ** 2 * p[0] + 2 * (1 - t) * t * p[1] + t ** 2 * p[2]
    return (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]


def loop_path(a, b, r=22, at=(0.35, 0.75), side=1, n=160):
    """hand-drawn arrow from a to b with one curl in the middle (the 'loop' arrows of the stories)"""
    a, b = np.float32(a), np.float32(b); d = b - a; L = np.linalg.norm(d); v = d / L; p = np.float32([-v[1], v[0]]) * side
    t = np.linspace(0, 1, n); u = np.clip((t - at[0]) / (at[1] - at[0]), 0, 1)
    base = a[None] + d[None] * t[:, None]
    off = r * np.sin(2 * np.pi * u)[:, None] * v[None] + r * (1 - np.cos(2 * np.pi * u))[:, None] * p[None]
    sway = (np.sin(np.pi * t) * r * 0.35)[:, None] * p[None]  # soft hand-drawn bow
    return base + off + sway


# ---------- photo backgrounds
SRC = ImageOps.exif_transpose(Image.open(PHOTO)).convert("RGB")


def cover(im, w, h, cx=0.5, cy=0.5, zoom=1.0):
    s = max(w / im.width, h / im.height) * zoom
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    x, y = int((im.width - w) * cx), int((im.height - h) * cy)
    return im.crop((x, y, x + w, y + h))


def photo_bg(w, h, blur=0, tint=NAVY, k=0.55, dark=0.0, cx=0.5, cy=0.5, zoom=1.0, gray=0.0):
    im = cover(SRC, w * S, h * S, cx, cy, zoom)
    if blur: im = im.filter(ImageFilter.GaussianBlur(blur * S))
    a = np.asarray(im).astype(np.float32) / 255
    if gray: g = a.mean(2, keepdims=True); a = a * (1 - gray) + g * gray
    a = a * (1 - k) + (np.float32(tint) / 255) * k * (0.35 + 0.65 * a.mean(2, keepdims=True) * 1.6)  # tone-map into the brand hue
    a *= 1 - dark
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))


def vshade(sl, y0, y1, a0, a1, col=DEEP):
    ys = np.arange(sl.h * S, dtype=np.float32); p = np.clip((ys - y0 * S) / ((y1 - y0) * S), 0, 1); p = p * p * (3 - 2 * p)
    arr = np.zeros((sl.h * S, sl.w * S, 4), np.uint8); arr[..., :3] = col; arr[..., 3] = ((a0 + (a1 - a0) * p) * 255)[:, None]
    sl.im.alpha_composite(Image.fromarray(arr, "RGBA"))


def logo(sl, y, col=POWDER, a=0.9, scale=1.0):
    """brand mark: circled A + ANNA TVAURI in thin spaced caps (same construction as the existing posts)"""
    r = 22 * scale; f = mont(int(34 * scale), 300); txt = "ANNA TVAURI"; sp = 0.06 * 34 * scale
    tw = sum(f.getlength(c) for c in txt) / S + sp * (len(txt) - 1); gap = 18 * scale
    x0 = (sl.w - (2 * r + gap + tw)) / 2; cx, cy = x0 + r, y
    al = int(255 * a); lw = 2 * scale
    sl.circle((cx, cy), r, col, lw, al)
    yt = cy - r * 0.62; xb = math.sqrt(r * r - (r * 0.8) ** 2); yb = cy + r * 0.8; xt = math.sqrt(r * r - (r * 0.62) ** 2)
    sl.poly([(cx - xt, yt), (cx + xt, yt)], col, lw, al)
    sl.poly([(cx - xb, yb), (cx, yt), (cx + xb, yb)], col, lw, al)
    k = (cy + r * 0.28 - yt) / (yb - yt); sl.poly([(cx - xb * k, cy + r * 0.28), (cx + xb * k, cy + r * 0.28)], col, lw, al)
    sl.text((cx + r + gap, cy + 1), txt, f, col, "lm", al, spacing=sp)


# ---------- icons (thin outline in a circle, like the reference posts)
def icon_circle(sl, c, kind, r=30, col=POWDER):
    x, y = c; sl.circle(c, r, col, 1.8, 230)
    if kind == "search":
        sl.circle((x - 3, y - 3), 9, col, 2.2); sl.poly([(x + 4, y + 4), (x + 12, y + 12)], col, 2.6)
    elif kind == "question":
        sl.text((x, y + 1), "?", mont(30, 300), col, "mm")
    elif kind == "chart":
        sl.poly([(x - 13, y + 11), (x - 13, y - 13)], col, 1.8); sl.poly([(x - 13, y + 11), (x + 14, y + 11)], col, 1.8)
        pts = [(x - 9, y + 5), (x - 2, y - 2), (x + 3, y + 2), (x + 12, y - 9)]; sl.poly(pts, col, 2.2); sl.head(pts, col, 2.2, 7)
    elif kind == "chat":
        sl.circle(c, r, col, 0, 255, fill=col + (255,))
        for dx in (-10, 0, 10): sl.dot((x + dx, y), 3.2, NAVY)
    elif kind == "check":
        sl.poly([(x - 11, y + 1), (x - 3, y + 9), (x + 12, y - 8)], col, 2.4)


def paperclip(sl, c, col=MAUVE, h=40):
    x, y = c  # one wire: down, U-turn, up, U-turn, down
    sl.poly([(x - 6, y - h * 0.45), (x - 6, y + h * 0.4)], col, 2.2)
    sl.poly([(x - 6 + 6 + 6 * math.cos(math.pi * (1 - s)), y + h * 0.4 + 6 * math.sin(math.pi * s)) for s in np.linspace(0, 1, 20)], col, 2.2)
    sl.poly([(x + 6, y + h * 0.4), (x + 6, y - h * 0.55)], col, 2.2)
    sl.poly([(x + 6 - 9 - 9 * math.cos(math.pi * s), y - h * 0.55 - 9 * math.sin(math.pi * s)) for s in np.linspace(0, 1, 20)], col, 2.2)
    sl.poly([(x - 12, y - h * 0.55), (x - 12, y + h * 0.2)], col, 2.2)


def bubble(sl, box, rows, f, side="left", fill=POWDER, col=INK, a=242, lh=1.3):
    x0, y0, x1, y1 = box; r = 34
    sl.shadow(box, r, 14, 70); sl.rrect(box, r, fill, a)
    tx = x0 + 36 if side == "left" else x1 - 36  # little tail at the bottom corner
    tail = [(tx - 4, y1 - 20), (tx + (-34 if side == "left" else 34), y1 + 12), (tx + (22 if side == "left" else -22), y1 - 4)]
    sl.d.polygon([(px * S, py * S) for px, py in tail], fill=fill + (a,))
    h = len(rows) * f.size / S * lh; cy = (y0 + y1) / 2 - h / 2
    for i, rw in enumerate(rows):
        sl.text(((x0 + x1) / 2, cy + i * f.size / S * lh + f.size / S * lh / 2), rw, f, col, "mm")


# =====================================================================================
# POSTS 1080x1350 — Bebas + Montserrat Light
# =====================================================================================
PW, PH = 1080, 1350


def post1():
    """cover: cinematic dark photo, powder Bebas headline left, chat icon + dashed curve to the question"""
    sl = Slide(PW, PH, photo_bg(PW, PH, blur=3, tint=NAVY, k=0.7, dark=0.55, cx=0.5, cy=0.3, zoom=1.15, gray=0.4))
    vshade(sl, 0, PH, 0.25, 0.55)
    logo(sl, 96)
    y = sl.lines((92, 300), ["ПЕРЕСТАНЬТЕ", "УБЕЖДАТЬ", "СЕБЯ,"], bebas(132), POWDER, lh=0.9)
    y = sl.lines((96, y + 18), ["что это вас", "не беспокоит."], mont(46, 300), WHITE, lh=1.2)
    icon_circle(sl, (128, y + 112), "chat", 30)
    sl.lines((186, y + 70), ["Вы привыкли говорить:"], mont(36, 400), WHITE)
    sl.lines((186, y + 116), ["«Да ладно, это", "не так важно»."], mont(36, 300), POWDER, lh=1.2)
    pts = bez((128, y + 170), (110, y + 330), (220, y + 400), (372, y + 392))
    sl.dashed(pts, WHITE, 2, 10, 9, 220); sl.head(pts, WHITE, 2, 14)
    sl.lines((400, y + 352), ["Но почему тогда вы", "снова смотрите на это", "в зеркале?"], mont(40, 300), POWDER, lh=1.2)
    return sl.save("post_01_cover.jpg")


def post2():
    """reasons list with outline icons over the OR photo"""
    sl = Slide(PW, PH, photo_bg(PW, PH, blur=0, tint=NAVY, k=0.5, dark=0.12, cx=0.5, cy=0.2, zoom=1.0))
    vshade(sl, 380, 980, 0.0, 0.88)
    logo(sl, 96)
    sl.text((96, 640), "поэтому", mont(34, 300), WHITE)
    y = sl.lines((92, 690), ["ИЗМЕНЕНИЯ НАЧИНАЮТСЯ", "С РЕШЕНИЯ РАЗОБРАТЬСЯ:"], bebas(96), POWDER, lh=0.92)
    for i, (ic, a, b) in enumerate([("search", "Что именно вас", "беспокоит?"), ("question", "Можно ли это", "исправить?"),
                                    ("chart", "Какой результат", "реально получить?")]):
        cy = y + 90 + i * 130; icon_circle(sl, (130, cy), ic, 34)
        sl.lines((196, cy - 40), [a, b], mont(36, 300), WHITE, lh=1.2)
    return sl.save("post_02_reasons.jpg")


def arch(sl, box, cx=0.5, cy=0.5, zoom=1.0, round_all=False):
    x0, y0, x1, y1 = box; w, h = int((x1 - x0) * S), int((y1 - y0) * S)
    ph = cover(SRC, w, h, cx, cy, zoom)
    ph = Image.fromarray((np.asarray(ph).astype(np.float32) * 0.92 + np.float32(POWDER) * 0.08).astype(np.uint8))  # warm into the palette
    m = Image.new("L", (w, h), 0); dm = ImageDraw.Draw(m)
    if round_all: dm.ellipse((0, 0, w - 1, h - 1), fill=255)
    else: dm.rounded_rectangle((0, 0, w - 1, h - 1 + w // 2), radius=w // 2, fill=255)  # arch: round top, square bottom
    sl.shadow((x0, y0, x1, y1), (x1 - x0) // 2 if round_all else 30, 16, 50)
    sl.im.paste(ph, (int(x0 * S), int(y0 * S)), m)
    return box


def post3():
    """light powder collage with arch frames, dotted path and a ruled list"""
    sl = Slide(PW, PH); sl.rrect((0, 0, PW, PH), 0, POWDER)
    sl.rrect((0, 0, PW, PH), 0, (238, 228, 229), a=110)
    arch(sl, (650, 70, 1020, 480), 0.85, 0.12, 1.5)                    # wall / door
    arch(sl, (680, 540, 940, 800), 0.5, 0.52, 2.3, round_all=True)     # face
    arch(sl, (560, 850, 900, 1290), 0.68, 0.95, 2.4)                   # hands at work
    path = bez((560, 300), (430, 500), (720, 520), (640, 700)); sl.dashed(path, MAUVE, 1.6, 4, 9, 200)
    path2 = bez((700, 800), (560, 860), (500, 900), (530, 1020)); sl.dashed(path2, MAUVE, 1.6, 4, 9, 200)
    for p in ((560, 300), (640, 700), (530, 1020)): sl.dot(p, 6, MAUVE)
    y = sl.lines((70, 150), ["ИНОГДА", "ЭТО", "ВЫГЛЯДИТ", "ТАК:"], bebas(110), NAVY, lh=0.88)
    items = [["выбираете одежду,", "которая скрывает"], ["избегаете", "некоторых фотографий"], ["ищете удачный", "ракурс"]]
    y += 30
    for it in items:
        sl.poly([(72, y), (150, y)], MAUVE, 1.6); sl.lines((72, y + 28), it, mont(36, 300), INK, lh=1.2); y += 175
    icon_circle(sl, (102, y + 40), "chat", 30, MAUVE)
    sl.lines((152, y + 4), ["говорите", "«мне всё равно»"], mont(36, 300), INK, lh=1.2)
    return sl.save("post_03_looks_like.jpg")


def post4():
    """timeline: glowing powder line with years, the 'time doesn't fix it' thought"""
    sl = Slide(PW, PH, photo_bg(PW, PH, blur=0, tint=DEEP, k=0.75, dark=0.62, cx=0.7, cy=0.3, zoom=1.05, gray=0.6))
    vshade(sl, 0, PH, 0.35, 0.2)
    logo(sl, 96)
    sl.text((92, 300), "И ЗДЕСЬ", mont(42, 300), WHITE, spacing=2)
    y = sl.lines((90, 350), ["ВАЖНО ПОНЯТЬ:"], bebas(112), POWDER, lh=0.9)
    y = sl.lines((92, y + 30), ["Со временем проблема редко", "становится меньше."], mont(40, 400), WHITE, lh=1.2)
    sl.lines((92, y), ["Чаще происходит другое."], mont(40, 300), POWDER, lh=1.2)
    # glowing wave
    xs = np.linspace(0, PW, 200); ys = 860 + 26 * np.sin(xs / PW * 2.6 * np.pi + 0.6)
    pts = np.stack([xs, ys], 1)
    glow = Slide(PW, PH, Image.new("RGBA", (PW * S, PH * S), (0, 0, 0, 0)))
    glow.poly(pts, POWDER, 6, 255)
    sl.im.alpha_composite(glow.im.filter(ImageFilter.GaussianBlur(14 * S)))
    sl.im.alpha_composite(glow.im.filter(ImageFilter.GaussianBlur(4 * S)))
    sl.poly(pts, (250, 238, 238), 1.6)
    for yr, x in (("2020", 160), ("2023", 470), ("2026", 860)):
        yy = float(np.interp(x, xs, ys)); sl.dot((x, yy), 7, WHITE); sl.circle((x, yy), 14, POWDER, 1.2, 160)
        sl.text((x, yy - 30), yr, mont(26, 400), WHITE, "ms")
    sl.lines((92, 1000), ["Человек просто привыкает", "жить с ней и откладывает себя"], mont(40, 300), WHITE, lh=1.25)
    sl.text((92, 1100), "ещё на годы.", mont(40, 700), POWDER)
    return sl.save("post_04_time.jpg")


def post5():
    """CTA: consultation card on navy"""
    sl = Slide(PW, PH, photo_bg(PW, PH, blur=14, tint=NAVY, k=0.8, dark=0.45, cy=0.4))
    logo(sl, 110)
    y = sl.lines((PW / 2, 260), ["НА КОНСУЛЬТАЦИИ", "РАЗБЕРЁМ ИМЕННО", "ВАШУ СИТУАЦИЮ"], bebas(112), POWDER, lh=0.9, anchor="ma")
    sl.poly([(PW / 2 - 70, y + 30), (PW / 2 + 70, y + 30)], POWDER, 2)
    box = (110, y + 90, PW - 110, y + 520); sl.shadow(box, 36, 20, 110); sl.rrect(box, 36, WHITE, 18, outline=POWDER, w=1.2)
    for i, (ic, t) in enumerate([("check", "что именно можно исправить"), ("check", "какой объём операции нужен"),
                                 ("check", "какой результат реален"), ("check", "как пройдёт восстановление")]):
        cy = box[1] + 80 + i * 92; icon_circle(sl, (box[0] + 76, cy), ic, 26)
        sl.text((box[0] + 130, cy), t, mont(36, 300), WHITE, "lm")
    sl.lines((PW / 2, box[3] + 70), ["Напишите в Direct"], mont(40, 300), WHITE, anchor="ma")
    sl.rrect((PW / 2 - 230, box[3] + 140, PW / 2 + 230, box[3] + 224), 42, POWDER)
    sl.text((PW / 2, box[3] + 183), "КОНСУЛЬТАЦИЯ", bebas(56), NAVY, "mm")
    return sl.save("post_05_cta.jpg")


# =====================================================================================
# STORIES 1080x1920 — classic Montserrat Bold + Light
# =====================================================================================
SW, SH = 1080, 1920
SAFE_T, SAFE_B = 250, 1670   # keep text out of the IG header and reply bar


def deco_dashes(sl, col=WHITE, a=120):
    """thin dashed arcs crossing the frame (signature of the reference stories)"""
    sl.dashed(bez((-20, 150), (300, 40), (700, 60), (1100, 230)), col, 1.3, 7, 8, a)
    sl.dashed(bez((-20, 1290), (260, 1180), (720, 1240), (1100, 1150)), col, 1.3, 7, 8, a)


def story1():
    """cover: blurred photo, staggered headline in light + bold with dot-arrow connector"""
    sl = Slide(SW, SH, photo_bg(SW, SH, blur=16, tint=MAUVE, k=0.6, dark=0.35, cx=0.55, cy=0.45, zoom=1.2, gray=0.5))
    vshade(sl, 900, SH, 0.0, 0.55, MAUVE)
    deco_dashes(sl)
    sl.glow_text((60, 560), "ЧТО МЕНЯЕТСЯ", mont(92, 300), WHITE, "ls")
    sl.glow_text((SW - 70, 640), "ПОСЛЕ ОПЕРАЦИИ", mont(48, 300), WHITE, "rs")
    sl.dot((590, 676), 6); sl.poly([(590, 676), (590, 780)], WHITE, 1.6); sl.head([(590, 700), (590, 780)], WHITE, 1.6, 12)
    sl.glow_text((60, 940), "КРОМЕ", mont(176, 700), WHITE, "ls")
    sl.glow_text((SW - 70, 1050), "ВНЕШНОСТИ?", mont(84, 300), POWDER, "rs")
    sl.lines((70, 1330), ["На консультациях пациентки почти никогда", "не говорят только про грудь или живот."],
             mont(32, 300), WHITE, lh=1.3)
    sl.text((70, 1440), "Они говорят совсем другие вещи.", mont(32, 700), WHITE)
    sl.dot((480, 1580), 6); sl.poly([(480, 1580), (590, 1580)], WHITE, 1.6); sl.head([(480, 1580), (590, 1580)], WHITE, 1.6, 12)
    return sl.save("story_01_cover.jpg")


def story2():
    """patient phrases as chat bubbles + white card with a paperclip"""
    sl = Slide(SW, SH, photo_bg(SW, SH, blur=5, tint=NAVY, k=0.45, dark=0.2, cx=0.5, cy=0.7, zoom=1.0))
    vshade(sl, 0, SH, 0.15, 0.35, NAVY)
    deco_dashes(sl, WHITE, 90)
    f = mont(33, 400)
    bubble(sl, (90, 330, 560, 450), ["Я перестала", "фотографироваться"], f, "left")
    bubble(sl, (430, 510, 1000, 670), ["Я выбираю одежду не потому,", "что нравится, а потому,", "что скрывает"], f, "right")
    bubble(sl, (90, 730, 640, 850), ["Я не люблю смотреть", "на себя сбоку"], f, "left")
    bubble(sl, (420, 910, 1000, 1070), ["Муж говорит, что всё", "нормально, а я сама", "так не чувствую"], f, "right")
    card = (130, 1170, SW - 130, 1430); sl.shadow(card, 26, 20, 110); sl.rrect(card, 26, WHITE)
    paperclip(sl, (SW / 2, 1168), MAUVE, 54)
    sl.text((SW / 2, 1250), "И вот это самое важное.", mont(36, 700), NAVY, "mm")
    sl.lines((SW / 2, 1300), ["Потому что пластическая операция", "начинается с усталости."], mont(34, 300), INK, lh=1.3, anchor="ma")
    return sl.save("story_02_bubbles.jpg")


def story3():
    """one big word with three looped arrows to the captions"""
    sl = Slide(SW, SH, photo_bg(SW, SH, blur=22, tint=POWDER, k=0.45, dark=0.05, cx=0.5, cy=0.5, zoom=1.3, gray=0.6))
    vshade(sl, 0, SH, 0.25, 0.1, MAUVE)
    sl.glow_text((SW / 2, 720), "УСТАЛА", mont(150, 300), WHITE, "ms", blur=14, a=110)
    L = [((470, 760), (350, 1000), -1, ["втягивать живот", "на каждом фото"], (310, 1040)),
         ((690, 760), (820, 1000), 1, ["прятаться", "в оверсайзе"], (840, 1040)),
         ((580, 900), (575, 1170), 1, ["делать вид,", "что всё устраивает"], (575, 1210))]
    for a, b, side, txt, tp in L:
        p = loop_path(a, b, 30, side=side); sl.dot(a, 5); sl.poly(p, WHITE, 2); sl.head(p, WHITE, 2, 14)
        sl.lines(tp, txt, mont(38, 300), WHITE, lh=1.2, anchor="ma")
    sl.lines((SW / 2, 1460), ["Знакомо? Отметьте, что про вас"], mont(32, 300), WHITE, anchor="ma")
    sl.rrect((SW / 2 - 270, 1520, SW / 2 + 270, 1610), 45, WHITE, 235)  # placeholder for the poll / quiz sticker
    sl.text((SW / 2, 1565), "место для опроса", mont(28, 300), MAUVE, "mm")
    return sl.save("story_03_tired.jpg")


def story4():
    """CTA on deep navy: bold question, loop arrow, 'write to Direct'"""
    sl = Slide(SW, SH, photo_bg(SW, SH, blur=10, tint=DEEP, k=0.6, dark=0.55, cx=0.5, cy=0.15, zoom=1.4, gray=0.7))
    vshade(sl, 700, 1500, 0.0, 0.95, DEEP)
    deco_dashes(sl, POWDER, 70)
    sl.lines((SW / 2, 1020), ["Если вы давно думаете об операции,", "но пока боитесь или сомневаетесь"],
             mont(38, 700), WHITE, lh=1.2, anchor="ma")
    p = loop_path((540, 1150), (540, 1330), 22, at=(0.3, 0.7)); sl.dot((540, 1150), 5); sl.poly(p, WHITE, 2); sl.head(p, WHITE, 2, 13)
    sl.lines((SW / 2, 1360), ["напишите", "в директ"], mont(72, 300), POWDER, lh=1.0, anchor="ma")
    sl.lines((SW / 2, 1545), ["разберём именно вашу ситуацию"], mont(32, 300), WHITE, anchor="ma")
    logo(sl, 1740, POWDER, 0.8, 0.85)
    return sl.save("story_04_cta.jpg")


def sheet(imgs, cols, cell_w, name, gap=24, bg=(14, 16, 22)):
    ims = [i.resize((cell_w, int(i.height * cell_w / i.width)), Image.LANCZOS) for i in imgs]
    ch = ims[0].height; rows = math.ceil(len(ims) / cols)
    out = Image.new("RGB", (cols * cell_w + (cols + 1) * gap, rows * ch + (rows + 1) * gap), bg)
    for k, im in enumerate(ims): out.paste(im, (gap + (k % cols) * (cell_w + gap), gap + (k // cols) * (ch + gap)))
    out.save(os.path.join(OUT, name), quality=92)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    posts = [post1(), post2(), post3(), post4(), post5()]
    stories = [story1(), story2(), story3(), story4()]
    sheet(posts, 5, 540, "00_posts_overview.jpg")
    sheet(stories, 4, 540, "00_stories_overview.jpg")
    print("ok")
