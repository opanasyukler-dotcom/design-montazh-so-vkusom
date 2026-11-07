"""@mcg.va — carousel «4 вещи, которые я бы 100% купила снова» (1080x1350).
Cover: her photo, the title in thin CAPS on the mug. Items: the product photo full-bleed, the text over it on a soft cream haze
(only where the text sits), number + thin CAPS heading, thin lowercase body, thin italic lead lines. Intro and outro: cream slides.
Her text is kept word for word (emoji included).
  python3 -I carousel_top4.py PHOTOS_DIR OUT_DIR
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
HERE = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(HERE, "..", "fonts", "stories")
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
W, H, M = 1080, 1350, 80
CREAM, INK, BROWN, MAUVE, BLUSH = (243, 236, 228), (43, 31, 26), (107, 74, 56), (139, 107, 92), (232, 201, 192)

def font(size, wght, it=False):
    f = ImageFont.truetype(os.path.join(FD, "Montserrat-Italic-VariableFont_wght.ttf" if it else "Montserrat-VariableFont_wght.ttf"), size)
    f.set_variation_by_axes([wght]); return f

_emo = {}
def emoji(ch, h):
    if (ch, h) not in _emo:
        f = ImageFont.truetype(EMOJI, 109); im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((80, 80), ch, font=f, embedded_color=True, anchor="mm"); im = im.crop(im.getbbox())
        _emo[(ch, h)] = im.resize((round(im.width * h / im.height), h), Image.LANCZOS)
    return _emo[(ch, h)]

isemo = lambda w: any(ord(c) > 0x2600 for c in w)
def wlen(w, f): return f.size * 1.15 if isemo(w) else f.getlength(w)
def llen(ws, f): return sum(wlen(w, f) for w in ws) + f.getlength(" ") * (len(ws) - 1)

def wrap(text, f, maxw):
    lines, cur = [], []
    for w in text.split():
        if cur and llen(cur + [w], f) > maxw: lines.append(cur); cur = [w]
        else: cur.append(w)
    return lines + ([cur] if cur else [])

def draw_words(d, canvas, x, y, ws, f, fill):
    sp = f.getlength(" ")
    for w in ws:
        if isemo(w):
            e = emoji(w, int(f.size * 0.95)); canvas.alpha_composite(e, (int(x + 2), int(y - f.size * 0.82))); x += wlen(w, f) + sp
        else: d.text((x, y), w, font=f, fill=fill, anchor="ls"); x += f.getlength(w) + sp

class Text:
    """number, heading (CAPS), paragraphs [(text, 'b'|'i')] laid out in one column"""
    def __init__(self, paras, width, size=30, num=None, head=None, kicker=None):
        self.items = []; self.width = width
        if kicker: self.items.append(("kick", kicker, font(21, 350), 46))
        if num: self.items.append(("num", num, font(64, 200), 78))
        if head:
            fh = font(38, 250)
            for l in wrap(head, fh, width): self.items.append(("head", " ".join(l), fh, 52))
            self.items.append(("gap", None, None, 26))
        for k, (t, st) in enumerate(paras):
            f = font(size, 300, st == "i")
            if k: self.items.append(("gap", None, None, int(size * 0.6)))
            for l in wrap(t, f, width): self.items.append(("body", l, f, int(size * 1.48)))
        self.h = sum(i[3] for i in self.items)

    def draw(self, canvas, x, y, col, accent):
        d = ImageDraw.Draw(canvas); yy = y
        for kind, v, f, step in self.items:
            if kind == "gap": yy += step; continue
            base = yy + step * 0.78
            if kind == "kick": tr = 5; xx = x; [d.text((xx := xx + (f.getlength(v[i - 1]) + tr if i else 0), base), c, font=f, fill=accent, anchor="ls") for i, c in enumerate(v)]
            elif kind == "num": d.text((x, base), v, font=f, fill=accent, anchor="ls")
            elif kind == "head": d.text((x, base), v, font=f, fill=col, anchor="ls")
            else: draw_words(d, canvas, x, base, v, f, col)
            yy += step

def crop(im, fx=0.5, fy=0.5):
    s = max(W / im.width, H / im.height); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    x = int((im.width - W) * fx); y = int((im.height - H) * fy); return im.crop((x, y, x + W, y + H))

def footer(canvas, n, total, col):
    d = ImageDraw.Draw(canvas); f = font(18, 350)
    d.text((M, H - 44), "@mcg.va", font=f, fill=col, anchor="ls"); d.text((W - M, H - 44), f"{n:02d} / {total:02d}", font=f, fill=col, anchor="rs")

def haze(canvas, box, colour, strength):
    """soft haze only under the text: a feathered rounded patch"""
    m = Image.new("L", (W, H), 0); ImageDraw.Draw(m).rounded_rectangle(box, 80, fill=255)
    h = Image.new("RGBA", (W, H), colour + (0,)); h.putalpha(m.filter(ImageFilter.GaussianBlur(55)).point(lambda v: v * strength))
    canvas.alpha_composite(h)

def lifted(src, up):
    """the photo at full width moved up by `up` px; the floor below it continues as its own mirror, softly blurred"""
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB"); im = im.resize((W, round(im.height * W / im.width)), Image.LANCZOS)
    bg = Image.new("RGB", (W, H)); bg.paste(im, (0, -up)); end = im.height - up
    if end < H:
        m = ImageOps.flip(im.crop((0, im.height - (H - end) - 60, W, im.height))).filter(ImageFilter.GaussianBlur(28))
        mask = Image.fromarray((np.clip(np.arange(m.height) / 60, 0, 1)[:, None].repeat(W, 1) * 255).astype(np.uint8))
        bg.paste(m, (0, end - 60), mask)
    return bg

def item_slide(src, num, head, paras, out, n, total, where="bottom", fy=0.5, size=30, strength=0, y=None, up=None, x=M, width=W - 2 * M):
    im = (lifted(src, up) if up else crop(ImageOps.exif_transpose(Image.open(src)).convert("RGB"), 0.5, fy)).convert("RGBA")
    t = Text(paras, width, size, num=num, head=head)
    ty = y if y is not None else (M if where == "top" else H - 96 - t.h)
    if strength: haze(im, (-120, ty - 70, W + 120, ty + t.h + 60), (12, 8, 6), strength)
    t.draw(im, x, ty, INK, BROWN); footer(im, n, total, MAUVE)
    im.convert("RGB").save(out, quality=95); print(out, "text", ty, ty + t.h)

def paper_slide(paras, out, n, total, kicker=None, head=None, extra=None, size=32):
    im = Image.new("RGBA", (W, H), CREAM + (255,))
    t = Text(paras, W - 2 * M, size, head=head, kicker=kicker); eh = extra[1] if extra else 0
    ty = (H - t.h - eh) // 2 - 10; t.draw(im, M, ty, INK, MAUVE)
    if extra: extra[0](im, ty + t.h)
    footer(im, n, total, MAUVE); im.convert("RGB").save(out, quality=95); print(out, "text", ty, ty + t.h + eh)

def cover_dark(src, out):
    """her on the brown backdrop: the title in thin cream CAPS on the empty brown above her head"""
    im = crop(ImageOps.exif_transpose(Image.open(src)).convert("RGB"), 0.5, 0.0).convert("RGBA"); d = ImageDraw.Draw(im)
    f = font(46, 250); y = 140
    for l in ["4 ВЕЩИ, КОТОРЫЕ", "Я БЫ 100% КУПИЛА СНОВА"]: d.text((W / 2, y), l, font=f, fill=CREAM, anchor="ms"); y += 60
    d.text((W / 2, y + 2), "если бы стала мамой во второй раз!", font=font(30, 300, True), fill=BLUSH, anchor="ms")
    im.convert("RGB").save(out, quality=95); print(out)

def cover(src, out):
    im = crop(ImageOps.exif_transpose(Image.open(src)).convert("RGB"), 0.5, 1.0).convert("RGBA"); d = ImageDraw.Draw(im)
    lines = ["4 ВЕЩИ, КОТОРЫЕ", "Я БЫ 100% КУПИЛА", "СНОВА"]; f = font(46, 250); cx = 572
    y = 1096
    for l in lines: d.text((cx, y), l, font=f, fill=INK, anchor="ms"); y += 58
    d.text((cx, y + 4), "если бы стала мамой во второй раз!", font=font(30, 300, True), fill=BROWN, anchor="ms")
    im.convert("RGB").save(out, quality=95); print(out)

ITEMS = [
    ("mubo.png", "01", "СТЕРИЛИЗАТОР MUBO", [
        ("Обожаю его за вместительность!", "i"),
        ("Я вообще не понимаю стерилизаторы, в которые помещаются всего 1–2 бутылочки. Мне хочется за один раз загрузить всё, что накопилось, а не стоять и стерилизовать по очереди.", "b"),],
     dict(where="bottom", fy=1.0, x=300, width=700, size=28)),
    ("warmer.png", "02", "ПОДОГРЕВАТЕЛЬ БУТЫЛОЧЕК", [
        ("Вот здесь я бы точно выбрала другую модель.", "i"),
        ("У меня сейчас достаточно бюджетный вариант, и, как оказалось, он больше подходит для поддержания температуры, чем для полноценного подогрева воды.", "b"),
        ("Поэтому во второй раз я бы не экономила и сразу взяла подогреватель, который действительно быстро и хорошо справляется со своей задачей.", "b")],
     dict(where="top", fy=1.0, size=28)),
    ("noise.jpg", "03", "БЕЛЫЙ ШУМ", [
        ("Вот это просто МАСТХЕВ с первых дней!", "i"),
        ("Я начала включать белый шум ещё в роддоме и пользуюсь им до сих пор.", "b")],
     dict(where="bottom", up=380, size=28, x=270, width=730)),
    ("machine_clean.png", "04", "МАШИНКА ДЛЯ ПРИГОТОВЛЕНИЯ СМЕСИ", [
        ("Мы пользуемся ей буквально с первых дней, и это просто лучшее, что можно было придумать для родителей на искусственном вскармливании.", "b")],
     dict(where="bottom", fy=0.0, size=28)),  # the bottle logo retouched out (cv2 inpaint) beforehand
]

def contents(im, y):
    d = ImageDraw.Draw(im); y += 70; d.line((M, y, W - M, y), fill=BLUSH, width=1)
    for k, (_, num, head, _p, _o) in enumerate(ITEMS):
        yy = y + 30 + k * 72; d.text((M, yy + 36), num, font=font(30, 200), fill=MAUVE, anchor="ls")
        d.text((M + 80, yy + 36), head, font=font(26, 300), fill=INK, anchor="ls")
        d.line((M, yy + 64, W - M, yy + 64), fill=BLUSH, width=1)

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); total = len(ITEMS) + 2
    cover_dark(os.path.join(src, "cover_belly.jpg"), os.path.join(out, "slide_01.jpg"))
    for k, (p, num, head, paras, o) in enumerate(ITEMS):
        item_slide(os.path.join(src, p), num, head, paras, os.path.join(out, f"slide_{k + 2:02d}.jpg"), k + 2, total, **o)
    paper_slide([("Конечно, у всех разные потребности, но лично мне эти четыре вещи очень облегчили первые месяцы материнства.", "b"),
                 ("А какая детская покупка у вас оказалась самой полезной? Та самая, которую вы бы купили ещё сто раз? Очень хочу почитать ваши находки!", "i")],
                os.path.join(out, f"slide_{total:02d}.jpg"), total, total, kicker="А У ВАС?")
