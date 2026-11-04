"""Instagram stories (1080x1920): the poll about the biggest fear before the operation -> anesthesia leads ->
the patients' emotions after the operation -> the highlight «Наркоз».

Old story design: Inter Bold caps headlines + Montserrat, white text with a soft shadow, no emojis, no dashes.
1. 01_opros.jpg: the operating photo, the question on top, the middle left free for the Instagram poll sticker.
2. 02_rezultat_oprosa.jpg: free room for the screenshot of the poll results, the comment below, «>>>».
5. 05_aktualnoe_narkoz.jpg: the surgeon's portrait, the call to the highlight in navy on the white jacket.
(3 and 4 are the patients' videos after the operation.)

Sources (SRC_DIR): operaciya.png (hands with the needle holder), portret.jpg (the surgeon).
usage: python3 stories_fear_anesthesia.py SRC_DIR OUT_DIR [1|2|5 ...]
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

W, H = 1080, 1920
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
WHITE, NAVY, GRAPHITE = (255, 255, 255), (2, 31, 71), (20, 23, 31)
SRC, OUT = sys.argv[1], sys.argv[2]
ONLY = {int(a) for a in sys.argv[3:]}
os.makedirs(OUT, exist_ok=True)
MARGIN = 84


def mont(size, wght=400):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([wght]); return f


def cover(name):
    im = ImageOps.exif_transpose(Image.open(os.path.join(SRC, name))).convert("RGB")
    s = max(W / im.width, H / im.height); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im.crop(((im.width - W) // 2, (im.height - H) // 2, (im.width - W) // 2 + W, (im.height - H) // 2 + H))


def lines_draw(lay, lines, x, y, fill=WHITE, anchor="la"):
    d = ImageDraw.Draw(lay)
    for t, f, step in lines:
        d.text((x, y), t, font=f, fill=fill, anchor=anchor); y += step
    return y


def shadowed(bg, lay, a=170, blur=9):
    """white text with the usual soft graphite shadow"""
    al = lay.getchannel("A")
    sh = Image.new("RGBA", (W, H), GRAPHITE + (0,)); sh.putalpha(al.point(lambda v: v * a // 255).filter(ImageFilter.GaussianBlur(blur)))
    sh2 = Image.new("RGBA", (W, H), GRAPHITE + (0,)); sh2.putalpha(al.point(lambda v: v * 140 // 255).filter(ImageFilter.GaussianBlur(3)))
    return Image.alpha_composite(Image.alpha_composite(Image.alpha_composite(bg.convert("RGBA"), sh), sh2), lay)


def top_fade(im, h, k):
    a = np.asarray(im, np.float32); ys = np.arange(H, dtype=np.float32)[:, None, None]
    a *= 1 - k * np.clip(1 - ys / h, 0, 1) ** 1.6
    return Image.fromarray(a.astype(np.uint8))


def save(im, name):
    out = os.path.join(OUT, name); im.convert("RGB").save(out, quality=94); print("wrote", out)


def s1():
    bg = top_fade(cover("operaciya.png"), 760, 0.55)       # the dark operating room on top only gets a little deeper
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hb = ImageFont.truetype(BOLD, 64)
    y = lines_draw(lay, [("ОПРОС", mont(30, 600), 70)], MARGIN, 150)
    ImageDraw.Draw(lay).line((MARGIN, y - 22, MARGIN + 70, y - 22), fill=WHITE, width=3)
    y = lines_draw(lay, [("КАКОЙ СТРАХ", hb, 76), ("ОСТАНАВЛИВАЕТ ВАС", hb, 76), ("ОТ ОПЕРАЦИИ?", hb, 104)], MARGIN, y + 6)
    lines_draw(lay, [("Отметьте ваш самый большой страх.", mont(36, 500), 48),
                     ("Опрос анонимный.", mont(36, 400), 48)], MARGIN, y)
    save(shadowed(bg, lay), "01_opros.jpg")                 # y 700-1500 left free for the poll sticker


def s2():
    """the screenshot of the results goes into the free upper part; the comment low under it"""
    bg = cover("operaciya.png")
    a = np.asarray(bg, np.float32) * 0.35                   # the same room, quietly in the dark, so the screenshot stands out
    bg = Image.fromarray(a.astype(np.uint8))
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hb = ImageFont.truetype(BOLD, 58); b = mont(38, 400)
    lines = [("ВИЖУ, ЧТО СТРАХ", hb, 70), ("АНЕСТЕЗИИ ЛИДИРУЕТ", hb, 96),
             ("Очень хочу показать вам", b, 50), ("эмоции моих пациенток", b, 50), ("после операции", b, 50)]
    top = H - 150 - sum(s for _, _, s in lines)
    lines_draw(lay, lines, MARGIN, top)
    ImageDraw.Draw(lay).text((W - MARGIN, H - 120), ">>>", font=mont(44, 500), fill=WHITE, anchor="rs")
    save(shadowed(bg, lay), "02_rezultat_oprosa.jpg")       # y 120 to ~1270 free for the results screenshot


def s5():
    bg = cover("portret.jpg")
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    hb = ImageFont.truetype(BOLD, 64); b = mont(37, 500)
    lines = [("ВСЁ ПРО НАРКОЗ", hb, 84), ("Как проходит анестезия,", b, 48),
             ("подготовка и пробуждение,", b, 48), ("смотрите в актуальном", b, 48), ("«Наркоз»", mont(37, 700), 0)]
    top = H - 110 - 84 - 48 * 3 - 34
    for (t, f, st), yy in zip(lines, np.cumsum([0] + [s for _, _, s in lines[:-1]])):
        d.text((W - MARGIN, top + yy), t, font=f, fill=NAVY + (255,), anchor="ra")
    # a soft white halo instead of a dark haze, so the lapel lines do not cut the letters
    al = lay.getchannel("A"); halo = Image.new("RGBA", (W, H), WHITE + (0,))
    halo.putalpha(al.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(10)).point(lambda v: min(255, v * 2) * 150 // 255))
    save(Image.alpha_composite(Image.alpha_composite(bg.convert("RGBA"), halo), lay), "05_aktualnoe_narkoz.jpg")


for k, f in [(1, s1), (2, s2), (5, s5)]:
    if not ONLY or k in ONLY: f()
