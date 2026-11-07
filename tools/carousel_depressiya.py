"""@mcg.va — carousel «Депрессия покидает моё тело, когда…» (1080x1350), her style: full-bleed photos, thin Montserrat over the
calm part of each photo (ink on light, white with a soft shadow on dark; a light black haze only where it would merge),
first sentence of each thought in thin italic, swipe arrows. Her text word for word.
  python3 -I carousel_depressiya.py PHOTOS_DIR OUT_DIR
"""
import os, sys
from PIL import Image, ImageDraw, ImageFilter, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carousel_top4 import W, H, M, CREAM, INK, BROWN, MAUVE, BLUSH, font, Text, crop, footer, haze, arrow, paper_slide

WHITE = (255, 255, 255)

def load(src, fy):
    return crop(ImageOps.exif_transpose(Image.open(src)).convert("RGB"), 0.5, fy).convert("RGBA")

def shadow_layer(im, draw_fn):
    """soft dark glyph shadow for white text: draw the text in black on a layer, blur, composite"""
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0)); draw_fn(lay, (20, 14, 10))
    a = lay.split()[3].filter(ImageFilter.GaussianBlur(6)).point(lambda v: int(v * 0.55)); sh = Image.new("RGBA", im.size, (20, 14, 10, 0))
    sh.putalpha(a); im.alpha_composite(sh)

def photo_slide(src, paras, out, n, total, x=M, y=None, width=W - 2 * M, size=30, col=INK, fy=0.5, hz=0.0, bottom=96):
    im = load(src, fy); t = Text(paras, width, size); ty = y if y is not None else H - bottom - t.h
    if hz: haze(im, (x - 90, ty - 70, x + width + 90, ty + t.h + 60), (12, 8, 6), hz)
    if col == WHITE: shadow_layer(im, lambda lay, c: t.draw(lay, x, ty + 2, c, c))
    t.draw(im, x, ty, col, col); footer(im, n, total, WHITE if col == WHITE else MAUVE)
    im.convert("RGB").save(out, quality=95); print(out, "text", ty, ty + t.h)

def cover(src, out):
    im = load(src, 0.5); d = ImageDraw.Draw(im); f = font(54, 250); y = 1080
    for l in ["ДЕПРЕССИЯ ПОКИДАЕТ", "МОЁ ТЕЛО, КОГДА…"]: d.text((W / 2, y), l, font=f, fill=INK, anchor="ms"); y += 70
    d.text((W - M - 104, H - 62), "листай", font=font(24, 300, True), fill=INK, anchor="rs"); arrow(im, W - M, H - 70, INK, 84)
    im.convert("RGB").save(out, quality=95); print(out)

def main(src, out):
    os.makedirs(out, exist_ok=True); p = lambda f: os.path.join(src, f); o = lambda n: os.path.join(out, f"slide_{n:02d}.jpg"); T = 9
    cover(p("bed.jpg"), o(1))
    photo_slide(p("mug.jpg"), [("Когда я пью горячий ароматный кофе.", "i"),
                               ("Желательно в тишине и пока он ещё горячий, а не разогретый в третий раз.", "b")],
                o(2), 2, T, x=250, y=1105, width=600, size=28, fy=1.0)
    photo_slide(p("lift.jpg"), [("Когда моя дочка улыбается мне и смотрит прямо в глаза.", "i"),
                                ("И я в этот момент забываю, как сильно устала.", "b")],
                o(3), 3, T, x=700, y=100, width=300, size=28, fy=0.36)
    paper_slide([("Когда муж возвращается с работы, мы укладываем малышку и наконец-то можем просто вместе посмотреть фильм.", "i"),
                 ("Когда муж заботится обо мне, обнимает, помогает с дочкой и мы проводим время всей семьёй.", "b")], o(4), 4, T)
    photo_slide(p("bottle.jpg"), [("Когда я иду на маникюр или массаж", "i"), ("и целый час могу просто ничего не делать.", "b")],
                o(5), 5, T, size=30, col=WHITE, fy=0.5, hz=0.45)
    photo_slide(p("square.jpg"), [("Когда я гуляю одна по городу.", "i"),
                                  ("Без коляски, без спешки, могу зайти за кофе, в магазин или просто идти куда глаза глядят.", "b")],
                o(6), 6, T, y=120, size=30, col=WHITE, fy=0.0, hz=0.25)
    photo_slide(p("baby.png"), [("Когда у дочки хорошее настроение, она спокойно играет, улыбается и не капризничает.", "b"),
                                ("И особенно когда ребёнок хорошо спит днём И НОЧЬЮ.", "i"),
                                ("Правда, такое у нас случается настолько редко, что хочется объявить этот день государственным праздником.", "b")],
                o(7), 7, T, size=28, col=WHITE, fy=0.0, hz=0.45)
    photo_slide(p("lift2.jpg"), [("Раньше я даже не замечала, сколько радости могут приносить такие обычные вещи.", "i"),
                                 ("А сейчас иногда достаточно одного спокойного часа, горячего кофе или улыбки дочки, чтобы почувствовать себя счастливой.", "b")],
                o(8), 8, T, size=29, col=WHITE, fy=0.3, hz=0.35)
    paper_slide([("А в какие моменты депрессия покидает ваше тело?", "i")], o(9), 9, T, kicker="А У ВАС?", size=40)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
