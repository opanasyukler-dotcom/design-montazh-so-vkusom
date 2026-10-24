"""@mcg.va — «подарочки из Италии» (Chicco set + Italian clothes), minimal stories 1080x1920.
  story 1 (cover): the heart plate full screen, the whole set in a big circle, white type over a bottom spray
  story 2: the cup full screen, spoon and cup-from-above in big circles with pastel pills, the «больше, чем своим покупкам» line
  story 3: the cherry sweater full screen, the jacket in a circle, the punchline on a white card
Fonts (stories_kit): main = Inter Bold, second = thin Montserrat, detail = Montserrat Italic.
  python3 -I stories_gifts.py spoon plate sweater jacket set cup_side cup_top out_dir
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import W, H, INK, font
from stories_pacifiers import circle, block, fill
from stories_walk_look import shadowed

PILL = (240, 218, 216)          # blush from the silicone
COCOA = (130, 93, 77)

def pill(img, text, cx, cy):
    f = font("detail", 30); w, h = f.getlength(text) + 44, 54; x0, y0 = cx - w / 2, cy - h / 2
    shadowed(img, (x0, y0, x0 + w, y0 + h), h // 2, blur=14, off=5, alpha=0.16, fill=PILL)
    ImageDraw.Draw(img).text((cx, cy + 1), text, font=f, fill=INK, anchor="mm")

def shade(img, start=0.6, k=0.72):
    """soft dark spray at the very bottom so white type reads"""
    a = np.asarray(img, np.float32); y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    a *= 1 - np.clip((y - start) / (1 - start), 0, 1) ** 1.3 * k
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))

def story1(plate, set_, out):
    """cover: the heart plate full screen, the whole set in a big circle, white type over the bottom spray"""
    img = shade(fill(plate), 0.55)
    circle(img, set_, 790, 520, 500, (131, 0, 1217, 1086))           # square crop of the landscape set
    d = ImageDraw.Draw(img)
    y = block(d, "Ой, как же я теперь люблю подарочки для дочки.", "main", 52, 1330, 64, 880, col=(255, 255, 255))
    block(d, "Приехал её дядя из Италии и привёз столько красотыыыы!", "second", 38, y + 20, 52, 880, col=(255, 255, 255))
    img.save(out, quality=95, subsampling=0)

def story2(cup_side, spoon, cup_top, out):
    """the cup full screen, spoon and cup-from-above in big circles, dark type on the linen above"""
    img = fill(cup_side); d = ImageDraw.Draw(img)
    block(d, "Я, кажется, радуюсь этим вещам больше, чем своим покупкам.", "main", 46, 290, 58, 880)
    circle(img, spoon, 830, 640, 420, (50, 310, 950, 1210))
    circle(img, cup_top, 830, 1180, 420, (167, 314, 927, 1074))
    pill(img, "ложечка", 830, 640 + 250)
    pill(img, "стаканчик", 830, 1180 + 250)
    img.save(out, quality=95, subsampling=0)

def story3(sweater, jacket, out):
    img = fill(sweater)
    circle(img, jacket, 300, 500, 480, (51, 194, 1051, 1194))
    x0, x1, y0, y1 = 110, W - 110, 1370, 1560
    shadowed(img, (x0, y0, x1, y1), 36)
    d = ImageDraw.Draw(img)
    block(d, "Человеку 5 месяцев, а она уже носит итальянские шмотки", "main", 44, y0 + 66, 58, 760)
    img.save(out, quality=95, subsampling=0)

if __name__ == "__main__":
    spoon, plate, sweater, jacket, set_, cup_side, cup_top = (Image.open(p).convert("RGB") for p in sys.argv[1:8]); od = sys.argv[8]
    story1(plate, set_, os.path.join(od, "podarki_1.jpg"))
    story2(cup_side, spoon, cup_top, os.path.join(od, "podarki_2.jpg"))
    story3(sweater, jacket, os.path.join(od, "podarki_3.jpg"))
