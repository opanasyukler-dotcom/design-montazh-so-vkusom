"""@mcg.va — «подарочки из Италии» (Chicco set + Italian clothes), minimal stories 1080x1920.
  story 1: soft blurred linen, the full Chicco set as a rounded card, the cup in a circle on its corner, the story opening
  story 2: blurred linen, the heart plate card + spoon and cup in circles, pastel pills, the «больше, чем своим покупкам» line
  story 3: the cherry sweater full screen, the jacket in a circle, the punchline on a white card
Fonts (stories_kit): main = Inter Bold, second = thin Montserrat, detail = Montserrat Italic.
  python3 -I stories_gifts.py spoon plate sweater jacket set cup_side cup_top out_dir
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import W, H, INK, font
from stories_pacifiers import rounded_card, circle, block, fill
from stories_walk_look import shadowed

PILL = (240, 218, 216)          # blush from the silicone
COCOA = (130, 93, 77)

def linen(ph, blur=30, lift=0.15):
    """the photo's own white linen, blurred and lifted a touch → a soft page"""
    a = np.asarray(fill(ph).filter(ImageFilter.GaussianBlur(blur)), np.float32)
    return Image.fromarray((a * (1 - lift) + np.array((250, 246, 243), np.float32) * lift).clip(0, 255).astype(np.uint8))

def pill(img, text, cx, cy):
    f = font("detail", 30); w, h = f.getlength(text) + 44, 54; x0, y0 = cx - w / 2, cy - h / 2
    shadowed(img, (x0, y0, x0 + w, y0 + h), h // 2, blur=14, off=5, alpha=0.16, fill=PILL)
    ImageDraw.Draw(img).text((cx, cy + 1), text, font=f, fill=INK, anchor="mm")

def story1(set_, cup_side, out):
    img = linen(set_, 30, 0.3); d = ImageDraw.Draw(img)
    y = block(d, "Ой, как же я теперь люблю подарочки для дочки.", "main", 52, 300, 64, 860)
    rounded_card(img, set_, (70, y + 40, W - 70, y + 40 + 760), 40)
    circle(img, cup_side, 850, y + 40 + 740, 300, (56, 280, 856, 1080))
    d = ImageDraw.Draw(img)
    block(d, "Приехал её дядя из Италии и привёз столько красотыыыы!", "second", 38, y + 40 + 760 + 120, 52, 640, x=90)
    img.save(out, quality=95, subsampling=0)

def story2(plate, spoon, cup_top, out):
    img = linen(plate, 40, 0.6); d = ImageDraw.Draw(img)
    y = block(d, "Я, кажется, радуюсь этим вещам больше, чем своим покупкам.", "main", 46, 300, 58, 860)
    top = y + 40
    rounded_card(img, plate, (70, top, 650, top + 780), 40)
    circle(img, spoon, 850, top + 165, 320, (50, 310, 950, 1210))
    circle(img, cup_top, 850, top + 610, 320, (167, 314, 927, 1074))
    pill(img, "тарелочка-сердце", 360, top + 780 + 70)
    pill(img, "ложечка", 850, top + 165 + 200)
    pill(img, "стаканчик", 850, top + 610 + 200)
    img.save(out, quality=95, subsampling=0)

def story3(sweater, jacket, out):
    img = fill(sweater)
    circle(img, jacket, 250, 400, 340, (51, 194, 1051, 1194))
    x0, x1, y0, y1 = 110, W - 110, 1370, 1560
    shadowed(img, (x0, y0, x1, y1), 36)
    d = ImageDraw.Draw(img)
    block(d, "Человеку 5 месяцев, а она уже носит итальянские шмотки", "main", 44, y0 + 66, 58, 760)
    img.save(out, quality=95, subsampling=0)

if __name__ == "__main__":
    spoon, plate, sweater, jacket, set_, cup_side, cup_top = (Image.open(p).convert("RGB") for p in sys.argv[1:8]); od = sys.argv[8]
    story1(set_, cup_side, os.path.join(od, "podarki_1.jpg"))
    story2(plate, spoon, cup_top, os.path.join(od, "podarki_2.jpg"))
    story3(sweater, jacket, os.path.join(od, "podarki_3.jpg"))
