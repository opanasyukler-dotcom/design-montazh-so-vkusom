"""Single editorial story: blurred full-bleed photo, long text on a frosted paper plate.

Usage: python3 tools/story_roddom.py
"""
import os
from PIL import Image, ImageDraw, ImageFilter

from stories_editorial import (BODY, BROWN, H, ROOT, UPLOADS, W, cover_crop, linen,
                               manrope, montserrat, wrap)

OUT = os.path.join(ROOT, "stories_roddom")
PHOTO = f"{UPLOADS}/1c049e08-image.jpg"

TITLE = "РОДДОМ"
PARAGRAPHS = [
    "Экскурсию по роддому, к сожалению, я вам не проведу, так как рожала "
    "экстренно и практически не успела побыть в родовой палате.",
    "Но по тому, что я успела увидеть, палаты показались мне немного устаревшими, "
    "хотя в целом вполне нормальными.",
]
ACCENT = "НО!"
PUNCHLINE = "Всё впечатление испортило послеродовое отделение."

BLUR = 30
PLATE_X, PAD, LINE = 60, 55, 56


def main():
    img = cover_crop(PHOTO, W, H, (0.5, 0.5)).filter(ImageFilter.GaussianBlur(BLUR))
    d = ImageDraw.Draw(img)

    tf, af = montserrat(78, "Bold"), montserrat(64, "Bold")
    bf, pf = manrope(38, "Light"), manrope(40, "Medium")
    tx = PLATE_X + PAD
    width = W - 2 * tx
    paras = [wrap(d, p, bf, width) for p in PARAGRAPHS]
    punch = wrap(d, PUNCHLINE, pf, width)

    # Measure the text block so the plate hugs it and sits centred
    block = 92 + 40 + sum(len(p) * LINE for p in paras) + 30 * (len(paras) - 1) \
        + 60 + 80 + len(punch) * 58
    top = (H - block) // 2 - 60  # a little above centre, clear of the reply bar
    box = (PLATE_X, top - PAD, W - PLATE_X, top + block + PAD)

    frost = img.crop(box).filter(ImageFilter.GaussianBlur(12))
    frost = Image.blend(frost, linen().crop(box), 0.86)
    m = Image.new("L", (frost.width * 4, frost.height * 4), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, m.width - 1, m.height - 1), radius=32 * 4, fill=255)
    img.paste(frost, box[:2], m.resize(frost.size, Image.LANCZOS))

    y = top
    d.text((tx, y), TITLE, font=tf, fill=BROWN)
    y += 92 + 40
    for k, lines in enumerate(paras):
        for line in lines:
            d.text((tx, y), line, font=bf, fill=BODY)
            y += LINE
        if k < len(paras) - 1:
            y += 30

    # Accent: thin rule, bold "НО!", then the punchline in a heavier weight
    y += 30
    d.line((tx, y, tx + 120, y), fill=BROWN, width=2)
    y += 30
    d.text((tx, y), ACCENT, font=af, fill=BROWN)
    y += 80
    for line in punch:
        d.text((tx, y), line, font=pf, fill=BROWN)
        y += 58

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "story_01.jpg")
    img.save(path, quality=95)
    print(path)


if __name__ == "__main__":
    main()
