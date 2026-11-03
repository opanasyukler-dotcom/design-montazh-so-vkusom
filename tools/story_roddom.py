"""Single editorial story: blurred full-bleed photo, long text on a frosted paper plate.

Usage: python3 tools/story_roddom.py
"""
import os
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from stories_editorial import (BODY, BROWN, H, PAPER, ROOT, UPLOADS, W, cover_crop, linen,
                               manrope, wrap)

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

# Story 2: angry statement, white type straight on a darkened blur
TEXT_2 = ("За четыре дня мне попадались просто УЖАСНЫЕ смены! Отношение максимально "
          "советское. Меня унижали, оскорбляли, а\u00a0когда я\u00a0просила помощи, "
          "вместо неё получала замечания.")
STATEMENT = ["Я НИКОГДА", "БОЛЬШЕ НЕ ПОШЛА", "БЫ РОЖАТЬ", "В РОДДОМ"]
HIGHLIGHT = "ИМЕНИ ЮДИНА!"

# Story 3: quiet ending, same dark blur, calmer type
LEAD_3 = "Для меня это был самый ужасный опыт общения с\u00a0медицинским персоналом."
TEXT_3 = ("У меня даже есть заключение специалиста о\u00a0психологической травме, "
          "полученной после этого опыта. Я\u00a0рассматривала возможность обратиться "
          "в\u00a0суд, но в\u00a0итоге решила, что не хочу больше тратить на\u00a0эту "
          "историю свои силы.")

# Bold lines use Inter Display Bold, the closest open match to Instagram's "Classic" font
CLASSIC = os.path.join(ROOT, "fonts", "commercial-safe", "Inter", "InterDisplay-Bold.otf")

BLUR = 30
PLATE_X, PAD, LINE = 60, 55, 56


def classic(size):
    return ImageFont.truetype(CLASSIC, size)


def blurred():
    return cover_crop(PHOTO, W, H, (0.5, 0.5)).filter(ImageFilter.GaussianBlur(BLUR))


def story_1():
    img = blurred()
    d = ImageDraw.Draw(img)

    tf, af = classic(80), classic(66)
    bf, pf = manrope(38, "Light"), classic(42)
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

    return img


def story_2():
    img = Image.blend(blurred(), Image.new("RGB", (W, H), (38, 24, 20)), 0.62)
    d = ImageDraw.Draw(img)
    x, width = 100, W - 200
    white = (250, 247, 242)

    bf = manrope(42, "Light")
    y = 330
    for line in wrap(d, TEXT_2, bf, width):
        d.text((x, y), line, font=bf, fill=white)
        y += 62

    y += 70
    d.line((x, y, x + 120, y), fill=white, width=2)
    y += 70

    sf = classic(84)
    for line in STATEMENT:
        d.text((x, y), line, font=sf, fill=white)
        y += 96

    # Marker-style paper highlight on the hospital name
    pad_x, pad_y = 22, 12
    l, t, r, b = d.textbbox((x + pad_x, y + 10), HIGHLIGHT, font=sf)
    d.rounded_rectangle((l - pad_x, t - pad_y, r + pad_x, b + pad_y), radius=10, fill=PAPER)
    d.text((x + pad_x, y + 10), HIGHLIGHT, font=sf, fill=BROWN)
    return img


def story_3():
    img = Image.blend(blurred(), Image.new("RGB", (W, H), (38, 24, 20)), 0.62)
    d = ImageDraw.Draw(img)
    x, width = 100, W - 200
    white = (250, 247, 242)
    soft = (226, 216, 206)

    lf, bf = classic(58), manrope(42, "Light")
    lead = wrap(d, LEAD_3, lf, width)
    body = wrap(d, TEXT_3, bf, width)
    block = len(lead) * 74 + 140 + len(body) * 62
    y = (H - block) // 2 - 60

    for line in lead:
        d.text((x, y), line, font=lf, fill=white)
        y += 74
    y += 60
    d.line((x, y, x + 120, y), fill=white, width=2)
    y += 80
    for line in body:
        d.text((x, y), line, font=bf, fill=soft)
        y += 62
    return img


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for k, make in enumerate((story_1, story_2, story_3), 1):
        path = os.path.join(OUT, f"story_{k:02d}.jpg")
        make().save(path, quality=95)
        print(path)
