"""Editorial-style Instagram stories: linen paper, brown caps title, arch photo.

Style reference: beige "обо мне" story series (linen background, running header
with counter, bold grotesque caps headline, light body text, arch motif).

Usage: python3 tools/stories_editorial.py
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS = "/root/.claude/uploads/407108f0-3ce4-585b-bf12-4b6e65a8be4f"
OUT = os.path.join(ROOT, "stories_baby")

W, H = 1080, 1920
PAPER = (241, 237, 231)
BROWN = (91, 45, 28)
BODY = (104, 84, 74)
HEADER = (150, 132, 122)

MONTSERRAT = os.path.join(ROOT, "fonts", "Montserrat-VariableFont_wght.ttf")
MANROPE = os.path.join(ROOT, "fonts", "commercial-safe", "Manrope")

SLIDES = [
    {
        "photo": f"{UPLOADS}/f696e9a3-image.png",
        "title": "СМЕСЬ",
        "text": "Каждый раз рада, что у нас не самая дорогая на свете смесь.",
        "focus": (0.45, 0.55),
    },
    {
        "photo": f"{UPLOADS}/2b5ba369-image.jpg",
        "title": "НОЧНЫЕ\nПОДГУЗНИКИ",
        "text": "Решила попробовать такие, так как у\u00a0нас появилась проблема: "
                "ночью всё стало протекать.",
        "focus": (0.45, 0.55),
    },
    {
        "photos": [f"{UPLOADS}/9361c464-image.png", f"{UPLOADS}/2c24d1e3-image.png"],
        "title": "БОДИКИ",
        "text": "Я не удержалась и взяла девчачьи бодики, а\u00a0то бабушка ругается, "
                "что мы как пацаны ходим.",
        "focus": (0.5, 0.45),
        "layout": "collage",
    },
]
HEADER_LABEL = "наш уход"

# Photo frame: arch | rounded | oval | passe (override with: python3 ... <frame>)
FRAME = "rounded"
BOTTOM = 1620  # lower edge of floating frames, clear of the Instagram reply bar


def montserrat(size, weight="Bold"):
    f = ImageFont.truetype(MONTSERRAT, size)
    f.set_variation_by_name(weight)
    return f


def manrope(size, style="Light"):
    return ImageFont.truetype(os.path.join(MANROPE, f"Manrope-{style}.ttf"), size)


def linen():
    """Warm off-white paper with a faint woven texture."""
    rng = np.random.default_rng(7)
    base = np.ones((H, W, 3), np.float32) * np.array(PAPER, np.float32)
    rows = rng.normal(0, 2.2, (H, 1, 1))
    cols = rng.normal(0, 2.2, (1, W, 1))
    grain = rng.normal(0, 3.0, (H, W, 1))
    img = np.clip(base + rows + cols + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(img).filter(ImageFilter.GaussianBlur(0.6))


def wrap(draw, text, font, width):
    lines, line = [], ""
    for word in text.split(" "):  # keep non-breaking spaces
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def cover_crop(path, box_w, box_h, focus):
    src = Image.open(path).convert("RGB")
    scale = max(box_w / src.width, box_h / src.height)
    src = src.resize((round(src.width * scale), round(src.height * scale)), Image.LANCZOS)
    left = int((src.width - box_w) * focus[0])
    top = int((src.height - box_h) * focus[1])
    return src.crop((left, top, left + box_w, top + box_h))


def shape_mask(w, h, shape, ss=4):
    """Supersampled mask: arch (round top), rounded rect, oval or plain rect."""
    m = Image.new("L", (w * ss, h * ss), 0)
    md = ImageDraw.Draw(m)
    if shape == "arch":
        r = w * ss // 2
        md.ellipse((0, 0, w * ss, 2 * r), fill=255)
        md.rectangle((0, r, w * ss, h * ss), fill=255)
    elif shape == "rounded":
        md.rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=36 * ss, fill=255)
    elif shape == "oval":
        md.ellipse((0, 0, w * ss - 1, h * ss - 1), fill=255)
    else:
        md.rectangle((0, 0, w * ss, h * ss), fill=255)
    return m.resize((w, h), Image.LANCZOS)


def outline(img, box, shape, offset=18, width=2):
    """Thin brown contour around the photo, like the circles in the reference."""
    x0, y0, x1, y1 = box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset
    ss = 4
    layer = Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0)
    ld = ImageDraw.Draw(layer)
    rect = (0, 0, layer.width - 1, layer.height - 1)
    if shape == "oval":
        ld.ellipse(rect, outline=255, width=width * ss)
    else:
        ld.rounded_rectangle(rect, radius=(36 + offset) * ss if shape == "rounded" else 0,
                             outline=255, width=width * ss)
    layer = layer.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    img.paste(Image.new("RGB", layer.size, BROWN), (x0, y0), layer)


def place_photo(img, path, x, y, w, focus, frame):
    """Draw one photo in the chosen frame style starting at (x, y), width w."""
    if frame == "arch":  # bleeds off the bottom edge
        h = H - y
        img.paste(cover_crop(path, w, h, focus), (x, y), shape_mask(w, h, "arch"))
        return
    h = BOTTOM - y
    if frame == "passe":  # white passe-partout card with a soft shadow
        pad = 22
        shadow = Image.new("L", img.size, 0)
        ImageDraw.Draw(shadow).rectangle((x - pad + 6, y - pad + 14, x + w + pad + 6, y + h + pad + 14), fill=70)
        shadow = shadow.filter(ImageFilter.GaussianBlur(22))
        img.paste(Image.new("RGB", img.size, (120, 100, 90)), (0, 0), shadow)
        ImageDraw.Draw(img).rectangle((x - pad, y - pad, x + w + pad, y + h + pad), fill=(252, 251, 249))
        img.paste(cover_crop(path, w, h, focus), (x, y))
        return
    img.paste(cover_crop(path, w, h, focus), (x, y), shape_mask(w, h, frame))
    outline(img, (x, y, x + w, y + h), frame)


def collage(img, paths, focus):
    """Whole photos as white-bordered prints, tilted and overlapping, no cropping."""
    cards = [(-4, 30, 660), (4, 500, 840)]  # (angle, x, y) per photo
    cw, pad = 520, 16
    for src, (angle, x, y) in zip(paths, cards):
        photo = Image.open(src).convert("RGB")
        ch = round(cw * photo.height / photo.width)
        card = Image.new("RGB", (cw + 2 * pad, ch + 2 * pad), (252, 251, 249))
        card.paste(photo.resize((cw, ch), Image.LANCZOS), (pad, pad))
        mask = Image.new("L", card.size, 255).rotate(angle, Image.BICUBIC, expand=True)
        card = card.rotate(angle, Image.BICUBIC, expand=True)
        shadow = Image.new("L", img.size, 0)
        shadow.paste(mask.point(lambda v: v * 80 // 255), (x + 8, y + 18))
        shadow = shadow.filter(ImageFilter.GaussianBlur(24))
        img.paste(Image.new("RGB", img.size, (110, 90, 80)), (0, 0), shadow)
        img.paste(card, (x, y), mask)


def render(i, slide):
    img = linen()
    d = ImageDraw.Draw(img)
    margin = 100

    # Running header, like a magazine page
    hf = manrope(26, "Regular")
    d.text((margin, 170), HEADER_LABEL, font=hf, fill=HEADER)
    counter = f"{i + 1}/{len(SLIDES)}"
    d.text((W - margin - d.textlength(counter, font=hf), 170), counter, font=hf, fill=HEADER)

    # Headline: bold grotesque caps, left-aligned
    tf = montserrat(78, "Bold")
    y = 300
    for line in slide["title"].split("\n"):
        d.text((margin, y), line, font=tf, fill=BROWN)
        y += 92

    # Body text shifted right for editorial asymmetry
    bf = manrope(38, "Light")
    bx = 300
    y += 30
    for line in wrap(d, slide["text"], bf, W - bx - margin):
        d.text((bx, y), line, font=bf, fill=BODY)
        y += 54

    # Photo block under the text
    top = max(y + 70, 760)
    if slide.get("layout") == "collage":
        collage(img, slide["photos"], slide["focus"])
    elif "photos" in slide:
        # Two narrow frames side by side, the second one stepped down
        aw, gap, step = 400, 70, 120
        x = (W - 2 * aw - gap) // 2
        for k, src in enumerate(slide["photos"]):
            place_photo(img, src, x + k * (aw + gap), top + k * step, aw, slide["focus"], FRAME)
    else:
        aw = 760
        place_photo(img, slide["photo"], (W - aw) // 2, top, aw, slide["focus"], FRAME)

    path = os.path.join(OUT, f"story_{i + 1:02d}.jpg")
    img.save(path, quality=95)
    return path


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        FRAME = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    paths = [render(i, s) for i, s in enumerate(SLIDES)]
    ims = [Image.open(p).resize((540, 960)) for p in paths]
    sheet = Image.new("RGB", (560 * len(ims) + 20, 1000), (255, 255, 255))
    for k, im in enumerate(ims):
        sheet.paste(im, (20 + k * 560, 20))
    sheet.save(os.path.join(OUT, "00_overview.jpg"), quality=92)
    print("\n".join(paths))
