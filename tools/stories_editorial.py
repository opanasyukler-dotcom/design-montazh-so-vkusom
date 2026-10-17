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
    },
]
HEADER_LABEL = "наш уход"


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


def arch_photo(path, box_w, box_h, focus):
    """Cover-crop the photo to box size and cut it into an arch (rounded top)."""
    src = Image.open(path).convert("RGB")
    scale = max(box_w / src.width, box_h / src.height)
    src = src.resize((round(src.width * scale), round(src.height * scale)), Image.LANCZOS)
    left = int((src.width - box_w) * focus[0])
    top = int((src.height - box_h) * focus[1])
    crop = src.crop((left, top, left + box_w, top + box_h))

    ss = 4  # supersample the mask for a smooth arch edge
    mask = Image.new("L", (box_w * ss, box_h * ss), 0)
    md = ImageDraw.Draw(mask)
    r = box_w * ss // 2
    md.ellipse((0, 0, box_w * ss, 2 * r), fill=255)
    md.rectangle((0, r, box_w * ss, box_h * ss), fill=255)
    return crop, mask.resize((box_w, box_h), Image.LANCZOS)


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

    # Arch photo bleeding off the bottom edge
    top = max(y + 70, 760)
    if "photos" in slide:
        # Two narrow arches side by side, the second one stepped down
        aw, gap, step = 420, 40, 120
        x = (W - 2 * aw - gap) // 2
        for k, src in enumerate(slide["photos"]):
            t = top + k * step
            photo, mask = arch_photo(src, aw, H - t, slide["focus"])
            img.paste(photo, (x + k * (aw + gap), t), mask)
    else:
        aw = 800
        photo, mask = arch_photo(slide["photo"], aw, H - top, slide["focus"])
        img.paste(photo, ((W - aw) // 2, top), mask)

    path = os.path.join(OUT, f"story_{i + 1:02d}.jpg")
    img.save(path, quality=95)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    paths = [render(i, s) for i, s in enumerate(SLIDES)]
    ims = [Image.open(p).resize((540, 960)) for p in paths]
    sheet = Image.new("RGB", (560 * len(ims) + 20, 1000), (255, 255, 255))
    for k, im in enumerate(ims):
        sheet.paste(im, (20 + k * 560, 20))
    sheet.save(os.path.join(OUT, "00_overview.jpg"), quality=92)
    print("\n".join(paths))
