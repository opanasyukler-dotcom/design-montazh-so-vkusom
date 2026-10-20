"""Stories kit for @mcg.va — the three type roles from the blog's own stories (1080x1920):
  title  — Times New Roman Italic look (Liberation Serif Italic): headlines, the main accent line
  body   — Helvetica look (Liberation Sans Regular): the main text
  detail — Helvetica Italic look (Liberation Sans Italic): small asides, details, captions
Sizes are matched to the reference story (centered stack on white). Fonts: fonts/stories (SIL OFL 1.1).
Instagram safe zone: keep text between y ≈ 250 and y ≈ 1580.
  python3 -I stories_kit.py demo out.png
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
SAFE_TOP, SAFE_BOTTOM = 250, 1580
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "stories")
ROLES = {
    "title": ("LiberationSerif-Italic.ttf", 62),
    "body": ("LiberationSans-Regular.ttf", 42),
    "detail": ("LiberationSans-Italic.ttf", 36),
}
INK = (17, 17, 17)

def font(role, size=None):
    f, s = ROLES[role]
    return ImageFont.truetype(os.path.join(FONT_DIR, f), size or s)

def wrap(text, fnt, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if fnt.getlength(t) <= maxw or not cur: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def draw_text(img, role, text, y, x=W // 2, align="center", size=None, color=INK, maxw=900, leading=1.25):
    """draw a (wrapped) text block whose first line is centred vertically on y; returns y after the block"""
    d = ImageDraw.Draw(img); fnt = font(role, size)
    anchor = {"center": "mm", "left": "lm", "right": "rm"}[align]
    for line in wrap(text, fnt, maxw):
        d.text((x, y), line, font=fnt, fill=color, anchor=anchor); y += fnt.size * leading
    return y

def demo(out):
    """the reference story re-set with the kit"""
    img = Image.new("RGB", (W, H), "white")
    draw_text(img, "title", "Какая сегодня солнечная погода !", 595)
    draw_text(img, "body", "очень красиво сегодня на улице", 712)
    draw_text(img, "detail", "мне так нравится гулять", 849)
    img.save(out)

if __name__ == "__main__":
    if sys.argv[1] == "demo": demo(sys.argv[2])
