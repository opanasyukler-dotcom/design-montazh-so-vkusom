"""@mcg.va — story mini-series in the brand guide style (brand/BRAND.md): full-bleed photo, short conversational line top-left
in Inter SemiBold (tracking −1 px), milk colour on a softly darkened top, no shadows/outlines; optional poll sticker.
  python3 -I brand_story.py OUT_DIR PHOTO1 PHOTO2 PHOTO3
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except ImportError: pass
FU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "user")
W, H = 1080, 1920
MILK, CREAM, PINK, ROSE, BERRY, WINE, COCOA, COFFEE = (251, 248, 244), (246, 240, 230), (240, 204, 213), (215, 160, 175), (166, 75, 100), (107, 48, 62), (121, 90, 74), (51, 43, 40)

def inter(size, w=600):
    f = ImageFont.truetype(os.path.join(FU, "Inter-Variable.ttf"), size); f.set_variation_by_axes([32 if size >= 32 else max(14, size), w]); return f

def track(d, xy, text, f, fill, t=-1.0):
    x, y = xy
    for c in text: d.text((x, y), c, font=f, fill=fill, anchor="ls"); x += f.getlength(c) + t

def wrap(text, f, maxw, t=-1.0):
    lines, cur = [], ""
    for w in text.split(" "):
        s = (cur + " " + w).strip()
        if cur and f.getlength(s) + t * len(s) > maxw: lines.append(cur); cur = w
        else: cur = s
    return lines + [cur]

def cover(im, fx=.5, fy=.5):
    s = max(W / im.width, H / im.height); im = im.resize((math.ceil(im.width * s), math.ceil(im.height * s)), Image.LANCZOS)
    x = int((im.width - W) * fx); y = int((im.height - H) * fy); return im.crop((x, y, x + W, y + H))

def warmth(im):
    a = np.asarray(im, np.float32) / 255; a = 0.03 + a * 0.95; a = a * np.array([1.02, 1.0, 0.97]); return Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8))

def story(src, text, out, fx=.5, fy=.5, poll=None):
    im = warmth(cover(ImageOps.exif_transpose(Image.open(src)).convert("RGB"), fx, fy)).convert("RGBA")
    g = np.zeros((H, 1), np.float32); g[:, 0] = np.clip(1 - np.arange(H) / 760, 0, 1) ** 1.6 * 0.42   # darkened top for milk text
    shade = Image.new("RGBA", (W, H), (51, 30, 30, 0)); shade.putalpha(Image.fromarray((np.repeat(g, W, 1) * 255).astype(np.uint8))); im.alpha_composite(shade)
    d = ImageDraw.Draw(im); f = inter(64, 600); y = 330
    for l in wrap(text, f, 760): track(d, (80, y), l, f, MILK); y += 78
    if poll:
        bw, bh = 820, 150; x0, y0 = (W - bw) // 2, 1440
        d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), 36, fill=MILK + (250,))
        d.line((W / 2, y0 + 30, W / 2, y0 + bh - 30), fill=PINK, width=3)
        for k, opt in enumerate(poll):
            cx = x0 + bw / 4 + k * bw / 2; fo = inter(46, 600); tw = fo.getlength(opt) - len(opt)
            track(d, (cx - tw / 2, y0 + bh / 2 + 16), opt, fo, BERRY)
    im.convert("RGB").save(out, quality=95); print(out)

if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); p = sys.argv[2:5]
    story(p[0], "Моё утро снова началось с бутылочки", os.path.join(out, "story_1.jpg"), .5, .6)
    story(p[1], "Зато у меня есть 10\u00a0минут. Кофе пью горячим", os.path.join(out, "story_2.jpg"), .42, .5)
    story(p[2], "Вам тоже нужно время для себя?", os.path.join(out, "story_3.jpg"), .5, .4, poll=("Да", "Очень"))
