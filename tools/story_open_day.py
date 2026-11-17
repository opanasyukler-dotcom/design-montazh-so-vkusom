"""Story 'День открытых дверей' in the 'quote on black' story design (same as stories_ptoz):
three powder dots + bold caps headline at the top, bold lead + regular line at the bottom,
a large free area in the middle for the reposted reel.

usage: python3 story_open_day.py OUT_DIR
Writes open_day.jpg (clean, for posting) and open_day_preview.jpg (with the repost area marked).
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
W, H, X = 1080, 1920, 78
POWDER = (226, 210, 211)
NAVY = (16, 46, 70)


def mont(size, w):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([w]); return f


def background():
    """black like the reference, with a very soft navy glow where the reel will sit"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.exp(-(((xx - W / 2) / 620) ** 2 + ((yy - 1060) / 760) ** 2)) * 0.55
    a = np.float32(NAVY)[None, None] * g[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")


def main():
    os.makedirs(OUT, exist_ok=True)
    im = background(); d = ImageDraw.Draw(im)
    for i in range(3):
        cx = X + 6 + i * 24; d.ellipse((cx - 6, 144, cx + 6, 156), fill=POWDER + (255,))
    fh = mont(76, 700); lh = 84
    head = ["ЖДУ ВАС В ЭТУ", "ПЯТНИЦУ НА ДЕНЬ", "ОТКРЫТЫХ ДВЕРЕЙ"]
    assert max(fh.getlength(s) for s in head) <= W - 2 * X, "headline too wide"
    for i, ln in enumerate(head): d.text((X, 210 + i * lh), ln, font=fh, fill=(255, 255, 255, 255))
    head_bottom = 210 + len(head) * lh

    fb, fr = mont(40, 600), mont(40, 400)
    lead, body = "Пишите, кому направить условия!", "Будут специальные условия в этот день!"
    y = 1790 - (50 + 22 + 50)
    text_top = y
    d.text((X, y), lead, font=fb, fill=(255, 255, 255, 255)); y += 72
    d.text((X, y), body, font=fr, fill=(255, 255, 255, 235))

    clean = im.convert("RGB"); clean.save(os.path.join(OUT, "open_day.jpg"), quality=95, subsampling=0)

    # preview: the free area for the reel repost (not for posting)
    pv = im.copy(); dp = ImageDraw.Draw(pv)
    top, bot = head_bottom + 50, text_top - 50
    rh = bot - top; rw = int(rh * 9 / 16); x0 = (W - rw) // 2
    for yy in range(top, bot, 22): dp.line((x0, yy, x0, yy + 11), fill=POWDER + (200,), width=3); dp.line((x0 + rw, yy, x0 + rw, yy + 11), fill=POWDER + (200,), width=3)
    for xx in range(x0, x0 + rw, 22): dp.line((xx, top, xx + 11, top), fill=POWDER + (200,), width=3); dp.line((xx, bot, xx + 11, bot), fill=POWDER + (200,), width=3)
    dp.text((W / 2, (top + bot) / 2), "место для репоста\nролика", font=mont(38, 400), fill=POWDER + (230,), anchor="mm", align="center")
    pv.convert("RGB").save(os.path.join(OUT, "open_day_preview.jpg"), quality=92)
    print("free area", top, bot, "=", rh, "px tall")


if __name__ == "__main__":
    main()
