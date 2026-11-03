"""Before / after collages 1080x1920 on black: after on top, before below, each body from the neck to the fingertips at
the same scale and on the same axis; caption between them in the clinic's type (Inter SemiBold).
Two results: 3.5 months (IMG_9288-9290, 09.10.2025) and 1 year 3 months (IMG_2052 / 2056 / 2058, 02.11.2026);
before: IMG_9285-9287 (29.07.2025). Front, 3/4, side.  No cut-out: the dark studio wall is pulled down to black, the body is left exactly as shot.

usage: python3 collages_ba_3m_1y3m.py SRC_DIR OUT_DIR      (SRC_DIR/drive holds <name>.png, the photos as RGB)
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC, OUT = sys.argv[1:3]; DRIVE = os.path.join(SRC, "drive"); os.makedirs(OUT, exist_ok=True)
W, H = 1080, 1920
SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
# name: (neck y, fingertips y) in the photo's pixels, read off zoomed grids of the neck and of the hands
LM = {"IMG_9285.JPG": (300, 3100), "IMG_9286.JPG": (250, 3300), "IMG_9287.JPG": (280, 3165),
      "IMG_9288.HEIC": (380, 3985), "IMG_9289.HEIC": (290, 3975), "IMG_9290.HEIC": (250, 3990),
      "IMG_2052.JPG": (160, 1730), "IMG_2056.JPG": (300, 2195), "IMG_2058.JPG": (290, 1910)}
BEFORE = ["IMG_9285.JPG", "IMG_9286.JPG", "IMG_9287.JPG"]
RESULTS = {"3_5_mesyaca": (["IMG_9288.HEIC", "IMG_9289.HEIC", "IMG_9290.HEIC"], "3,5 месяца после операции"),
           "1_god_3_mesyaca": (["IMG_2052.JPG", "IMG_2056.JPG", "IMG_2058.JPG"], "1 год 3 месяца после операции")}
VIEWS = ["pryamo", "3-4", "sboku"]
BAND = 860                     # neck .. a little below the fingertips takes this many pixels
EXT = 0.06                     # how far below the fingertips (share of neck..fingertips), as far as the photo goes
TOP_Y, BOT_Y = 30, 1050        # where the after / before necks sit


def body(name):
    """the photo itself (no cut-out): the wall pulled down to black row by row (its level = the darker of the two outer
    6 % strips), so the body, stockings and jeans stay exactly as shot; from the neck (no face) to a little below the
    fingertips; the body axis from the bright (skin) pixels between neck and fingertips"""
    im = np.asarray(Image.open(os.path.join(DRIVE, name + ".png")).convert("RGB"), np.float32) / 255
    h, w = im.shape[:2]; k = int(w * 0.06)
    bg = np.minimum(np.median(im[:, :k], 1), np.median(im[:, -k:], 1))[:, None, :]          # per row, per channel
    bg = np.minimum(bg, 0.35)
    out = np.clip((im - bg) / (1 - bg), 0, 1)
    neck, tips = LM[name]; span = tips - neck
    bot = min(h, int(tips + EXT * span))
    skin = out[neck:tips].mean(2) > 0.25; xs = np.nonzero(skin)[1]; axis = np.median(xs)
    s = BAND / (span * (1 + EXT))
    out = out[neck:bot]
    # the photo's edges melt into black (no visible rectangle); 7 % of the width at the sides, a little at the bottom
    hh, ww = out.shape[:2]; fx = int(ww * 0.07)
    gx = np.clip(np.minimum(np.arange(ww), ww - 1 - np.arange(ww)) / fx, 0, 1); gx = gx * gx * (3 - 2 * gx)
    fy = int(hh * 0.03); gy = np.clip((hh - 1 - np.arange(hh)) / max(fy, 1), 0, 1)
    out = out * gx[None, :, None] * gy[:, None, None]
    img = Image.fromarray((out * 255).astype(np.uint8))
    img = img.resize((int(w * s), int((bot - neck) * s)), Image.LANCZOS)
    return img, axis * s


def collage(after, before, caption):
    cv = Image.new("RGB", (W, H), (0, 0, 0))
    for name, y in [(after, TOP_Y), (before, BOT_Y)]:
        img, ax = body(name); x = int(W / 2 - ax)
        lim = (BOT_Y - 140) if y == TOP_Y else H                                       # the after stops above the caption
        cv.paste(img.crop((0, 0, img.width, min(img.height, lim - y))), (x, y))
    d = ImageDraw.Draw(cv); f = ImageFont.truetype(SEMI, 46)
    for k, t in enumerate(["Результат спустя", caption]):
        d.text((W / 2, BOT_Y - 120 + k * 56), t, font=f, fill=(255, 255, 255), anchor="mm")
    return cv


for key, (afters, cap) in RESULTS.items():
    for a, b, v in zip(afters, BEFORE, VIEWS):
        out = os.path.join(OUT, f"kollazh_{key}_{v}.jpg"); collage(a, b, cap).save(out, quality=95); print("wrote", out)
