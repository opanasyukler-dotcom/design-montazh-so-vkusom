"""Before / after collages 1080x1920 on black: after on top, before below, each body from the neck to the fingertips at
the same scale and on the same axis; caption between them in the clinic's type (Inter SemiBold).
Two results: 3.5 months (IMG_9288-9290, 09.10.2025) and 1 year 3 months (IMG_2052 / 2056 / 2058, 02.11.2026);
before: IMG_9285-9287 (29.07.2025). Front, 3/4, side.  Backgrounds cut to black with rembg (<name>.mask.png).

usage: python3 collages_ba_3m_1y3m.py SRC_DIR OUT_DIR      (SRC_DIR/drive holds <name>.png + <name>.mask.png)
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC, OUT = sys.argv[1:3]; DRIVE = os.path.join(SRC, "drive"); os.makedirs(OUT, exist_ok=True)
W, H = 1080, 1920
SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
# name: (neck y, fingertips y) in the photo's pixels, read off a grid; the body axis is the mask's centroid between them
LM = {"IMG_9285.JPG": (300, 2980), "IMG_9286.JPG": (250, 3150), "IMG_9287.JPG": (280, 3100),
      "IMG_9288.HEIC": (380, 3840), "IMG_9289.HEIC": (290, 3950), "IMG_9290.HEIC": (250, 3700),
      "IMG_2052.JPG": (160, 1700), "IMG_2056.JPG": (300, 2010), "IMG_2058.JPG": (290, 1900)}
BEFORE = ["IMG_9285.JPG", "IMG_9286.JPG", "IMG_9287.JPG"]
RESULTS = {"3_5_mesyaca": (["IMG_9288.HEIC", "IMG_9289.HEIC", "IMG_9290.HEIC"], "3,5 месяца после операции"),
           "1_god_3_mesyaca": (["IMG_2052.JPG", "IMG_2056.JPG", "IMG_2058.JPG"], "1 год 3 месяца после операции")}
VIEWS = ["pryamo", "3-4", "sboku"]
BAND = 830                     # neck..fingertips of every body takes this many pixels
TOP_Y, BOT_Y = 40, 1050        # where the after / before necks sit


def body(name):
    im = np.asarray(Image.open(os.path.join(DRIVE, name + ".png")).convert("RGB"), np.float32)
    m = np.asarray(Image.open(os.path.join(DRIVE, name + ".mask.png")).convert("L"), np.float32) / 255
    neck, tips = LM[name]
    ys, xs = np.nonzero(m[neck:tips] > 0.5); axis = xs.mean()
    s = BAND / (tips - neck)
    img = Image.fromarray((im * m[..., None]).astype(np.uint8))
    img = img.crop((0, neck, img.width, min(img.height, int(tips + 15 / s))))        # from the neck: no face
    img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    return img, axis * s


def collage(after, before, caption):
    cv = Image.new("RGB", (W, H), (0, 0, 0))
    for name, y in [(after, TOP_Y), (before, BOT_Y)]:
        img, ax = body(name); x = int(W / 2 - ax)
        lim = (BOT_Y - 150) if y == TOP_Y else H                                       # the after stops above the caption
        cv.paste(img.crop((0, 0, img.width, min(img.height, lim - y))), (x, y))
    d = ImageDraw.Draw(cv); f = ImageFont.truetype(SEMI, 46)
    for k, t in enumerate(["Результат спустя", caption]):
        d.text((W / 2, BOT_Y - 120 + k * 56), t, font=f, fill=(255, 255, 255), anchor="mm")
    return cv


for key, (afters, cap) in RESULTS.items():
    for a, b, v in zip(afters, BEFORE, VIEWS):
        out = os.path.join(OUT, f"kollazh_{key}_{v}.jpg"); collage(a, b, cap).save(out, quality=95); print("wrote", out)
