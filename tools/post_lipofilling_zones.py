"""Carousel 'Липофилинг — это не только про грудь' (1080x1350) in the reference post style:
dark cinematic cover, powder Bebas headlines, thin Montserrat, light powder collage with arch /
circle frames and a dotted path, one slide per zone with the before/after photo in a card.

usage: python3 post_lipofilling_zones.py MATERIALS_DIR OUT_DIR
MATERIALS_DIR holds the Drive files (IMG_0668 thighs, IMG_0669 shins, IMG_0671-0673 lower lids,
IMG_0678 hands, clip.mp4 face marking).
"""
import os, sys, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageOps

SRC, OUT = sys.argv[1:3]
TMP = tempfile.mkdtemp()
cover_png = os.path.join(TMP, "cover.png")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "15", "-i", os.path.join(SRC, "clip.mp4"), "-frames:v", "1", cover_png], check=True)
# the clip carries burnt-in subtitles (y 925-1022 of 720x1280); they sit under the dark cover shade but
# would still ghost through, so the band is inpainted before the frame becomes the cover background
import cv2  # noqa: E402
_f = cv2.imread(cover_png); _m = np.zeros(_f.shape[:2], np.uint8); _m[922:1026, 110:610] = 255
cv2.imwrite(cover_png, cv2.inpaint(_f, _m, 9, cv2.INPAINT_TELEA))
sys.argv = [sys.argv[0], cover_png, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surgeon_style as ss  # noqa: E402
from surgeon_style import (Slide, bebas, mont, logo, vshade, icon_circle, POWDER, NAVY, DEEP, WHITE, INK, MAUVE, S)  # noqa: E402

PW, PH = 1080, 1350
TOTAL = 6


def photo(name):
    return ImageOps.exif_transpose(Image.open(os.path.join(SRC, name))).convert("RGB")


def cover_fit(im, w, h, cx=0.5, cy=0.5):
    s = max(w / im.width, h / im.height)
    im = im.resize((max(w, round(im.width * s)), max(h, round(im.height * s))), Image.LANCZOS)
    x, y = int((im.width - w) * cx), int((im.height - h) * cy)
    return im.crop((x, y, x + w, y + h))


def place(sl, im, box, shape="rect", cx=0.5, cy=0.5, r=28, shadow=True):
    """photo into a frame: rect (rounded), arch (round top) or circle"""
    x0, y0, x1, y1 = box; w, h = int((x1 - x0) * S), int((y1 - y0) * S)
    ph = cover_fit(im, w, h, cx, cy)
    m = Image.new("L", (w, h), 0); dm = ImageDraw.Draw(m)
    if shape == "circle": dm.ellipse((0, 0, w - 1, h - 1), fill=255)
    elif shape == "arch": dm.rounded_rectangle((0, 0, w - 1, h - 1 + w // 2), radius=w // 2, fill=255)
    else: dm.rounded_rectangle((0, 0, w - 1, h - 1), radius=r * S, fill=255)
    if shadow: sl.shadow(box, (x1 - x0) // 2 if shape != "rect" else r, 20, 110)
    sl.im.paste(ph, (int(x0 * S), int(y0 * S)), m)


def counter(sl, n, col=WHITE, a=170):
    sl.text((PW - 80, 1286), f"{n:02d} / {TOTAL:02d}", mont(24, 400), col, "rs", a=a, spacing=2)


def swipe(sl, x, y, col=POWDER):
    """'>>>' hint as three chevrons, the way the reference posts lead to the next slide"""
    for i in range(3):
        cx = x + i * 26; sl.poly([(cx, y - 13), (cx + 13, y), (cx, y + 13)], col, 2.4, 255 - i * 60)


# ------------------------------------------------------------------ 1 cover
def s1():
    bg = ss.photo_bg(PW, PH, blur=0, tint=NAVY, k=0.4, dark=0.2, cx=1.0, cy=0.736, zoom=1.2, gray=0.25)
    sl = Slide(PW, PH, bg)
    vshade(sl, 0, 260, 0.6, 0.0, DEEP); vshade(sl, 560, 960, 0.0, 0.97, DEEP)
    logo(sl, 92)
    y = sl.lines((80, 760), ["А ВЫ ЗНАЛИ,", "ЧТО ЛИПОФИЛИНГ"], bebas(108), POWDER, lh=0.9)
    y = sl.lines((84, y + 14), ["(увеличение жировой тканью)"], mont(36, 300), WHITE, lh=1.2)
    y = sl.lines((80, y + 26), ["ЭТО НЕ ТОЛЬКО ПРО ГРУДЬ."], bebas(84), WHITE, lh=0.92)
    swipe(sl, 86, y + 56)
    counter(sl, 1)
    return sl.save("01_cover.jpg")


# ------------------------------------------------------------------ 2 list collage (light)
ZONES = [("Коррекция бёдер", "IMG_0668.JPG", (0.25, 0.45)),
         ("Липофилинг голеней", "IMG_0669.JPG", (0.25, 0.75)),
         ("Липофилинг нижних век", "IMG_0673.JPG", (0.5, 0.2)),
         ("Липофилинг кистей рук", "IMG_0678.JPG", (0.25, 0.5))]


def s2():
    sl = Slide(PW, PH); sl.rrect((0, 0, PW, PH), 0, POWDER); sl.rrect((0, 0, PW, PH), 0, (238, 228, 229), a=110)
    logo(sl, 92, MAUVE, 0.8)
    y = sl.lines((76, 190), ["А ТАКЖЕ", "ЭТО ЕЩЁ:"], bebas(124), NAVY, lh=0.88)
    # list on the left with short rules, like the reference
    yy = y + 50
    for i, (t, *_) in enumerate(ZONES):
        sl.poly([(78, yy), (156, yy)], MAUVE, 1.8)
        sl.text((78, yy + 26), f"0{i + 1}", mont(26, 500), MAUVE, spacing=2)
        rows = t.split(" ", 1) if len(t) > 16 else [t]
        sl.lines((78, yy + 66), rows, mont(38, 300), INK, lh=1.15)
        yy += 210
    # frames on the right connected by a dotted path
    frames = [((600, 120, 980, 470), "arch"), ((660, 520, 920, 780), "circle"),
              ((560, 830, 860, 1110), "rect"), ((790, 1010, 1010, 1230), "circle")]
    sl.dashed(ss.bez((560, 330), (480, 520), (640, 560), (660, 650)), MAUVE, 1.6, 4, 9, 200)
    sl.dashed(ss.bez((700, 780), (560, 820), (520, 900), (560, 960)), MAUVE, 1.6, 4, 9, 200)
    sl.dashed(ss.bez((860, 980), (920, 975), (930, 990), (900, 1012)), MAUVE, 1.6, 4, 9, 200)
    for (box, shape), (t, f, (cx, cy)) in zip(frames, ZONES):
        im = photo(f)
        place(sl, im, box, shape, cx, cy, r=26)
    for p in ((560, 330), (660, 650), (560, 960)): sl.dot(p, 6, MAUVE)
    counter(sl, 2, MAUVE, 200)
    return sl.save("02_list.jpg")


# ------------------------------------------------------------------ 3-6 one zone per slide
def zone(n, title, fname, box, kicker="липофилинг", cy=0.5):
    sl = Slide(PW, PH); sl.rrect((0, 0, PW, PH), 0, NAVY); vshade(sl, 0, PH, 0.0, 0.5, DEEP)
    logo(sl, 92)
    sl.text((82, 200), f"0{n - 2}", mont(30, 500), POWDER, spacing=3)
    sl.poly([(140, 216), (230, 216)], POWDER, 1.6)
    y = sl.lines((80, 236), title, bebas(118), POWDER, lh=0.9)
    place(sl, photo(fname), box, "rect", 0.5, cy, r=28)
    counter(sl, n)
    return sl.save(f"0{n}_{os.path.splitext(fname)[0]}.jpg")


def main():
    os.makedirs(OUT, exist_ok=True)
    out = [s1(), s2(),
           zone(3, ["КОРРЕКЦИЯ", "БЁДЕР"], "IMG_0668.JPG", (60, 500, 1020, 1213)),
           zone(4, ["ЛИПОФИЛИНГ", "ГОЛЕНЕЙ"], "IMG_0669.JPG", (60, 500, 1020, 1213)),
           zone(5, ["ЛИПОФИЛИНГ", "НИЖНИХ ВЕК"], "IMG_0673.JPG", (190, 480, 890, 1180)),
           zone(6, ["ЛИПОФИЛИНГ", "КИСТЕЙ РУК"], "IMG_0678.JPG", (60, 520, 1020, 1160))]
    ss.sheet(out, 3, 540, "00_overview.jpg")
    print("ok")


if __name__ == "__main__":
    main()
