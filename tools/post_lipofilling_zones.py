"""Carousel 'Липофилинг — это не только про грудь' (1080x1350) in the reference post style:
navy slides, powder Bebas headlines, thin Montserrat, before/after photos in rounded cards.

  01 cover: breast lipofilling result (front + three-quarter) under the headline
  02 shins: front and back result, stacked, with 'спереди' / 'сзади' labels
  03 lower eyelids: straight-on and three-quarter result
  04 hands: the hands result
  (a hips slide is added when HIPS files are given)

usage: python3 post_lipofilling_zones.py MATERIALS_DIR OUT_DIR [HIPS_1.jpg HIPS_2.jpg]
"""
import os, sys
from PIL import Image, ImageDraw, ImageOps

SRC, OUT = sys.argv[1:3]
HIPS = sys.argv[3:5]
sys.argv = [sys.argv[0], os.path.join(SRC, "IMG_0665.JPG"), OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surgeon_style as ss  # noqa: E402
from surgeon_style import Slide, bebas, mont, logo, vshade, POWDER, NAVY, DEEP, WHITE, S  # noqa: E402

PW, PH = 1080, 1350


def photo(name):
    p = name if os.path.isabs(name) else os.path.join(SRC, name)
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


def fit(im, w, h, cx=0.5, cy=0.5):
    s = max(w / im.width, h / im.height)
    im = im.resize((max(w, round(im.width * s)), max(h, round(im.height * s))), Image.LANCZOS)
    x, y = int((im.width - w) * cx), int((im.height - h) * cy)
    return im.crop((x, y, x + w, y + h))


def card(sl, im, box, r=26, cx=0.5, cy=0.5):
    x0, y0, x1, y1 = box; w, h = int((x1 - x0) * S), int((y1 - y0) * S)
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=r * S, fill=255)
    sl.shadow(box, r, 20, 120)
    sl.im.paste(fit(im, w, h, cx, cy), (int(x0 * S), int(y0 * S)), m)


def base():
    sl = Slide(PW, PH); sl.rrect((0, 0, PW, PH), 0, NAVY); vshade(sl, 0, PH, 0.0, 0.5, DEEP)
    logo(sl, 92); return sl


def counter(sl, n, total):
    sl.text((PW - 80, 1300), f"{n:02d} / {total:02d}", mont(24, 400), WHITE, "rs", a=170, spacing=2)


def swipe(sl, x, y, col=POWDER):
    for i in range(3):
        cx = x + i * 26; sl.poly([(cx, y - 13), (cx + 13, y), (cx, y + 13)], col, 2.4, 255 - i * 60)


def cover(n, total):
    sl = base()
    y = sl.lines((80, 160), ["А ВЫ ЗНАЛИ, ЧТО ЛИПОФИЛИНГ"], bebas(92), POWDER, lh=0.9)
    y = sl.lines((84, y + 10), ["(увеличение жировой тканью)"], mont(34, 300), WHITE, lh=1.2)
    y = sl.lines((80, y + 18), ["ЭТО НЕ ТОЛЬКО ПРО ГРУДЬ."], bebas(78), WHITE, lh=0.92)
    swipe(sl, 940, y - 34)
    top = y + 40
    h = 1270 - top; w = int(h * 720 / 1280)          # the breast collages are 9:16 (after on top, before below)
    gap = 30; x0 = (PW - 2 * w - gap) // 2
    card(sl, photo("IMG_0665.JPG"), (x0, top, x0 + w, top + h))
    card(sl, photo("IMG_0666.JPG"), (x0 + w + gap, top, x0 + 2 * w + gap, top + h))
    sl.text((x0 + 18, top + h - 20), "липофилинг груди", mont(26, 400), WHITE, "ls", a=220)
    counter(sl, n, total)
    return sl.save(f"{n:02d}_cover.jpg")


def pill(sl, x, y, text):
    f = mont(26, 400); tw = f.getlength(text) / S
    sl.rrect((x, y, x + tw + 36, y + 46), 23, NAVY, 225)
    sl.text((x + 18, y + 23), text, f, WHITE, "lm")


def two_stacked(n, total, title, items, name):
    """one-line title, two results stacked and centred, each labelled with a small pill on the card"""
    sl = base()
    sl.lines((80, 170), title, bebas(104), POWDER, lh=0.9)
    gap, top, bottom = 26, 300, 1265
    y = top
    for label, fname, aspect, cy in items:
        h = int((bottom - top - gap) / 2); w = int(min(h * aspect, 960)); h = int(w / aspect)
        x = (PW - w) // 2
        card(sl, photo(fname), (x, y, x + w, y + h), cy=cy)
        pill(sl, x + 18, y + 18, label)
        y += h + gap
    counter(sl, n, total)
    return sl.save(name)


def single(n, total, title, fname, name):
    sl = base()
    sl.lines((80, 170), title, bebas(104), POWDER, lh=0.9)
    im = photo(fname); w = 960; h = int(w * im.height / im.width)
    y = 300 + (1265 - 300 - h) // 2
    card(sl, im, (60, y, 60 + w, y + h))
    counter(sl, n, total)
    return sl.save(name)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".jpg"): os.remove(os.path.join(OUT, f))
    total = 5 if HIPS else 4
    out = [cover(1, total)]
    k = 2
    if HIPS:
        out.append(two_stacked(k, total, ["КОРРЕКЦИЯ БЁДЕР"], [("результат 1", HIPS[0], 1.345, 0.5), ("результат 2", HIPS[1], 1.345, 0.5)],
                               f"{k:02d}_hips.jpg")); k += 1
    out.append(two_stacked(k, total, ["ЛИПОФИЛИНГ ГОЛЕНЕЙ"],
                           [("спереди", "IMG_0668.JPG", 1024 / 761, 0.5), ("сзади", "IMG_0669.JPG", 1024 / 761, 0.5)],
                           f"{k:02d}_shins.jpg")); k += 1
    out.append(two_stacked(k, total, ["ЛИПОФИЛИНГ НИЖНИХ ВЕК"],
                           [("анфас", "IMG_0673.JPG", 1.0, 0.5), ("в три четверти", "IMG_0671.JPG", 1.0, 0.5)],
                           f"{k:02d}_eyelids.jpg")); k += 1
    out.append(single(k, total, ["ЛИПОФИЛИНГ КИСТЕЙ РУК"], "IMG_0678.JPG", f"{k:02d}_hands.jpg"))
    ss.sheet(out, len(out), 432, "00_overview.jpg")
    print("ok", len(out))


if __name__ == "__main__":
    main()
