"""ДО/ПОСЛЕ collages 1080x1920, ПОСЛЕ on top, ДО below, neck to knees, background removed to black,
'1 НЕДЕЛЯ ПОСЛЕ ОПЕРАЦИИ' on the seam (same look as Collage_body_*.jpg).

Both halves of a pair use the same scale, anchored on the nipple line, so the torso is compared 1:1.

usage: python3 collage_week1.py SRC_DIR OUT_DIR
SRC_DIR holds b_front/b_34/b_side (before) and a_front/a_34/a_side (after) .jpg
"""
import os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rembg import remove, new_session

SRC, OUT = sys.argv[1:3]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEBAS = os.path.join(ROOT, "skills/montage-svg-typography/Шрифты/Коллекция шрифтов/BebasNeueCyrillic.ttf")
W, H, HALF = 1080, 1920, 960
ABOVE, BELOW = 560, 1720          # source px kept above / below the nipple line: from under the chin to the knees
S = 900 / (ABOVE + BELOW)
# nipple line y and body centre x, source pixels (images are 2576 px tall)
ANCH = {"b_front": (657, 566), "a_front": (631, 690),
        "b_34": (670, 550), "a_34": (631, 690),
        "b_side": (631, 532), "a_side": (580, 711)}
SESSION = new_session("u2net_human_seg")   # keeps the whole silhouette in profile (plain u2net lost legs and back)


def half(name):
    im = ImageOps.exif_transpose(Image.open(os.path.join(SRC, name + ".jpg"))).convert("RGB")
    m = np.asarray(remove(im, session=SESSION, only_mask=True, post_process_mask=True)).astype(np.float32) / 255
    m = cv2.GaussianBlur(m, (0, 0), 1.2)[..., None]
    img = np.asarray(im).astype(np.float32)
    # the dark studio backdrop that shows between the legs stays inside the mask: drop dark, grey pixels too
    hsv = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2HSV).astype(np.float32)
    bgk = np.clip((60 - hsv[..., 2]) / 20, 0, 1) * np.clip((70 - hsv[..., 1]) / 30, 0, 1)
    m = m * (1 - cv2.GaussianBlur(bgk, (0, 0), 2)[..., None])
    img = img * m                                                   # background -> black
    ny, cx = ANCH[name]; top = ny - ABOVE
    M = np.float32([[S, 0, 540 - S * cx], [0, S, 30 - S * top]])
    out = cv2.warpAffine(img, M, (W, HALF), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))
    y = np.arange(HALF, dtype=np.float32)[:, None, None]
    fade = np.clip((y - 30) / 22 + 0.5, 0, 1) * np.clip((930 - y) / 22 + 0.5, 0, 1)   # soft cut under the chin and at the knees
    return out * fade


def pill(d, x, y, txt, col):
    f = ImageFont.truetype(BEBAS, 60); tw = f.getlength(txt); h = 66; w = int(tw + 60)
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=col)
    d.text((x + w / 2, y + h / 2 + 2), txt, font=f, fill=(28, 24, 40), anchor="mm")


def collage(view):
    top = half("a_" + view); bot = half("b_" + view)
    fr = np.concatenate([top, bot], 0)
    fr[HALF - 3:HALF + 3] = (34, 34, 44)
    im = Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    pill(d, 54, 40, "ПОСЛЕ", (233, 214, 242)); pill(d, 54, HALF + 72, "ДО", (232, 212, 206))
    cap = "1 НЕДЕЛЯ ПОСЛЕ ОПЕРАЦИИ"; f = ImageFont.truetype(BEBAS, 84); tw = f.getlength(cap)
    arr = np.asarray(im).astype(np.float32)
    lay = Image.new("L", (W, H), 0); ImageDraw.Draw(lay).text((W / 2, HALF + 2), cap, font=f, fill=255, anchor="mm")
    t = np.asarray(lay).astype(np.float32) / 255
    sh = np.clip(cv2.GaussianBlur(cv2.dilate(t, np.ones((9, 9), np.uint8)), (0, 0), 10) * 1.6, 0, 1)[..., None]
    arr = arr * (1 - sh)
    arr[HALF - 3:HALF + 3, int(W / 2 - tw / 2 - 26):int(W / 2 + tw / 2 + 26)] = 0
    arr = arr * (1 - t[..., None]) + 255 * t[..., None]
    out = os.path.join(OUT, f"Collage_week1_{view}.jpg")
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(out, quality=95)
    print("saved", out)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for v in ("front", "34", "side"): collage(v)
