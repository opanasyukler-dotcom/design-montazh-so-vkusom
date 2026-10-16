"""Reels cover 1080x1920: before | after side by side on black, cut straight from the neck to the fingertips,
Bebas title on top; the skin of the before photo is matched to the after one.
Tattoos are removed (dark line work inside hand-picked boxes -> inpainted with the surrounding skin).
Everything sits inside the central 3:4 zone (y 240..1680) so the grid crop in the profile shows it all.

usage: python3 cover_breast_implants.py BEFORE.jpg AFTER.jpg OUT.jpg [masks.jpg]
"""
import os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

BEFORE, AFTER, OUT = sys.argv[1:4]
DEBUG = sys.argv[4] if len(sys.argv) > 4 else None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEBAS = os.path.join(ROOT, "skills/montage-svg-typography/Шрифты/Коллекция шрифтов/BebasNeueCyrillic.ttf")
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
W, H = 1080, 1920
TITLE = ["УВЕЛИЧЕНИЕ ГРУДИ", "ИМПЛАНТАМИ"]

# per photo: tattoo areas (polygons in source pixels), scale, shoulder line y (column x=500), body centre x
SPEC = {
    "before": dict(tattoos=[[(602, 1202), (615, 1202), (615, 1230), (602, 1230)],              # tiny marks in the side crease
                            [(601, 1250), (613, 1250), (613, 1270), (601, 1270)]], s=0.79, shoulder=722, cx=723),
    "after": dict(tattoos=[
        [(430, 1238), (530, 1238), (532, 1562), (468, 1562), (428, 1420), (424, 1320)],              # left forearm
        [(553, 1232), (592, 1232), (592, 1305), (553, 1305)],                                         # ribs
        [(908, 1138), (942, 1138), (952, 1200), (966, 1300), (975, 1400), (986, 1500), (992, 1580),
         (940, 1580), (930, 1500), (924, 1400), (896, 1372), (889, 1322), (899, 1260), (906, 1200)],   # right forearm
    ], s=0.848, shoulder=799, cx=724),
}
# one shoulder line for both; s equalises shoulder->fingertips (before 722->1733, after 799->1740 px)
SHOULDER_Y = 730
TOP = SHOULDER_Y - 29          # straight cut at the neck (the after photo starts at its y 759)
BOT = SHOULDER_Y + 806         # straight cut just under the fingertips
TITLE_Y = 380


def remove_tattoos(img, polys, small=False):
    """polys: outlines drawn just inside the limb, so the dark contour of the arm against the body stays"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    area = np.zeros(gray.shape, np.uint8)
    for poly in polys: cv2.fillPoly(area, [np.array(poly, np.int32)], 1)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    mask = ((bh > 6) & (gray > 40)).astype(np.uint8) & area
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8)) & cv2.dilate(area, np.ones((5, 5), np.uint8))
    if small:                                                     # a couple of dots in the shadow crease: median keeps the edge
        med = cv2.medianBlur(img, 17)
        soft = cv2.GaussianBlur(area.astype(np.float32), (0, 0), 3)[..., None]
        return (med * soft + img * (1 - soft)).astype(np.uint8), area
    # grey-level closing wipes thin dark lines but keeps the light and shading of the skin around them
    clean = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    clean = cv2.GaussianBlur(clean, (0, 0), 2.5)
    clean = (clean.astype(np.float32) * 0.95)                     # closing brightens a touch
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)[..., None]
    out = (clean * soft + img * (1 - soft)).clip(0, 255).astype(np.uint8)
    return out, mask


def skin_mask(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return ((lab[..., 1] > 136) & (lab[..., 0] > 70)).astype(np.uint8)


def match_skin(img, ref):
    """bring the skin of img to the tone of ref: LAB mean/std transfer, applied on skin only (stickers, briefs stay)"""
    li = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32); lr = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    mi, mr = skin_mask(img) > 0, skin_mask(ref) > 0
    out = li.copy()
    for c in range(3):
        a, b = li[..., c][mi], lr[..., c][mr]
        out[..., c] = (li[..., c] - a.mean()) * (b.std() / a.std()) + b.mean()
    out = cv2.cvtColor(out.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    # the whole body, except the white briefs and the navy stickers
    L, A, B = li[..., 0], li[..., 1], li[..., 2]
    keep = (L > 35) & ~((L > 175) & (np.abs(A - 128) < 6) & (np.abs(B - 128) < 10)) & ~(B < 118)
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 2)[..., None]
    return (out * soft + img * (1 - soft)).astype(np.uint8)


def panel(name, img):
    """scale, then put the shoulder line on SHOULDER_Y; centre the body in a W/2 column; straight cut top/bottom"""
    sp = SPEC[name]; s = sp["s"]
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ty = int(round(SHOULDER_Y - sp["shoulder"] * s))
    cx = int(round(sp["cx"] * s)); w = W // 2
    p = np.zeros((H, w, 3), np.uint8)
    for y in range(TOP, BOT):
        sy = y - ty
        if 0 <= sy < img.shape[0]:
            x0 = cx - w // 2
            p[y] = img[sy, x0:x0 + w]
    return p


def main():
    before, mb = remove_tattoos(cv2.imread(BEFORE), SPEC["before"]["tattoos"], True)
    after, ma = remove_tattoos(cv2.imread(AFTER), SPEC["after"]["tattoos"])
    before = match_skin(before, after)
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:, :W // 2] = panel("before", before)
    canvas[:, W // 2:] = panel("after", after)

    pil = Image.fromarray(canvas[..., ::-1])
    d = ImageDraw.Draw(pil)
    ft = ImageFont.truetype(BEBAS, 132)
    for i, line in enumerate(TITLE):
        tw = d.textlength(line, font=ft)
        d.text(((W - tw) / 2, TITLE_Y + i * 122), line, font=ft, fill=(255, 255, 255))
    pil.save(OUT, quality=95)
    if DEBUG:
        cv2.imwrite(DEBUG, np.hstack([cv2.resize(mb * 255, (725, 1288)), cv2.resize(ma * 255, (725, 1288))]))
    print("wrote", OUT)


main()
