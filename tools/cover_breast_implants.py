"""Reels cover 1080x1920: before | after side by side on black, Bebas title on top, thin Montserrat caption below.
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
TITLE, CAPTION = "ДО / ПОСЛЕ", "увеличение груди имплантами"

# per photo: tattoo areas (polygons in source pixels), near-nipple point, crop x-range
SPEC = {
    "before": dict(tattoos=[[(602, 1202), (615, 1202), (615, 1230), (602, 1230)],              # tiny marks in the side crease
                            [(601, 1250), (613, 1250), (613, 1270), (601, 1270)]], small=True, nip=(798, 1045), x=(366, 1016)),
    "after": dict(tattoos=[
        [(430, 1238), (530, 1238), (532, 1562), (468, 1562), (428, 1420), (424, 1320)],              # left forearm
        [(553, 1232), (592, 1232), (592, 1305), (553, 1305)],                                         # ribs
        [(908, 1138), (942, 1138), (952, 1200), (966, 1300), (975, 1400), (986, 1500), (992, 1580),
         (940, 1580), (930, 1500), (924, 1400), (896, 1372), (889, 1322), (899, 1260), (906, 1200)],   # right forearm
    ],
                  nip=(760, 1065), x=(398, 1048)),
}
S = 0.84             # both photos are shot at the same distance -> one scale
NIP_Y = 900          # nipple line on the cover
TOP, BOT = 470, 1530  # photo band


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


def panel(name):
    sp = SPEC[name]
    img = cv2.imread(BEFORE if name == "before" else AFTER)
    img, mask = remove_tattoos(img, sp["tattoos"], sp.get("small", False))
    x0, x1 = sp["x"]
    crop = img[:, x0:x1]
    crop = cv2.resize(crop, None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
    ty = int(round(NIP_Y - sp["nip"][1] * S))
    p = np.zeros((H, crop.shape[1], 3), np.uint8)
    a, b = max(ty, 0), min(ty + crop.shape[0], H)
    p[a:b] = crop[a - ty:b - ty]
    return p, mask


def main():
    canvas = np.zeros((H, W, 3), np.uint8)
    pb, mb = panel("before"); pa, ma = panel("after")
    wb = W // 2
    canvas[:, :wb] = pb[:, (pb.shape[1] - wb) // 2:(pb.shape[1] - wb) // 2 + wb]
    canvas[:, wb:] = pa[:, (pa.shape[1] - (W - wb)) // 2:(pa.shape[1] - (W - wb)) // 2 + (W - wb)]
    # photo band: hard top/bottom with a short fade to black
    y = np.arange(H, dtype=np.float32)[:, None, None]
    k = np.clip((y - TOP) / 40, 0, 1) * np.clip((BOT - y) / 60, 0, 1)
    canvas = (canvas * k).astype(np.uint8)

    pil = Image.fromarray(canvas[..., ::-1])
    d = ImageDraw.Draw(pil)
    ft = ImageFont.truetype(BEBAS, 118)
    tw = d.textlength(TITLE, font=ft)
    d.text(((W - tw) / 2, 300), TITLE, font=ft, fill=(255, 255, 255))
    fc = ImageFont.truetype(MONT, 46); fc.set_variation_by_axes([300])
    cw = d.textlength(CAPTION, font=fc)
    d.text(((W - cw) / 2, 1572), CAPTION, font=fc, fill=(255, 255, 255))
    pil.save(OUT, quality=95)
    if DEBUG:
        cv2.imwrite(DEBUG, np.hstack([cv2.resize(mb * 255, (725, 1288)), cv2.resize(ma * 255, (725, 1288))]))
    print("wrote", OUT)


main()
