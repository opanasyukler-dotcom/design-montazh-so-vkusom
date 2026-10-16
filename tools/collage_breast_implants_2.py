"""Before | after collage 1080x1920 in the style of cover_breast_implants: black background, straight cut from the neck
(under the chin) down, Bebas title on top. The after photo is colour-matched to the before one.

usage: python3 collage_breast_implants_2.py BEFORE.jpg AFTER.jpg OUT.jpg
"""
import os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

BEFORE, AFTER, OUT = sys.argv[1:4]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEBAS = os.path.join(ROOT, "skills/montage-svg-typography/Шрифты/Коллекция шрифтов/BebasNeueCyrillic.ttf")
W, H = 1080, 1920
TITLE = "УВЕЛИЧЕНИЕ ГРУДИ ИМПЛАНТАМИ"

# both photos are shot from the same distance -> one scale; aligned on the near sticker
# before: chin ends at y~60, after: chin ~100 + chain; frames end at 2016 (after: sticker+1316)
# nipples (near, far) get the clinic sticker on top, d = diameter in source px (covers the whole areola);
# the far one hangs over the breast edge like on the reference cover
SPEC = {"before": dict(sticker=565, cx=652, nipples=[(697, 571), (975, 648)], d=112),
        "after": dict(sticker=700, cx=660, nipples=[(811, 700), (1052, 655)], d=128)}
STICKER = os.path.join(ROOT, "collage_lipofiling_lift/sticker_anna_tvauri.png")
S = 0.58
UP, DOWN = 490, 1310             # cut: sticker-UP (neck, under both chins) .. sticker+DOWN (bottom of the after frame)
STICKER_Y = 590 + int(UP * S)    # photo band starts at y 590
TITLE_Y = 472


def body_mask(img, name):
    """body vs backdrop; holes (stickers) filled.
    before: slate-blue backdrop (L<70, b<121) vs pale warm skin / white briefs; after: near-black backdrop (L<30)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    m = ((L > 105) | (b > 127)) if name == "before" else ((L > 45) | (a > 134))
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m, 8)
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > big * 0.02: keep[lab_ == i] = 1
    ff = np.pad(keep, 1); h, w = ff.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    keep[ff[1:-1, 1:-1] == 0] = 1
    return cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 2.0)


def skin_mask(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    return (L > 90) & (L < 240) & (a > 124) & (b > 131)          # warm skin; jeans, white briefs, stickers, backdrop out


def match_skin(img, ref):
    """bring the skin of img to the tone of ref (LAB mean/std on skin); stickers, jeans, briefs keep their colour"""
    li = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32); lr = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    mi, mr = skin_mask(img), skin_mask(ref)
    out = li.copy()
    for c in range(3):
        a, b = li[..., c][mi], lr[..., c][mr]
        out[..., c] = (li[..., c] - a.mean()) * (b.std() / a.std()) + b.mean()
    out = cv2.cvtColor(out.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    keep = cv2.morphologyEx(mi.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    keep &= (li[..., 2] > 126).astype(np.uint8)                    # never the blue jeans
    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 3)[..., None]
    return (out * soft + img * (1 - soft)).astype(np.uint8)


def panel(name, img):
    sp = SPEC[name]
    img = (img * body_mask(img, name)[..., None]).astype(np.uint8)
    img = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
    ty = int(round(STICKER_Y - sp["sticker"] * S)); w = W // 2
    x0 = int(round(sp["cx"] * S)) - w // 2
    top, bot = STICKER_Y - int(UP * S), STICKER_Y + int(DOWN * S)
    p = np.zeros((H, w, 3), np.uint8)
    src = img[top - ty:bot - ty]
    xa, xb = max(x0, 0), min(x0 + w, img.shape[1])
    p[top:top + src.shape[0], xa - x0:xb - x0] = src[:, xa:xb]
    return p


# bruises on the after photo: under the breast and in the cleavage (source pixels)
BRUISE_ZONE = [(520, 600), (1000, 600), (1010, 900), (520, 900)]
BRUISE_KEEP = 0.35          # "slightly": keep 35 % of the bruise colour


def soften_bruises(img):
    """pull yellow / green-grey patches towards the surrounding skin tone (median of a wide neighbourhood)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    zone = np.zeros(img.shape[:2], np.uint8); cv2.fillPoly(zone, [np.array(BRUISE_ZONE, np.int32)], 1)
    ref = lab.copy()
    for c in range(3):                                            # wide median (~250 px) as the "clean skin" reference
        small = cv2.resize(lab[..., c], None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA).astype(np.uint8)
        med = cv2.medianBlur(cv2.medianBlur(small, 31), 15)
        ref[..., c] = cv2.GaussianBlur(cv2.resize(med.astype(np.float32), (img.shape[1], img.shape[0])), (0, 0), 20)
    dev = np.maximum(lab[..., 2] - ref[..., 2] - 2, 0) / 6 + np.maximum(ref[..., 0] - lab[..., 0] - 10, 0) / 20
    sticker = cv2.dilate(((lab[..., 2] < 128) | (lab[..., 0] < 90)).astype(np.uint8), np.ones((9, 9), np.uint8))
    m = np.clip(dev, 0, 1) * zone * (1 - sticker)
    m = cv2.GaussianBlur(m, (0, 0), 6)[..., None] * (1 - BRUISE_KEEP)
    out = lab.copy()
    out[..., 1:] = lab[..., 1:] * (1 - m) + ref[..., 1:] * m      # colour of the bruise -> skin colour
    out[..., :1] = lab[..., :1] * (1 - m * 0.8) + ref[..., :1] * m * 0.8
    return cv2.cvtColor(out.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def warm(img, da=4.0, db=8.0, dl=2.0):
    """warmer, livelier skin: shift a/b towards peach on skin pixels only (briefs, jeans, stickers stay)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = np.clip((lab[..., 2] - 124) / 6, 0, 1) * np.clip((lab[..., 0] - 40) / 30, 0, 1)   # warm-ish pixels only:
    m = cv2.GaussianBlur(m, (0, 0), 2)                                                     # cool briefs/jeans/stickers stay
    lab[..., 0] += dl * m; lab[..., 1] += da * m; lab[..., 2] += db * m
    return cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def put_stickers(pil, name, x_off):
    sp = SPEC[name]; w = W // 2
    x0 = int(round(sp["cx"] * S)) - w // 2
    d = int(round(sp["d"] * S))
    st = Image.open(STICKER).convert("RGBA").resize((d, d), Image.LANCZOS)
    for x, y in sp["nipples"]:
        cx = x * S - x0 + x_off; cy = STICKER_Y + (y - sp["sticker"]) * S
        pil.alpha_composite(st, (int(round(cx - d / 2)), int(round(cy - d / 2))))


def main():
    before, after = cv2.imread(BEFORE), cv2.imread(AFTER)
    after = soften_bruises(after)
    after = match_skin(after, before)
    before, after = warm(before), warm(after)
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:, :W // 2] = panel("before", before)
    canvas[:, W // 2:] = panel("after", after)
    pil = Image.fromarray(canvas[..., ::-1]).convert("RGBA")
    put_stickers(pil, "before", 0); put_stickers(pil, "after", W // 2)
    d = ImageDraw.Draw(pil)
    ft = ImageFont.truetype(BEBAS, 84)
    d.text(((W - d.textlength(TITLE, font=ft)) / 2, TITLE_Y), TITLE, font=ft, fill=(255, 255, 255))
    pil.convert("RGB").save(OUT, quality=95)
    print("wrote", OUT)


main()
