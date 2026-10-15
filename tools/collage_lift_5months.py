"""Swap the AFTER (top) half of the «5 месяцев после операции» collages for new after photos.

usage: python3 collage_lift_5months.py SRC OUT
  SRC holds old_{front,34,profile}.jpg (1320x2346 collages: after on top, before below)
        and new_{front,34,profile}.jpg (new after photos on a dark-gray backdrop).
The before half is kept as is; the old caption at the seam is inpainted and drawn again over the new top.
"""
import os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SRC, OUT = sys.argv[1:3]
SEAM = 1173                     # first row of the before photo
INTER = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"

# new photo -> old top half: scale s, then the new point (nx, ny) lands on the old point (ox, oy)
# (front: sticker midpoint; 3/4 and profile: the near sticker; s from sticker spacing / torso width / sticker-to-waist)
FIT = {
    "front":   dict(s=0.61, new=(835, 750), old=(690, 410), chin=110, lines=["5 месяцев", "после операции"]),
    "34":      dict(s=0.62, new=(890, 620), old=(800, 340), chin=0, lines=["5 месяцев после", "операции"]),
    "profile": dict(s=0.76, new=(1100, 730), old=(880, 380), chin=225, lines=["5 месяцев после", "операции"]),
}


def body_mask(bgr):
    """skin / hair (brighter than the backdrop) + blue jeans (darker but bluish); backdrop -> 0"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    # the backdrop is a neutral gray (a~128, b~124-129, L<70); skin/hair are warm, jeans are blue
    m = ((a > 132) | (b < 120) | ((b < 123) & (L < 30)) | (b > 134) | (L > 95)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m, 8)
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > big * 0.02: keep[lab_ == i] = 1
    ff = np.pad(keep, 1); h, w = ff.shape                      # zero frame: the backdrop is one region around the body
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    keep[ff[1:-1, 1:-1] == 0] = 1                                # fill holes only
    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 2.5)
    # dark jeans in shadow next to an almost black backdrop: take what is brighter than the backdrop of that column
    Lf = L.astype(np.float32); bgL = np.array([np.median(Lf[keep[:, x] == 0, x]) if (keep[:, x] == 0).any() else 0
                                                for x in range(Lf.shape[1])], np.float32)
    bgL = cv2.GaussianBlur(bgL[None], (0, 0), 15)[0]
    near = cv2.dilate(keep, np.ones((61, 61), np.uint8)).astype(np.float32)
    extra = np.clip((Lf - bgL[None] - 3) / 8, 0, 1) * ((b < 126) | (a > 130)) * near
    extra = cv2.GaussianBlur(cv2.morphologyEx(extra, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), (0, 0), 2)
    # where the backdrop itself is almost black, keep the photo as is (black point below) instead of cutting
    dark = np.clip((18 - bgL) / 6, 0, 1)[None].repeat(Lf.shape[0], 0) * near
    return np.maximum(np.maximum(soft, extra), cv2.GaussianBlur(dark, (0, 0), 8))


def top_half(name, W, H):
    f = FIT[name]; s = f["s"]
    im = cv2.imread(os.path.join(SRC, f"new_{name}.jpg"))
    m = body_mask(im)
    im = np.clip((im.astype(np.float32) - 14) * 255 / 241, 0, 255).astype(np.uint8)   # black point: dark backdrop -> black
    # cut the face: fade to black just under the chin
    y = np.arange(im.shape[0], dtype=np.float32)[:, None]
    m = m * np.clip((y - f["chin"] - 10) / 40, 0, 1)
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA)
    tx = int(round(f["old"][0] - f["new"][0] * s)); ty = int(round(f["old"][1] - f["new"][1] * s))
    canvas = np.zeros((H, W, 3), np.float32); mc = np.zeros((H, W), np.float32)
    x0, y0 = max(tx, 0), max(ty, 0); x1, y1 = min(tx + im.shape[1], W), min(ty + im.shape[0], H)
    canvas[y0:y1, x0:x1] = im[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
    mc[y0:y1, x0:x1] = m[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
    return (canvas * mc[..., None]).astype(np.uint8)


def clean_bottom(old):
    """remove the old white caption from the before photo (rows SEAM..~1290)"""
    part = old[SEAM:SEAM + 150].copy()
    hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
    txt = ((hsv[..., 2] > 200) & (hsv[..., 1] < 45)).astype(np.uint8)
    txt[:, :250] = 0; txt[:, 1080:] = 0
    txt = cv2.dilate(txt, np.ones((9, 9), np.uint8))
    part = cv2.inpaint(part, txt, 9, cv2.INPAINT_TELEA)
    out = old.copy(); out[SEAM:SEAM + 150] = part
    return out


def caption(img, lines, cy=1145, size=84, lead=82):
    pil = Image.fromarray(img[..., ::-1])
    f = ImageFont.truetype(INTER, size)
    lay = Image.new("L", pil.size, 0); d = ImageDraw.Draw(lay)
    pos = []
    for i, t in enumerate(lines):
        w = d.textlength(t, font=f)
        pos.append(((pil.width - w) / 2, cy + (i - len(lines) / 2) * lead))
    for (x, y), t in zip(pos, lines): d.text((x, y), t, font=f, fill=255)
    shadow = lay.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.55))
    pil.paste((0, 0, 0), (0, 3), shadow)
    pil.paste((255, 255, 255), (0, 0), lay)
    return np.asarray(pil)[..., ::-1]


os.makedirs(OUT, exist_ok=True)
names = {"front": "1_anfas", "34": "2_tri_chetverti", "profile": "3_profil"}
for name, fn in names.items():
    old = cv2.imread(os.path.join(SRC, f"old_{name}.jpg"))
    H, W = old.shape[:2]
    out = clean_bottom(old)
    top = top_half(name, W, SEAM)
    out[:SEAM] = top
    # soft seam: top photo fades into black over the last rows
    k = np.clip((SEAM - np.arange(SEAM - 30, SEAM, dtype=np.float32)) / 30, 0, 1)[:, None, None]
    out[SEAM - 30:SEAM] = (out[SEAM - 30:SEAM] * k).astype(np.uint8)
    out = caption(out, FIT[name]["lines"])
    cv2.imwrite(os.path.join(OUT, fn + ".jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("wrote", fn)
