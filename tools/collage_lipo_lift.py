"""Before/after collages 4:5 (1080x1350) "Липофилинг + подтяжка груди".

Left = ДО, right = ПОСЛЕ.  Each pair is scaled to the same body size (shoulder line -> reference line),
everything above the chin line is cut away, the backdrop becomes pure black, the "before" skin is
colour-matched to the "after" photo (LAB mean/std transfer on skin pixels) and the clinic's nipple
clinic's round sticker (collage_lipofiling_lift/sticker_anna_tvauri.png) covers the nipples on the "before" photos
only, clipped to the body outline; the "after" photos keep their own stickers.

usage: python3 collage_lipo_lift.py SRC_DIR OUT_DIR [lr|tb]   (lr = 4:5 before|after, tb = 9:16 after over before)
SRC_DIR holds b_front/b_profile/b_34 and a_front/a_profile/a_34 (.jpg)
"""
import os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1080, 1350
TOP = 150                       # title band
HW = (W - 4) // 2               # half width
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEBAS = os.path.join(ROOT, "skills/montage-svg-typography/Шрифты/Коллекция шрифтов/BebasNeueCyrillic.ttf")
WHITE, LILAC, POWDER, GRAPHITE = (255, 255, 255), (252, 211, 255), (233, 207, 199), (20, 23, 31)
SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# per photo (fractions of width/height): reference line top/bottom, body centre x, chin cut, stickers
# stickers on "after" = where to lift them from; on "before" = nipple positions + which lifted sticker to use
P = {
    "front": dict(
        b=dict(ref=(0.15, 0.83), cx=0.51, cut=0.08, nip=[(0.3825, 0.373, "a0"), (0.662, 0.390, "a1")]),
        a=dict(ref=(0.14, 0.78), cx=0.52, cut=0.06, floor=0.945, stk=[(0.294, 0.320), (0.685, 0.325)]), span=690, yref=TOP + 150),
    "profile": dict(
        b=dict(ref=(0.09, 0.86), cx=0.52, cut=0.04, nip=[(0.722, 0.331, "a0")]),
        a=dict(ref=(0.155, 0.85), cx=0.47, cut=0.12, stk=[(0.655, 0.340)]), span=690, yref=TOP + 150),
    "34": dict(
        b=dict(ref=(0.10, 0.545), cx=0.50, cut=0.06, nip=[(0.565, 0.342, "a0"), (0.751, 0.350, "pf")]),
        a=dict(ref=(0.155, 0.55), cx=0.55, cut=0.125, floor=0.925, stk=[(0.553, 0.345), (0.712, 0.345)]), span=455, yref=TOP + 150),
}


def load(n): return np.asarray(Image.open(os.path.join(SRC, n + ".jpg")).convert("RGB"))


def person_mask(rgb):
    """everything that is not the dark backdrop (largest blobs, holes filled, soft edge)"""
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    m = (L > 62).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    big = st[1:, cv2.CC_STAT_AREA].max()
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > big * 0.02: keep[lab == i] = 1
    ff = keep.copy(); h, w = ff.shape; fm = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, fm, (0, 0), 1); holes = (ff == 0)
    keep[holes] = 1
    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 2.0)
    return np.clip(soft * np.clip((L - 30) / 40, 0, 1) + soft * (L > 70), 0, 1)


def skin_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    return ((hsv[..., 0] < 25) & (hsv[..., 1] > 45) & (hsv[..., 2] > 90)).astype(np.float32)


def color_match(src, ref):
    """move the before photo's skin tone onto the after photo's (LAB mean/std on skin pixels)"""
    ms, mr = skin_mask(src) > 0, skin_mask(ref) > 0
    ls = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32); lr = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = ls.copy()
    for c in range(3):
        a, b = ls[..., c][ms], lr[..., c][mr]
        k = np.clip(b.std() / max(a.std(), 1e-3), 0.8, 1.25)
        out[..., c] = (ls[..., c] - a.mean()) * k + b.mean()
    res = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV).astype(np.float32)
    cloth = ((hsv[..., 1] < 22) & (hsv[..., 2] > 185)).astype(np.uint8)     # stockings / panties stay white
    n, lab, st, _ = cv2.connectedComponentsWithStats(cloth, 8)                # ...but small skin highlights are not cloth
    big = np.zeros(n, bool); big[1:] = st[1:, cv2.CC_STAT_AREA] > cloth.size * 0.004
    cloth = big[lab].astype(np.float32)
    w = 1 - cv2.GaussianBlur(cloth, (0, 0), 3)
    w = (w * person_mask(src))[..., None]
    return (src * (1 - w) + res * w).astype(np.uint8)


def lift_sticker(rgb, cx, cy, dark=True):
    """navy round sticker around (cx, cy) px -> RGBA crop (ellipse mask)"""
    r = int(rgb.shape[1] * 0.09)
    x0, y0 = int(cx - r), int(cy - r)
    box = rgb[y0:y0 + 2 * r, x0:x0 + 2 * r].astype(np.int32)
    hsv = cv2.cvtColor(box.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.int32)
    # navy body, its glossy highlight and the white lettering: everything that is not skin-toned
    navy = (((hsv[..., 0] > 95) & (hsv[..., 0] < 140) & (hsv[..., 1] > 40)) | ((hsv[..., 2] < 70) & dark)).astype(np.uint8)
    navy = cv2.morphologyEx(navy, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    cnts, _ = cv2.findContours(navy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mid = np.array([r, r])
    c = min((k for k in cnts if cv2.contourArea(k) > 400), key=lambda k: np.linalg.norm(k.reshape(-1, 2).mean(0) - mid))
    c = cv2.convexHull(c)
    (ex, ey), (ew, eh), ang = cv2.fitEllipse(c)
    m = np.zeros(navy.shape, np.uint8)
    cv2.ellipse(m, ((ex, ey), (ew * 0.98, eh * 0.98), ang), 255, -1)
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    x, y, w, h = cv2.boundingRect((m > 8).astype(np.uint8))
    rgba = np.dstack([box.astype(np.uint8), m])[y:y + h, x:x + w]
    return Image.fromarray(rgba, "RGBA")


def place(rgb, spec, span, yref, mask, cw=HW, ch=H, top=TOP):
    """scale/translate one photo into a half canvas (cw x ch); black above the chin cut"""
    h, w = rgb.shape[:2]
    s = span / ((spec["ref"][1] - spec["ref"][0]) * h)
    tx = cw / 2 - spec["cx"] * w * s; ty = yref - spec["ref"][0] * h * s
    M = np.float32([[s, 0, tx], [0, s, ty]])
    img = cv2.warpAffine(rgb, M, (cw, ch), flags=cv2.INTER_AREA, borderValue=(0, 0, 0)).astype(np.float32)
    m = cv2.warpAffine(mask, M, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=0)
    ycut = spec["cut"] * h * s + ty
    ys = np.arange(ch, dtype=np.float32); cut = np.clip((ys - ycut) / 40, 0, 1)[:, None]
    if "floor" in spec:                                   # the white floor strip under the backdrop
        yf = spec["floor"] * h * s + ty; cut = cut * np.clip((yf - ys) / 6, 0, 1)[:, None]
    m = m * cut; m[:top] = 0
    return img * m[..., None], (s, tx, ty), m


NAVY = (30, 56, 110)
STK_D = {"front": 0.072, "profile": 0.07, "34": 0.095}    # sticker covers the whole areola (share of the body span)
STICKER = os.path.join(ROOT, "collage_lipofiling_lift/sticker_anna_tvauri.png")


def logo_sticker(d):
    """the clinic's round sticker (PNG with a black surround) cut to its circle, d px wide"""
    im = Image.open(STICKER).convert("RGB"); n = min(im.size); im = im.crop((0, 0, n, n))
    k = 4; big = im.resize((d * k, d * k), Image.LANCZOS)
    m = Image.new("L", big.size, 0); ImageDraw.Draw(m).ellipse((k, k, d * k - 1 - k, d * k - 1 - k), fill=255)
    big.putalpha(m); return big.resize((d, d), Image.LANCZOS)


def brand_sticker(d):
    """flat clinic sticker: navy disc, faint inner ring, ANNA / TVAURI in white (as on the clinic's collages)"""
    k = 4; D = d * k
    im = Image.new("RGBA", (D, D), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    dr.ellipse((0, 0, D - 1, D - 1), fill=NAVY + (255,))
    r = int(D * 0.05); dr.ellipse((r, r, D - 1 - r, D - 1 - r), outline=(150, 170, 210, 120), width=max(1, D // 90))
    f = ImageFont.truetype(os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf"), int(D * 0.21)); f.set_variation_by_axes([500])
    dr.text((D / 2, D * 0.40), "ANNA", font=f, fill=WHITE + (255,), anchor="mm")
    dr.text((D / 2, D * 0.62), "TVAURI", font=f, fill=WHITE + (255,), anchor="mm")
    return im.resize((d, d), Image.LANCZOS)


def stick(half, mask, pts, d):
    """stickers centred on pts, clipped to the body silhouette (a sticker on the breast edge is cut by it)"""
    st = logo_sticker(d)
    layer = Image.new("RGBA", half.size, (0, 0, 0, 0))
    for px, py in pts: layer.alpha_composite(st, (int(px - d / 2), int(py - d / 2)))
    a = np.asarray(layer).astype(np.float32)
    body = cv2.morphologyEx((mask > 0.25).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))  # dark old stickers are not holes
    a[..., 3] *= cv2.GaussianBlur(body.astype(np.float32), (0, 0), 1.0)
    half.alpha_composite(Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGBA"))


def chip(text, fill):
    f = ImageFont.truetype(BEBAS, 44); w = int(f.getlength(text)) + 48
    im = Image.new("RGBA", (w, 60), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, 59), radius=30, fill=fill + (240,))
    d.text((w / 2, 32), text, font=f, fill=GRAPHITE + (255,), anchor="mm"); return im


def title(d, cw, y=70):
    f = ImageFont.truetype(BEBAS, 84)                     # Bebas Cyrillic draws "+" as "±", so the plus is drawn apart
    fp = ImageFont.truetype(os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf"), 60); fp.set_variation_by_axes([700])
    a_, b_ = "ЛИПОФИЛИНГ", "ПОДТЯЖКА ГРУДИ"; gap = 22
    wa, wb, wp = f.getlength(a_), f.getlength(b_), fp.getlength("+")
    x = cw / 2 - (wa + wb + wp + 2 * gap) / 2
    d.text((x, y), a_, font=f, fill=WHITE + (255,), anchor="lm"); x += wa + gap
    d.text((x, y - 4), "+", font=fp, fill=LILAC + (255,), anchor="lm"); x += wp + gap
    d.text((x, y), b_, font=f, fill=WHITE + (255,), anchor="lm")
    d.rectangle((cw / 2 - 80, y + 52, cw / 2 + 80, y + 55), fill=LILAC + (255,))


def collage(name, layout="lr"):
    """lr: 4:5, before left / after right.  tb: 9:16, after on top / before below."""
    cfg = P[name]
    if layout == "tb":
        CH = (1920 - TOP) // 2; k = 640 / 690              # title band sits between the halves
        span, yref, cw, ch, top = cfg["span"] * k, 110, W, CH, 0
    else:
        span, yref, cw, ch, top = cfg["span"], cfg["yref"], HW, H, TOP
    b, a = load("b_" + name), load("a_" + name)
    b = color_match(b, a)
    hb, tb, mb = place(b, cfg["b"], span, yref, person_mask(b), cw, ch, top)
    ha, ta, ma = place(a, cfg["a"], span, yref, person_mask(a), cw, ch, top)
    d = int(round(STK_D[name] * span))
    left = Image.fromarray(np.clip(hb, 0, 255).astype(np.uint8)).convert("RGBA")
    stick(left, mb, [(x * b.shape[1] * tb[0] + tb[1], y * b.shape[0] * tb[0] + tb[2]) for x, y, _ in cfg["b"]["nip"]], d)
    right = Image.fromarray(np.clip(ha, 0, 255).astype(np.uint8)).convert("RGBA")
    if layout == "tb":
        canvas = Image.new("RGBA", (W, 1920), (0, 0, 0, 255))
        canvas.alpha_composite(right, (0, 0)); canvas.alpha_composite(left, (0, ch + TOP))
        d = ImageDraw.Draw(canvas)
        title(d, W, ch + TOP // 2 - 14)
        canvas.alpha_composite(chip("ПОСЛЕ", LILAC), (40, 40))
        canvas.alpha_composite(chip("ДО", POWDER), (40, ch + TOP + 20))
        out = os.path.join(OUT, f"lipofiling_podtyazhka_{name}_9x16.jpg")
    else:
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        canvas.alpha_composite(left, (0, 0)); canvas.alpha_composite(right, (W - HW, 0))
        d = ImageDraw.Draw(canvas)
        d.line((W // 2, TOP + 40, W // 2, H - 40), fill=(60, 60, 66, 255), width=2)
        title(d, W)
        canvas.alpha_composite(chip("ДО", POWDER), (40, H - 100))
        canvas.alpha_composite(chip("ПОСЛЕ", LILAC), (W - HW + 40, H - 100))
        out = os.path.join(OUT, f"lipofiling_podtyazhka_{name}.jpg")
    canvas.convert("RGB").save(out, quality=95); print("wrote", out)


LAYOUT = sys.argv[3] if len(sys.argv) > 3 else "lr"
for n in ["front", "34", "profile"]: collage(n, LAYOUT)

