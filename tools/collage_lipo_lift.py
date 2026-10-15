"""Before/after collages 4:5 (1080x1350) "Липофилинг + подтяжка груди".

Left = ДО, right = ПОСЛЕ.  Each pair is scaled to the same body size (shoulder line -> reference line),
everything above the chin line is cut away, the backdrop becomes pure black, the "before" skin is
colour-matched to the "after" photo (LAB mean/std transfer on skin pixels) and the clinic's nipple
stickers are lifted from the "after" photos and placed on the "before" nipples.

usage: python3 collage_lipo_lift.py SRC_DIR OUT_DIR
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
        b=dict(ref=(0.15, 0.83), cx=0.51, cut=0.08, nip=[(0.3825, 0.373, "a0"), (0.6625, 0.390, "a1")]),
        a=dict(ref=(0.14, 0.78), cx=0.52, cut=0.06, floor=0.945, stk=[(0.294, 0.320), (0.680, 0.320)]), span=690, yref=TOP + 150),
    "profile": dict(
        b=dict(ref=(0.09, 0.86), cx=0.52, cut=0.04, nip=[(0.700, 0.333, "a0")]),
        a=dict(ref=(0.155, 0.85), cx=0.47, cut=0.12, stk=[(0.655, 0.340)]), span=690, yref=TOP + 150),
    "34": dict(
        b=dict(ref=(0.10, 0.545), cx=0.50, cut=0.06, nip=[(0.567, 0.343, "a0"), (0.750, 0.352, "pf")]),
        a=dict(ref=(0.155, 0.55), cx=0.55, cut=0.125, floor=0.925, stk=[(0.553, 0.345)]), span=455, yref=TOP + 150),
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
    cloth = ((hsv[..., 1] < 22) & (hsv[..., 2] > 185)).astype(np.float32)   # stockings / panties stay white
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


def place(rgb, spec, span, yref, mask):
    """scale/translate one photo into a half canvas; black above the chin cut"""
    h, w = rgb.shape[:2]
    s = span / ((spec["ref"][1] - spec["ref"][0]) * h)
    tx = HW / 2 - spec["cx"] * w * s; ty = yref - spec["ref"][0] * h * s
    M = np.float32([[s, 0, tx], [0, s, ty]])
    img = cv2.warpAffine(rgb, M, (HW, H), flags=cv2.INTER_AREA, borderValue=(0, 0, 0)).astype(np.float32)
    m = cv2.warpAffine(mask, M, (HW, H), flags=cv2.INTER_LINEAR, borderValue=0)
    ycut = spec["cut"] * h * s + ty
    ys = np.arange(H, dtype=np.float32); cut = np.clip((ys - ycut) / 40, 0, 1)[:, None]
    if "floor" in spec:                                   # the white floor strip under the backdrop
        yf = spec["floor"] * h * s + ty; cut = cut * np.clip((yf - ys) / 6, 0, 1)[:, None]
    m = m * cut; m[:TOP] = 0
    return img * m[..., None], (s, tx, ty)


def chip(text, fill):
    f = ImageFont.truetype(BEBAS, 44); w = int(f.getlength(text)) + 48
    im = Image.new("RGBA", (w, 60), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, 59), radius=30, fill=fill + (240,))
    d.text((w / 2, 32), text, font=f, fill=GRAPHITE + (255,), anchor="mm"); return im


def collage(name):
    cfg = P[name]
    b, a = load("b_" + name), load("a_" + name)
    stickers = [lift_sticker(a, x * a.shape[1], y * a.shape[0], dark=name != "profile") for x, y in cfg["a"]["stk"]]
    prof = load("a_profile"); pf = lift_sticker(prof, P["profile"]["a"]["stk"][0][0] * prof.shape[1], P["profile"]["a"]["stk"][0][1] * prof.shape[0], dark=False)
    b = color_match(b, a)
    hb, tb = place(b, cfg["b"], cfg["span"], cfg["yref"], person_mask(b))
    ha, ta = place(a, cfg["a"], cfg["span"], cfg["yref"], person_mask(a))
    left = Image.fromarray(np.clip(hb, 0, 255).astype(np.uint8)).convert("RGBA")
    # stickers on the before nipples, at the size they have on the after photo
    for x, y, which in cfg["b"]["nip"]:
        st = pf if which == "pf" else stickers[int(which[1:])]
        sa = (P["profile"] if which == "pf" else cfg)  # whose scale the sticker comes from
        s_after = cfg["span"] / ((sa["a"]["ref"][1] - sa["a"]["ref"][0]) * (prof if which == "pf" else a).shape[0])
        st = st.resize((max(1, int(st.width * s_after)), max(1, int(st.height * s_after))), Image.LANCZOS)
        px = x * b.shape[1] * tb[0] + tb[1]; py = y * b.shape[0] * tb[0] + tb[2]
        left.alpha_composite(st, (int(px - st.width / 2), int(py - st.height / 2)))
    right = Image.fromarray(np.clip(ha, 0, 255).astype(np.uint8)).convert("RGBA")
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    canvas.alpha_composite(left, (0, 0)); canvas.alpha_composite(right, (W - HW, 0))
    d = ImageDraw.Draw(canvas)
    d.line((W // 2, TOP + 40, W // 2, H - 40), fill=(60, 60, 66, 255), width=2)
    f = ImageFont.truetype(BEBAS, 84)                     # Bebas Cyrillic draws "+" as "±", so the plus is drawn apart
    fp = ImageFont.truetype(os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf"), 60); fp.set_variation_by_axes([700])
    a_, b_ = "ЛИПОФИЛИНГ", "ПОДТЯЖКА ГРУДИ"; gap = 22
    wa, wb, wp = f.getlength(a_), f.getlength(b_), fp.getlength("+")
    x = W / 2 - (wa + wb + wp + 2 * gap) / 2
    d.text((x, 70), a_, font=f, fill=WHITE + (255,), anchor="lm"); x += wa + gap
    d.text((x, 66), "+", font=fp, fill=LILAC + (255,), anchor="lm"); x += wp + gap
    d.text((x, 70), b_, font=f, fill=WHITE + (255,), anchor="lm")
    d.rectangle((W / 2 - 80, 122, W / 2 + 80, 125), fill=LILAC + (255,))
    canvas.alpha_composite(chip("ДО", POWDER), (40, H - 100))
    canvas.alpha_composite(chip("ПОСЛЕ", LILAC), (W - HW + 40, H - 100))
    out = os.path.join(OUT, f"lipofiling_podtyazhka_{name}.jpg")
    canvas.convert("RGB").save(out, quality=95); print("wrote", out)


for n in ["front", "34", "profile"]: collage(n)
