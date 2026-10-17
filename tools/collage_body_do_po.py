# ДО/ПОСЛЕ collages: ДО on top, ПОСЛЕ below, each cropped from the neck down to the fingertips
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
SP = "/tmp/claude-0/-home-user-design-montazh-so-vkusom/b26b20a4-6ac5-50f8-8404-dcbad88c88d3/scratchpad/"
def osw(s): return ImageFont.truetype(SP + "v14/fonts/BN-Bold.ttf", s)
W, H, HALF = 1080, 1920, 960
# (top just under the chin, bottom just under the fingertips, body centre x) in source pixels
SPAN = {"do_f": (262, 1930, 740), "po_f": (8, 1480, 721),
        "do_q": (190, 2080, 721), "po_q": (222, 1850, 721),
        "do_s": (185, 2090, 721), "po_s": (262, 1800, 773)}
from retouch import skin_mask
def skin_stats(rgb):
    m = skin_mask(rgb)[..., 0] > 0.5; return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[m].astype(np.float32).mean(0)
def match_to(rgb, ref_rgb):
    """ПОСЛЕ skin brought to the brightness and tone of ДО (same angle); dark underwear stays dark"""
    mu, mr = skin_stats(rgb), skin_stats(ref_rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); out = lab.copy()
    out[..., 0] *= mr[0] / mu[0]; out[..., 1:] += mr[1:] - mu[1:]
    k = np.clip((lab[..., 0] - 50) / 40, 0, 1)[..., None]                                # skin, not the black briefs
    return cv2.cvtColor(np.clip(lab * (1 - k) + out * k, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
BG = np.float32([0, 0, 0])                                         # pure black, like the example
def half(name):
    img = np.array(Image.open(f"dl/{name}.jpg").convert("RGB"))
    img = img.astype(np.float32)
    m = np.load(f"mask_{name}.npy").astype(np.float32)                                 # person mask (mediapipe selfie segmenter)
    m = cv2.GaussianBlur(np.clip((m - 0.3) / 0.4, 0, 1), (0, 0), 1.5)[..., None]
    img = img * m                                                                       # wall, floor and baseboard -> black
    top, bot, cx = SPAN[name]; s = 900 / (bot - top)
    M = np.float32([[s, 0, 540 - s * cx], [0, s, 30 - s * top]])
    out = cv2.warpAffine(img, M, (W, HALF), flags=cv2.INTER_AREA, borderValue=tuple(map(float, BG)))
    y = np.arange(HALF, dtype=np.float32)[:, None, None]
    x = np.arange(W, dtype=np.float32)[None, :, None]; xl, xr = 540 - s * cx, 540 + s * (img.shape[1] - cx)
    m = np.clip((y - 30) / 22 + 0.5, 0, 1) * np.clip((930 - y) / 22 + 0.5, 0, 1)        # soft cut at the neck and under the fingertips

    return out * m + BG * (1 - m)
def pill(d, x, y, txt, col):                                                       # label pill as in the example
    f = osw(50); tw = f.getlength(txt); h = 56; w = int(tw + 52)
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=col)
    d.text((x + w / 2, y + h / 2 + 2), txt, font=f, fill=(28, 24, 40), anchor="mm")
LOGO = Image.open("logo_clean.png").convert("RGBA"); LOGO = LOGO.resize((300, int(LOGO.size[1] * 300 / LOGO.size[0])), Image.LANCZOS)
for ang in ("f", "q", "s"):
    top = np.clip(half(f"do_{ang}"), 0, 255).astype(np.uint8); bot = np.clip(half(f"po_{ang}"), 0, 255).astype(np.uint8)
    for _ in range(2): bot = match_to(bot, top)                                          # match on exactly what is shown
    fr = np.concatenate([top, bot], 0).astype(np.float32)
    fr[HALF - 3:HALF + 3] = (34, 34, 44)                                              # thin dark divider instead of the white line
    im = Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    pill(d, 54, 40, "ДО", (232, 212, 206)); pill(d, 54, HALF + 40, "ПОСЛЕ", (233, 214, 242))
    cap = "4,5 МЕСЯЦА ПОСЛЕ ОПЕРАЦИИ"; f = osw(54); tw = f.getlength(cap)                # caption on the seam
    arr = np.asarray(im).astype(np.float32)
    lay = Image.new("L", (W, H), 0); ImageDraw.Draw(lay).text((W / 2, HALF + 2), cap, font=f, fill=255, anchor="mm")
    t = np.asarray(lay).astype(np.float32) / 255
    sh = np.clip(cv2.GaussianBlur(cv2.dilate(t, np.ones((9, 9), np.uint8)), (0, 0), 10) * 1.6, 0, 1)[..., None]   # soft dark glow, no hard box
    band = np.zeros((H, W), np.float32); band[HALF - 3:HALF + 3, int(W / 2 - tw / 2 - 26):int(W / 2 + tw / 2 + 26)] = 1   # gap in the divider
    arr = arr * (1 - sh) ; arr[band > 0] = 0; arr = arr * (1 - t[..., None]) + 255 * t[..., None]
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    im.save(f"collage_{ang}.jpg", quality=95)
