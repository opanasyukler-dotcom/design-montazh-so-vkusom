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
BG = np.float32([10, 10, 12])
def half(name):
    img = np.array(Image.open(f"dl/{name}.jpg").convert("RGB")).astype(np.float32)
    top, bot, cx = SPAN[name]; s = 900 / (bot - top)
    M = np.float32([[s, 0, 540 - s * cx], [0, s, 30 - s * top]])
    out = cv2.warpAffine(img, M, (W, HALF), flags=cv2.INTER_AREA, borderValue=tuple(map(float, BG)))
    y = np.arange(HALF, dtype=np.float32)[:, None, None]
    x = np.arange(W, dtype=np.float32)[None, :, None]; xl, xr = 540 - s * cx, 540 + s * (img.shape[1] - cx)
    m = np.clip((y - 30) / 22 + 0.5, 0, 1) * np.clip((930 - y) / 22 + 0.5, 0, 1)        # soft cut at the neck and under the fingertips
    m = m * np.clip((x - xl) / 60, 0, 1) * np.clip((xr - x) / 60, 0, 1)                  # photo edges melt into the black
    return out * m + BG * (1 - m)
LOGO = Image.open("logo_clean.png").convert("RGBA"); LOGO = LOGO.resize((300, int(LOGO.size[1] * 300 / LOGO.size[0])), Image.LANCZOS)
for ang in ("f", "q", "s"):
    fr = np.concatenate([half(f"do_{ang}"), half(f"po_{ang}")], 0)
    fr[HALF - 2:HALF + 2] = 235                                                      # thin divider
    im = Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im); f = osw(60)
    d.text((44, 60 + 52), "ДО", font=f, fill=(255, 255, 255), anchor="ls")
    d.text((44, HALF + 60 + 52), "ПОСЛЕ", font=f, fill=(228, 210, 210), anchor="ls")
    im.paste(LOGO, ((W - LOGO.size[0]) // 2, 40), LOGO)
    im.save(f"collage_{ang}.jpg", quality=95)
