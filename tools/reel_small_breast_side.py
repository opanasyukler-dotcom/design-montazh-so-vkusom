# side-view ДО/ПОСЛЕ: crop above the neck, round ANNA TVAURI stickers on the nipple, warm the result, build the collage
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
SP = "/tmp/claude-0/-home-user-design-montazh-so-vkusom/b26b20a4-6ac5-50f8-8404-dcbad88c88d3/scratchpad/"
def osw(s): return ImageFont.truetype(SP + "v14/fonts/BN-Bold.ttf", s)
def mont(w, s): return ImageFont.truetype(SP + f"fonts2/M{w}.ttf", s)
NAVY = (16, 46, 70)
def sticker(r):
    S = 4 * r; im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = S // 2; R = 2 * r
    d.ellipse((c - R, c - R, c + R, c + R), fill=NAVY + (255,))
    d.ellipse((c - R + 9, c - R + 9, c + R - 9, c + R - 9), outline=(255, 255, 255, 200), width=3)
    f = mont(500, int(R * 0.36))
    d.text((c, c - R * 0.2), "ANNA", font=f, fill=(255, 255, 255, 255), anchor="mm"); d.text((c, c + R * 0.24), "TVAURI", font=f, fill=(255, 255, 255, 255), anchor="mm")
    return np.asarray(im.resize((2 * r, 2 * r), Image.LANCZOS)).astype(np.float32) / 255
def put(img, st, cx, cy):
    h, w = st.shape[:2]; x0, y0 = int(cx - w / 2), int(cy - h / 2)
    a = st[..., 3:4]; reg = img[y0:y0 + h, x0:x0 + w].astype(np.float32)
    img[y0:y0 + h, x0:x0 + w] = (reg * (1 - a) + st[..., :3] * 255 * a).astype(np.uint8)
def warm(img, k=1.0):
    f = img.astype(np.float32); g = f.mean(2, keepdims=True); f = g + (f - g) * (1 + 0.06 * k)
    f *= np.float32([1 + 0.07 * k, 1 + 0.025 * k, 1 - 0.09 * k]); return np.clip(f, 0, 255).astype(np.uint8)
def top_cut(img, y, f=60):                                              # nothing above the neck: chin/cap fade to black
    bg = np.float32([14, 14, 16]); m = np.clip((np.arange(img.shape[0], dtype=np.float32) - y) / f, 0, 1)[:, None, None]
    return (img * m + bg * (1 - m)).astype(np.uint8)
B = np.array(Image.open("dl/side_before.jpg").convert("RGB")); A = np.array(Image.open("dl/side_after.jpg").convert("RGB"))
put(B, sticker(82), 838, 702); put(A, sticker(82), 1012, 714)
def deepen(img):                                                        # grey studio wall -> close to black, skin untouched
    f = img.astype(np.float32); lum = f.mean(2, keepdims=True); k = np.clip((lum - 40) / 50, 0, 1)
    return np.clip(f * (0.45 + 0.55 * k), 0, 255).astype(np.uint8)
B = top_cut(deepen(B), 160); A = top_cut(deepen(warm(A)), 290)
cv2.imwrite("side_do.png", cv2.cvtColor(B, cv2.COLOR_RGB2BGR)); cv2.imwrite("side_po.png", cv2.cvtColor(A, cv2.COLOR_RGB2BGR))
# collage in the same layout as the first one: title on top, ДО left, ПОСЛЕ right
W, H = 1080, 1920; col = np.zeros((H, W, 3), np.uint8); col[:] = (8, 8, 9)
def panel(img, cx, x0):
    crop = img[200:2048, int(cx - 508):int(cx + 508)].astype(np.float32)          # 1016 x 1848
    h, w = crop.shape[:2]; m = np.ones((h, w), np.float32); e = 70
    m *= np.clip(np.arange(w) / e, 0, 1)[None] * np.clip((w - 1 - np.arange(w)) / e, 0, 1)[None]
    m *= (np.clip((h - 1 - np.arange(h)) / e, 0, 1) * np.clip(np.arange(h) / 40, 0, 1))[:, None]; crop = crop * m[..., None] + 8 * (1 - m[..., None])
    return cv2.resize(crop.astype(np.uint8), (526, 956), interpolation=cv2.INTER_AREA)
col[640:1596, 14:540] = panel(B, 645, 14); col[640:1596, 548:1074] = panel(A, 702, 548)
im = Image.fromarray(col); d = ImageDraw.Draw(im); f = osw(92)
d.text((540, 540), "УВЕЛИЧЕНИЕ ГРУДИ ИМПЛАНТАМИ", font=f, fill=(255, 255, 255), anchor="ms")
im.save("collage_side.jpg", quality=95)
