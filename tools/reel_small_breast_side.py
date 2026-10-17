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
from retouch import retouch
def debruise(rgb, cx=880, cy=760, ax=200, ay=230):
    """soften bruises, tape and marker lines on the side view: skin colour is evened out, texture kept"""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    ref = lab[420:560, 720:930].reshape(-1, 3); ref = ref[ref[:, 0] > 120]; a_ref, b_ref = np.median(ref[:, 1]), np.median(ref[:, 2])
    m = np.zeros(L.shape, np.float32); cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)
    m *= (L > 90).astype(np.float32); m = cv2.GaussianBlur(m, (0, 0), 25)
    lowL = cv2.GaussianBlur(L, (0, 0), 3); hi = L - lowL; bigL = cv2.GaussianBlur(L, (0, 0), 40)
    L2 = bigL + (lowL - bigL) * 0.45 + hi                               # strip and lines fade, pores stay
    a2 = a_ref + (a - a_ref) * 0.25; b2 = b_ref + (b - b_ref) * 0.25    # yellow/green/blue tints -> skin tone
    out = np.stack([L2, a2, b2], -1); k = m[..., None] * 0.85
    return cv2.cvtColor(np.clip(lab * (1 - k) + out * k, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
B = np.array(Image.open("dl/side_before.jpg").convert("RGB")); A = np.array(Image.open("dl/side_after.jpg").convert("RGB"))
A = retouch(debruise(A), k_red=0.35, k_band=0.3, k_tex=0.05)                 # light retouch like the 3/4 view
put(B, sticker(82), 838, 702); put(A, sticker(82), 1012, 714)
def deepen(img):                                                        # grey studio wall -> close to black, skin untouched
    f = img.astype(np.float32); lum = f.mean(2, keepdims=True); k = np.clip((lum - 40) / 50, 0, 1)
    return np.clip(f * (0.45 + 0.55 * k), 0, 255).astype(np.uint8)
from retouch import skin_mask
def match_skin(src, ref):
    """bring the skin tone of `src` to the one of `ref` (same light and colour on both sides of the collage)"""
    ms, mr = skin_mask(src)[..., 0], skin_mask(ref)[..., 0]
    ls = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32); lr = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    mu_s = ls[ms > 0.5].mean(0); mu_r = lr[mr > 0.5].mean(0)
    k = cv2.GaussianBlur((ls[..., 0] > 70).astype(np.float32), (0, 0), 15)[..., None]   # body only, the black background stays black
    out = ls.copy(); out[..., 0] = ls[..., 0] * (mu_r[0] / mu_s[0]); out[..., 1:] = ls[..., 1:] + (mu_r[1:] - mu_s[1:])
    out = ls * (1 - k) + out * k
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
A = match_skin(A, B)
B = top_cut(deepen(warm(B, 0.6)), 130); A = top_cut(deepen(warm(A, 0.6)), 215)          # same gentle warmth on both
cv2.imwrite("side_do.png", cv2.cvtColor(B, cv2.COLOR_RGB2BGR)); cv2.imwrite("side_po.png", cv2.cvtColor(A, cv2.COLOR_RGB2BGR))
# collage in the same layout as the first one: title on top, ДО left, ПОСЛЕ right
W, H = 1080, 1920; col = np.zeros((H, W, 3), np.uint8); col[:] = (8, 8, 9)
def panel(img, cx, x0):
    crop = img[110:2048, int(cx - 508):int(cx + 508)].astype(np.float32)          # 1016 x 1848
    h, w = crop.shape[:2]; m = np.ones((h, w), np.float32); e = 70
    m *= np.clip(np.arange(w) / e, 0, 1)[None] * np.clip((w - 1 - np.arange(w)) / e, 0, 1)[None]
    m *= (np.clip((h - 1 - np.arange(h)) / e, 0, 1) * np.clip(np.arange(h) / 40, 0, 1))[:, None]; crop = crop * m[..., None] + 8 * (1 - m[..., None])
    return cv2.resize(crop.astype(np.uint8), (526, 1001), interpolation=cv2.INTER_AREA)
col[600:1601, 14:540] = panel(B, 645, 14); col[600:1601, 548:1074] = panel(A, 702, 548)
im = Image.fromarray(col); d = ImageDraw.Draw(im); f = osw(92)
d.text((540, 540), "УВЕЛИЧЕНИЕ ГРУДИ ИМПЛАНТАМИ", font=f, fill=(255, 255, 255), anchor="ms")
im.save("collage_side.jpg", quality=95)
