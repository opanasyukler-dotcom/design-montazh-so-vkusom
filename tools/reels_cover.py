"""Reels cover 1080x1920 in the brand style: photo cropped to 9:16, soft INK shade at the top,
Biro Script title in BLUSH + thin Montserrat subtitle, kept inside the 3:4 profile-grid crop.
  python3 -I reels_cover.py photo.jpg fonts_dir out.jpg [title] [subtitle] [center_x 0..1]
"""
import sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
def hx(s): return np.array([int(s[i:i + 2], 16) for i in (1, 3, 5)]) / 255
INK, BLUSH, WHITE = hx("#2F0600"), hx("#F7DFDD"), np.ones(3)

def text_mask(txt, font, cx, baseline, dil=0):
    L = Image.new("L", (W, H), 0); ImageDraw.Draw(L).text((cx, baseline), txt, font=font, fill=255, anchor="ms")
    m = np.asarray(L).astype(np.float32) / 255
    return cv2.dilate(m, np.ones((dil, dil), np.uint8)) if dil else m

def comp(a, m, col, sh=0.45, r=10):
    s = np.minimum(1, cv2.GaussianBlur(m, (0, 0), r) * 1.4) * sh
    a = a * (1 - s[..., None])
    return a * (1 - m[..., None]) + col[None, None, :] * m[..., None]

def main(photo, fonts, out, title="мини влог", sub="с 4,5 месячным малышом", cxr=0.47):
    im = Image.open(photo).convert("RGB"); w, h = im.size
    cw = min(w, int(h * 9 / 16)); x0 = int(np.clip(cxr * w - cw / 2, 0, w - cw))
    im = im.crop((x0, 0, x0 + cw, int(cw * 16 / 9))).resize((W, H), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255
    y = np.linspace(0, 1, H)[:, None, None]; shade = np.clip(1 - y / 0.36, 0, 1) ** 1.6 * 0.55
    a = a * (1 - shade) + INK * shade
    fb = ImageFont.truetype(f"{fonts}/BiroScriptUSPlus-Regular.ttf", 200)
    fm = ImageFont.truetype(f"{fonts}/Montserrat-VariableFont_wght.ttf", 50); fm.set_variation_by_axes([300])
    a = comp(a, text_mask(title, fb, W // 2, 395, dil=3), BLUSH)
    a = comp(a, text_mask(sub, fm, W // 2, 470), WHITE, sh=0.5, r=6)
    Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out, quality=95, subsampling=0)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *sys.argv[4:6], *(float(v) for v in sys.argv[6:7]))
