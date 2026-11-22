"""Trend «blur band under subtitles» (ref: assets/refs_nov/blur_band_ref.png): a wide horizontal strip of the frame behind
the subtitle is frosted (strong gaussian blur) and slightly darkened/warmed, edges feathered left-right and top-bottom so the
band melts into the picture; text on it is white, lowercase, a geometric sans (Montserrat 500–600 / Inter Tight), centred.
  from blur_band import blur_band; img = blur_band(img_rgb, y_center, text, font)
  python3 -I blur_band.py IN.jpg OUT.jpg "повторяй за мной"   — demo
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

def blur_band(im, yc, text=None, font=None, band_h=None, width=0.82, blur=None, tint=(70, 42, 26), tint_a=0.45, fill=(255, 255, 255)):
    W, H = im.size; band_h = band_h or int(H * 0.095); blur = blur or max(18, W // 40)
    x0, x1 = int(W * (1 - width) / 2), int(W * (1 + width) / 2); y0, y1 = int(yc - band_h / 2), int(yc + band_h / 2)
    pad = blur * 3; crop = im.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad)).filter(ImageFilter.GaussianBlur(blur))
    frost = Image.new("RGB", im.size); frost.paste(crop, (x0 - pad, y0 - pad))
    a = np.asarray(frost, np.float32); a = a * (1 - tint_a) + np.array(tint, np.float32) * tint_a; frost = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    m = np.zeros((H, W), np.float32); fx, fy = int(W * 0.06), int(band_h * 0.18)     # feathered mask: soft sides, softer top/bottom
    m[y0:y1, x0:x1] = 1
    mask = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(fy)).filter(ImageFilter.GaussianBlur(fx // 3))
    out = Image.composite(frost, im, mask)
    if text:
        d = ImageDraw.Draw(out); sh = Image.new("L", im.size, 0); ImageDraw.Draw(sh).text((W / 2, yc + 2), text, font=font, fill=110, anchor="mm")
        out.paste((20, 12, 8), (0, 0), sh.filter(ImageFilter.GaussianBlur(4))); d.text((W / 2, yc), text, font=font, fill=fill, anchor="mm")
    return out

if __name__ == "__main__":
    im = Image.open(sys.argv[1]).convert("RGB"); W, H = im.size
    f = ImageFont.truetype(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "stories", "Montserrat-VariableFont_wght.ttf"), int(W * 0.06)); f.set_variation_by_axes([560])
    blur_band(im, int(H * 0.6), sys.argv[3], f).save(sys.argv[2], quality=92)
