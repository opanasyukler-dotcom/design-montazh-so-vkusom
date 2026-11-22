"""Trend «blur band under subtitles» — measured on assets/refs_nov/blur_band_ref.png (1320x2868):
- a HARD-edged rectangle (edge transition ≤ 4 px, corner radius ≈ 0.5 % of width), not feathered;
- height ≈ 8.4 % of the frame height (241 px), width = the text + generous side padding (≈ 66 % of the frame for a 3-word
  phrase), centred on the text;
- inside: the picture is smeared VERTICALLY (column streaks: vertical detail ≈ 0, colours change only across x) plus a
  moderate horizontal soften; no tint, no darkening — the band keeps the local colours of the frame;
- text: flat white, lowercase, Montserrat Alternates SemiBold (single-storey «а», wide geometric), x-height ≈ 1/5 of the band,
  no shadow, centred.
  from blur_band import blur_band; img = blur_band(img_rgb, y_center, "повторяй за мной")
  python3 -I blur_band.py IN.jpg OUT.jpg "повторяй за мной" [Y_FRACTION]
"""
import os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

FU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "user")

def band_font(W):
    return ImageFont.truetype(os.path.join(FU, "MontserratAlternates-SemiBold.ttf"), round(W * 0.057))

def blur_band(im, yc, text, font=None, fill=(255, 255, 255)):
    W, H = im.size; font = font or band_font(W)
    bh = round(H * 0.084); pad = round(W * 0.075)
    tw = font.getlength(text); x0 = round(max(W * 0.04, W / 2 - tw / 2 - pad)); x1 = round(min(W * 0.96, W / 2 + tw / 2 + pad))
    y0, y1 = round(yc - bh / 2), round(yc + bh / 2)
    a = np.asarray(im.convert("RGB")).copy()
    m = round(bh * 1.2); ya, yb = max(0, y0 - m), min(H, y1 + m); xa, xb = max(0, x0 - m), min(W, x1 + m)
    reg = a[ya:yb, xa:xb].astype(np.float32)
    ky = max(3, round(bh * 1.6)) | 1; kx = max(3, round(W * 0.03)) | 1        # long vertical smear, short horizontal soften
    reg = cv2.blur(reg, (1, ky)); reg = cv2.blur(reg, (1, ky)); reg = cv2.GaussianBlur(reg, (kx, 1), 0)
    blurred = a.copy(); blurred[ya:yb, xa:xb] = np.clip(reg, 0, 255).astype(np.uint8)
    mask = Image.new("L", (W * 4, H * 4), 0); r = round(W * 0.005) * 4            # hard rectangle, tiny radius, anti-aliased
    ImageDraw.Draw(mask).rounded_rectangle((x0 * 4, y0 * 4, x1 * 4, y1 * 4), r, fill=255)
    mask = mask.resize((W, H), Image.LANCZOS)
    out = Image.composite(Image.fromarray(blurred), im.convert("RGB"), mask)
    ImageDraw.Draw(out).text((W / 2, yc), text, font=font, fill=fill, anchor="mm")
    return out

if __name__ == "__main__":
    im = Image.open(sys.argv[1]).convert("RGB"); yf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
    blur_band(im, int(im.height * yf), sys.argv[3]).save(sys.argv[2], quality=92)
