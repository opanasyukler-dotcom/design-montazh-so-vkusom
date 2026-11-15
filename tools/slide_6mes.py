"""@mcg.va — single slide «День с нами / малышке почти 6 месяцев!» (1080x1350) in her fonts: Inter (thin) + Lora Italic.
The text sits on the calm light table under the mug and the Kinder bar.
  python3 -I slide_6mes.py PHOTO OUT.jpg
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont, ImageOps
FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "user")
W, H = 1080, 1350
INK, BROWN = (43, 31, 26), (107, 74, 56)

def font(name, size, axes):
    f = ImageFont.truetype(os.path.join(FD, name), size); f.set_variation_by_axes(axes); return f

def main(src, out):
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB"); s = W / im.width
    im = im.resize((W, round(im.height * s)), Image.LANCZOS); im = im.crop((0, im.height - H, W, im.height))   # keep the table, trim the wall
    d = ImageDraw.Draw(im)
    title = font("Inter-Variable.ttf", 96, [32, 250]); sub = font("Lora-Italic-Variable.ttf", 54, [400])
    d.text((W / 2, 1190), "День с нами", font=title, fill=INK, anchor="ms")
    d.text((W / 2, 1275), "малышке почти 6 месяцев!", font=sub, fill=BROWN, anchor="ms")
    im.save(out, quality=95); print(out)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
