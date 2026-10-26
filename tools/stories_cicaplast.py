"""@mcg.va — «крем на холода» (La Roche-Posay Cicaplast B5+), one collage story 1080x1920.
Two photos stacked full-bleed with a thin white seam; a white rounded card with a soft shadow sits on the seam.
Fonts (stories_kit): main = Inter Bold, second = thin Montserrat.
  python3 -I stories_cicaplast.py box.png tube.png out.jpg
"""
import os, sys
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import W, H, INK, font, wrap
from stories_walk_look import shadowed

SEAM = 8

def crop(ph, h, cy):
    """photo scaled to the story width, a window of height h centred at cy (0..1 of the photo)"""
    s = W / ph.width; p = ph.resize((W, round(ph.height * s)), Image.LANCZOS)
    y = int(min(max(cy * p.height - h / 2, 0), p.height - h))
    return p.crop((0, y, W, y + h))

def story(box, tube, out):
    img = Image.new("RGB", (W, H), (255, 255, 255)); half = (H - SEAM) // 2
    img.paste(crop(box, half, 0.52), (0, 0)); img.paste(crop(tube, half, 0.47), (0, half + SEAM))
    fm, fs = font("main", 46), font("second", 38)
    main = wrap("Купила на холода классный крем от раздражений на коже,", fm, 800)
    sec = wrap("а то у Агаты уже от холода и отопления красные щёки", fs, 800)
    h = 60 + 62 * len(main) + 18 + 52 * len(sec) + 44; y0 = H // 2 - h // 2
    shadowed(img, (70, y0, W - 70, y0 + h), 40, blur=30, off=12, alpha=0.22)
    d = ImageDraw.Draw(img); y = y0 + 60 + 22
    for l in main: d.text((W / 2, y), l, font=fm, fill=INK, anchor="mm"); y += 62
    y += 18 - 62 + 52
    for l in sec: d.text((W / 2, y), l, font=fs, fill=INK, anchor="mm"); y += 52
    img.save(out, quality=95, subsampling=0)

if __name__ == "__main__":
    box, tube = (Image.open(p).convert("RGB") for p in sys.argv[1:3])
    story(box, tube, sys.argv[3])
