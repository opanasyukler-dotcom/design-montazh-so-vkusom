"""@mcg.va — «крем на холода» (La Roche-Posay Cicaplast B5+), one collage story 1080x1920.
Two photos stacked full-bleed edge to edge; white type with a soft glyph shadow sits on the seam, no card.
Fonts (stories_kit): main = Inter Bold, second = thin Montserrat.
  python3 -I stories_cicaplast.py box.png tube.png out.jpg
"""
import os, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import W, H
from stories_pacifiers import shadow_text

SEAM = 0                                        # the two wood tables meet without a line under the type

def crop(ph, h, cy):
    """photo scaled to the story width, a window of height h centred at cy (0..1 of the photo)"""
    s = W / ph.width; p = ph.resize((W, round(ph.height * s)), Image.LANCZOS)
    y = int(min(max(cy * p.height - h / 2, 0), p.height - h))
    return p.crop((0, y, W, y + h))

def story(box, tube, out):
    img = Image.new("RGB", (W, H), (255, 255, 255)); half = (H - SEAM) // 2
    img.paste(crop(box, half, 0.52), (0, 0)); img.paste(crop(tube, half, 0.47), (0, half + SEAM))
    items = [("main", 50, 64, "Купила на холода классный крем от раздражений на коже,", 18),
             ("second", 40, 54, "а то у Агаты уже от холода и отопления красные щёки", 0)]
    hgt = 2 * 64 + 18 + 2 * 54                                    # two lines each, centred on the seam
    shadow_text(img, items, H // 2 - hgt // 2 + 32, maxw=860)
    img.save(out, quality=95, subsampling=0)

if __name__ == "__main__":
    box, tube = (Image.open(p).convert("RGB") for p in sys.argv[1:3])
    story(box, tube, sys.argv[3])
