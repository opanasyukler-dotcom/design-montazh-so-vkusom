"""@mcg.va — «обновили пустышки» (Paomma), minimal stories 1080x1920.
  story 1: warm off-white page, the main photo as a rounded card (the cover), the second photo in a circle on its corner,
           headline (main) + first paragraph (second)
  story 2: the rest of the text on the same page, the question in bold, room at the bottom for a native question sticker
Fonts (stories_kit): main = Inter Bold, second = thin Montserrat, detail = Montserrat Italic.
  python3 -I stories_pacifiers.py main.jpg side.jpg stroller.jpg baby.jpg out_dir
"""
import os, sys
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import W, H, INK, font, wrap

PAPER = (247, 244, 240)
GREY = (120, 112, 108)
LILAC = (214, 196, 204)

def shadow(img, mask, xy, blur=26, off=12, alpha=0.20):
    sh = Image.new("L", img.size, 0); m = mask.point(lambda v: int(v * alpha))
    sh.paste(m, (xy[0], xy[1] + off)); sh = sh.filter(ImageFilter.GaussianBlur(blur))
    img.paste(Image.new("RGB", img.size, (60, 45, 40)), (0, 0), sh)

def rounded_card(img, ph, box, r):
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
    s = max(w / ph.width, h / ph.height); p = ph.resize((round(ph.width * s), round(ph.height * s)), Image.LANCZOS)
    p = p.crop(((p.width - w) // 2, (p.height - h) // 2, (p.width - w) // 2 + w, (p.height - h) // 2 + h))
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), r, fill=255)
    shadow(img, m, (x0, y0)); img.paste(p, (x0, y0), m)

def circle(img, ph, cx, cy, d, crop):
    p = ph.crop(crop).resize((d, d), Image.LANCZOS)
    m = Image.new("L", (d * 4, d * 4), 0); ImageDraw.Draw(m).ellipse((0, 0, d * 4 - 1, d * 4 - 1), fill=255)
    m = m.resize((d, d), Image.LANCZOS)
    ring = d + 20; rm = Image.new("L", (ring * 4, ring * 4), 0); ImageDraw.Draw(rm).ellipse((0, 0, ring * 4 - 1, ring * 4 - 1), fill=255)
    rm = rm.resize((ring, ring), Image.LANCZOS)
    shadow(img, rm, (cx - ring // 2, cy - ring // 2), blur=22, off=10, alpha=0.25)
    img.paste(Image.new("RGB", (ring, ring), (255, 255, 255)), (cx - ring // 2, cy - ring // 2), rm)
    img.paste(p, (cx - d // 2, cy - d // 2), m)

def block(d, text, role, size, y, lead, maxw=880, col=INK, x=None):
    f = font(role, size)
    for l in wrap(text, f, maxw):
        if x is None: d.text((W / 2, y), l, font=f, fill=col, anchor="mm")
        else: d.text((x, y), l, font=f, fill=col, anchor="lm")
        y += lead
    return y

def story1(main, side, out):
    """cover: the main photo full screen, the second photo in a big circle, text over a soft bottom shade"""
    import numpy as np
    sc = H / main.height; p = main.resize((round(main.width * sc), H), Image.LANCZOS)
    x = (p.width - W) // 2; img = p.crop((x, 0, x + W, H))
    a = np.asarray(img, np.float32); y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    a *= 1 - np.clip((y - 0.58) / 0.42, 0, 1) ** 1.3 * 0.72                        # bottom shade so white text reads
    img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    circle(img, side, 270, 330, 420, (430, 640, 1230, 1440))                        # square crop → no distortion
    d = ImageDraw.Draw(img)
    y = block(d, "А ещё обновили пустышки.", "main", 52, 1450, 64, col=(255, 255, 255))
    block(d, "Мы пользуемся Paomma и BIBS, потому что другую форму соски человек категорически отказывается воспринимать.",
          "second", 36, y + 22, 50, 880, col=(255, 255, 255))
    img.save(out, quality=95, subsampling=0)

def fill(ph):
    s = max(W / ph.width, H / ph.height); p = ph.resize((round(ph.width * s), round(ph.height * s)), Image.LANCZOS)
    x, y = (p.width - W) // 2, (p.height - H) // 2
    return p.crop((x, y, x + W, y + H))

def shadow_text(img, items, y, x=None, maxw=880):
    """white lines with a soft dark glyph shadow; items = [(role, size, lead, text, gap_after)]"""
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0)); sh = Image.new("L", img.size, 0)
    d, ds = ImageDraw.Draw(lay), ImageDraw.Draw(sh)
    for role, size, lead, t, gap in items:
        f = font(role, size)
        for l in wrap(t, f, maxw):
            d.text((W / 2, y), l, font=f, fill=(255, 255, 255, 255), anchor="mm")
            ds.text((W / 2, y + 3), l, font=f, fill=170, anchor="mm"); y += lead
        y += gap
    img.paste(Image.new("RGB", img.size, (30, 22, 20)), (0, 0), sh.filter(ImageFilter.GaussianBlur(9)))
    img.paste(lay, (0, 0), lay)
    return y

def story2(stroller, out):
    """blurred stroller photo, the search story over it"""
    import numpy as np
    img = fill(stroller).filter(ImageFilter.GaussianBlur(26))
    a = np.asarray(img, np.float32) * 0.78; img = Image.fromarray(a.clip(0, 255).astype(np.uint8))   # a bit darker for white type
    items = [("second", 38, 52, "Раньше я всегда покупала латексные, но они ужасно быстро изнашиваются, и приходится постоянно менять.", 34),
             ("second", 38, 52, "Поэтому решила найти силиконовые такой же круглой формы.", 34),
             ("main", 50, 64, "И вы не представляете, какой это оказался квест!", 34),
             ("second", 38, 52, "Почему-то найти красивую пустышку с круглой силиконовой соской практически невозможно.", 34),
             ("detail", 38, 52, "В итоге нашла подходящую только у Paomma.", 0)]
    shadow_text(img, items, 560)
    img.save(out, quality=95, subsampling=0)

def story3(baby, out):
    """baby photo full screen, the question on the white sheet above her head, dark type"""
    img = fill(baby); d = ImageDraw.Draw(img)
    y = block(d, "Если знаете ещё бренды с красивыми силиконовыми пустышками именно круглой формы, поделитесь!",
              "main", 40, 270, 54, 880)
    block(d, "Потому что я уже весь интернет перерыла.", "detail", 34, y + 14, 46, 900, INK)
    img.save(out, quality=95, subsampling=0)

if __name__ == "__main__":
    main, side, stroller, baby = (Image.open(p).convert("RGB") for p in sys.argv[1:5]); od = sys.argv[5]
    story1(main, side, os.path.join(od, "pustyshki_1.jpg"))
    story2(stroller, os.path.join(od, "pustyshki_2.jpg"))
    story3(baby, os.path.join(od, "pustyshki_3.jpg"))
