"""@mcg.va — «обновили пустышки» (Paomma), minimal stories 1080x1920.
  story 1: warm off-white page, the main photo as a rounded card (the cover), the second photo in a circle on its corner,
           headline (main) + first paragraph (second)
  story 2: the rest of the text on the same page, the question in bold, room at the bottom for a native question sticker
Fonts (stories_kit): main = Inter Bold, second = thin Montserrat, detail = Montserrat Italic.
  python3 -I stories_pacifiers.py main.jpg side.jpg out_dir
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

def story2(main, out):
    img = Image.new("RGB", (W, H), PAPER); d = ImageDraw.Draw(img)
    # small round thumb of the cover photo as a visual link to story 1
    circle(img, main, W // 2, 380, 170, (180, 700, 1380, 1900))
    X = 110; y = 560
    paras = [
        ("second", "Раньше я всегда покупала латексные, но они ужасно быстро изнашиваются, и приходится постоянно менять."),
        ("second", "Поэтому решила найти силиконовые такой же круглой формы."),
        ("main", "И вы не представляете, какой это оказался квест!"),
        ("second", "Почему-то найти красивую пустышку с круглой силиконовой соской практически невозможно."),
        ("detail", "В итоге нашла подходящую только у Paomma."),
    ]
    sizes = {"main": 44, "second": 36, "detail": 36}; leads = {"main": 56, "second": 50, "detail": 50}
    for role, t in paras:
        y = block(d, t, role, sizes[role], y, leads[role], 860, GREY if role == "detail" else INK, X) + 28
    d.line((X, y + 6, X + 90, y + 6), fill=LILAC, width=3); y += 56
    y = block(d, "Если знаете ещё бренды с красивыми силиконовыми пустышками именно круглой формы, поделитесь!",
              "main", 40, y, 54, 860, INK, X) + 18
    block(d, "Потому что я уже весь интернет перерыла.", "detail", 34, y, 48, 860, GREY, X)
    img.save(out, quality=95, subsampling=0)
    return y

if __name__ == "__main__":
    main, side = Image.open(sys.argv[1]).convert("RGB"), Image.open(sys.argv[2]).convert("RGB"); od = sys.argv[3]
    story1(main, side, os.path.join(od, "pustyshki_1.jpg"))
    print("text ends at y", story2(main, os.path.join(od, "pustyshki_2.jpg")))
