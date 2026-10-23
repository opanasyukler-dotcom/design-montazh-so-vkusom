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
    img = Image.new("RGB", (W, H), PAPER)
    cw, ch = 720, 960; x0, y0 = (W - cw) // 2, 250
    rounded_card(img, main, (x0, y0, x0 + cw, y0 + ch), 36)
    circle(img, side, x0 + cw - 20, y0 + 70, 300, (156, 640, 1356, 1340))             # nipples through the clear case, top corner
    d = ImageDraw.Draw(img)
    y = block(d, "А ещё обновили пустышки.", "main", 50, y0 + ch + 120, 62)
    y = block(d, "Мы пользуемся Paomma и BIBS, потому что другую форму соски человек категорически отказывается воспринимать.",
              "second", 36, y + 24, 50, 860)
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
