"""@mcg.va stories — «образ на прогулку для Агатика» (2 stories, 1080x1920), v3.
Aesthetic-stories look: full-bleed photo, centred white headline with a soft shadow, white rounded cards with a soft
drop shadow, pastel highlighter pills for the pieces of the look.
Fonts (stories_kit): main = Instagram-Classic-like bold (Inter Bold), second = thin Montserrat, detail = Montserrat Italic.
  python3 -I stories_walk_look.py kiss.jpg clothes.jpg out_dir
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import W, H, INK, font, wrap

EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
PILL = (236, 222, 230)          # pale lilac from the bodysuit
GREY = (105, 98, 102)

def emoji(ch, size):
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=ImageFont.truetype(EMOJI, 109), embedded_color=True)
    im = im.crop(im.getbbox()); k = size / im.height
    return im.resize((int(im.width * k), size), Image.LANCZOS)

def bg(photo, ox):
    """photo scaled to the story height, window starting at ox; slightly muted grade"""
    ph = Image.open(photo).convert("RGB"); s = H / ph.height
    ph = ph.resize((round(ph.width * s), H), Image.LANCZOS)
    a = np.asarray(ph.crop((ox, 0, ox + W, H)), np.float32) / 255
    a = a * 0.92 + a.mean(2, keepdims=True) * 0.08
    return Image.fromarray((np.clip(a, 0, 1) * 255 + .5).astype(np.uint8)), s

def shadowed(img, box, radius, blur=28, off=10, alpha=0.20, fill=(255, 255, 255)):
    x0, y0, x1, y1 = box
    sh = Image.new("L", img.size, 0); ImageDraw.Draw(sh).rounded_rectangle((x0, y0 + off, x1, y1 + off), radius, fill=int(255 * alpha))
    img.paste(Image.new("RGB", img.size, (40, 30, 35)), (0, 0), sh.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(img).rounded_rectangle(box, radius, fill=fill)

def text_shadow(img, xy, text, fnt, fill=(255, 255, 255), strength=0.55, blur=10):
    m = Image.new("L", img.size, 0); ImageDraw.Draw(m).text((xy[0], xy[1] + 3), text, font=fnt, fill=int(255 * strength), anchor="mm")
    img.paste(Image.new("RGB", img.size, (30, 25, 28)), (0, 0), m.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(img).text(xy, text, font=fnt, fill=fill, anchor="mm")

def pill(img, text, cx, cy):
    fnt = font("detail", 31); tw = fnt.getlength(text)
    w, h = tw + 46, 56; x0 = min(max(cx - w / 2, 40), W - 40 - w); y0 = cy - h / 2
    shadowed(img, (x0, y0, x0 + w, y0 + h), h // 2, blur=14, off=5, alpha=0.18, fill=PILL)
    ImageDraw.Draw(img).text((x0 + w / 2, cy + 1), text, font=fnt, fill=INK, anchor="mm")

def story1(photo, out):
    img, _ = bg(photo, 280)                                       # baby + mum, the wall above is free for type
    fm = font("main", 50)
    text_shadow(img, (W / 2, 272), "последние дни", fm)
    text_shadow(img, (W / 2, 334), "столько всего происходит", fm)
    fs, fd = font("second", 32), font("detail", 30)
    lines = ["ездили делать прививку,", "смотрели, как папа выступает,", "много гуляли, сходили с ума,", "гости приезжали"]
    x0, x1, y0 = 150, W - 150, 392
    y1 = y0 + 54 + 46 * len(lines) + 88
    shadowed(img, (x0, y0, x1, y1), 34)
    d = ImageDraw.Draw(img); y = y0 + 54
    for l in lines: d.text((W / 2, y), l, font=fs, fill=INK, anchor="mm"); y += 46
    d.line((W / 2 - 40, y - 4, W / 2 + 40, y - 4), fill=PILL, width=3)
    t = "но всё никак не могу выйти в сторис"; tw = fd.getlength(t); es = 38
    xs = (W - tw - 12 - es) / 2; yy = y1 - 46
    d.text((xs, yy), t, font=fd, fill=GREY, anchor="lm")
    e = emoji("🫠", es); img.paste(e, (int(xs + tw + 12), int(yy - es / 2)), e)
    img.save(out)

def story2(photo, out):
    ox = 180
    img, s = bg(photo, ox)
    P = lambda x, y: (x * s - ox, y * s)                          # photo px → story px
    x0, x1, y0, y1 = 90, W - 90, 270, 540
    shadowed(img, (x0, y0, x1, y1), 34)
    d = ImageDraw.Draw(img)
    d.text((W / 2, y0 + 64), "кажется, будто нечего рассказать,", font=font("detail", 31), fill=GREY, anchor="mm")
    d.text((W / 2, y0 + 122), "но что у нас точно неизменно —", font=font("second", 36), fill=INK, anchor="mm")
    d.text((W / 2, y0 + 196), "образ на прогулку для Агатика", font=font("main", 48), fill=INK, anchor="mm")
    pill(img, "вязаный комплект", *P(560, 960))
    pill(img, "кофточка с капюшоном", *P(1200, 720))
    pill(img, "нагрудник из муслина", 905, 1500)
    pill(img, "боди с рюшами", *P(420, 1560))
    img.save(out)

if __name__ == "__main__":
    kiss, clothes, od = sys.argv[1:4]
    story1(kiss, os.path.join(od, "story_1.png")); story2(clothes, os.path.join(od, "story_2.png"))
