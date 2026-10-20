"""@mcg.va stories — «образ на прогулку для Агатика» (2 stories, 1080x1920), v2 in the aesthetic-stories look:
full-bleed muted photo, centred white headline with a soft shadow, white rounded cards with a soft drop shadow
(like native IG reply cards / paper notes), pastel highlighter pills for the pieces of the look.
Type roles from stories_kit: title = serif italic, body = sans, detail = sans italic.
  python3 -I stories_walk_look.py photo.jpg out_dir
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import W, H, INK, font

EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
PILL = (236, 222, 230)          # pale lilac from the bodysuit
GREY = (110, 104, 108)

def emoji(ch, size):
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=ImageFont.truetype(EMOJI, 109), embedded_color=True)
    im = im.crop(im.getbbox()); k = size / im.height
    return im.resize((int(im.width * k), size), Image.LANCZOS)

def bg(photo, scale, ox, oy, mute=0.0):
    """photo scaled by `scale`, story window at (ox, oy) in scaled px; slightly muted grade"""
    ph = Image.open(photo).convert("RGB"); ph = ph.resize((int(ph.width * scale), int(ph.height * scale)), Image.LANCZOS)
    a = np.asarray(ph.crop((ox, oy, ox + W, oy + H)), np.float32) / 255
    a = a * 0.9 + a.mean(2, keepdims=True) * 0.1
    a = a * (1 - mute)
    return Image.fromarray((np.clip(a, 0, 1) * 255 + .5).astype(np.uint8))

def shadowed(img, box, radius, blur=28, off=10, alpha=0.20, fill=(255, 255, 255)):
    """rounded card with a soft drop shadow"""
    x0, y0, x1, y1 = box
    sh = Image.new("L", img.size, 0); ImageDraw.Draw(sh).rounded_rectangle((x0, y0 + off, x1, y1 + off), radius, fill=int(255 * alpha))
    img.paste(Image.new("RGB", img.size, (40, 30, 35)), (0, 0), sh.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(img).rounded_rectangle(box, radius, fill=fill)

def text_shadow(img, xy, text, fnt, fill=(255, 255, 255), strength=0.45, blur=9):
    m = Image.new("L", img.size, 0); ImageDraw.Draw(m).text((xy[0], xy[1] + 3), text, font=fnt, fill=int(255 * strength), anchor="mm")
    img.paste(Image.new("RGB", img.size, (30, 25, 28)), (0, 0), m.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(img).text(xy, text, font=fnt, fill=fill, anchor="mm")

def pill(img, text, cx, cy):
    fnt = font("detail", 31); tw = fnt.getlength(text)
    w, h = tw + 44, 54; x0 = min(max(cx - w / 2, 40), W - 40 - w); y0 = cy - h / 2
    shadowed(img, (x0, y0, x0 + w, y0 + h), h // 2, blur=14, off=5, alpha=0.18, fill=PILL)
    ImageDraw.Draw(img).text((x0 + w / 2, cy + 1), text, font=fnt, fill=INK, anchor="mm")

def story1(photo, out):
    img = bg(photo, 1.3, 0, 0, mute=0.06)                                       # bedding on top, knits under the card
    ft = font("title", 66)
    text_shadow(img, (W / 2, 330), "Последние дни", ft)
    text_shadow(img, (W / 2, 408), "столько всего происходит", ft)
    items = ["ездили делать прививку", "смотрели, как папа выступает", "много гуляли", "сходили с ума", "гости приезжали"]
    fb, fd = font("body", 38), font("detail", 33)
    x0, x1, y0, row = 150, W - 150, 560, 74
    y1 = y0 + 70 + row * len(items) + 110
    shadowed(img, (x0, y0, x1, y1), 34)
    d = ImageDraw.Draw(img); y = y0 + 70
    for i, t in enumerate(items):
        d.ellipse((x0 + 34, y - 5, x0 + 44, y + 5), fill=PILL)
        d.text((x0 + 62, y), t, font=fb, fill=INK, anchor="lm")
        if i < len(items) - 1: d.line((x0 + 62, y + row / 2, x1 - 50, y + row / 2), fill=(232, 226, 230), width=2)
        y += row
    t = "но всё никак не могу выйти в сторис"; tw = fd.getlength(t); es = 40
    xs = (W - tw - 14 - es) / 2; yy = y1 - 62
    d.text((xs, yy), t, font=fd, fill=GREY, anchor="lm")
    e = emoji("🫠", es); img.paste(e, (int(xs + tw + 14), int(yy - es / 2)), e)
    img.save(out)

def story2(photo, out):
    s = H / 2000; ox = 180
    img = bg(photo, s, ox, 0)
    P = lambda x, y: (x * s - ox, y * s)                                        # photo px → story px
    x0, x1, y0, y1 = 80, W - 80, 270, 560
    shadowed(img, (x0, y0, x1, y1), 34)
    d = ImageDraw.Draw(img)
    d.text((W / 2, y0 + 70), "кажется, будто нечего рассказать,", font=font("detail", 33), fill=GREY, anchor="mm")
    d.text((W / 2, y0 + 132), "но что у нас точно неизменно —", font=font("body", 38), fill=INK, anchor="mm")
    d.text((W / 2, y0 + 218), "образ на прогулку для Агатика", font=font("title", 60), fill=INK, anchor="mm")
    pill(img, "вязаный комплект", *P(560, 960))
    pill(img, "кофточка с капюшоном", *P(1200, 720))
    pill(img, "нагрудник из муслина", 905, 1500)
    pill(img, "боди с рюшами", *P(420, 1560))
    img.save(out)

if __name__ == "__main__":
    photo, od = sys.argv[1], sys.argv[2]
    story1(photo, os.path.join(od, "story_1.png")); story2(photo, os.path.join(od, "story_2.png"))
