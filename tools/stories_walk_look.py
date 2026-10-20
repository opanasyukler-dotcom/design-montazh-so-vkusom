"""@mcg.va stories — «образ на прогулку для Агатика» (2 stories, 1080x1920) set with stories_kit roles.
  1: white page, the week in a centred stack (title / body list / detail + 🫠), like the blog's own text stories
  2: white band that melts into the flat-lay photo; the look's pieces marked with small detail-font labels
  python3 -I stories_walk_look.py photo.jpg out_dir
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import W, H, INK, draw_text, font

EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

def emoji(ch, size):
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=ImageFont.truetype(EMOJI, 109), embedded_color=True)
    im = im.crop(im.getbbox()); k = size / im.height
    return im.resize((int(im.width * k), size), Image.LANCZOS)

def line_with_emoji(img, role, text, ch, y):
    fnt = font(role); tw = fnt.getlength(text); es = int(fnt.size * 1.05); gap = fnt.size * 0.35
    x0 = (W - (tw + gap + es)) / 2
    ImageDraw.Draw(img).text((x0, y), text, font=fnt, fill=INK, anchor="lm")
    e = emoji(ch, es); img.paste(e, (int(x0 + tw + gap), int(y - es / 2)), e)

def story1(out):
    img = Image.new("RGB", (W, H), "white")
    y = draw_text(img, "title", "Последние дни", 600)
    draw_text(img, "title", "столько всего происходит", y - 6)
    y = 850
    for t in ["ездили делать прививку", "смотрели, как папа выступает", "много гуляли", "сходили с ума", "гости приезжали"]:
        draw_text(img, "body", t, y); y += 66
    line_with_emoji(img, "detail", "но всё никак не могу выйти в сторис", "🫠", y + 80)
    img.save(out)

def label(img, text, tx, ty, dot, align="left"):
    d = ImageDraw.Draw(img); fnt = font("detail", 32)
    d.text((tx, ty), text, font=fnt, fill=INK, anchor="lm" if align == "left" else "rm")
    tw = fnt.getlength(text); x0, x1 = (tx, tx + tw) if align == "left" else (tx - tw, tx)
    sx = min(max(dot[0], x0 + 12), x1 - 12); sy = ty + 26 if dot[1] > ty else ty - 26   # leader leaves from under the label
    d.line((sx, sy, dot[0], dot[1]), fill=INK, width=2)
    r = 6; d.ellipse((dot[0] - r, dot[1] - r, dot[0] + r, dot[1] + r), fill=INK)

def story2(photo, out):
    ph = Image.open(photo).convert("RGB")
    ph = ph.resize((W, int(ph.height * W / ph.width)), Image.LANCZOS)        # 1500x2000 → 1080x1440
    top = H - ph.height                                                         # 480 px white band on top
    img = Image.new("RGB", (W, H), "white")
    a = np.clip((np.arange(ph.height) - 0) / 200, 0, 1) ** 1.3                  # photo melts in from the white band
    mask = Image.fromarray((np.repeat(a[:, None], W, 1) * 255).astype(np.uint8))
    img.paste(ph, (0, top), mask)
    draw_text(img, "detail", "кажется, будто нечего рассказать,", 285)
    draw_text(img, "body", "но что у нас точно неизменно —", 350)
    draw_text(img, "title", "образ на прогулку для Агатика", 440)
    s = W / 1500
    P = lambda x, y: (x * s, top + y * s)                                       # photo px → story px
    label(img, "вязаный комплект", 70, 690, P(560, 640))
    label(img, "кофточка с капюшоном", 1010, 640, P(1180, 470), "right")
    label(img, "нагрудник из муслина", 1010, 1190, P(1330, 1260), "right")
    label(img, "боди с рюшами", 70, 1290, P(230, 1480))
    img.save(out)

if __name__ == "__main__":
    photo, od = sys.argv[1], sys.argv[2]
    story1(os.path.join(od, "story_1.png")); story2(photo, os.path.join(od, "story_2.png"))
