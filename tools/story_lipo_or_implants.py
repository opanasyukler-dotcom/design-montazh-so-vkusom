# story fix: new headline ("Глядя на предыдущую сторис, …") + arrows to the breast with fat added and a caption
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
SP = "/tmp/claude-0/-home-user-design-montazh-so-vkusom/b26b20a4-6ac5-50f8-8404-dcbad88c88d3/scratchpad/"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
im = Image.open("src.jpg").convert("RGB"); W, H = im.size; BG = (1, 30, 70)
d = ImageDraw.Draw(im); d.rectangle((0, 120, W, 488), fill=BG)                       # remove the old headline
# size so that the widest line matches the original headline width (~800 px)
lines = ["ГЛЯДЯ НА ПРЕДЫДУЩУЮ СТОРИС,", "КАК ДУМАЕТЕ, ГДЕ ГРУДЬ", "УВЕЛИЧЕНА СОБСТВЕННЫМ", "ЖИРОМ, А ГДЕ УСТАНОВЛЕНЫ", "ИМПЛАНТЫ?"]
size = 60
while ImageFont.truetype(BOLD, size).getlength(max(lines, key=lambda l: ImageFont.truetype(BOLD, size).getlength(l))) > 1000: size -= 1
f = ImageFont.truetype(BOLD, size); lh = int(size * 1.17); y0 = 425 - lh * len(lines)
txt = Image.new("L", (W, H), 0); dt = ImageDraw.Draw(txt)
for i, l in enumerate(lines): dt.text((W / 2, y0 + i * lh), l, font=f, fill=255, anchor="mt")
sh = txt.filter(ImageFilter.GaussianBlur(6)).point(lambda v: int(v * 0.55))
im = Image.composite(Image.new("RGB", (W, H), (0, 8, 25)), im, sh); im = Image.composite(Image.new("RGB", (W, H), (255, 255, 255)), im, txt)
# caption card on the drape, arrows down to the breast where fat was added
cap = ["Здесь, кстати, у нас уже", "добавлена жировая ткань", "в одну грудь"]
fc = ImageFont.truetype(SP + "fonts2/M300.ttf", 40); cw = int(max(fc.getlength(c) for c in cap)) + 56; chh = 52 * len(cap) + 36   # thin Montserrat
cx0, cy0 = W - cw - 70, 690
layer = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dl = ImageDraw.Draw(layer)
dl.rounded_rectangle((cx0, cy0, cx0 + cw, cy0 + chh), radius=28, fill=(255, 255, 255, 245))
for i, c in enumerate(cap): dl.text((cx0 + 28, cy0 + 20 + i * 52), c, font=fc, fill=(16, 30, 60, 255))
def bez(p0, p1, p2, n=40):
    t = np.linspace(0, 1, n)[:, None]; p0, p1, p2 = map(np.float32, (p0, p1, p2)); return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
def arrow(path, col=(255, 255, 255, 255), w=5, hl=20, hw=11):
    dl.line([tuple(p) for p in path], fill=col, width=w, joint="curve")
    a, b = path[-4], path[-1]; v = (b - a) / np.linalg.norm(b - a); n = np.float32([-v[1], v[0]])
    dl.polygon([tuple(b + v * 4), tuple(b - v * hl + n * hw), tuple(b - v * hl - n * hw)], fill=col)
target = (812, 1157)                                                                      # left breast (on the photo): fat already added
arrow(bez((cx0 + 60, cy0 + chh + 14), (cx0 + 20, cy0 + chh + 70), (target[0] + 40, target[1] - 62)))      # small, neat
arrow(np.float32([[440, 462], [300, 462], [200, 462], [150, 462]]), w=5, hl=22, hw=12)   # straight, under the headline: back to the previous story
shadow = layer.filter(ImageFilter.GaussianBlur(8)); sa = np.asarray(shadow)[..., 3].astype(np.float32) * 0.35
base = np.asarray(im).astype(np.float32); base *= 1 - sa[..., None] / 255
im = Image.fromarray(base.astype(np.uint8)); im.paste(layer, (0, 0), layer)
im.save("story_fixed.jpg", quality=95); print("font", size)
