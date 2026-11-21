"""@mcg.va — a 9-post profile feed in the Pinterest-moodboard motherhood style: each post 1080x1350 (text kept inside the
centre 3:4 that the profile grid shows), plus a phone mock-up of the profile with the 3x3 grid.
Rhythm: photo / text tile / photo in a checkerboard; colour and b&w alternate; tiles = pink stripe, pink tile, cream paper,
blue stripe (all drawn here). Built on moodboard_demo.py.
  python3 -I feed_grid.py OUT_DIR P1 ... P9_photos (see main for the order) AVATAR
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moodboard_demo import (inter, lora, load, film, cover_crop, stripes, paper, shadow, polaroid, arrow, tracked,
                            MILK, POWDER, LATTE, OLIVE, MOCHA, CHOC, PINK, PINK2, BLUE, MARK)
W, H = 1080, 1350; SAFE = (34, W - 34)                     # the grid crops 34 px each side (3:4)

def tiles_bg(w, h):
    im = Image.new("RGB", (w, h), (238, 196, 190)); d = ImageDraw.Draw(im); s = 180
    for x in range(-20, w, s): d.line((x, 0, x, h), fill=(250, 238, 234), width=7)
    for y in range(-40, h, s): d.line((0, y, w, y), fill=(250, 238, 234), width=7)
    a = np.asarray(im, np.float32) + np.random.default_rng(4).normal(0, 4, (h, w, 1))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))

def photo_post(src, out, bw=False, fx=0.5, fy=0.5, seed=0, label=None):
    im = film(cover_crop(load(src), W, H, fx, fy), bw=bw, seed=seed).convert("RGBA")
    if label:
        d = ImageDraw.Draw(im); f = lora(44)
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0)); ImageDraw.Draw(sh).text((80, H - 92), label, font=f, fill=(20, 14, 10, 150), anchor="ls")
        im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(5))); d.text((80, H - 94), label, font=f, fill=MILK, anchor="ls")
    im.convert("RGB").save(out, quality=95)

def post_stripe_cover(out):                                  # 2 — pink stripe: «день с нами»
    cv = stripes(W, H, PINK, PINK2).convert("RGBA"); shadow(cv, (150, 300, 930, 1050)); cv.paste(paper(780, 750), (150, 300)); d = ImageDraw.Draw(cv)
    d.text((540, 420), "digital diary", font=inter(26, 400), fill=OLIVE, anchor="ms")
    tracked(d, (540, 620), "ДЕНЬ", inter(150, 650, 32), CHOC, -0.045, "m"); tracked(d, (540, 770), "С НАМИ", inter(150, 650, 32), CHOC, -0.045, "m")
    d.text((540, 880), "малышке почти 6 месяцев", font=lora(46), fill=MOCHA, anchor="ms")
    d.text((540, 990), "(карусель)", font=inter(24, 400), fill=OLIVE, anchor="ms")
    cv.convert("RGB").save(out, quality=95)

def post_quote(out):                                          # 4 — cream paper: reel cover quote
    cv = paper(W, H).convert("RGBA"); d = ImageDraw.Draw(cv)
    d.text((540, 160), "@mcg.va  ·  reels", font=inter(24, 400), fill=OLIVE, anchor="ms")
    tracked(d, (540, 520), "МАТЕРИНСТВО", inter(112, 650, 32), CHOC, -0.045, "m")
    d.text((540, 640), "это прекрасно", font=lora(72), fill=MOCHA, anchor="ms")
    d.text((540, 730), "и одновременно", font=inter(52, 300), fill=CHOC, anchor="ms")
    f = inter(64, 500); tw = f.getlength("пи**ец"); d.rectangle((540 - tw / 2 - 26, 790, 540 + tw / 2 + 26, 880), fill=MARK)
    d.text((540, 858), "пи**ец", font=f, fill=CHOC, anchor="ms")
    d.text((540, 1180), "« смотреть до конца »", font=inter(26, 400), fill=OLIVE, anchor="ms")
    cv.convert("RGB").save(out, quality=95)

def post_grid(srcs, out):                                     # 6 — 2x3 mini grid, colour + b&w, «uncurated life»
    cv = Image.new("RGBA", (W, H), MILK + (255,)); cw, ch = W // 2, H // 3
    for k, p in enumerate(srcs[:6]):
        r, c = divmod(k, 2); cv.paste(film(cover_crop(load(p), cw, ch), bw=(k in (1, 2, 5)), seed=k + 20), (c * cw, r * ch))
    lab = Image.new("RGBA", (440, 100), (0, 0, 0, 0)); ImageDraw.Draw(lab).rectangle((0, 0, 439, 99), fill=MILK + (238,))
    ImageDraw.Draw(lab).text((220, 66), "uncurated life", font=lora(50), fill=CHOC, anchor="ms"); cv.alpha_composite(lab, ((W - 440) // 2, H // 2 - 50))
    cv.convert("RGB").save(out, quality=95)

def post_tiles(src, out):                                     # 8 — pink tile + polaroid: «4 вещи…»
    cv = tiles_bg(W, H).convert("RGBA"); d = ImageDraw.Draw(cv)
    d.text((540, 190), "(топ-4)", font=inter(30, 400), fill=CHOC, anchor="ms")
    tracked(d, (540, 300), "4 ВЕЩИ,", inter(92, 650, 32), CHOC, -0.045, "m")
    d.text((540, 380), "которые я бы купила снова", font=lora(48), fill=CHOC, anchor="ms")
    polaroid(cv, film(cover_crop(load(src), 440, 470)), 540, 820, 440, 470, -4, "mubo", True); d = ImageDraw.Draw(cv)
    arrow(d, ((800, 1180), (880, 1120), (790, 1060)), CHOC)
    d.text((540, 1260), "листай →", font=inter(26, 400), fill=CHOC, anchor="ms")
    cv.convert("RGB").save(out, quality=95)

def post_blue(src, out):                                      # alt text tile: blue stripe + note (not used in the 9)
    cv = stripes(W, H, BLUE, (236, 240, 246), 40).convert("RGBA"); polaroid(cv, film(cover_crop(load(src), 560, 620)), 540, 600, 560, 620, 3, "мы", True)
    cv.convert("RGB").save(out, quality=95)

def mockup(posts, avatar, out):
    """iPhone-ish profile: header, bio, highlights, 3x3 grid (3:4 cells, 2 px gaps)"""
    PW = 1179; cell = (PW - 4) // 3; chh = round(cell * 4 / 3)
    top = 760; PH = top + 3 * chh + 4 + 160
    cv = Image.new("RGB", (PW, PH), (255, 255, 255)); d = ImageDraw.Draw(cv)
    d.text((PW / 2, 90), "mcg.va", font=inter(40, 600), fill=(20, 20, 20), anchor="ms")
    av = cover_crop(film(load(avatar)), 230, 230, 0.5, 0.3); m = Image.new("L", (230, 230), 0); ImageDraw.Draw(m).ellipse((0, 0, 229, 229), fill=255)
    ring = Image.new("RGB", (246, 246), (255, 255, 255)); ImageDraw.Draw(ring).ellipse((0, 0, 245, 245), outline=PINK, width=5)
    cv.paste(ring, (52, 142)); cv.paste(av, (60, 150), m)
    for i, (n, l) in enumerate((("128", "публикаций"), ("4 210", "подписчиков"), ("312", "подписок"))):
        x = 470 + i * 240; d.text((x, 250), n, font=inter(40, 600), fill=(20, 20, 20), anchor="ms"); d.text((x, 296), l, font=inter(28, 400), fill=(20, 20, 20), anchor="ms")
    d.text((60, 440), "мама Агаты", font=inter(32, 600), fill=(20, 20, 20), anchor="ls")
    for i, l in enumerate(("slow motherhood & uncurated life", "день с нами · находки для малышей · честно о материнстве")):
        d.text((60, 488 + i * 44), l, font=inter(30, 400), fill=(20, 20, 20), anchor="ls")
    for i, (lab, col) in enumerate((("роды", POWDER), ("день", PINK2), ("находки", LATTE), ("мы", BLUE), ("прикорм", MILK))):
        cx = 120 + i * 230; d.ellipse((cx - 70, 600 - 70, cx + 70, 600 + 70), fill=col, outline=(220, 220, 220), width=3)
        d.text((cx, 600 + 12), lab[0].upper(), font=lora(56), fill=CHOC, anchor="ms"); d.text((cx, 712), lab, font=inter(26, 400), fill=(20, 20, 20), anchor="ms")
    for k, p in enumerate(posts):
        r, c = divmod(k, 3); im = Image.open(p).convert("RGB"); im = cover_crop(im, cell, chh)
        cv.paste(im, (c * (cell + 2), top + r * (chh + 2)))
    cv.save(out, quality=92)

if __name__ == "__main__":
    out = sys.argv[1]; a = sys.argv[2:]; os.makedirs(out, exist_ok=True)
    coffee_bottle, baby_bw, lift, walk, us, mubo, avatar = a[:7]; grid6 = a[7:13]
    o = lambda n: os.path.join(out, f"post_{n}.jpg")
    photo_post(coffee_bottle, o(1), fy=0.6, seed=1)
    post_stripe_cover(o(2))
    photo_post(baby_bw, o(3), bw=True, seed=3)
    post_quote(o(4))
    photo_post(lift, o(5), fx=0.4, seed=5)
    post_grid(grid6, o(6))
    photo_post(walk, o(7), seed=7, fy=0.6)
    post_tiles(mubo, o(8))
    photo_post(us, o(9), bw=True, seed=9, fy=0.35)
    mockup([o(i) for i in range(1, 10)], avatar, os.path.join(out, "profile_mockup.jpg")); print("ok")
