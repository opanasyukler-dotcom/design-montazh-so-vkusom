"""@mcg.va — demo of the Pinterest-moodboard style: a 4-slide carousel (1080x1350) «день с нами» and one story (1080x1920).
Warm «latte» film treatment, pink / blue stripe and cream paper backgrounds, Inter + Lora Italic, digital-diary timeline,
3x4 grid mixing colour and b&w, polaroid with tape, hand-drawn arrows.
  python3 -I moodboard_demo.py OUT_DIR PHOTO_cover PHOTO_coffee PHOTO_puree PHOTO_bath PHOTO_toys PHOTO_walk PHOTO_baby PHOTO_us [grid extras ...]
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except ImportError: pass
FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "user")
MILK, POWDER, LATTE, OLIVE, MILKCOFFEE, MOCHA, CHOC = (242, 241, 236), (217, 201, 196), (186, 169, 158), (149, 140, 126), (119, 108, 97), (94, 81, 68), (59, 44, 34)
PINK, PINK2, BLUE, MARK = (232, 169, 166), (246, 226, 222), (183, 195, 218), (240, 214, 205)

def inter(size, w=400, opsz=None):
    f = ImageFont.truetype(os.path.join(FD, "Inter-Variable.ttf"), size); f.set_variation_by_axes([opsz or min(32, max(14, size)), w]); return f
def lora(size, w=400):
    f = ImageFont.truetype(os.path.join(FD, "Lora-Italic-Variable.ttf"), size); f.set_variation_by_axes([w]); return f

def load(p): return ImageOps.exif_transpose(Image.open(p)).convert("RGB")

def film(im, bw=False, seed=0):
    """warm latte film: lifted blacks, soft contrast, warm cast, fine grain"""
    a = np.asarray(im, np.float32) / 255
    if bw: g = a @ np.array([0.3, 0.59, 0.11]); a = np.repeat(g[..., None], 3, 2)
    a = 0.06 + a * 0.9; a = a + (a - 0.5) * -0.08
    if not bw: a = a * np.array([1.04, 1.0, 0.93]); m = a.mean(2, keepdims=True); a = a * 0.85 + m * 0.15
    else: a = a * np.array([1.02, 1.0, 0.97])
    rng = np.random.default_rng(seed); a = a + rng.normal(0, 0.022, a.shape[:2])[..., None]
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

def cover_crop(im, w, h, fx=0.5, fy=0.5):
    s = max(w / im.width, h / im.height); im = im.resize((math.ceil(im.width * s), math.ceil(im.height * s)), Image.LANCZOS)
    x = int((im.width - w) * fx); y = int((im.height - h) * fy); return im.crop((x, y, x + w, y + h))

def stripes(w, h, c1, c2, width=46):
    im = Image.new("RGB", (w, h), c2); d = ImageDraw.Draw(im)
    for x in range(0, w, width * 2): d.rectangle((x, 0, x + width - 1, h), fill=c1)
    n = np.random.default_rng(1).normal(0, 4, (h, w, 1)); a = np.clip(np.asarray(im, np.float32) + n, 0, 255).astype(np.uint8)
    return Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.6))

def paper(w, h, c=MILK):
    a = np.asarray(Image.new("RGB", (w, h), c), np.float32) + np.random.default_rng(2).normal(0, 3, (h, w, 1))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def shadow(cv, box, r=22, op=70):
    s = Image.new("RGBA", cv.size, (0, 0, 0, 0)); ImageDraw.Draw(s).rectangle((box[0], box[1] + 10, box[2], box[3] + 10), fill=(60, 40, 30, op))
    cv.alpha_composite(s.filter(ImageFilter.GaussianBlur(r)))

def polaroid(cv, ph, x, y, w, h, angle=0, caption=None, tape=True):
    b = 22; card = Image.new("RGBA", (w + 2 * b, h + b + 90), (252, 250, 246, 255)); card.paste(ph.resize((w, h)) if ph.size != (w, h) else ph, (b, b))
    if caption: ImageDraw.Draw(card).text((card.width / 2, h + b + 60), caption, font=lora(34), fill=MOCHA, anchor="ms")
    lay = Image.new("RGBA", (card.width + 80, card.height + 80), (0, 0, 0, 0)); sh = Image.new("RGBA", lay.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle((40, 52, 40 + card.width, 52 + card.height), fill=(60, 40, 30, 75)); lay.alpha_composite(sh.filter(ImageFilter.GaussianBlur(16)))
    lay.alpha_composite(card, (40, 40)); lay = lay.rotate(angle, resample=Image.BICUBIC, expand=True)
    cv.alpha_composite(lay, (int(x - lay.width / 2), int(y - lay.height / 2)))
    if tape:
        t = Image.new("RGBA", (190, 54), (232, 220, 202, 185)); t = t.rotate(angle - 4, expand=True, resample=Image.BICUBIC)
        top = y - (h + b + 90) / 2 - 8; cv.alpha_composite(t, (int(x - t.width / 2 + math.sin(math.radians(angle)) * 60), int(top - t.height / 2 + 10)))

def arrow(d, pts, col=MOCHA, w=3):
    (x0, y0), (cx, cy), (x1, y1) = pts; P = []
    for u in np.linspace(0, 1, 50):
        P.append(((1 - u) ** 2 * x0 + 2 * (1 - u) * u * cx + u * u * x1, (1 - u) ** 2 * y0 + 2 * (1 - u) * u * cy + u * u * y1))
    d.line(P, fill=col, width=w, joint="curve"); ax, ay = P[-1]; bx, by = P[-5]; a = math.atan2(ay - by, ax - bx)
    for s in (-0.5, 0.5): d.line((ax, ay, ax - 22 * math.cos(a + s), ay - 22 * math.sin(a + s)), fill=col, width=w)

def topline(d, W, left, right, y=62, col=MOCHA):
    d.text((60, y), left, font=inter(22, 400), fill=col, anchor="ls"); d.text((W - 60, y), right, font=inter(22, 400), fill=col, anchor="rs")

def tracked(d, xy, text, f, fill, track=-0.03, anchor="l"):
    """tight tracking for caps headlines"""
    ws = [f.getlength(c) + f.size * track for c in text]; tw = sum(ws); x, y = xy
    if anchor == "m": x -= tw / 2
    for c, w in zip(text, ws): d.text((x, y), c, font=f, fill=fill, anchor="ls"); x += w
    return tw

# ── slides ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def slide1(P, out):
    W, H = 1080, 1350; cv = stripes(W, H, PINK, PINK2).convert("RGBA"); d = ImageDraw.Draw(cv)
    topline(d, W, "@mcg.va", "digital diary  ·  1/4", col=CHOC)
    ph = film(cover_crop(load(P["cover"]), 760, 900, 0.5, 0.55)); bx = ((W - 760) // 2, 300)
    shadow(cv, (bx[0], bx[1], bx[0] + 760, bx[1] + 900)); cv.paste(ph, bx)
    d = ImageDraw.Draw(cv)
    f = inter(118, 600, 32); tw = tracked(d, (W / 2, 255), "ДЕНЬ С НАМИ", f, CHOC, -0.045, "m")
    d.text((W / 2, 1268), "малышке почти 6 месяцев", font=lora(46), fill=CHOC, anchor="ms")
    d.text((bx[0] + 30, bx[1] + 900 - 30), "(01)", font=inter(30, 300), fill=MILK, anchor="ls")
    d.text((W - 60, 1328), "листай →", font=inter(22, 400), fill=CHOC, anchor="rs")
    cv.convert("RGB").save(out, quality=95)

def slide2(P, out):
    W, H = 1080, 1350; cv = paper(W, H).convert("RGBA"); d = ImageDraw.Draw(cv)
    topline(d, W, "@mcg.va", "день с нами  ·  2/4")
    d.text((60, 160), "November", font=inter(84, 650, 32), fill=CHOC, anchor="ls")
    days = [("пн", 17), ("вт", 18), ("ср", 19), ("чт", 20), ("пт", 21), ("сб", 22), ("вс", 23)]
    for i, (n, k) in enumerate(days):
        x = 90 + i * 150; d.text((x, 215), n, font=inter(22, 400), fill=OLIVE, anchor="ms")
        if k == 20: d.ellipse((x - 34, 232, x + 34, 300), fill=PINK)
        d.text((x, 280), str(k), font=inter(34, 400), fill=CHOC if k != 20 else MILK, anchor="ms")
    rows = [("07:00", "кофе, пока он ещё горячий", "coffee"), ("09:30", "тыква — первый прикорм", "puree"),
            ("11:00", "ванна после прикорма", "bath"), ("14:00", "играем на коврике", "toys"), ("16:30", "прогулка под сон", "walk")]
    y0 = 360; step = 190
    for i, (t, txt, key) in enumerate(rows):
        y = y0 + i * step; d.line((60, y, W - 60, y), fill=POWDER, width=2); d.text((60, y + 46), t, font=inter(26, 400), fill=OLIVE, anchor="ls")
        tw = inter(32, 400).getlength(txt)
        if i == 0: d.rectangle((196, y + 16, 196 + tw + 20, y + 58), fill=MARK)
        d.text((206, y + 48), txt, font=inter(32, 400), fill=CHOC, anchor="ls")
        ph = film(cover_crop(load(P[key]), 230, 160), bw=(i % 2 == 1), seed=i); px = W - 60 - 230 - (i % 2) * 70; py = y + 14
        shadow(cv, (px, py, px + 230, py + 160), 12, 50); cv.paste(ph, (px, py)); d = ImageDraw.Draw(cv)
    d.text((60, 1300), "slow day rituals", font=lora(40), fill=MOCHA, anchor="ls")
    arrow(d, ((420, 1280), (520, 1250), (560, 1190)))
    cv.convert("RGB").save(out, quality=95)

def slide3(P, out):
    W, H = 1080, 1350; cv = Image.new("RGBA", (W, H), MILK + (255,)); cw, ch = W // 3, (H - 0) // 4
    for k, p in enumerate(P["grid"][:12]):
        r, c = divmod(k, 3); bw = (k * 7) % 3 == 0
        cv.paste(film(cover_crop(load(p), cw, ch), bw=bw, seed=k), (c * cw, r * ch))
    d = ImageDraw.Draw(cv)
    lab = Image.new("RGBA", (420, 96), (0, 0, 0, 0)); ld = ImageDraw.Draw(lab); ld.rectangle((0, 0, 419, 95), fill=MILK + (235,))
    ld.text((210, 64), "uncurated life", font=lora(48), fill=CHOC, anchor="ms"); cv.alpha_composite(lab, ((W - 420) // 2, ch * 2 - 48))
    d.text((W - 40, 50), "3/4", font=inter(22, 500), fill=MILK, anchor="rs")
    cv.convert("RGB").save(out, quality=95)

def slide4(P, out):
    W, H = 1080, 1350; cv = stripes(W, H, BLUE, (236, 240, 246), 40).convert("RGBA"); d = ImageDraw.Draw(cv)
    topline(d, W, "@mcg.va", "день с нами  ·  4/4", col=CHOC)
    card = paper(880, 1040); cv.alpha_composite(Image.new("RGBA", (1, 1)), (0, 0)); shadow(cv, (100, 170, 980, 1210)); cv.paste(card, (100, 170)); d = ImageDraw.Draw(cv)
    polaroid(cv, film(cover_crop(load(P["us"]), 420, 480, 0.5, 0.35)), 540, 520, 420, 480, -3, "мы", True); d = ImageDraw.Draw(cv)
    lines = ["Раньше я ждала", "«идеального» дня.", "А сейчас понимаю:", "самые тёплые дни —", "вот такие, обычные."]
    y = 880
    for i, l in enumerate(lines):
        d.text((W / 2, y), l, font=lora(46) if i % 2 == 0 else inter(40, 300), fill=CHOC, anchor="ms"); y += 62
    d.text((W / 2, 1180), "а как проходит ваш день?", font=inter(26, 500), fill=MOCHA, anchor="ms")
    cv.convert("RGB").save(out, quality=95)

def story(P, out):
    W, H = 1080, 1920; cv = stripes(W, H, PINK, PINK2, 52).convert("RGBA"); d = ImageDraw.Draw(cv)
    d.text((80, 330), "доброе утро,", font=lora(64), fill=CHOC, anchor="ls")
    tracked(d, (80, 440), "ДЕНЬ С НАМИ", inter(104, 600, 32), CHOC, -0.045)
    polaroid(cv, film(cover_crop(load(P["coffee"]), 640, 760, 0.45, 0.5)), 540, 1030, 640, 760, 3, "07:00 · кофе горячий!", True); d = ImageDraw.Draw(cv)
    d.rectangle((80, 1490, 80 + inter(36, 400).getlength("уже успех ✓") + 24, 1540), fill=MARK)
    d.text((92, 1530), "уже успех", font=inter(36, 400), fill=CHOC, anchor="ls")
    arrow(d, ((300, 1505), (330, 1450), (380, 1425)))
    d.text((80, 1640), "сегодня у нас, как всегда,", font=inter(34, 300), fill=CHOC, anchor="ls")
    d.text((80, 1700), "женская тусовка", font=lora(54), fill=CHOC, anchor="ls")
    d.text((W - 80, 1540), "(01)", font=inter(32, 300), fill=MOCHA, anchor="rs")
    cv.convert("RGB").save(out, quality=95)

if __name__ == "__main__":
    out = sys.argv[1]; k = ["cover", "coffee", "puree", "bath", "toys", "walk", "baby", "us"]
    P = dict(zip(k, sys.argv[2:10])); P["grid"] = sys.argv[3:10] + sys.argv[10:]
    os.makedirs(out, exist_ok=True)
    slide1(P, os.path.join(out, "slide_01.jpg")); slide2(P, os.path.join(out, "slide_02.jpg"))
    slide3(P, os.path.join(out, "slide_03.jpg")); slide4(P, os.path.join(out, "slide_04.jpg")); story(P, os.path.join(out, "story_01.jpg"))
    print("ok")
