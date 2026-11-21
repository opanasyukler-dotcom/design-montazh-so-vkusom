"""@mcg.va — her whole visible grid (24 posts) redone as new covers in the styles of her Pinterest board «Мой профиль»:
full warm/b&w photo with headline, «digital diary» split (photo over paper), colour+b&w collage, stripe / pink-tile card with
a taped polaroid, calendar page, oval frame card. Her own photos only (the pins are style references).
Type: Inter Tight (letter-spacing −1 %) caps + Playfair Display italic. Tiles are 3:4, the profile-grid view.
  python3 -I feed_pinstyle.py SHOT1 SHOT2 SCRATCH OUT_DIR
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moodboard_demo import film, cover_crop, stripes, paper, polaroid, MILK, POWDER, LATTE, OLIVE, MOCHA, CHOC, PINK, PINK2, MARK
from feed_recover import tiles, up, TW, TH, K, grid
from feed_unified import ui
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except ImportError: pass
FU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "user")
W, H = TW * K, TH * K                                         # 1314 x 1752
WHITE = (255, 255, 255)

def it_(size, w=500):
    f = ImageFont.truetype(os.path.join(FU, "InterTight-Variable.ttf"), size); f.set_variation_by_axes([w]); return f
def pf(size, w=400, italic=True):
    f = ImageFont.truetype(os.path.join(FU, "PlayfairDisplay-Italic-Variable.ttf" if italic else "PlayfairDisplay-Variable.ttf"), size)
    f.set_variation_by_axes([w]); return f

def caps(d, x, y, text, f, fill, anchor="m"):
    """Inter Tight caps, letter-spacing −1 % of the size"""
    tr = -0.01 * f.size; ws = [f.getlength(c) + tr for c in text]; tw = sum(ws) - tr
    x = x - tw / 2 if anchor == "m" else x
    for c, w in zip(text, ws): d.text((x, y), c, font=f, fill=fill, anchor="ls"); x += w
    return tw

def fit(text, w_max, size, weight=500):
    while size > 40 and it_(size, weight).getlength(text) * 0.99 > w_max: size -= 4
    return it_(size, weight)

def load(p): return ImageOps.exif_transpose(Image.open(p)).convert("RGB")

def headline(cv, title, sub, y, col=WHITE, shade=True, w_max=1150, size=170):
    d = ImageDraw.Draw(cv); f = fit(title, w_max, size); fs = pf(78)
    if shade:
        sh = Image.new("RGBA", cv.size, (0, 0, 0, 0)); ds = ImageDraw.Draw(sh)
        caps(ds, W / 2, y + 3, title, f, (15, 10, 8, 150)); ds.text((W / 2, y + 112), sub, font=fs, fill=(15, 10, 8, 150), anchor="ms")
        cv.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)))
    caps(d, W / 2, y, title, f, col + (255,)); d.text((W / 2, y + 108), sub, font=fs, fill=col + (255,), anchor="ms")

def haze(cv, y, strength=0.5, r=420):
    g = Image.new("L", (1, H))
    for yy in range(H): g.putpixel((0, yy), int(255 * strength * max(0.0, 1 - abs(yy - y) / r) ** 1.3))
    hz = Image.new("RGBA", (W, H), (20, 13, 10, 0)); hz.putalpha(g.resize((W, H))); cv.alpha_composite(hz)

# ── templates ───────────────────────────────────────────────────────────────────────────────────────────────────────
def t_photo(src, title, sub, fx=.5, fy=.5, bw=False, y=1290, seed=0):
    cv = film(cover_crop(src, W, H, fx, fy), bw=bw, seed=seed).convert("RGBA"); haze(cv, y + 20); headline(cv, title, sub, y); return cv

def t_split(src, title, sub, fx=.5, fy=.5, bw=False, label="digital diary", seed=0):
    cv = paper(W, H).convert("RGBA"); ph = 1060; cv.paste(film(cover_crop(src, W, ph, fx, fy), bw=bw, seed=seed), (0, 0)); d = ImageDraw.Draw(cv)
    headline(cv, title, sub, 1330, CHOC, False, 1100, 150); d.text((80, H - 70), label, font=it_(30, 400), fill=OLIVE, anchor="ls")
    d.text((W - 80, H - 70), "@mcg.va", font=it_(30, 400), fill=OLIVE, anchor="rs"); return cv

def t_card(src, title, sub, bg="stripe", fx=.5, fy=.5, angle=-3, seed=0):
    cv = (stripes(W, H, PINK, PINK2, 56) if bg == "stripe" else stripes(W, H, (226, 214, 196), MILK, 40) if bg == "beige" else tiles_bg()).convert("RGBA")
    polaroid(cv, film(cover_crop(src, 760, 860, fx, fy), seed=seed), W / 2, 700, 760, 860, angle, None, True)
    d = ImageDraw.Draw(cv); pw = 980; d.rectangle(((W - pw) / 2, 1250, (W + pw) / 2, 1560), fill=MILK + (255,))
    headline(cv, title, sub, 1400, CHOC, False, 900, 130); return cv

def tiles_bg():
    im = Image.new("RGB", (W, H), (238, 196, 190)); d = ImageDraw.Draw(im); s = 220
    for x in range(-30, W, s): d.line((x, 0, x, H), fill=(250, 238, 234), width=9)
    for yy in range(-60, H, s): d.line((0, yy, W, yy), fill=(250, 238, 234), width=9)
    a = np.asarray(im, np.float32) + np.random.default_rng(4).normal(0, 4, (H, W, 1)); return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def t_collage(srcs, title, sub, seed=0):
    cv = Image.new("RGBA", (W, H), MILK + (255,)); cw, ch = W // 2, H // 3
    for k, s in enumerate(srcs[:6]):
        r, c = divmod(k, 2); cv.paste(film(cover_crop(s, cw, ch), bw=(k in (1, 2, 5)), seed=seed + k), (c * cw, r * ch))
    d = ImageDraw.Draw(cv); d.rectangle((150, H / 2 - 170, W - 150, H / 2 + 170), fill=MILK + (240,))
    headline(cv, title, sub, H / 2 + 10, CHOC, False, 900, 130); return cv

def t_calendar(srcs, title, sub, times):
    cv = paper(W, H).convert("RGBA"); d = ImageDraw.Draw(cv)
    caps(d, 90, 200, title, fit(title, 1100, 140, 600), CHOC + (255,), "l"); d.text((90, 290), sub, font=pf(70), fill=MOCHA, anchor="ls")
    for i, (t, s) in enumerate(zip(times, srcs)):
        y = 380 + i * 270; d.line((90, y, W - 90, y), fill=POWDER, width=3); d.text((90, y + 70), t[0], font=it_(40, 400), fill=OLIVE, anchor="ls")
        if i == 0: d.rectangle((270, y + 26, 270 + it_(46, 400).getlength(t[1]) + 26, y + 86), fill=MARK)
        d.text((283, y + 74), t[1], font=it_(46, 400), fill=CHOC, anchor="ls")
        cv.paste(film(cover_crop(s, 300, 220), bw=i % 2 == 1, seed=i), (W - 90 - 300, y + 24)); d = ImageDraw.Draw(cv)
    return cv

def t_oval(src, title, sub):
    cv = Image.new("RGBA", (W, H), (88, 62, 44, 255)); d = ImageDraw.Draw(cv)
    d.rectangle((90, 90, W - 90, H - 90), fill=(196, 202, 160)); d.rectangle((130, 130, W - 130, H - 130), outline=(150, 160, 110), width=6)
    m = Image.new("L", (W, H), 0); ImageDraw.Draw(m).ellipse((250, 240, W - 250, H - 240), fill=255)
    cv.paste(film(cover_crop(src, W, H)), (0, 0), m); d = ImageDraw.Draw(cv)
    d.ellipse((240, 230, W - 240, H - 230), outline=(150, 160, 110), width=10)
    haze(cv, 1180, 0.6, 320); headline(cv, title, sub, 1160, WHITE, True, 700, 130); return cv

# ── the grid ────────────────────────────────────────────────────────────────────────────────────────────────────────
def main(s1, s2, sc, out):
    A, B = tiles(s1, 0), tiles(s2, 1); P = lambda *a: load(os.path.join(sc, *a)); repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    R = lambda p, box=None: load(os.path.join(repo, p)).crop(box) if box else load(os.path.join(repo, p))
    T = lambda t, y0=60, y1=505, x0=0, x1=TW: up(t).crop((x0 * K, y0 * K, x1 * K, y1 * K))
    st = lambda n: P("prot", "stills", f"s{n:02d}.jpg"); s3 = lambda h: load(next(os.path.join(sc, "st3", "dl", f) for f in os.listdir(os.path.join(sc, "st3", "dl")) if f.startswith(h)))
    upl = "/root/.claude/uploads/33ffb1b3-db43-534f-8dee-debb988d835d/c086bd35-image.jpg"
    G0 = {
        (0, 0): (t_split(T(A[(0, 0)], 20, 380), "СРЕДСТВА", "для малышки и мамы", fy=.3, seed=1), "302", "car"),
        (0, 1): (t_photo(P("st3", "dl", "B61AB164-F3AC-461E-A95C-C0D4E0E40DE3.png"), "ПАТЧИ", "пять минут для себя", .6, .5, True, seed=2), "318", "car"),
        (0, 2): (t_photo(P("prot", "dl", "72BC58DA-326D-4D9C-933E-BE9E71C75B77.png"), "МАТЕРИНСТВО", "прекрасно и одновременно…", .5, .45, seed=3), "407", "reel"),
        (1, 0): (t_card(T(A[(1, 0)]), "УХОД", "для мамы и малышки", "beige", seed=4), "401", "reel"),
        (1, 1): (t_photo(P("dep", "bed.jpg"), "ДЕПРЕССИЯ", "покидает моё тело, когда…", .5, .2, True, seed=5), "352", "car"),
        (1, 2): (t_photo(P("mat", "cover_bg.png"), "МАТЕРИНСТВО", "это не про ребёнка", .5, .2, seed=6), "1 609", "reel"),
        (2, 0): (t_collage([P("top4", "mubo.png"), P("top4", "warmer.png"), P("top4", "noise.jpg"), P("top4", "machine_clean.png"), s3("329009F6"), s3("51EA4A83")], "4 ВЕЩИ", "которые я бы купила снова"), "353", "car"),
        (2, 1): (t_card(T(A[(2, 1)]), "ГАРДЕРОБ", "малышки на осень", "tile", angle=3, seed=7), "341", "reel"),
        (2, 2): (t_photo(T(A[(2, 2)], 20, 410), "СПАСИБО МУЖУ", "за то, что всегда рядом", .5, .3, seed=8), "763", "car"),
        (3, 0): (t_calendar([load(upl), s3("51EA4A83"), s3("A09CF331"), s3("BE0BD576"), s3("6391B2B6")], "ДЕНЬ С НАМИ", "малышке почти 6 месяцев",
                            [("07:00", "кофе, пока горячий"), ("09:30", "тыква"), ("11:00", "ванна"), ("14:00", "игры"), ("16:30", "прогулка")]), "1 934", "reel"),
        (3, 1): (t_split(T(A[(3, 1)]), "БЕЛЫЙ ШУМ", "мастхэв с первых дней", seed=9), "335", "reel"),
        (3, 2): (t_photo(T(A[(3, 2)]), "БЕРЕМЕННОСТЬ", "как это было у меня", .5, .4, True, seed=10), "799", "reel"),
    }
    G1 = {
        (0, 0): (t_photo(R("reels/rody_part1_cover.jpg", (0, 560, 1080, 1920)), "РОДЫ", "часть первая", .5, .2, seed=11), "1 419", "reel"),
        (0, 1): (t_card(T(B[(0, 1)]), "ПЕРВЫЕ ДНИ", "дома с малышкой", "stripe", seed=12), "475", "reel"),
        (0, 2): (t_photo(R("sling_cover.jpg", (0, 180, 1080, 1290)), "СЛИНГ", "лучшая покупка для мамы", .5, .3, seed=13), "416", "reel"),
        (1, 0): (t_oval(T(B[(1, 0)]), "НА ИВ", "наш опыт"), "390", "reel"),
        (1, 1): (t_collage([R("reels_cover_mini_vlog.jpg", (0, 470, 1080, 1920)), st(2), st(9), st(5), st(14), st(13)], "МИНИ-ВЛОГ", "с 4,5-месячным малышом", 20), "1 202", "reel"),
        (1, 2): (t_split(T(B[(1, 2)], 60, 470, 10, 235), "ЧТО ДЕЛАТЬ", "шпаргалка для беременных", label="шпаргалка", seed=14), "322", "car"),
        (2, 0): (t_photo(T(B[(2, 0)]), "ПАПА И ДОЧКА", "наши тихие утра", seed=15), "509", "reel"),
        (2, 1): (t_photo(T(B[(2, 1)]), "УСТАЛОСТЬ", "честно о материнстве", .5, .3, True, seed=16), "364", "reel"),
        (2, 2): (t_photo(T(B[(2, 2)], 0, 385), "Я ДУМАЛА,", "нас это обойдёт…", .5, .15, seed=17), "6 300", "reel"),
        (3, 0): (t_card(T(B[(3, 0)]), "СОСКА", "брать или нет", "beige", angle=3, seed=18), "525", "reel"),
        (3, 1): (t_split(T(B[(3, 1)], 30, 360), "ДВЕ СМЕСИ", "почему мы кормим так", fy=.4, seed=19), "2 928", "reel"),
        (3, 2): (t_photo(T(B[(3, 2)], 190, 505), "ПОКУПКИ", "для малышки · 68 размер", seed=20), "552", "reel"),
    }
    os.makedirs(os.path.join(out, "covers"), exist_ok=True); Gs = ({}, {})
    for s, G in enumerate((G0, G1)):
        for (r, c), (cv, v, kind) in G.items():
            cv = cv.convert("RGB"); cv.save(os.path.join(out, "covers", f"s{s + 1}_r{r + 1}c{c + 1}.jpg"), quality=92)
            Gs[s][(r, c)] = ui(cv.resize((TW, TH), Image.LANCZOS), v, kind)
    full = Image.new("RGB", (3 * TW + 6, 8 * TH + 21), (20, 20, 20)); full.paste(grid(Gs[0]), (0, 0)); full.paste(grid(Gs[1]), (0, 4 * TH + 12))
    full.save(os.path.join(out, "feed_pinstyle.jpg"), quality=90); print("ok")

if __name__ == "__main__":
    main(*sys.argv[1:5])
