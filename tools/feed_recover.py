"""@mcg.va — her real profile grid (two screenshots) re-shown with new unified covers: thin Montserrat caps + thin italic
over a warm photo (like «МАТЕРИНСТВО» / «Спасибо мужу»), or a cream paper card with the photo in a window.
Tiles are 438x584 (3:4) cut from the 1320x2868 screenshots; new text is drawn at 3x and scaled down.
  python3 -I feed_recover.py SHOT1 SHOT2 REPO_DIR OUT_DIR
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(HERE, "..", "fonts", "stories")
TW, TH, K = 438, 584, 3
COLS = [(0, 438), (441, 879), (882, 1320)]; ROW0 = (191, 207); STEP = 587
WHITE, INK, MOCHA, MILK = (255, 255, 255), (43, 31, 26), (107, 74, 56), (243, 236, 228)

def mont(size, w=200, it=False):
    f = ImageFont.truetype(os.path.join(FD, "Montserrat-Italic-VariableFont_wght.ttf" if it else "Montserrat-VariableFont_wght.ttf"), size)
    f.set_variation_by_axes([w]); return f

def tiles(shot, s):
    im = Image.open(shot).convert("RGB")
    return {(r, c): im.crop((x0, ROW0[s] + r * STEP, x1, ROW0[s] + r * STEP + TH)) for r in range(4) for c, (x0, x1) in enumerate(COLS)}

def cover(im, w, h, fx=0.5, fy=0.5):
    s = max(w / im.width, h / im.height); im = im.resize((math.ceil(im.width * s), math.ceil(im.height * s)), Image.LANCZOS)
    x = int((im.width - w) * fx); y = int((im.height - h) * fy); return im.crop((x, y, x + w, y + h))

def warm(im):
    a = np.asarray(im, np.float32) / 255; a = 0.04 + a * 0.93; a = a * np.array([1.03, 1.0, 0.95])
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

def spaced(d, x, y, text, f, fill, track):
    ws = [f.getlength(c) + track for c in text]; x -= (sum(ws) - track) / 2
    for c, w in zip(text, ws): d.text((x, y), c, font=f, fill=fill, anchor="ls"); x += w

def overlay(bg, caps, it, y=0.74, haze=True, kicker=None, col=WHITE):
    """bg at 3x; caps lines in thin spaced Montserrat, italic line under, soft dark band only behind the text"""
    W, H = bg.size; cv = bg.convert("RGBA"); y0 = int(H * y)
    if haze:
        g = Image.new("L", (1, H)); c = y0 + 60; r = 330
        for yy in range(H): g.putpixel((0, yy), int(255 * 0.45 * max(0.0, 1 - abs(yy - c) / r) ** 1.2))
        hz = Image.new("RGBA", (W, H), (20, 13, 10, 0)); hz.putalpha(g.resize((W, H))); cv.alpha_composite(hz)
    txt = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(txt); sh = Image.new("L", (W, H), 0); ds = ImageDraw.Draw(sh)
    fc = mont(118 if max(len(l) for l in caps) <= 9 else 92, 200); fi = mont(56, 250, True); yy = y0
    if kicker:
        for dd, fill in ((ds, 160), (d, col + (255,))): spaced(dd, W / 2, yy - 150, kicker, mont(34, 400), fill, 8)
    for l in caps:
        for dd, fill in ((ds, 160), (d, col + (255,))): spaced(dd, W / 2, yy, l, fc, fill, 10)
        yy += int(fc.size * 1.08)
    if it:
        for dd, fill in ((ds, 160), (d, col + (255,))): dd.text((W / 2, yy + 6), it, font=fi, fill=fill, anchor="ms")
    s = Image.new("RGBA", (W, H), (15, 10, 8, 0)); s.putalpha(sh.filter(ImageFilter.GaussianBlur(9)))
    if col == WHITE: cv.alpha_composite(s)
    cv.alpha_composite(txt); return cv.convert("RGB")

def paper_card(photo, caps, it):
    W, H = TW * K, TH * K; a = np.asarray(Image.new("RGB", (W, H), MILK), np.float32) + np.random.default_rng(3).normal(0, 3, (H, W, 1))
    cv = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA"); d = ImageDraw.Draw(cv)
    spaced(d, W / 2, 230, caps[0], mont(104, 250), INK + (255,), 8)
    if len(caps) > 1: spaced(d, W / 2, 330, caps[1], mont(64, 300), INK + (255,), 8)
    d.text((W / 2, 430), it, font=mont(60, 250, True), fill=MOCHA + (255,), anchor="ms")
    pw, ph = 900, 1040; p = warm(cover(photo, pw, ph)); x, y = (W - pw) // 2, 540
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(sh).rectangle((x, y + 18, x + pw, y + ph + 18), fill=(60, 40, 30, 80))
    cv.alpha_composite(sh.filter(ImageFilter.GaussianBlur(26))); cv.paste(p, (x, y))
    t = Image.new("RGBA", (260, 70), (230, 216, 198, 190)).rotate(-5, expand=True); cv.alpha_composite(t, (W // 2 - t.width // 2, y - 40))
    return cv.convert("RGB")

def ui(tile, views, reel=True):
    """the grid's own overlays: reel icon top-right, eye + view count bottom-left"""
    t = tile.convert("RGBA"); s = 4; L = Image.new("RGBA", (TW * s, TH * s), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    if reel:
        d.rounded_rectangle((382 * s, 16 * s, 414 * s, 48 * s), 8 * s, fill=(255, 255, 255, 240))
        d.polygon([(394 * s, 24 * s), (394 * s, 40 * s), (406 * s, 32 * s)], fill=(40, 40, 40, 255))
    sh = Image.new("RGBA", L.size, (0, 0, 0, 0)); dsh = ImageDraw.Draw(sh)
    for dd, col in ((dsh, (0, 0, 0, 120)), (d, (255, 255, 255, 255))):
        dd.ellipse((16 * s, 548 * s, 40 * s, 564 * s), outline=col, width=3 * s); dd.ellipse((24 * s, 552 * s, 32 * s, 560 * s), fill=col)
        dd.text((48 * s, 566 * s), views, font=mont(30 * s, 600), fill=col, anchor="ls")
    t.alpha_composite(sh.filter(ImageFilter.GaussianBlur(3 * s)).resize((TW, TH), Image.LANCZOS)); t.alpha_composite(L.resize((TW, TH), Image.LANCZOS))
    return t.convert("RGB")

def up(tile): return tile.resize((TW * K, TH * K), Image.LANCZOS)
def down(im): return im.resize((TW, TH), Image.LANCZOS)

def build(shot1, shot2, repo):
    A, B = tiles(shot1, 0), tiles(shot2, 1); R = lambda p: Image.open(os.path.join(repo, p)).convert("RGB")
    # screen 1 — clean reels get a caption over the existing tile (its own icon and counter stay)
    A[(1, 0)] = down(overlay(up(A[(1, 0)]), ["УХОД"], "для мамы и малышки", 0.70))
    A[(2, 1)] = down(overlay(up(A[(2, 1)]), ["ГАРДЕРОБ"], "малышки на осень", 0.70))
    A[(3, 0)] = down(overlay(up(A[(3, 0)]), ["ДЕНЬ С НАМИ"], "малышке почти 6 месяцев", 0.72))
    A[(3, 1)] = down(overlay(up(A[(3, 1)]), ["БЕЛЫЙ ШУМ"], "мастхэв с первых дней", 0.70))
    A[(3, 2)] = down(overlay(up(A[(3, 2)]), ["БЕРЕМЕННОСТЬ"], "как это было у меня", 0.72))
    # screen 2 — old-style covers rebuilt from clean parts of the photo
    B[(0, 0)] = ui(down(overlay(warm(cover(R("reels/rody_part1_cover.jpg").crop((0, 560, 1080, 1920)), TW * K, TH * K, 0.5, 0.2)), ["РОДЫ"], "часть 1", 0.74)), "1 419")
    B[(0, 2)] = ui(down(overlay(warm(cover(R("sling_cover.jpg").crop((0, 180, 1080, 1290)), TW * K, TH * K, 0.5, 0.3)), ["СЛИНГ"], "лучшая покупка для мамы", 0.76, kicker="ОБЗОР")), "416")
    B[(1, 1)] = ui(down(overlay(warm(cover(R("reels_cover_mini_vlog.jpg").crop((0, 470, 1080, 1920)), TW * K, TH * K, 0.5, 0.4)), ["МИНИ-ВЛОГ"], "с 4,5-месячным малышом", 0.74)), "1 202")
    B[(2, 2)] = ui(down(overlay(warm(cover(up(B[(2, 2)]).crop((0, 0, TW * K, 385 * K)), TW * K, TH * K, 0.5, 0.15)), ["Я ДУМАЛА,"], "нас это обойдёт…", 0.74)), "6 300")
    B[(3, 1)] = ui(down(overlay(warm(cover(up(B[(3, 1)]).crop((0, 30 * K, TW * K, 360 * K)), TW * K, TH * K, 0.5, 0.4)), ["ДВЕ СМЕСИ"], "почему мы кормим так", 0.72)), "2 928")
    B[(3, 2)] = ui(down(paper_card(up(B[(3, 2)]).crop((0, 190 * K, TW * K, 505 * K)), ["ПОКУПКИ", "ДЛЯ МАЛЫШКИ"], "68 размер")), "552")
    return A, B

def grid(T, gap=3):
    im = Image.new("RGB", (3 * TW + 2 * gap, 4 * TH + 3 * gap), (20, 20, 20))
    for (r, c), t in T.items(): im.paste(t, (c * (TW + gap), r * (TH + gap)))
    return im

if __name__ == "__main__":
    s1, s2, repo, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
    A0, B0 = tiles(s1, 0), tiles(s2, 1); A, B = build(s1, s2, repo)
    full = Image.new("RGB", (3 * TW + 6, 8 * TH + 21), (20, 20, 20)); full.paste(grid(A), (0, 0)); full.paste(grid(B), (0, 4 * TH + 12))
    full.save(os.path.join(out, "feed_new.jpg"), quality=90)
    ba = Image.new("RGB", (2 * (3 * TW + 6) + 60, 8 * TH + 21), (255, 255, 255))
    for k, (a, b) in enumerate(((A0, B0), (A, B))):
        col = Image.new("RGB", (3 * TW + 6, 8 * TH + 21), (20, 20, 20)); col.paste(grid(a), (0, 0)); col.paste(grid(b), (0, 4 * TH + 12))
        ba.paste(col, (k * (3 * TW + 66), 0))
    ba.save(os.path.join(out, "feed_before_after.jpg"), quality=85); print("ok")
