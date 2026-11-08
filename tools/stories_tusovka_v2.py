"""@mcg.va — stories «женская тусовка» v2, in the style of her posts but 9:16: light background, the whole photo (no zoom)
in a frame, thin Montserrat text under it (thin CAPS for the main line, thin body, thin italic accents). Nothing overlaps.
  python3 -I stories_tusovka_v2.py PHOTOS_DIR OUT_DIR
"""
import os, sys
from PIL import Image, ImageDraw, ImageFilter, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carousel_top4 import font, wrap, INK, MAUVE, BLUSH
W, H, M = 1080, 1920, 80
BG = (248, 245, 240)
TOP, BOTTOM = 300, 1560                                      # Instagram safe zone for stories
STY = {"main": (lambda: font(48, 250), 1.3, True), "second": (lambda: font(36, 300), 1.5, False), "script": (lambda: font(36, 300, True), 1.5, False)}

def lines_of(blocks, width=W - 2 * M):
    out = []
    for k, (kind, text) in enumerate(blocks):
        mk, lead, caps = STY[kind]; f = mk(); t = text.upper() if caps else text
        if k: out.append((None, None, 18))
        out += [(" ".join(l), f, int(f.size * lead)) for l in wrap(t, f, width)]
    return out

def story(src, blocks, out):
    im = Image.new("RGB", (W, H), BG)
    ph = ImageOps.exif_transpose(Image.open(src)).convert("RGB"); w = W - 2 * M; h = round(ph.height * w / ph.width)
    ph = ph.resize((w, h), Image.LANCZOS)
    ls = lines_of(blocks); th = sum(s for _, _, s in ls); gap = 70
    y0 = TOP + max(0, (BOTTOM - TOP - h - gap - th) // 2)       # photo + text block centred in the safe zone
    sh = Image.new("RGBA", (w + 80, h + 80), (0, 0, 0, 0)); ImageDraw.Draw(sh).rectangle((40, 50, w + 40, h + 50), fill=(60, 40, 30, 60))
    im.paste(sh.filter(ImageFilter.GaussianBlur(18)), (M - 40, y0 - 40), sh.filter(ImageFilter.GaussianBlur(18)))
    im.paste(ph, (M, y0)); d = ImageDraw.Draw(im); y = y0 + h + gap
    for text, f, step in ls:
        if text: d.text((M, y + f.size), text, font=f, fill=INK, anchor="ls")
        y += step
    im.save(out, quality=95); print(out, y0, y0 + h, y)

def full(src, blocks, out, fx=0.5, where="bottom", width=W - 2 * M):
    """full-screen photo, the text in the calm top or bottom of the safe zone: ink on calm light ground, otherwise white
    with a soft shadow over a light black gradient"""
    import numpy as np
    ph = ImageOps.exif_transpose(Image.open(src)).convert("RGB"); s = H / ph.height; ph = ph.resize((round(ph.width * s), H), Image.LANCZOS)
    x = int(min(max(fx * ph.width - W / 2, 0), ph.width - W)); im = ph.crop((x, 0, x + W, H)).convert("RGBA")
    ls = lines_of(blocks, width); th = sum(st for _, _, st in ls); y0 = (BOTTOM - th) if where == "bottom" else TOP + 40
    reg = np.asarray(im.convert("L"), np.float32)[y0:y0 + th, M:W - M]; light = reg.mean() > 175 and reg.std() < 30
    col = INK if light else (255, 255, 255)
    if not light:
        g = Image.new("L", (1, H)); c = y0 + th / 2; r = th / 2 + 260
        for yy in range(H): g.putpixel((0, yy), int(255 * 0.42 * max(0.0, 1 - abs(yy - c) / r) ** 0.8))
        hz = Image.new("RGBA", (W, H), (12, 8, 6, 0)); hz.putalpha(g.resize((W, H))); im.alpha_composite(hz)
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ds = ImageDraw.Draw(sh); y = y0
        for text, f, step in ls:
            if text: ds.text((M, y + f.size + 2), text, font=f, fill=(15, 10, 8, 170), anchor="ls")
            y += step
        im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(5)))
    d = ImageDraw.Draw(im); y = y0
    for text, f, step in ls:
        if text: d.text((M, y + f.size), text, font=f, fill=col, anchor="ls")
        y += step
    im.convert("RGB").save(out, quality=95); print(out, "full", where, "ink" if light else "white")

FULL = {1: dict(fx=0.5, where="bottom"), 3: dict(fx=0.55, where="top"), 4: dict(fx=0.48, where="top"),
        6: dict(fx=0.5, where="top", width=400), 7: dict(fx=0.42, where="top")}

P = lambda src, h: os.path.join(src, next(f for f in os.listdir(src) if f.startswith(h)))
S = [("54E3497C", [("second", "Сегодня у нас, как всегда,"), ("main", "женская тусовка!")]),
     ("1D8618DB", [("second", "Папа уехал на тренировку,"), ("script", "а мы с Агатиком уже успели столько всего сделать.")]),
     ("51EA4A83", [("main", "Попробовали тыкву!"), ("script", "Нам понравилось,")]),
     ("8441869F", [("second", "правда, потом вся эта тыква оказалась на кровати.")]),
     ("A09CF331", [("second", "После прикорма, естественно, пошли принимать ванну,")]),
     ("9CC28CF9", [("second", "потому что чистыми после еды мы теперь не остаёмся.")]),
     ("329009F6", [("second", "Мама успела выпить холодный кофе"), ("script", "и даже позавтракать!")]),
     ("B61AB164", [("main", "Уже успех.")]),
     ("BE0BD576", [("second", "Потом мы много-много играли,"), ("script", "а впереди у нас ещё любимая мамина прогулка под сон Агатика.")]),
     ("6391B2B6", [("main", "Обожаю гулять, пока она спит."), ("second", "Можно хотя бы немного побыть со своими мыслями"), ("script", "и насладиться тишиной.")])]

if __name__ == "__main__":
    src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    for i, (h, b) in enumerate(S, 1):
        o = os.path.join(out, f"story_{i:02d}.jpg")
        if i in FULL: full(P(src, h), b, o, **FULL[i])
        else: story(P(src, h), b, o)
