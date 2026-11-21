"""@mcg.va — the whole visible grid (24 posts from two screenshots) redone in ONE cover style: warm photo, thin spaced
Montserrat caps headline + thin italic line at the same height on every tile, soft haze only behind the text.
Clean sources are used where they exist; otherwise the clean part of the screenshot tile. Headlines marked «?» in TOPIC_GUESS
are guesses from the picture.
  python3 -I feed_unified.py SHOT1 SHOT2 SCRATCH OUT_DIR
"""
import os, sys
from PIL import Image, ImageDraw, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feed_recover as FR
FR.HAZE, FR.CAPS_W = 0.6, 260
from feed_recover import tiles, cover, warm, overlay, down, up, grid, TW, TH, K, mont
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except ImportError: pass

def ui(tile, views, kind):
    t = tile.convert("RGBA"); s = 4; from PIL import ImageFilter
    L = Image.new("RGBA", (TW * s, TH * s), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    if kind == "reel":
        d.rounded_rectangle((382 * s, 16 * s, 414 * s, 48 * s), 8 * s, fill=(255, 255, 255, 240))
        d.polygon([(394 * s, 24 * s), (394 * s, 40 * s), (406 * s, 32 * s)], fill=(40, 40, 40, 255))
    else:
        d.rounded_rectangle((388 * s, 16 * s, 414 * s, 42 * s), 5 * s, outline=(255, 255, 255, 240), width=3 * s)
        d.rounded_rectangle((382 * s, 22 * s, 408 * s, 48 * s), 5 * s, fill=(255, 255, 255, 240))
    sh = Image.new("RGBA", L.size, (0, 0, 0, 0)); dsh = ImageDraw.Draw(sh)
    for dd, col in ((dsh, (0, 0, 0, 120)), (d, (255, 255, 255, 255))):
        dd.ellipse((16 * s, 548 * s, 40 * s, 564 * s), outline=col, width=3 * s); dd.ellipse((24 * s, 552 * s, 32 * s, 560 * s), fill=col)
        dd.text((48 * s, 566 * s), views, font=mont(30 * s, 600), fill=col, anchor="ls")
    t.alpha_composite(sh.filter(ImageFilter.GaussianBlur(3 * s)).resize((TW, TH), Image.LANCZOS)); t.alpha_composite(L.resize((TW, TH), Image.LANCZOS))
    return t.convert("RGB")

def clean_tile(t, y0=60, y1=505):            # screenshot tile without its old icon / counter (and old text if any)
    return up(t).crop((0, y0 * K, TW * K, y1 * K))

def file(p, box=None):
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB"); return im.crop(box) if box else im

def main(s1, s2, sc, repo, out):
    A, B = tiles(s1, 0), tiles(s2, 1); P = lambda *a: os.path.join(sc, *a); R = lambda p: os.path.join(repo, p)
    # (screen, row, col): (source image, fx, fy, caps lines, italic, views, kind)
    S = {
     (0, 0, 0): (clean_tile(A[(0, 0)], 20, 380), .5, .3, ["СРЕДСТВА"], "для малышки и мамы", "302", "car"),
     (0, 0, 1): (file(P("st3/dl/B61AB164-F3AC-461E-A95C-C0D4E0E40DE3.png")), .6, .5, ["ПАТЧИ"], "5 минут для себя", "318", "car"),
     (0, 0, 2): (file(P("prot/dl/72BC58DA-326D-4D9C-933E-BE9E71C75B77.png")), .5, .45, ["МАТЕРИНСТВО"], "прекрасно и одновременно…", "407", "reel"),
     (0, 1, 0): (clean_tile(A[(1, 0)]), .5, .5, ["УХОД"], "для мамы и малышки", "401", "reel"),
     (0, 1, 1): (file(P("dep/bed.jpg")), .5, .2, ["ДЕПРЕССИЯ"], "покидает моё тело, когда…", "352", "car"),
     (0, 1, 2): (file(P("mat/cover_bg.png")), .5, .2, ["МАТЕРИНСТВО"], "это не про ребёнка", "1 609", "reel"),
     (0, 2, 0): (file(P("top4/cover_belly.jpg")), .5, .45, ["4 ВЕЩИ"], "которые я бы купила снова", "353", "car"),
     (0, 2, 1): (clean_tile(A[(2, 1)]), .5, .5, ["ГАРДЕРОБ"], "малышки на осень", "341", "reel"),
     (0, 2, 2): (clean_tile(A[(2, 2)], 20, 410), .5, .3, ["СПАСИБО МУЖУ"], "за то, что всегда рядом", "763", "car"),
     (0, 3, 0): (clean_tile(A[(3, 0)]), .5, .4, ["ДЕНЬ С НАМИ"], "малышке почти 6 месяцев", "1 934", "reel"),
     (0, 3, 1): (clean_tile(A[(3, 1)]), .5, .5, ["БЕЛЫЙ ШУМ"], "мастхэв с первых дней", "335", "reel"),
     (0, 3, 2): (clean_tile(A[(3, 2)]), .5, .5, ["БЕРЕМЕННОСТЬ"], "как это было у меня", "799", "reel"),
     (1, 0, 0): (file(R("reels/rody_part1_cover.jpg"), (0, 560, 1080, 1920)), .5, .2, ["РОДЫ"], "часть 1", "1 419", "reel"),
     (1, 0, 1): (clean_tile(B[(0, 1)]), .5, .5, ["ПЕРВЫЕ ДНИ"], "дома с малышкой", "475", "reel"),
     (1, 0, 2): (file(R("sling_cover.jpg"), (0, 180, 1080, 1290)), .5, .3, ["СЛИНГ"], "лучшая покупка для мамы", "416", "reel"),
     (1, 1, 0): (clean_tile(B[(1, 0)]), .5, .5, ["НА ИВ"], "наш опыт", "390", "reel"),
     (1, 1, 1): (file(R("reels_cover_mini_vlog.jpg"), (0, 470, 1080, 1920)), .5, .4, ["МИНИ-ВЛОГ"], "с 4,5-месячным малышом", "1 202", "reel"),
     (1, 1, 2): (up(B[(1, 2)]).crop((10 * K, 60 * K, 235 * K, 470 * K)), .5, .5, ["ЧТО ДЕЛАТЬ"], "шпаргалка для беременных", "322", "car"),
     (1, 2, 0): (clean_tile(B[(2, 0)]), .5, .5, ["ПАПА И ДОЧКА"], "", "509", "reel"),
     (1, 2, 1): (clean_tile(B[(2, 1)]), .5, .3, ["УСТАЛОСТЬ"], "честно о материнстве", "364", "reel"),
     (1, 2, 2): (clean_tile(B[(2, 2)], 0, 385), .5, .15, ["Я ДУМАЛА,"], "нас это обойдёт…", "6 300", "reel"),
     (1, 3, 0): (clean_tile(B[(3, 0)]), .5, .5, ["СОСКА"], "брать или нет", "525", "reel"),
     (1, 3, 1): (clean_tile(B[(3, 1)], 30, 360), .5, .4, ["ДВЕ СМЕСИ"], "почему мы кормим так", "2 928", "reel"),
     (1, 3, 2): (clean_tile(B[(3, 2)], 190, 505), .5, .5, ["ПОКУПКИ"], "для малышки · 68 размер", "552", "reel"),
    }
    G = ({}, {})
    for (s, r, c), (src, fx, fy, caps, it, v, kind) in S.items():
        G[s][(r, c)] = ui(down(overlay(warm(cover(src, TW * K, TH * K, fx, fy)), caps, it, 0.74)), v, kind)
    os.makedirs(out, exist_ok=True)
    full = Image.new("RGB", (3 * TW + 6, 8 * TH + 21), (20, 20, 20)); full.paste(grid(G[0]), (0, 0)); full.paste(grid(G[1]), (0, 4 * TH + 12))
    full.save(os.path.join(out, "feed_unified.jpg"), quality=90); print("ok")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], ".", sys.argv[4])
