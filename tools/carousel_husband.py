"""@mcg.va — carousel post «спасибо мужу» (1080x1350): a full-bleed cover, then cream «paper» slides — her photo in a frame,
thin Montserrat text (small spaced CAPS kicker, thin lowercase body, thin italic accents), photo above / below the text in turns.
Every word of her text is kept, in order.
  python3 -I carousel_husband.py PHOTOS_DIR OUT_DIR
"""
import os, sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
HERE = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(HERE, "..", "fonts", "stories")
W, H, M = 1080, 1350, 76
CREAM, INK, BROWN, MAUVE, BLUSH = (243, 236, 228), (43, 31, 26), (107, 74, 56), (139, 107, 92), (232, 201, 192)

def font(size, wght, it=False):
    f = ImageFont.truetype(os.path.join(FD, "Montserrat-Italic-VariableFont_wght.ttf" if it else "Montserrat-VariableFont_wght.ttf"), size)
    f.set_variation_by_axes([wght]); return f

def wrap(text, f, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if f.getlength(t) <= maxw or not cur: cur = t
        else: lines.append(cur); cur = w
    return lines + ([cur] if cur else [])

def spaced(d, xy, text, f, fill, track, anchor_center=False):
    widths = [f.getlength(c) + track for c in text]; tw = sum(widths) - track; x, y = xy
    if anchor_center: x -= tw / 2
    for c, w in zip(text, widths): d.text((x, y), c, font=f, fill=fill, anchor="ls"); x += w
    return tw

def crop(im, w, h, fx, fy):
    s = max(w / im.width, h / im.height); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    x = int(min(max(fx * im.width - w / 2, 0), im.width - w)); y = int(min(max(fy * im.height - h / 2, 0), im.height - h))
    return im.crop((x, y, x + w, y + h))

BODY = (34, 300, 1.5); ITAL = (34, 300, 1.5)
def text_block(paras, width):
    """paras: [(text, 'b'|'i')] → list of (line, font) with paragraph gaps (None)"""
    out = []
    for k, (t, st) in enumerate(paras):
        f = font(BODY[0], BODY[1], it=(st == "i"))
        if k: out.append(None)
        out += [(l, f) for l in wrap(t, f, width)]
    return out

def text_h(lines):
    return sum(int(BODY[0] * 0.55) if l is None else int(BODY[0] * BODY[2]) for l in lines)

def paper(src, kicker, paras, out, fx=0.5, fy=0.4, photo_top=True, n=None, total=None):
    bg = Image.new("RGB", (W, H), CREAM); d = ImageDraw.Draw(bg)
    lines = text_block(paras, W - 2 * M); th = 46 + text_h(lines)                 # kicker + body
    ph = H - 2 * M - th - 64 - 40                                                  # photo gets what the text leaves
    pic = crop(ImageOps.exif_transpose(Image.open(src)).convert("RGB"), W - 2 * M, ph, fx, fy)
    py = M if photo_top else H - M - 40 - ph; ty = M + ph + 64 if photo_top else M + 10
    bg.paste(pic, (M, py))
    spaced(d, (M, ty + 22), kicker, font(22, 300), MAUVE, 5)
    y = ty + 46 + int(BODY[0] * 1.05)
    for l in lines:
        if l is None: y += int(BODY[0] * 0.55); continue
        d.text((M, y), l[0], font=l[1], fill=INK, anchor="ls"); y += int(BODY[0] * BODY[2])
    fy_ = H - M + 14 if photo_top else H - M + 14                                   # footer: page and handle, hairline
    f = font(18, 300); d.line((M, H - M - 12, W - M, H - M - 12), fill=BLUSH, width=1)
    d.text((M, fy_), "@mcg.va", font=f, fill=MAUVE, anchor="ls"); d.text((W - M, fy_), f"{n:02d} / {total:02d}", font=f, fill=MAUVE, anchor="rs")
    if not photo_top: pass
    bg.save(out, quality=95); print(out, "photo h", ph, "overflow" if ph < 360 else "")

def cover(src, out, fx=0.5, fy=0.45):
    im = crop(ImageOps.exif_transpose(Image.open(src)).convert("RGB"), W, H, fx, fy).convert("RGBA")
    g = Image.new("L", (1, H)); [g.putpixel((0, y), int(255 * max(0, (y - H * 0.6) / (H * 0.4)) ** 1.3 * 0.62)) for y in range(H)]
    hz = Image.new("RGBA", (W, H), (20, 13, 10, 0)); hz.putalpha(g.resize((W, H))); im.alpha_composite(hz)
    t = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sh = Image.new("L", (W, H), 0); d, ds = ImageDraw.Draw(t), ImageDraw.Draw(sh)
    for dd, col in ((ds, 140), (d, (255, 255, 255, 255))):
        spaced(dd, (W / 2, H - 300), "ПОСТ БЛАГОДАРНОСТИ", font(24, 300), col, 7, True)
        spaced(dd, (W / 2, H - 200), "СПАСИБО МУЖУ", font(76, 200), col, 10, True)
        dd.text((W / 2, H - 132), "за то, что всегда рядом", font=font(36, 250, True), fill=col, anchor="ms")
    s = Image.new("RGBA", (W, H), (15, 10, 8, 0)); s.putalpha(sh.filter(ImageFilter.GaussianBlur(8))); im.alpha_composite(s); im.alpha_composite(t)
    im.convert("RGB").save(out, quality=95); print(out)

S = [  # photo, kicker, [(paragraph, style)], fx, fy
    ("IMG_5644.jpg", "ПОСТ БЛАГОДАРНОСТИ", [
        ("Этот пост благодарности я решила посвятить своему мужу!", "i"),
        ("Я, если честно, вообще не представляю, как девушки проходят через беременность, роды и воспитывают деток без поддержки мужчины. Для меня вы просто героини!", "b")], 0.5, 0.3),
    ("0L2A9185.jpg", "ВЕСЬ ЭТОТ ПУТЬ", [
        ("И я хочу сказать огромное спасибо своему мужу за то, что он всегда был и остаётся рядом.", "b"),
        ("Мы вместе прошли весь этот путь. От новости о беременности до родов, того самого ада в послеродовом отделении и сейчас вместе проживаем одновременно самое прекрасное и самое сложное время в нашей жизни.", "b")], 0.45, 0.6),
    ("DSC06164.jpg", "БЕРЕМЕННОСТЬ", [
        ("Я бесконечно благодарна ему за то, что он с самого начала был вовлечён во всё.", "i"),
        ("За то, как во время беременности он мог в два часа ночи поехать за чем-нибудь сладеньким для меня. Как приносил цветочки просто так. Как делал массаж, когда отекали ноги. Как заботился обо мне и старался сделать всё, чтобы мне было легче.", "b")], 0.5, 0.25),
    ("DSC06185.jpg", "РОДЫ", [
        ("За то, что был рядом на родах. И я сейчас вообще не преувеличиваю, когда говорю, что без него я бы не справилась. Стоило ему хотя бы немного отойти от меня, как у меня сразу пропадали силы. Мне настолько важно было чувствовать, что он рядом.", "b")], 0.5, 0.55),
    ("0L2A7733.jpg", "РОДДОМ", [
        ("Я никогда не забуду, как он сидел со мной в роддоме до двух часов ночи. Как чуть ли не ногой выбивал двери, чтобы его пустили к нам. Как защищал нас и каждый день приезжал, чтобы побыть рядом.", "b")], 0.5, 0.33),
    ("DSC06475 - н.jpg", "ПАПА С ПЕРВЫХ СЕКУНД", [
        ("И как с первых секунд после рождения дочки он начал заботиться о ней.", "i"),
        ("Мне даже не пришлось объяснять, что мне нужна помощь с ребёнком. Он просто с самого начала был папой. Не человеком, который иногда помогает маме, а полноценным родителем.", "b")], 0.5, 0.5),
    ("IMG_0218.JPG", "ВМЕСТЕ", [
        ("Как мы вместе проходили бессонные ночи, учились понимать нашу малышку, переживали, уставали, иногда вообще не понимали, что делать.", "b"),
        ("И до сих пор он каждый день заботится о нас, помогает, поддерживает и делает столько всего, что я даже не всегда успеваю это замечать.", "b")], 0.5, 0.62),
    ("IMG_5705.jpg", "ПОВЕЗЛО", [
        ("Я смотрю на него с дочкой и каждый раз думаю, как же ей повезло с папой. А мне с мужем.", "i")], 0.5, 0.38),
    ("wed-110.jpg", "ЧЕСТНО", [
        ("Конечно, у нас тоже бывают сложные моменты. Мы устаём, можем спорить, иногда оба на пределе. Но мне очень ценно, что через всё это мы проходим вместе.", "b")], 0.5, 0.35),
    ("0L2A9008.jpg", "БЕЗ НЕГО", [
        ("Я не знаю, как сложилось бы моё материнство без него рядом. И даже не хочу представлять.", "b")], 0.5, 0.45),
    ("DSC06189.jpg", "СПАСИБО ТЕБЕ", [
        ("Спасибо тебе за нашу семью, за твою любовь, заботу и за то, каким папой ты стал для нашей дочки.", "b"),
        ("Я вас очень люблю!", "i")], 0.5, 0.3),
]

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); total = len(S) + 1
    cover(os.path.join(src, "IMG_5772.jpg"), os.path.join(out, "slide_01.jpg"), 0.5, 0.42)
    for i, (p, k, paras, fx, fy) in enumerate(S):
        paper(os.path.join(src, p), k, paras, os.path.join(out, f"slide_{i + 2:02d}.jpg"), fx, fy, photo_top=(i % 2 == 0), n=i + 2, total=total)
