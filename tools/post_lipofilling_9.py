"""9-slide carousel 'Неожиданные возможности собственной жировой ткани' (1080x1350).

Post style of the brand (reference posts): navy, powder Bebas headlines, thin Montserrat text,
ANNA TVAURI mark. Zone slides show the patient's ДО and ПОСЛЕ side by side (or stacked) with
annotated arrows: what was there before (white) and what changed after (powder).

usage: python3 post_lipofilling_9.py MATERIALS_DIR OUT_DIR
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter

SRC, OUT = sys.argv[1:3]
sys.argv = [sys.argv[0], os.path.join(SRC, "DSC01141.JPG"), OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surgeon_style as ss  # noqa: E402
from surgeon_style import Slide, bebas, mont, logo, vshade, icon_circle, POWDER, NAVY, DEEP, WHITE, INK, S  # noqa: E402

PW, PH = 1080, 1350
TOTAL = 9
_cache = {}


def photo(name, max_side=2400):
    if name not in _cache:
        im = ImageOps.exif_transpose(Image.open(os.path.join(SRC, name))).convert("RGB")
        if max(im.size) > max_side: im.thumbnail((max_side, max_side), Image.LANCZOS)
        _cache[name] = im
    return _cache[name]


def fit(im, w, h, cx=0.5, cy=0.5):
    s = max(w / im.width, h / im.height)
    im = im.resize((max(w, round(im.width * s)), max(h, round(im.height * s))), Image.LANCZOS)
    x, y = int((im.width - w) * cx), int((im.height - h) * cy)
    return im.crop((x, y, x + w, y + h))


def card(sl, im, box, r=24):
    x0, y0, x1, y1 = box; w, h = int((x1 - x0) * S), int((y1 - y0) * S)
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=r * S, fill=255)
    sl.shadow(box, r, 20, 120)
    sl.im.paste(fit(im, w, h), (int(x0 * S), int(y0 * S)), m)


def bg_photo(name, cx=0.5, cy=0.5, tint=NAVY, k=0.45, dark=0.35, blur=0, gray=0.2):
    im = fit(photo(name), PW * S, PH * S, cx, cy)
    if blur: im = im.filter(ImageFilter.GaussianBlur(blur * S))
    a = np.asarray(im).astype(np.float32) / 255
    a = a * (1 - gray) + a.mean(2, keepdims=True) * gray
    a = a * (1 - k) + np.float32(tint) / 255 * k * (0.4 + 0.6 * a.mean(2, keepdims=True) * 1.5)
    return Image.fromarray(np.clip(a * (1 - dark) * 255, 0, 255).astype(np.uint8))


def base():
    sl = Slide(PW, PH); sl.rrect((0, 0, PW, PH), 0, NAVY); vshade(sl, 0, PH, 0.0, 0.5, DEEP)
    logo(sl, 84); return sl


def counter(sl, n):
    sl.text((PW - 80, 1306), f"{n:02d} / {TOTAL:02d}", mont(24, 400), WHITE, "rs", a=170, spacing=2)


def swipe(sl, x, y, col=POWDER):
    for i in range(3):
        cx = x + i * 26; sl.poly([(cx, y - 13), (cx + 13, y), (cx, y + 13)], col, 2.4, 255 - i * 60)


def wrap(text, f, width):
    out, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if f.getlength(t) / S <= width or not line: line = t
        else: out.append(line); line = w
    return out + [line]


def para(sl, x, y, text, size=29, w=300, col=WHITE, width=920, lh=1.32, a=255):
    f = mont(size, w); rows = wrap(text, f, width)
    for i, r in enumerate(rows): sl.text((x, y + i * size * lh), r, f, col, a=a)
    return y + len(rows) * size * lh


def heart(sl, cx, cy, r, col=POWDER):
    t = np.linspace(0, 2 * np.pi, 120)
    xs = 16 * np.sin(t) ** 3; ys = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = [((cx + x * r / 16) * S, (cy + y * r / 16) * S) for x, y in zip(xs, ys)]
    sl.d.polygon(pts, fill=col + (255,))


# ------------------------------------------------------------------ annotations
def pill(sl, x, y, text, bg, fg, size=24, w=500, anchor="lt"):
    f = mont(size, w); tw = f.getlength(text) / S; ph = size + 22
    if anchor == "mm": x, y = x - (tw + 32) / 2, y - ph / 2
    sl.rrect((x, y, x + tw + 32, y + ph), ph / 2, bg, 235)
    sl.text((x + 16, y + ph / 2), text, f, fg, "lm")
    return (x, y, x + tw + 32, y + ph)


def annotate(sl, box, label, lpos, pts, col):
    """label pill at lpos (fractions of the card), an arrow from it to every point (fractions)"""
    x0, y0, x1, y1 = box; W_, H_ = x1 - x0, y1 - y0
    fg = NAVY
    lx, ly = x0 + lpos[0] * W_, y0 + lpos[1] * H_
    pb = pill(sl, lx, ly, label, col, fg, 23, 500, "mm")
    for px, py in pts:
        tx, ty = x0 + px * W_, y0 + py * H_
        # start on the pill edge facing the target
        sx = min(max(tx, pb[0] + 22), pb[2] - 22); sy = pb[1] if ty < ly else pb[3]
        if abs(ty - ly) < 30: sy = ly; sx = pb[0] if tx < lx else pb[2]
        mx, my = (sx + tx) / 2, (sy + ty) / 2
        bend = 0.18 * (1 if tx > sx else -1)
        ctrl = (mx - (ty - sy) * bend, my + (tx - sx) * bend)
        path = ss.bez((sx, sy), ctrl, (tx, ty), n=40)
        # stop a little before the point so the ring stays visible
        path = path[:-4]
        # dark under-stroke so the arrows read on light photos (hands, eyes) as well as on black
        sl.poly(path, DEEP, 6, 120); sl.head(path, DEEP, 6, 16, 120); sl.circle((tx, ty), 9, DEEP, 5.5, 120)
        sl.poly(path, col, 2.6); sl.head(path, col, 2.6, 15)
        sl.circle((tx, ty), 9, col, 2.4)


def compare(sl, area, before, after, layout="side", gap=22):
    """before/after = (image, [annotations]); annotation = (label, lpos, [points])"""
    ax0, ay0, ax1, ay1 = area; aw, ah = ax1 - ax0, ay1 - ay0
    asp = before[0].width / before[0].height
    if layout == "side":
        w = (aw - gap) / 2; h = w / asp
        if h > ah: h = ah; w = h * asp
        x = ax0 + (aw - 2 * w - gap) / 2; y = ay0 + (ah - h) / 2
        boxes = [(x, y, x + w, y + h), (x + w + gap, y, x + 2 * w + gap, y + h)]
    else:
        h = (ah - gap) / 2; w = h * asp
        if w > aw: w = aw; h = w / asp
        x = ax0 + (aw - w) / 2; y = ay0 + (ah - 2 * h - gap) / 2
        boxes = [(x, y, x + w, y + h), (x, y + h + gap, x + w, y + 2 * h + gap)]
    for (im, notes), box, tag, col in zip((before, after), boxes, ("ДО", "ПОСЛЕ"), (WHITE, POWDER)):
        card(sl, im, tuple(int(v) for v in box))
        for label, lpos, pts in notes: annotate(sl, box, label, lpos, pts, col)
        pill(sl, box[0] + 16, box[1] + 16, tag, NAVY if tag == "ДО" else POWDER, WHITE if tag == "ДО" else NAVY, 24, 600)


def zone_slide(n, kicker, quote, body, before, after, layout="side", quote_size=72, body_size=29):
    sl = base()
    sl.text((80, 150), kicker, mont(24, 500), POWDER, spacing=3, a=220)
    y = sl.lines((80, 186), quote, bebas(quote_size), POWDER, lh=0.92)
    y += 14
    for p in body:
        if isinstance(p, tuple):  # bullets
            for b in p:
                sl.dot((92, y + body_size * 0.62), 5, POWDER)
                y = para(sl, 112, y, b, body_size, width=890) + 4
        else:
            y = para(sl, 80, y, p, body_size) + 8
    compare(sl, (60, y + 18, 1020, 1272), before, after, layout)
    counter(sl, n)
    return sl.save(f"{n:02d}.jpg")


def crop(name, box):
    return photo(name, 4000).crop(box)


# ------------------------------------------------------------------ slides
def s1():
    sl = Slide(PW, PH, bg_photo("DSC01141.JPG", cx=0.55, cy=0.18, k=0.4, dark=0.25))
    vshade(sl, 0, 240, 0.55, 0.0, DEEP); vshade(sl, 470, 1000, 0.0, 0.95, DEEP)
    logo(sl, 84)
    y = sl.lines((80, 700), ["А ВЫ ЗНАЛИ, ЧТО СОБСТВЕННЫМ", "ЖИРОМ МОЖНО СКОРРЕКТИРОВАТЬ", "ДАЖЕ ФОРМУ НОГ?"], bebas(84), POWDER, lh=0.92)
    sl.poly([(82, y + 30), (200, y + 30)], POWDER, 2)
    y = para(sl, 80, y + 60, "5 возможностей липофилинга, о которых вы могли не знать.", 34, 300, WHITE, 760)
    swipe(sl, 86, y + 50)
    counter(sl, 1)
    return sl.save("01.jpg")


def step_icon(sl, c, kind, r=58):
    x, y = c; sl.circle(c, r, POWDER, 2)
    if kind == "take":     # cannula drawing fat out: drop + arrow up
        sl.d.polygon([(px * S, py * S) for px, py in [(x, y - 26), (x + 16, y + 2), (x + 16, y + 8)] +
                      [(x + 16 * math.cos(a), y + 8 + 16 * math.sin(a)) for a in np.linspace(0, math.pi, 30)] +
                      [(x - 16, y + 8), (x - 16, y + 2)]], outline=POWDER + (255,), width=int(2.4 * S))
    elif kind == "prep":   # syringe
        sl.poly([(x - 30, y + 14), (x + 14, y - 30)], POWDER, 2.4)
        sl.d.polygon([((x - 22 + dx) * S, (y + 22 + dy) * S) for dx, dy in [(0, 0), (-8, -8), (28, -44), (36, -36)]],
                     outline=POWDER + (255,), width=int(2.2 * S))
        sl.poly([(x + 10, y - 34), (x + 26, y - 18)], POWDER, 2.4); sl.poly([(x + 18, y - 26), (x + 30, y - 38)], POWDER, 2.4)
        sl.poly([(x - 26, y + 26), (x - 36, y + 36)], POWDER, 2.4)
    elif kind == "place":  # target / volume added
        sl.circle(c, 26, POWDER, 2.2); sl.circle(c, 12, POWDER, 2.2)
        sl.poly([(x, y - 34), (x, y + 34)], POWDER, 1.6, 140); sl.poly([(x - 34, y), (x + 34, y)], POWDER, 1.6, 140)


def s2():
    sl = Slide(PW, PH, bg_photo("DSC00132.JPG", cx=0.5, cy=0.55, k=0.55, dark=0.55, blur=2))
    vshade(sl, 0, 700, 0.75, 0.35, DEEP); vshade(sl, 900, PH, 0.35, 0.85, DEEP)
    logo(sl, 84)
    y = sl.lines((80, 170), ["ЖИР, ОТ КОТОРОГО ВЫ ХОТИТЕ", "ИЗБАВИТЬСЯ, МОЖЕТ", "НАМ ПРИГОДИТЬСЯ"], bebas(84), POWDER, lh=0.92)
    y = para(sl, 80, y + 24, "Во время липосакции мы забираем собственную жировую ткань, подготавливаем её "
                             "и пересаживаем в те зоны, где не хватает объёма.", 31, 300, WHITE, 900)
    # 3-step scheme
    cy = y + 150; xs = (220, 540, 860)
    for (x, kind, t1, t2) in zip(xs, ("take", "prep", "place"),
                                 ("забираем", "подготавливаем", "пересаживаем"), ("при липосакции", "жировую ткань", "в нужную зону")):
        sl.circle((x, cy), 58, NAVY, 0, 255, fill=NAVY + (200,))
        step_icon(sl, (x, cy), kind)
        sl.text((x, cy + 92), t1, mont(30, 600), WHITE, "ma")
        sl.text((x, cy + 132), t2, mont(26, 300), WHITE, "ma", a=220)
    for a, b in ((xs[0], xs[1]), (xs[1], xs[2])):
        pts = ss.bez((a + 72, cy), ((a + b) / 2, cy - 30), (b - 72, cy), n=30)
        sl.dashed(pts, POWDER, 2, 9, 8, 230); sl.head(pts, POWDER, 2, 13)
    y = cy + 230
    sl.poly([(82, y), (200, y)], POWDER, 2)
    y = para(sl, 80, y + 30, "А теперь покажу на примерах моих пациенток, что именно можно скорректировать.", 32, 400, WHITE, 900)
    swipe(sl, 86, y + 40)
    counter(sl, 2)
    return sl.save("02.jpg")


def s3():
    b = crop("IMG_0665.JPG", (40, 700, 680, 1240)); a = crop("IMG_0665.JPG", (40, 40, 680, 580))
    return zone_slide(3, "01 · ГРУДЬ", ["«ХОЧУ УВЕЛИЧИТЬ ГРУДЬ,", "НО НЕ ХОЧУ ИМПЛАНТЫ»"],
                      ["С помощью липофилинга мы можем:",
                       ("добавить объём и наполненность груди;", "скорректировать небольшую асимметрию;",
                        "сохранить естественную мягкость груди.")],
                      (b, [("мало объёма", (0.5, 0.86), [(0.30, 0.43), (0.67, 0.45)])]),
                      (a, [("объём и наполненность", (0.5, 0.86), [(0.27, 0.36), (0.68, 0.38)])]))


def s4():
    b = crop("IMG_0614.JPG", (0, 40, 627, 940)); a = crop("IMG_0614.JPG", (627, 40, 1254, 940))
    return zone_slide(4, "02 · ЯГОДИЦЫ И БЁДРА", ["«ХОЧЕТСЯ БОЛЕЕ ОКРУГЛЫХ ЯГОДИЦ", "И ВЫРАЖЕННОЙ ТАЛИИ»"],
                      ["Сочетая липосакцию и липофилинг ягодиц, мы можем перераспределить объёмы, скорректировать "
                       "западения в области бёдер и изменить пропорции всей фигуры."],
                      (b, [("лишний объём", (0.40, 0.06 + 0.08), [(0.76, 0.17)]),
                           ("нет чёткого контура", (0.52, 0.66), [(0.27, 0.50)])]),
                      (a, [("талия", (0.40, 0.06 + 0.08), [(0.78, 0.16)]),
                           ("округлые ягодицы", (0.55, 0.66), [(0.27, 0.38)])]), quote_size=66)


def s5():
    b = crop("IMG_0668.JPG", (0, 150, 512, 761)); a = crop("IMG_0668.JPG", (512, 150, 1024, 761))
    return zone_slide(5, "03 · ГОЛЕНИ", ["«МНЕ НИКОГДА НЕ НРАВИЛАСЬ", "ФОРМА МОИХ НОГ»"],
                      ["Иногда визуальная кривизна ног связана с недостатком мягких тканей в области голеней.",
                       "С помощью липофилинга можно восполнить недостающий объём и сделать контур ног более ровным."],
                      (b, [("зазор между голенями", (0.5, 0.88), [(0.485, 0.47)])]),
                      (a, [("ровный контур", (0.5, 0.88), [(0.29, 0.55), (0.76, 0.55)])]))


def s6():
    b = crop("IMG_0673.JPG", (110, 180, 970, 540)); a = crop("IMG_0673.JPG", (110, 720, 970, 1080))
    return zone_slide(6, "04 · НИЖНИЕ ВЕКИ", ["«Я ВЫСЫПАЮСЬ, НО ВСЁ РАВНО", "ВЫГЛЯЖУ УСТАВШЕЙ»"],
                      ["Причиной может быть дефицит объёма в области нижних век.",
                       "С помощью собственной жировой ткани мы можем восполнить этот объём, смягчить "
                       "носослёзную борозду и сделать взгляд более свежим."],
                      (b, [("носослёзная борозда", (0.5, 0.86), [(0.27, 0.62), (0.76, 0.62)])]),
                      (a, [("объём восполнен", (0.5, 0.86), [(0.27, 0.62), (0.76, 0.62)])]), layout="stack")


def s7():
    b = crop("IMG_0678.JPG", (0, 100, 640, 853)); a = crop("IMG_0678.JPG", (640, 100, 1280, 853))
    return zone_slide(7, "05 · КИСТИ РУК", ["«ЗА ЛИЦОМ УХАЖИВАЮ, А РУКИ", "ВСЁ РАВНО ВЫДАЮТ ВОЗРАСТ»"],
                      ["Со временем кисти рук теряют объём, становятся более заметными вены и сухожилия.",
                       "Липофилинг позволяет восполнить утраченный объём и визуально омолодить кисти рук."],
                      (b, [("вены и сухожилия", (0.5, 0.88), [(0.28, 0.45), (0.74, 0.57)])]),
                      (a, [("гладкий контур", (0.5, 0.88), [(0.36, 0.40), (0.63, 0.60)])]))


def s8():
    sl = base()
    y = sl.lines((80, 170), ["А МОЖНО ЛИ СКОРРЕКТИРОВАТЬ", "НЕСКОЛЬКО ЗОН", "ЗА ОДНУ ОПЕРАЦИЮ?"], bebas(86), POWDER, lh=0.92)
    y = para(sl, 80, y + 26, "Да, в некоторых случаях это возможно.", 34, 600, WHITE, 900)
    y = para(sl, 80, y + 18, "Поэтому, если вы уже планируете липосакцию или комплексную операцию, расскажите мне "
                             "обо всех зонах, которые вас беспокоят.", 30, 300, WHITE, 560)
    y = para(sl, 80, y + 18, "Возможно, собственная жировая ткань пригодится для решения ещё одной вашей задачи.",
             30, 300, POWDER, 560)
    # surgeon at work in an arch on the right
    x0, x1, top, bot = 680, 1010, 560, 1260
    w, h = int((x1 - x0) * S), int((bot - top) * S)
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1 + w // 2), radius=w // 2, fill=255)
    sl.shadow((x0, top, x1, bot), 60, 22, 120)
    sl.im.paste(fit(photo("IMG_9779.JPG"), w, h, 0.62, 0.3), (int(x0 * S), int(top * S)), m)
    # zones: one donor area feeding several targets
    pts = ss.bez((300, y + 50), (330, y + 160), (520, y + 200), (660, y + 150))
    sl.dashed(pts, POWDER, 2, 10, 9, 210); sl.head(pts, POWDER, 2, 14)
    counter(sl, 8)
    return sl.save("08.jpg")


def s9():
    sl = Slide(PW, PH, bg_photo("DSC01116.JPG", cx=0.5, cy=0.35, k=0.6, dark=0.55, blur=6))
    vshade(sl, 0, PH, 0.35, 0.75, DEEP)
    logo(sl, 84)
    y = sl.lines((80, 170), ["А В КАКУЮ ЗОНУ ВЫ БЫ", "ХОТЕЛИ ДОБАВИТЬ ОБЪЁМ?"], bebas(88), POWDER, lh=0.92)
    y = para(sl, 80, y + 26, "На консультации обсудим ваши пожелания, оценим донорские зоны, возможности липофилинга "
                             "и рассчитаем стоимость операции.", 31, 300, WHITE, 900)
    y += 30
    hx = 80; f = mont(31, 500)
    t = "Больше результатов показываю в актуальном «ЛИПОФИЛИНГ»"
    rows = wrap(t, f, 820)
    for i, r in enumerate(rows): sl.text((hx, y + i * 42), r, f, WHITE)
    heart(sl, hx + f.getlength(rows[-1]) / S + 26, y + (len(rows) - 1) * 42 + 20, 16)
    y += len(rows) * 42 + 40
    # contact card: the WhatsApp/Telegram number and the separate MAX number
    x, y0 = 100, y
    t1 = ("По всем вопросам, пожалуйста, сразу пишите в WhatsApp, Telegram по удобным контактам "
          "в шапке профиля, оставляю номер и здесь:")
    n1 = len(wrap(t1, mont(26, 300), 840))
    y_t1 = y0 + 44; y_p1 = y_t1 + n1 * 26 * 1.3 + 10
    y_t2 = y_p1 + 92; y_p2 = y_t2 + 26 * 1.32 + 10
    box = (60, y0, 1020, int(y_p2 + 96))
    sl.shadow(box, 30, 20, 110); sl.rrect(box, 30, WHITE, 22, outline=POWDER, w=1.2)
    para(sl, x, y_t1, t1, 26, 300, WHITE, 840, 1.3)
    sl.text((x, y_p1), "+7 926 636-30-00", mont(44, 600), WHITE)
    sl.text((box[2] - 40, y_p1 + 22), "WhatsApp · Telegram", mont(24, 400), POWDER, "ra")
    para(sl, x, y_t2, "Также связаться можно и в МАХ:", 26, 300, WHITE, 840)
    sl.text((x, y_p2), "+7 925 310-63-64", mont(44, 600), WHITE)
    sl.text((box[2] - 40, y_p2 + 22), "MAX", mont(24, 400), POWDER, "ra")
    counter(sl, 9)
    return sl.save("09.jpg")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out = [s1(), s2(), s3(), s4(), s5(), s6(), s7(), s8(), s9()]
    ss.sheet(out, 3, 540, "00_overview.jpg")
    print("ok")
