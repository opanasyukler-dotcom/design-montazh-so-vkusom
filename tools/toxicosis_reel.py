"""@mcg.va — «Как справляться с токсикозом?»: a fully animated, voiced vertical reel (1080x1920, 30 fps), built for viewers with ADHD:
something moves or appears on almost every key word, the picture confirms what the voice says, a progress line and a «N / 5»
counter keep the place, captions run word by word with the spoken word on a pill, and every visual beat has its sound.
  python3 toxicosis_reel.py WORK_DIR VOICE_MODEL.onnx OUT.mp4
Voice: Piper TTS (ru_RU-irina-medium), word timings from faster-whisper on the voice itself. Art: simple flat icons drawn here.
"""
import difflib, json, math, os, re, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import posuda_sfx as FX
from rody_reel import key_sample

W, H, FPS, SR = 1080, 1920, 30, 48000
FD = os.path.join(HERE, "..", "fonts", "stories")
CREAM, INK, BROWN, MAUVE, GREY, SKY, BLUSH = (243, 236, 228), (43, 31, 26), (107, 74, 56), (139, 107, 92), (173, 166, 158), (220, 225, 231), (232, 201, 192)
LEMON, LEAF, WATER, COLA, POM, ORANGE, RED, SAGE = (242, 208, 75), (126, 166, 96), (156, 199, 224), (74, 38, 26), (194, 57, 75), (242, 155, 56), (206, 72, 60), (190, 205, 180)
CX, CY = 540, 930                                                          # the stage centre

def font(size, wght=400, italic=False):
    f = ImageFont.truetype(os.path.join(FD, "Montserrat-Italic-VariableFont_wght.ttf" if italic else "Montserrat-VariableFont_wght.ttf"), size)
    f.set_variation_by_axes([wght]); return f

def ease_back(x, k=1.7):
    x = min(max(x, 0.0), 1.0) - 1; return 1 + (k + 1) * x ** 3 + k * x ** 2

def ease(x): x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)

# ---------- icons: flat shapes, drawn at 2× and scaled down for clean edges ----------
def art(fn, size=520):
    S = size * 2; im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    P = lambda *v: tuple(round(a * S) for a in v); fn(d, P, S)
    return im.resize((size, size), Image.LANCZOS)

def _outline(S): return max(2, round(S * 0.018))

def ic_face(d, P, S, sick=True):
    lw = _outline(S); d.ellipse(P(.12, .12, .88, .88), fill=(196, 222, 170) if sick else (247, 214, 160), outline=INK, width=lw)
    for cx in (.37, .63):                                                  # swirly eyes / smiling eyes
        if sick:                                                           # squeezed «> <» eyes
            sx = 1 if cx < .5 else -1; d.line([P(cx - .05 * sx, .37), P(cx + .05 * sx, .42), P(cx - .05 * sx, .47)], fill=INK, width=lw, joint="curve")
        else: d.arc(P(cx - .06, .38, cx + .06, .48), 200, 340, fill=INK, width=lw)
    if sick:
        pts = [P(.32 + i * .06, .66 + (.025 if i % 2 else -.025)) for i in range(7)]; d.line(pts, fill=INK, width=lw, joint="curve")
        d.polygon([P(.80, .22), P(.75, .34), P(.85, .34)], fill=WATER); d.ellipse(P(.75, .30, .85, .40), fill=WATER)   # sweat drop
    else: d.arc(P(.34, .52, .66, .74), 20, 160, fill=INK, width=lw)

def ic_smell(d, P, S):
    lw = round(S * .03)
    for k, x0 in enumerate((.25, .48, .71)):
        pts = [P(x0 + .06 * math.sin(i / 3 + k), .85 - i * .05) for i in range(15)]; d.line(pts, fill=(150, 180, 120), width=lw, joint="curve")

def ic_plate(d, P, S):
    lw = _outline(S); d.ellipse(P(.15, .30, .85, .80), fill=(255, 255, 255), outline=INK, width=lw); d.ellipse(P(.27, .40, .73, .70), outline=GREY, width=lw)
    d.ellipse(P(.38, .46, .62, .62), fill=(222, 140, 90))                  # food
    d.line([P(.07, .30), P(.07, .80)], fill=INK, width=lw * 2); d.line([P(.93, .30), P(.93, .80)], fill=INK, width=lw * 2)

def ic_thought(d, P, S):
    lw = _outline(S)
    for c in ((.30, .45, .26), (.55, .38, .28), (.74, .50, .22), (.50, .60, .26)):
        d.ellipse(P(c[0] - c[2], c[1] - c[2] * .8, c[0] + c[2], c[1] + c[2] * .8), fill=(255, 255, 255), outline=INK, width=lw)
    for c in ((.50, .60, .26), (.30, .45, .26), (.55, .38, .28), (.74, .50, .22)):
        d.ellipse(P(c[0] - c[2] + .02, c[1] - c[2] * .8 + .02, c[0] + c[2] - .02, c[1] + c[2] * .8 - .02), fill=(255, 255, 255))
    for x, y, r in ((.22, .82, .05), (.12, .92, .03)): d.ellipse(P(x - r, y - r, x + r, y + r), fill=(255, 255, 255), outline=INK, width=lw)

def ic_lemon(d, P, S):
    lw = _outline(S); d.ellipse(P(.14, .26, .86, .78), fill=LEMON, outline=INK, width=lw)
    d.ellipse(P(.06, .46, .18, .58), fill=LEMON, outline=INK, width=lw); d.ellipse(P(.82, .44, .94, .56), fill=LEMON, outline=INK, width=lw)
    d.ellipse(P(.15, .27, .85, .77), fill=LEMON); d.ellipse(P(.30, .34, .48, .44), fill=(250, 232, 150))
    d.polygon([P(.55, .27), P(.70, .10), P(.78, .22)], fill=LEAF, outline=INK)

def ic_slice(d, P, S):
    lw = _outline(S); d.ellipse(P(.1, .1, .9, .9), fill=LEMON, outline=INK, width=lw); d.ellipse(P(.17, .17, .83, .83), fill=(252, 240, 180))
    for k in range(8):
        a = k * math.pi / 4; d.line([P(.5, .5), P(.5 + .33 * math.cos(a), .5 + .33 * math.sin(a))], fill=LEMON, width=lw)
    d.ellipse(P(.46, .46, .54, .54), fill=LEMON)

def ic_glass(d, P, S, liquid=WATER, bubbles=True, straw=False, ice=False):
    lw = _outline(S); top, bot = (.24, .14, .76, .14), (.31, .90, .69, .90)
    d.polygon([P(.27, .40), P(.73, .40), P(.69, .90), P(.31, .90)], fill=liquid)
    if ice:
        for x, y in ((.38, .44), (.52, .48)): d.rectangle(P(x, y, x + .12, y + .11), fill=(235, 245, 250), outline=(200, 220, 230), width=lw)
    if bubbles:
        for x, y, r in ((.40, .78, .025), (.55, .66, .02), (.47, .56, .03), (.60, .82, .02)): d.ellipse(P(x - r, y - r, x + r, y + r), outline=(255, 255, 255), width=lw)
    if straw: d.line([P(.58, .02), P(.50, .70)], fill=RED, width=lw * 3)
    d.line([P(.24, .14), P(.31, .90), P(.69, .90), P(.76, .14)], fill=INK, width=lw, joint="curve")

def ic_bottle(d, P, S, body=(120, 170, 150), label=(240, 232, 220), cap=(70, 90, 80)):
    lw = _outline(S); d.rounded_rectangle(P(.30, .30, .70, .92), radius=round(S * .07), fill=body, outline=INK, width=lw)
    d.polygon([P(.40, .30), P(.44, .14), P(.56, .14), P(.60, .30)], fill=body, outline=INK); d.rectangle(P(.42, .06, .58, .14), fill=cap, outline=INK, width=lw)
    d.rectangle(P(.30, .50, .70, .70), fill=label); d.line([P(.36, .56), P(.64, .56)], fill=GREY, width=lw); d.line([P(.38, .63), P(.62, .63)], fill=GREY, width=lw)
    d.rectangle(P(.34, .36, .38, .86), fill=(255, 255, 255, 90))

def ic_pom(d, P, S):
    lw = _outline(S); d.ellipse(P(.12, .22, .88, .92), fill=POM, outline=INK, width=lw)
    d.polygon([P(.40, .26), P(.42, .10), P(.47, .20), P(.50, .08), P(.53, .20), P(.58, .10), P(.60, .26)], fill=POM, outline=INK)
    d.ellipse(P(.26, .36, .40, .48), fill=(230, 110, 120))

def ic_orange(d, P, S):
    lw = _outline(S); d.ellipse(P(.1, .1, .9, .9), fill=ORANGE, outline=INK, width=lw); d.ellipse(P(.17, .17, .83, .83), fill=(250, 190, 110))
    for k in range(10):
        a = k * math.pi / 5; d.line([P(.5, .5), P(.5 + .33 * math.cos(a), .5 + .33 * math.sin(a))], fill=ORANGE, width=lw)

def ic_flame(d, P, S):
    outer = [P(.5, .06), P(.66, .30), P(.80, .52), P(.78, .76), P(.62, .92), P(.38, .92), P(.22, .76), P(.24, .52), P(.40, .36)]
    inner = [P(.5, .40), P(.60, .58), P(.62, .76), P(.50, .88), P(.38, .76), P(.42, .58)]
    d.polygon(outer, fill=(240, 120, 60), outline=INK); d.polygon(inner, fill=(250, 200, 90))

def ic_blister(d, P, S):
    lw = _outline(S); d.rounded_rectangle(P(.14, .22, .86, .78), radius=round(S * .06), fill=(214, 220, 226), outline=INK, width=lw)
    for i in range(4):
        for j in range(2):
            x, y = .26 + i * .16, .38 + j * .24; d.ellipse(P(x - .06, y - .06, x + .06, y + .06), fill=(255, 255, 255), outline=GREY, width=lw)

def ic_med(d, P, S):
    lw = _outline(S); d.ellipse(P(.1, .1, .9, .9), fill=(255, 255, 255), outline=INK, width=lw)
    d.rounded_rectangle(P(.42, .24, .58, .76), radius=round(S * .03), fill=(80, 160, 140)); d.rounded_rectangle(P(.24, .42, .76, .58), radius=round(S * .03), fill=(80, 160, 140))

def ic_clock(d, P, S):
    lw = _outline(S); d.ellipse(P(.1, .1, .9, .9), fill=(255, 255, 255), outline=INK, width=lw * 2)
    for k in range(12):
        a = k * math.pi / 6; d.line([P(.5 + .34 * math.cos(a), .5 + .34 * math.sin(a)), P(.5 + .38 * math.cos(a), .5 + .38 * math.sin(a))], fill=INK, width=lw)

def ic_hands(d, P, S):
    lw = _outline(S); d.line([P(.5, .5), P(.5, .22)], fill=INK, width=lw * 2); d.line([P(.5, .5), P(.68, .5)], fill=MAUVE, width=lw * 2)

def ic_snow(d, P, S):
    lw = round(S * .03)
    for k in range(3):
        a = k * math.pi / 3; dx, dy = .38 * math.cos(a), .38 * math.sin(a); d.line([P(.5 - dx, .5 - dy), P(.5 + dx, .5 + dy)], fill=(120, 170, 210), width=lw)

def ic_drop(d, P, S):
    d.polygon([P(.5, .1), P(.72, .55), P(.28, .55)], fill=WATER); d.ellipse(P(.28, .38, .72, .86), fill=WATER)

def ic_cross(d, P, S):
    lw = round(S * .09); d.line([P(.2, .2), P(.8, .8)], fill=RED, width=lw); d.line([P(.8, .2), P(.2, .8)], fill=RED, width=lw)

def ic_check(d, P, S):
    lw = round(S * .09); d.line([P(.18, .52), P(.42, .76), P(.84, .26)], fill=(90, 160, 90), width=lw, joint="curve")

def ic_qmark(d, P, S):
    f = font(round(S * .8), 800); d.text(P(.5, .5), "?", font=f, fill=MAUVE, anchor="mm")

def ic_heart(d, P, S):
    t = np.linspace(0, 2 * math.pi, 80); x = 16 * np.sin(t) ** 3; y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    d.polygon([P(.5 + a / 40, .48 + b / 40) for a, b in zip(x, y)], fill=(222, 120, 130), outline=INK)

def ic_comment(d, P, S):
    lw = _outline(S); d.rounded_rectangle(P(.08, .18, .92, .72), radius=round(S * .12), fill=(255, 255, 255), outline=INK, width=lw)
    d.polygon([P(.24, .70), P(.20, .88), P(.40, .71)], fill=(255, 255, 255), outline=INK)
    d.rectangle(P(.25, .66, .40, .73), fill=(255, 255, 255))
    for x in (.34, .50, .66): d.ellipse(P(x - .04, .41, x + .04, .49), fill=MAUVE)

def ic_list(d, P, S):
    lw = _outline(S); d.rounded_rectangle(P(.18, .08, .82, .92), radius=round(S * .05), fill=(255, 255, 255), outline=INK, width=lw)
    for k in range(5):
        y = .2 + k * .15; d.rounded_rectangle(P(.26, y, .34, y + .08), radius=round(S * .015), outline=MAUVE, width=lw); d.line([P(.40, y + .04), P(.74, y + .04)], fill=GREY, width=lw)

ICONS = {"sick": lambda: art(lambda d, P, S: ic_face(d, P, S, True)), "happy": lambda: art(lambda d, P, S: ic_face(d, P, S, False)),
         "smell": lambda: art(ic_smell), "plate": lambda: art(ic_plate), "thought": lambda: art(ic_thought), "lemon": lambda: art(ic_lemon),
         "slice": lambda: art(ic_slice), "water": lambda: art(ic_glass),
         "cola": lambda: art(lambda d, P, S: ic_glass(d, P, S, liquid=COLA, ice=True, straw=True)),
         "mineral": lambda: art(ic_bottle), "colabottle": lambda: art(lambda d, P, S: ic_bottle(d, P, S, body=(90, 50, 35), label=(206, 72, 60), cap=(206, 72, 60))),
         "pom": lambda: art(ic_pom), "orange": lambda: art(ic_orange), "flame": lambda: art(ic_flame), "blister": lambda: art(ic_blister),
         "med": lambda: art(ic_med), "clock": lambda: art(ic_clock), "hands": lambda: art(ic_hands), "snow": lambda: art(ic_snow),
         "drop": lambda: art(ic_drop), "cross": lambda: art(ic_cross), "check": lambda: art(ic_check), "qmark": lambda: art(ic_qmark),
         "heart": lambda: art(ic_heart), "comment": lambda: art(ic_comment), "list": lambda: art(ic_list)}
_IC = {}
def icon(name):
    if name not in _IC: _IC[name] = ICONS[name]()
    return _IC[name]

def chip_img(text, size=46, fill=BLUSH, col=INK):
    f = font(size, 600); w = int(f.getlength(text)) + 64; h = int(size * 1.9)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=fill); d.text((w / 2, h / 2), text, font=f, fill=col, anchor="mm")
    return im

def stamp_img(text):
    f = font(64, 800); w = int(f.getlength(text)) + 90; h = 130
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((6, 6, w - 7, h - 7), radius=26, outline=RED, width=10); d.text((w / 2, h / 2 + 2), text, font=f, fill=RED, anchor="mm")
    return im.rotate(-6, resample=Image.BICUBIC, expand=True)

# ---------- the script: what she says, the headline, and the beats that show it ----------
# beat: (word key | seconds from scene start, kind, args). kinds: spr (name, x, y, size, enter, motion[, until]), chip (text, x, y),
# stamp (text), shake, cross (x, y, size), orbit (names, radius, size), sfx (name)
SCENES = [
    dict(head="Как справляться с токсикозом?", text="Как справляться с токсикозом?",
         beats=[(0.0, "spr", ("sick", CX, CY + 60, 520, "pop", "wiggle")), ("токсик", "shake", ())]),
    dict(head="Отдельная история", text="Ой, токсикоз — это вообще отдельная история беременности.",
         beats=[(0.0, "spr", ("sick", CX, CY, 440, "pop", "wiggle")), ("беремен", "chip", ("беременность", CX, 1260))]),
    dict(head="Тошнит от всего", text="Когда тебя тошнит буквально от всего, даже от запахов, которые раньше вообще не замечала.",
         beats=[(0.0, "spr", ("sick", CX, CY + 40, 420, "pop", "shake")), ("тошнит", "shake", ()),
                ("запах", "spr", ("smell", 250, 760, 300, "fade", "float")), ("запах", "spr", ("smell", 830, 740, 300, "fade", "float")),
                ("запах", "chip", ("даже запахи", CX, 1270))]),
    dict(head="Хочется есть… но нет", text="И самое обидное, что ты вроде хочешь поесть, но от одной мысли о еде уже плохо.",
         beats=[(0.0, "spr", ("sick", 320, 1040, 340, "pop", "wiggle")), ("поесть", "spr", ("thought", 720, 790, 440, "pop", "float")),
                ("поесть", "spr", ("plate", 735, 770, 230, "pop", "float")), ("плохо", "cross", (735, 770, 300)), ("плохо", "shake", ())]),
    dict(head="Что можно попробовать", text="Хочу поделиться тем, что можно попробовать, чтобы хоть немного облегчить своё состояние.",
         beats=[(0.0, "spr", ("list", CX, CY, 440, "pop", "float", "можно")),
                ("можно", "spr", ("lemon", 140, CY, 190, "pop", "float")), ("можно", "spr", ("mineral", 340, CY, 190, "pop", "float", None, 0.25)),
                ("можно", "spr", ("cola", 540, CY, 190, "pop", "float", None, 0.5)), ("можно", "spr", ("pom", 740, CY, 190, "pop", "float", None, 0.75)),
                ("можно", "spr", ("blister", 940, CY, 190, "pop", "float", None, 1.0)), ("облегч", "chip", ("облегчить состояние", CX, 1220))]),
    dict(num="1 / 5", head="Лимон", text="Лимон. Иногда достаточно просто понюхать или добавить пару долек в воду. Кислое некоторым действительно помогает справляться с тошнотой.",
         beats=[(0.0, "spr", ("lemon", CX, CY, 480, "pop", "wiggle", "добав")), ("понюх", "spr", ("smell", CX, 640, 260, "fade", "float", "добав")),
                ("добав", "spr", ("lemon", 300, 1010, 300, "pop", "float")), ("добав", "spr", ("water", 740, 980, 400, "pop", "none")),
                ("дол", "spr", ("slice", 740, 900, 170, "drop", "none")), ("воду", "sfx", ("splash",)),
                ("кисл", "chip", ("кислое помогает", CX, 1290))]),
    dict(num="2 / 5", head="Боржоми", text="Боржоми. Холодная минералка маленькими глотками. Но если от газировки становится хуже, лучше обычная вода.",
         beats=[(0.0, "spr", ("mineral", CX, CY, 500, "pop", "float", "вода")), ("холодн", "spr", ("snow", 800, 700, 200, "pop", "spin", "газир")),
                ("глотк", "spr", ("drop", 300, 1150, 90, "pop", "float", "газир")), ("глотк", "spr", ("drop", 380, 1190, 70, "pop", "float", "газир", 0.15)),
                ("глотк", "spr", ("drop", 450, 1150, 55, "pop", "float", "газир", 0.3)), ("газир", "sfx", ("fizz",)), ("хуже", "cross", (CX, CY, 380, "вода")),
                ("вода", "spr", ("water", CX, CY, 440, "pop", "float")), ("вода", "spr", ("check", 790, 760, 200, "pop", "none"))]),
    dict(num="3 / 5", head="Кола", text="Кока-кола. Да-да, звучит странно, но некоторым помогает несколько глотков холодной колы. Только не стоит пить её литрами из-за сахара и кофеина.",
         beats=[(0.0, "spr", ("cola", CX, CY, 480, "pop", "float", "литрам")), ("кола", "sfx", ("fizz",)),
                ("странно", "spr", ("qmark", 820, 700, 220, "pop", "wiggle", "некотор")), ("странно", "shake", ()),
                ("глотков", "chip", ("пара глотков", CX, 1270)), ("литрам", "spr", ("colabottle", CX, CY, 520, "pop", "none")),
                ("литрам", "cross", (CX, CY, 420)), ("сахар", "chip", ("сахар", 330, 1290)), ("кофеин", "chip", ("кофеин", 760, 1290))]),
    dict(num="4 / 5", head="Гранат и цитрусы", text="Гранат и цитрусы. Кислые фрукты тоже могут немного облегчать тошноту. Главное, смотреть на свою реакцию, потому что у кого-то они, наоборот, вызывают изжогу.",
         beats=[(0.0, "spr", ("pom", 340, CY, 380, "pop", "float")), ("цитрус", "spr", ("orange", 750, CY + 30, 340, "pop", "float")),
                ("реакц", "chip", ("следи за реакцией", CX, 1270)), ("изжог", "spr", ("flame", CX, 690, 300, "pop", "wiggle")), ("изжог", "shake", ())]),
    dict(num="5 / 5", head="Дуоника", text="Дуоника. Если токсикоз сильный, не нужно просто терпеть. Можно обсудить с врачом препараты от тошноты, в том числе Дуонику. Самостоятельно начинать приём во время беременности не стоит.",
         beats=[(0.0, "spr", ("blister", CX, CY, 480, "pop", "float")), ("терпеть", "chip", ("не нужно терпеть", CX, 1215)),
                ("врач", "spr", ("med", 820, 700, 230, "pop", "pulse")), ("самостоят", "stamp", ("ТОЛЬКО С ВРАЧОМ",))]),
    dict(head="Маленькими порциями", text="Ещё мне кажется важным не заставлять себя есть большими порциями. Иногда проще перекусывать понемногу в течение дня, чем пытаться съесть полноценный обед.",
         beats=[(0.0, "spr", ("plate", CX, CY, 520, "pop", "float", "перекус")), ("больш", "cross", (CX, CY, 420, "перекус")),
                ("перекус", "spr", ("plate", 230, 1000, 240, "pop", "float")), ("перекус", "spr", ("plate", 540, 1000, 240, "pop", "float", None, 0.2)),
                ("перекус", "spr", ("plate", 850, 1000, 240, "pop", "float", None, 0.4)), ("дня", "spr", ("clock", CX, 690, 250, "pop", "none")),
                ("дня", "spr", ("hands", CX, 690, 250, "pop", "spin")), ("обед", "chip", ("понемногу в течение дня", CX, 1290))]),
    dict(head="Нет одного средства для всех", text="И вообще, не существует одного средства, которое помогает абсолютно всем. Тут приходится пробовать и смотреть, от чего становится легче именно тебе.",
         beats=[(0.0, "orbit", (("lemon", "mineral", "cola", "pom", "orange", "blister"), 300, 190)),
                ("всем", "spr", ("qmark", CX, CY, 230, "pop", "wiggle", "пробов")), ("тебе", "spr", ("heart", CX, CY, 300, "pop", "pulse"))]),
    dict(head="А что спасало вас?", text="А что вас спасало от токсикоза? Мне кажется, у каждой беременной есть свой список самых неожиданных средств.",
         beats=[(0.0, "spr", ("comment", 320, 820, 300, "pop", "float")), (0.5, "spr", ("comment", 760, 940, 270, "pop", "float")),
                ("кажется", "spr", ("comment", 400, 1120, 240, "pop", "float")), ("список", "spr", ("list", 780, 700, 260, "pop", "float")),
                ("неожидан", "spr", ("heart", 560, 1000, 200, "pop", "pulse"))]),
]

# ---------- voice ----------
def norm(w): return re.sub(r"[^\wё-]", "", w.lower())

def voice(work, model):
    """one Piper take per scene, a short breath between scenes; returns the voice track and each scene's word timings"""
    from faster_whisper import WhisperModel
    asr = WhisperModel("small", device="cpu", compute_type="int8"); parts, scenes, t = [], [], 0.6
    lead = np.zeros((int(0.6 * SR), 2), np.float32); parts.append(lead)
    for k, sc in enumerate(SCENES):
        wav = os.path.join(work, f"v{k:02d}.wav")
        if not os.path.exists(wav):
            subprocess.run([sys.executable, "-m", "piper", "-m", model, "-f", wav, "--length_scale", "0.96", "--sentence_silence", "0.22"],
                           input=sc["text"].encode(), check=True, capture_output=True)
        x = FX.read(wav); dur = len(x) / SR
        segs, _ = asr.transcribe(wav, language="ru", word_timestamps=True, beam_size=5)
        ws = [(w.word.strip(), w.start, w.end) for s in segs for w in s.words]
        dst = sc["text"].split(); src = [norm(w[0]) for w in ws]; tm = [None] * len(dst)
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=src, b=[norm(w) for w in dst], autojunk=False).get_opcodes():
            if op == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2)): tm[j] = (ws[i][1], ws[i][2])
        j = 0                                                              # unmatched words: spread evenly between the known neighbours
        while j < len(dst):
            if tm[j]: j += 1; continue
            e = j
            while e < len(dst) and tm[e] is None: e += 1
            a = tm[j - 1][1] if j else 0.0; b = tm[e][0] if e < len(dst) else dur; step = max((b - a) / (e - j), 0.12)
            for i in range(j, e): tm[i] = (a + (i - j) * step, a + (i - j + 1) * step)
            j = e
        scenes.append(dict(sc, t0=t, dur=dur, words=[(w, t + a, t + b) for w, (a, b) in zip(dst, tm)]))
        parts.append(x); gap = 0.35; parts.append(np.zeros((int(gap * SR), 2), np.float32)); t += dur + gap
    v = np.concatenate(parts + [np.zeros((int(1.2 * SR), 2), np.float32)])
    raw = os.path.join(work, "voice_raw.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", raw], input=v.tobytes(), check=True)
    meas = subprocess.run(["ffmpeg", "-hide_banner", "-i", raw, "-af", "loudnorm=I=-15:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                          capture_output=True, text=True).stderr
    mj = json.loads(meas[meas.rindex("{"):meas.rindex("}") + 1])
    ln = (f"loudnorm=I=-15:TP=-1.5:LRA=11:linear=true:measured_I={mj['input_i']}:measured_TP={mj['input_tp']}:"
          f"measured_LRA={mj['input_lra']}:measured_thresh={mj['input_thresh']}:offset={mj['target_offset']}")
    out = os.path.join(work, "voice.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af", ln + ",alimiter=limit=0.89:level=false", "-ar", str(SR), out], check=True)
    return FX.read(out), scenes, len(v) / SR

# ---------- sound effects ----------
def sfx_bank():
    rng = np.random.default_rng(5); st = lambda y: np.stack([y, y], 1).astype(np.float32)
    def splash():
        n = int(0.5 * SR); t = np.arange(n) / SR; f = 900 * np.exp(-t * 7) + 250
        bl = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * 0.6; nz = rng.standard_normal(n) * np.exp(-t * 14) * 0.35
        nz = np.convolve(nz, np.ones(6) / 6, "same"); return st((bl + nz) * FX.db(-14))
    def fizz():
        n = int(0.9 * SR); y = np.zeros(n); idx = rng.integers(0, n - 200, 260)
        for i in idx: y[i:i + 60] += rng.standard_normal(60) * np.exp(-np.arange(60) / 12) * rng.uniform(0.2, 1)
        y *= np.linspace(1, 0.3, n); return st(y * FX.db(-22))
    def thud():
        n = int(0.6 * SR); t = np.arange(n) / SR; y = np.sin(2 * np.pi * (60 + 50 * np.exp(-t * 18)) * t) * np.exp(-t * 7); return st(y * FX.db(-10))
    return dict(pop=lambda: FX.pop(FX.db(-20)), whoosh=lambda: FX.noise_sweep(0.45, 400, 4200, FX.db(-26)),
                ding=lambda: FX.bells([1318.5, 1760.0], FX.db(-24), 0.06), splash=splash, fizz=fizz, thud=thud,
                shake=lambda: thud() * 0.5, key=lambda: key_sample(rng, FX.db(-19)))

# ---------- render ----------
class Spr:
    def __init__(self, img, x, y, size, t0, t1, enter="pop", motion="float", seed=0):
        self.img, self.x, self.y, self.size, self.t0, self.t1, self.enter, self.motion, self.seed = img, x, y, size, t0, t1, enter, motion, seed
        self.cache = {}

    def frame(self, t):
        if t < self.t0 or t > self.t1 + 0.22: return None
        u = t - self.t0; s, a, dx, dy, al = 1.0, 0.0, 0.0, 0.0, 1.0
        if self.enter == "pop": s *= ease_back(u / 0.35)
        elif self.enter == "fade": al = ease(u / 0.4); dy += 30 * (1 - ease(u / 0.4))
        elif self.enter == "drop":                                            # falls from above and bounces in
            k = min(u / 0.45, 1); dy -= 520 * (1 - k) ** 2 if k < 1 else 0; dy += 18 * math.sin(min(u - 0.45, 0.3) / 0.3 * math.pi) if u > 0.45 else 0
        if self.motion == "float": dy += 12 * math.sin(t * 2.2 + self.seed)
        elif self.motion == "wiggle": a += 6 * math.sin(t * 5 + self.seed)
        elif self.motion == "shake": dx += 6 * math.sin(t * 37 + self.seed); a += 3 * math.sin(t * 29)
        elif self.motion == "pulse": s *= 1 + 0.06 * math.sin(t * 6)
        elif self.motion == "spin": a += (t - self.t0) * 140
        if t > self.t1: k = (t - self.t1) / 0.22; s *= 1 - ease(k); al *= 1 - ease(k)
        if s <= 0.02 or al <= 0.01: return None
        sz = max(int(self.size * s), 2); key = (sz, int(a) % 360)
        im = self.cache.get(key)
        if im is None:
            im = self.img.resize((sz, sz), Image.BILINEAR)
            if abs(a) > 0.5: im = im.rotate(a, resample=Image.BICUBIC, expand=True)
            if len(self.cache) < 400: self.cache[key] = im
        if al < 1: im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * al)))
        return im, int(self.x + dx - im.width / 2), int(self.y + dy - im.height / 2)

def word_time(sc, key):
    if isinstance(key, (int, float)): return sc["t0"] + key
    for w, a, b in sc["words"]:
        if norm(w).startswith(key): return a
    return sc["t0"]

def build(scenes, total):
    sprs, events, shakes, chips = [], [], [], []
    for k, sc in enumerate(scenes):
        end = sc["t0"] + sc["dur"] + 0.2 if k + 1 < len(scenes) else total - 0.3
        events.append((sc["t0"] - 0.25, "whoosh"))
        if sc.get("num"): events.append((sc["t0"] + 0.05, "ding"))
        for key, kind, args in sc["beats"]:
            t = word_time(sc, key)
            if kind == "spr":
                name, x, y, size, enter, motion, *rest = args; until = rest[0] if rest else None; delay = rest[1] if len(rest) > 1 else 0.0
                t1 = word_time(sc, until) if until else end; t += delay
                sprs.append(Spr(icon(name), x, y, size, t, t1, enter, motion, seed=len(sprs))); events.append((t, "pop"))
            elif kind == "chip":
                text, x, y = args; sprs.append(Spr(chip_img(text), x, y, None, t, end, "pop", "none")); events.append((t, "pop"))
            elif kind == "cross":                                          # optional 4th arg: the word where it goes away
                x, y, size, *u = args; sprs.append(Spr(icon("cross"), x, y, size, t, word_time(sc, u[0]) if u else end, "pop", "none"))
                events.append((t, "thud"))
            elif kind == "stamp":
                im = stamp_img(args[0]); sp = Spr(im, CX, 1355, None, t, end, "pop", "none"); sprs.append(sp); events.append((t, "thud")); shakes.append(t)
            elif kind == "shake": shakes.append(t); events.append((t, "shake"))
            elif kind == "sfx": events.append((t, args[0]))
            elif kind == "orbit":
                names, rad, size = args
                for i, nm in enumerate(names):
                    sp = Spr(icon(nm), CX, CY, size, t + i * 0.15, end, "pop", "none", seed=i); sp.orbit = (rad, i * 2 * math.pi / len(names)); sprs.append(sp)
                    events.append((t + i * 0.15, "pop"))
    return sprs, events, shakes

def caption_frames(scenes):
    """phrases of ≤ 3 words (≤ 22 letters), broken at punctuation; each word's own time span"""
    out, cf = [], font(58, 700)
    for sc in scenes:
        cur = []
        for w in sc["words"]:
            if cur and cf.getlength(" ".join(x[0] for x in cur + [w])) > 900: out.append(cur); cur = []   # never wider than the frame
            cur.append(w)
            if len(cur) >= 3 or len(" ".join(x[0] for x in cur)) >= 20 or re.search(r"[.,!?…—]$", w[0]): out.append(cur); cur = []
        if cur: out.append(cur)
    return out

def paste(fr, im, x, y):
    """alpha-composite im at (x, y), clipped to the frame"""
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + im.width, W), min(y + im.height, H)
    if x1 > x0 and y1 > y0: fr.alpha_composite(im.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))

def render(work, scenes, total, out_video):
    sprs, events, shakes = build(scenes, total)
    phrases = caption_frames(scenes)
    hf, nf, cf = font(78, 800), font(40, 600), font(58, 700)
    # background: cream with three soft drifting blobs
    base = Image.new("RGBA", (W, H), CREAM + (255,))
    blobs = []
    for col, r in ((BLUSH, 380), (SKY, 420), (SAGE, 340)):
        a = Image.new("L", (r * 2, r * 2), 0); ImageDraw.Draw(a).ellipse((r * .3, r * .3, r * 1.7, r * 1.7), fill=150)
        b = Image.new("RGBA", (r * 2, r * 2), col + (0,)); b.putalpha(a.filter(ImageFilter.GaussianBlur(r * .22))); blobs.append(b)
    keyt = []                                                              # typed headline clicks
    for sc in scenes:
        for i, c in enumerate(sc["head"]):
            if c != " ": keyt.append(sc["t0"] + 0.1 + i * 0.03)
    wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", os.path.join(work, "video.mp4")], stdin=subprocess.PIPE)
    n = int(total * FPS)
    for fi in range(n):
        t = fi / FPS; fr = base.copy()
        for i, b in enumerate(blobs):
            x = int(W * (0.2 + 0.6 * (0.5 + 0.5 * math.sin(t * 0.21 + i * 2.1))) - b.width / 2)
            y = int(H * (0.25 + 0.5 * (0.5 + 0.5 * math.sin(t * 0.17 + i * 1.3 + 1))) - b.height / 2); paste(fr, b, x, y)
        d = ImageDraw.Draw(fr)
        d.rounded_rectangle((60, 236, W - 60, 244), radius=4, fill=(222, 212, 202)); d.rounded_rectangle((60, 236, 60 + int((W - 120) * t / total), 244), radius=4, fill=BROWN)
        sc = max([s for s in scenes if s["t0"] - 0.3 <= t] or [scenes[0]], key=lambda s: s["t0"])
        if sc.get("num"):
            ci = chip_img(sc["num"], 38, fill=BROWN, col=(255, 255, 255)); k = ease_back((t - sc["t0"]) / 0.3)
            if k > 0.05: ci = ci.resize((max(int(ci.width * k), 1), max(int(ci.height * k), 1))); fr.alpha_composite(ci, (int(CX - ci.width / 2), int(300 - ci.height / 2)))
        nch = int((t - sc["t0"] - 0.1) / 0.03) + 1; head = sc["head"][:max(nch, 0)]
        if head:
            lines, cur = [], ""
            for w in head.split(" "):
                tt = (cur + " " + w).strip()
                if hf.getlength(tt) <= 940 or not cur: cur = tt
                else: lines.append(cur); cur = w
            lines.append(cur)
            for j, l in enumerate(lines): d.text((CX, 420 + j * 92), l, font=hf, fill=INK, anchor="mm")
        for sp in sprs:
            if hasattr(sp, "orbit"):
                rad, ph = sp.orbit; sp.x = CX + rad * math.cos(ph + t * 0.6); sp.y = CY + rad * 0.85 * math.sin(ph + t * 0.6)
            if sp.size is None:                                            # chips / stamps: scale relative to their own size
                u = t - sp.t0
                if u < 0 or t > sp.t1 + 0.22: continue
                k = (1.6 - 0.6 * ease(u / 0.25)) if sp.img.size[1] > 120 else ease_back(u / 0.3)
                if t > sp.t1: k *= 1 - ease((t - sp.t1) / 0.22)
                if k <= 0.02: continue
                im = sp.img.resize((max(int(sp.img.width * k), 1), max(int(sp.img.height * k), 1)), Image.BILINEAR)
                paste(fr, im, int(sp.x - im.width / 2), int(sp.y - im.height / 2)); continue
            r = sp.frame(t)
            if r: paste(fr, *r)
        ph = next((p for p in phrases if p[0][1] - 0.05 <= t <= p[-1][2] + 0.25), None)  # captions
        if ph:
            widths = [cf.getlength(w[0]) for w in ph]; total_w = sum(widths) + 22 * (len(ph) - 1); x = CX - total_w / 2; y = 1480
            for (w, a, b), wd in zip(ph, widths):
                on = a - 0.03 <= t <= b + 0.05
                if on:
                    k = 1 + 0.12 * (1 - ease((t - a) / 0.15)); pw, phh = wd * k + 36, 86 * k
                    d.rounded_rectangle((x + wd / 2 - pw / 2, y - phh / 2, x + wd / 2 + pw / 2, y + phh / 2), radius=int(phh / 2), fill=BLUSH)
                d.text((x + wd / 2, y), w, font=cf, fill=INK if (on or t > b) else GREY, anchor="mm"); x += wd + 22
        img = np.asarray(fr.convert("RGB"))
        sh = [s for s in shakes if 0 <= t - s < 0.3]
        if sh:
            k = 1 - (t - sh[-1]) / 0.3; img = np.roll(img, (int(10 * k * math.sin(t * 90)), int(12 * k * math.cos(t * 77))), (0, 1))
        wr.stdin.write(np.ascontiguousarray(img).tobytes())
    wr.stdin.close(); wr.wait()
    return events, keyt

def mix(work, v, events, keyt, out_video):
    bank = sfx_bank(); N = len(v); bus = np.zeros((N, 2), np.float32)
    def place(x, t):
        i = int(t * SR); n_ = min(len(x), N - i)
        if n_ > 0 and i >= 0: bus[i:i + n_] += x[:n_]
    for t, name in events: place(bank[name](), t)
    for t in keyt: place(bank["key"](), t)
    lv = np.convolve(np.abs(v).mean(1), np.ones(960) / 960, "same"); g = np.where(lv > FX.db(-40), FX.db(-4), 1.0)
    y = v + bus * g[:, None].astype(np.float32); y *= min(1, FX.db(-1) / (np.abs(y).max() + 1e-9))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-i", os.path.join(work, "video.mp4"),
                    "-map", "1:v", "-map", "0:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_video],
                   input=y.astype(np.float32).tobytes(), check=True)

if __name__ == "__main__":
    work, model, out = sys.argv[1:4]; os.makedirs(work, exist_ok=True)
    v, scenes, total = voice(work, model)
    json.dump([{k: s[k] for k in ("t0", "dur", "words")} for s in scenes], open(os.path.join(work, "timing.json"), "w"), ensure_ascii=False)
    events, keyt = render(work, scenes, total, out)
    mix(work, v, events, keyt, out)
