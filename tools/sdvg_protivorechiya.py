"""@mcg.va — reel «противоречия материнства», montage system «СДВГ» (animated-montage style, her type and palette).
Her voice, cleaned and tightened (pauses and the false start cut, studio chain), synthesized sound effects, no music.
Built from the primitives of sdvg_materinstvo.py; every time below is written in seconds of the ORIGINAL voice recording
and converted to the tightened reel with N().
  python3 sdvg_protivorechiya.py MATERIALS_DIR CUTOUTS_DIR STILLS_DIR WORDS.json OUT.mp4 [--preview O1,O2,...]   (preview times: original seconds)
"""
import json, math, os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdvg_materinstvo import (W, H, FPS, CX, CREAM, INK, BROWN, MAUVE, BLUSH, WHITE, font, ease, Stream, treat, zoom, paper,
                              text_img, plaque_img, shadowed, Sprite, Typed, Fn, big, plaque, window_frame, polaroid, cutout_img,
                              arrow_fn, bubble_img, tag_img, pill_img, subs_draw, shadow_text)
import sdvg_audio as AU

VOICE = "Новая запись.m4a"
DROP = [(73.6, 75.2), (108.47, 108.95)]                     # false start «И отдельная моя люб…»; unfinished «…обсуждать это с…»
PAD, TAIL = (0.03, 0.08), 1.8

# ── timeline: speech runs of the voice, pauses cut ──────────────────────────────────────────────────────────────────
def speech_runs(wav):
    a = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
                                     capture_output=True, check=True).stdout, np.float32)
    hop = 480; n = len(a) // hop; e = 20 * np.log10(np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    e = np.convolve(e, np.ones(3) / 3, "same"); sp = e > -52; runs, i = [], 0
    while i < n:
        if sp[i]:
            j = i
            while j < n and sp[j]: j += 1
            runs.append([i / 100, j / 100]); i = j
        else: i += 1
    m = [runs[0]]
    for r in runs[1:]:
        if r[0] - m[-1][1] < 0.25: m[-1][1] = r[1]
        else: m.append(r)
    m = [r for r in m if r[1] - r[0] > 0.06 and r[1] > 1.8 and r[0] < 129.8 and not any(a <= r[0] and r[1] <= b for a, b in DROP)]
    out = []
    for a, b in m:
        a, b = a - PAD[0], b + PAD[1]
        if out and a <= out[-1][1]: out[-1][1] = b
        else: out.append([a, b])
    for da, db in DROP:                                     # cut the stumbles out of the runs
        cut = []
        for a, b in out:
            if b <= da or a >= db: cut.append([a, b]); continue
            if a < da - 0.05: cut.append([a, da])
            if b > db + 0.05: cut.append([db, b])
        out = cut
    return out

class Timeline:
    def __init__(self, runs):
        self.runs = runs
        self.gaps = [(0.12 if runs[k + 1][0] - runs[k][1] >= 0.35 else 0.03) for k in range(len(runs) - 1)] + [0.0]
        self.starts = list(np.cumsum([0.0] + [b - a + g for (a, b), g in zip(runs, self.gaps)])[:-1])
        self.speech_end = self.starts[-1] + runs[-1][1] - runs[-1][0]; self.end = self.speech_end + TAIL

    def N(self, o):
        for k, (a, b) in enumerate(self.runs):
            if o < a: return self.starts[k]
            if o <= b: return self.starts[k] + (o - a)
        return self.speech_end + (o - self.runs[-1][1])

# ── subtitles ───────────────────────────────────────────────────────────────────────────────────────────────────────
FIX = {"пиздец.": "пи**ец.", "весит": "бесит", "разрыветься.": "разреветься.", "чатик,": "часик,", "ее": "её", "ребенка": "ребёнка",
       "ребенок": "ребёнок", "подрастет": "подрастёт", "растешь.": "растёшь?", "все.": "всё.", "всем.": "всём.", "Ну": "ну"}
FIX_AT = {101.36: "во", 101.56: "мне", 68.38: "каждые", 111.70: "всё"}

def cues_of(words_path, N):
    raw = json.load(open(words_path)); ws = []
    for w, a, b in raw:
        w = w.strip()
        if w.startswith("-") and ws: ws[-1][0] += w; ws[-1][2] = b; continue
        ws.append([w, a, b])
    for w in ws:
        w[0] = FIX_AT.get(round(w[1], 2), FIX.get(w[0], w[0]))
    cues, cur = [], []
    for w, a, b in ws:
        cur.append([w, N(a), N(b)])
        if w[-1] in ".?!" or (w[-1] == "," and len(cur) >= 5) or len(cur) >= 9: cues.append(cur); cur = []
    if cur: cues.append(cur)
    for c in cues:
        if c[0][0] != "Агата": c[0][0] = c[0][0][0].lower() + c[0][0][1:]
    return cues

# ── extra building blocks ───────────────────────────────────────────────────────────────────────────────────────────
def gallery_fn(stills, t0, x0=110, y0=360, w=520, h=900):
    """a phone with a photo grid that scrolls (she re-watches the baby photos)"""
    th = (w - 40 - 2 * 8) // 3; ims = []
    for p in stills:
        im = Image.open(p).convert("RGB"); s = max(th / im.width, th / im.height); im = im.resize((int(im.width * s) + 1, int(im.height * s) + 1))
        ims.append(im.crop(((im.width - th) // 2, (im.height - th) // 2, (im.width - th) // 2 + th, (im.height - th) // 2 + th)))
    rows = math.ceil(len(ims) * 3 / 3); strip = Image.new("RGB", (w - 40, (th + 8) * 12), (250, 248, 245))
    for k in range(36): strip.paste(ims[k % len(ims)], ((k % 3) * (th + 8), (k // 3) * (th + 8)))
    body = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(body)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 64, fill=INK + (255,)); mask = Image.new("L", (w - 40, h - 40), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 41, h - 41), 48, fill=255)
    frame = shadowed(body, 20, 0.3, (0, 12)); off = 40
    def fn(cv, t):
        k = ease((t - t0) / 0.35); y = y0 + (1 - k) * 200; sc = int(max(0, t - t0 - 0.5) * 260) % ((th + 8) * 6)
        cv.alpha_composite(frame, (int(x0 - off), int(y - off)))
        view = strip.crop((0, sc, w - 40, sc + h - 40)); cv.paste(view, (x0 + 20, int(y) + 20), mask)
        ImageDraw.Draw(cv).rounded_rectangle((x0 + w / 2 - 70, y + 34, x0 + w / 2 + 70, y + 60), 13, fill=INK + (255,))
    return fn

def checklist_fn(items, t0, x=100, y=430):
    """«ага, дома:» — to-do lines that appear and get crossed out"""
    f = font(48, 300); fh = font(34, 300, True)
    def fn(cv, t):
        d = ImageDraw.Draw(cv); d.text((x, y), "план на день:", font=fh, fill=MAUVE + (255,), anchor="ls")
        for k, (txt, ta, tx) in enumerate(items):
            if t < ta: continue
            yy = y + 110 + k * 120; a = ease((t - ta) / 0.25)
            d.rounded_rectangle((x, yy - 42, x + 46, yy + 4), 8, outline=INK + (int(255 * a),), width=3)
            d.text((x + 76, yy), txt, font=f, fill=INK + (int(255 * a),), anchor="ls")
            if t >= tx:
                k2 = ease((t - tx) / 0.3); L = f.getlength(txt)
                d.line((x + 6, yy - 36, x + 6 + 34 * k2, yy - 2), fill=MAUVE + (255,), width=5)
                d.line((x + 40, yy - 36, x + 40 - 34 * k2, yy - 2), fill=MAUVE + (255,), width=5)
                d.line((x + 70, yy - 16, x + 70 + (L + 12) * k2, yy - 16), fill=MAUVE + (255,), width=3)
    return fn

def ticket_img():
    w, h = 640, 260; im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 26, fill=BLUSH + (255,))
    for r in (0, h): d.ellipse((440 - 22, r - 22, 440 + 22, r + 22), fill=(0, 0, 0, 0))
    for yy in range(26, h - 20, 22): d.line((440, yy, 440, yy + 10), fill=INK + (120,), width=2)
    d.text((36, 58), "МАРШРУТ", font=font(24, 400), fill=MAUVE + (255,), anchor="ls")
    d.text((36, 150), "куда-то  →", font=font(66, 300, True), fill=INK + (255,), anchor="ls")
    d.text((36, 214), "без коляски · без графика", font=font(26, 300), fill=INK + (200,), anchor="ls")
    d.text((540, 120), "1", font=font(70, 200), fill=INK + (255,), anchor="mm"); d.text((540, 180), "мама", font=font(24, 300), fill=MAUVE + (255,), anchor="mm")
    return shadowed(im, 14, 0.25, (0, 8))

def bar_fn(t0, t1, label, x0=150, x1=800, y=1290):
    def fn(cv, t):
        d = ImageDraw.Draw(cv); k = ease((t - t0) / (t1 - t0))
        shadow_text(cv, (x0, y - 34), label, font(32, 300, True), "ls"); d.text((x0, y - 34), label, font=font(32, 300, True), fill=WHITE + (255,), anchor="ls")
        d.rounded_rectangle((x0, y - 16, x1, y + 16), 16, fill=CREAM + (200,), outline=WHITE + (255,), width=2)
        if k > 0.02: d.rounded_rectangle((x0 + 4, y - 12, x0 + 4 + (x1 - x0 - 8) * k, y + 12), 12, fill=BLUSH + (255,))
    return fn

# ── the reel ────────────────────────────────────────────────────────────────────────────────────────────────────────
def build(M, C, ST, T):
    N = T.N; dl = lambda n: os.path.join(M, n); cut = lambda n, h: cutout_img(os.path.join(C, n), h)
    V = lambda n, ss, **k: (lambda dur: Stream(dl(n), ss, dur, **k))
    sc, el, sfx, whips = [], [], [], []
    def S(o0, o1, kind, spec=None, whip=False, **kw):
        sc.append(dict(t0=N(o0), t1=N(o1), kind=kind, spec=spec, **kw))
        if whip: whips.append(N(o0))
        elif sc[:-1]: sfx.append((N(o0) - 0.1, "soft"))
    def E(e, kind=None, at=None):
        el.append(e)
        if kind: sfx.append((e.t0 if at is None else at, kind))
    BIG = lambda txt, y, o0, o1, size=150, *a, **k: E(big(txt, CX, y, N(o0), N(o1), size, *a, **k), "swell")
    PL = lambda txt, x, y, o0, o1, st="blush", ang=-3, size=46, **k: E(plaque(txt, x, y, N(o0), N(o1), st, ang, size, **k), "pop")
    IT = lambda txt, x, y, o0, o1, size=72, col=WHITE, glow=(50, 34, 26): E(Sprite(text_img(txt, font(size, 250, True), col, glow=glow), x, y, N(o0), N(o1), "pop"), "pop")
    TY = lambda txt, x, y, o0, o1, size=50, col=INK, sh=False: [E(Typed(txt, x, y, N(o0), N(o1), font(size, 300, True), col, 0.045, shadow=sh))] + \
        [sfx.append((N(o0) + i * 0.045, "key")) for i, c in enumerate(txt) if c != " "]

    # 1 hook — прекрасно и одновременно пи**ец
    S(1.8, 5.6, "clip", V("IMG_4709.MOV", 0.5))
    BIG("ПРЕКРАСНО", 1080, 2.98, 5.6, 130); IT("и одновременно", CX, 1190, 3.64, 5.6, 60); PL("пи**ец.", CX, 1290, 4.42, 5.6, "ink", -2, 58, shake=3)
    # 2 what surprised me — I contradict myself
    S(5.6, 11.6, "clip", V("IMG_4992.MOV", 2.0))
    PL("больше всего удивило", CX, 900, 7.70, 9.6, "blush", -2); IT("сама себе", 300, 830, 10.06, 11.6, 46); BIG("ПРОТИВОРЕЧУ", 915, 10.62, 11.6, 100)
    # 3 yesterday — dreamt she'd fall asleep
    S(11.6, 15.7, "window", V("IMG_9977.MOV", 0.5, w=840, h=760))
    E(Sprite(pill_img("вчера"), CX, 1200, N(12.1), N(15.7), "pop"), "tick"); TY("наконец-то уснула", 120, 1290, 14.52, 15.7)
    # 4 she fell asleep, I lay down
    S(15.7, 18.0, "clip", V("IMG_5638.MOV", 0.5), mode="soft", whip=True)
    PL("она уснула", 300, 420, 15.78, 18.0, "blush", -3, 50); PL("я легла отдыхать", 420, 520, 16.74, 18.0, "ink", 2, 50)
    # 5 …and 10 minutes later I was scrolling her photos
    S(18.0, 22.6, "paper")
    E(Sprite(cut("IMG_7960.png", 820), 830, 1000, N(18.0), N(22.6), "up", dur=0.45), "swish")
    IT("и что вы думаете?", CX, 300, 18.08, 22.6, 66, INK, None)
    E(Sprite(pill_img("+10 минут"), 800, 440, N(19.14), N(22.6), "pop"), "tick")
    stills = sorted(os.path.join(ST, f) for f in os.listdir(ST))
    E(Fn(gallery_fn(stills, N(20.0), 90, 380, 500, 860), N(20.0), N(22.6)), "swish", N(20.0))
    # 6 and so in everything
    S(22.6, 24.5, "clip", V("IMG_6383.MOV", 3.0, speed=0.5), whip=True)
    BIG("ВО ВСЁМ.", 1150, 23.82, 24.5, 150)
    # 7 dream of silence — start missing her
    S(24.5, 29.6, "split", [V("IMG_9041.MOV", 3.0, h=950), V("copy_1CD5B7E7-302D-46C1-B814-3C032E0A8B9B.mov", 9.0, h=950)], whip=True)
    PL("(1) мечтаешь о тишине", 330, 820, 25.08, 29.6, "blush", -2); PL("(2) начинаешь скучать", 330, 1070, 27.88, 29.6, "ink", 2)
    # 8 want to be left alone
    S(29.6, 32.9, "clip", V("IMG_7146.MOV", 1.0), whip=True)
    PL("оставили одну", CX, 1160, 31.62, 32.9, "blush", -3, 52)
    # 9 free time — what to do with it
    S(32.9, 37.3, "paper")
    E(Sprite(cut("E7CA53F1-372D-41B8-AFE2-7AFFC580A2CE.png", 760), 720, 1030, N(32.9), N(37.3), "up", dur=0.45), "swish")
    E(Sprite(pill_img("свободное время"), 330, 480, N(33.84), N(37.3), "pop"), "tick")
    TY("что с ним делать?", 100, 640, 35.64, 37.3, 64)
    # 10 a day when everything annoys you — rapid b&w cuts
    S(37.3, 38.3, "clip", V("IMG_0799.MOV", 1.0), mode="bw", whip=True)
    S(38.3, 39.1, "clip", V("C0690.MP4", 1.0), mode="bw", whip=True)
    S(39.1, 40.4, "clip", V("IMG_5626.MOV", 1.0), mode="bw", whip=True)
    BIG("БЕСИТ", 1110, 38.90, 40.4, 170); PL("абсолютно всё", CX, 940, 39.24, 40.4, "ink", 2, 52, shake=4)
    # 11 so tired you want nothing
    S(40.4, 43.2, "clip", V("IMG_8820.MOV", 1.0), mode="bw")
    PL("ничего не хочешь", CX, 1200, 42.16, 43.2, "blush", -2, 50)
    # 12 …but you look at your child: the best thing in your life
    S(43.2, 48.8, "clip", V("1924850015330377605.MP4", 8.0), mode="warm", whip=True)
    BIG("ЛУЧШЕЕ", 1130, 46.60, 48.8, 170); sfx.append((N(46.6), "shimmer"))
    # 13 when Agata grows up it'll be easier
    S(48.8, 53.6, "clip", V("IMG_9217.MOV", 3.0), mode="warm", whip=True)
    E(Fn(bar_fn(N(50.7), N(52.6), "подрастёт Агата…"), N(50.5), N(53.6)), "tick", N(50.5))
    PL("станет чуть полегче", CX, 1050, 52.24, 53.6, "blush", -2, 46)
    # 14 photos from a month ago — where are you growing so fast
    S(53.6, 59.6, "paper")
    E(Sprite(polaroid(os.path.join(ST, "s00.jpg"), 420, 1.15, -5), 330, 700, N(54.64), N(59.6), "drop", -5, dur=0.3), "shutter")
    E(Sprite(polaroid(os.path.join(ST, "s05.jpg"), 420, 1.15, 5), 740, 960, N(56.54), N(59.6), "drop", 5, dur=0.3), "shutter")
    TY("месяц назад", 120, 1075, 55.38, 59.6, 40); TY("сейчас", 640, 1300, 56.74, 59.6, 40)
    E(Fn(arrow_fn(((300, 1105), (380, 1250), (560, 1230)), N(57.0)), N(57.0), N(59.6)))
    IT("куда ты так быстро?", CX, 300, 57.42, 59.6, 64, INK, None)
    # 15 so sad, want to cry
    S(59.6, 63.8, "clip", V("2026-09-13 12-40-26.MP4", 18.0), mode="soft", whip=True)
    PL("хочется разреветься", CX, 1180, 61.72, 63.8, "blush", -2, 48)
    # 16 someone take her for an hour — and then «как она?» every 5 minutes
    S(63.8, 73.0, "paper")
    E(Fn(lambda cv, t: ImageDraw.Draw(cv).text((90, 430), "кому:  няня", font=font(44, 300), fill=MAUVE + (255,), anchor="ls"), N(63.9), N(73.0)))
    E(Sprite(tag_img("1 час"), 760, 470, N(66.76), N(73.0), "drop", -6), "ding")
    for k, (o, txt, hh) in enumerate(((68.38, "как она?", "12:00"), (68.80, "как она??", "12:05"), (69.40, "всё хорошо?", "12:10"))):
        b = bubble_img(txt); E(Sprite(b, 90 + b.width / 2, 600 + k * 150, N(o), N(73.0), "pop"), "bloop")
        E(Fn((lambda hh, k: lambda cv, t: ImageDraw.Draw(cv).text((100, 690 + k * 150), hh, font=font(24, 350), fill=MAUVE + (255,), anchor="ls"))(hh, k), N(o), N(73.0)))
    E(Sprite(cut("IMG_7960.png", 640), 820, 1080, N(64.0), N(73.0), "up", dur=0.45), "swish")
    PL("особенно когда она с няней", CX, 1250, 70.78, 73.0, "ink", -2, 44)
    # 17 «ну ты же целый день дома»
    S(76.0, 80.1, "clip", V("IMG_6883.MOV", 4.0, speed=0.5), whip=True)
    E(Sprite(pill_img("дома"), CX, 1020, N(79.34), N(80.1), "pop"), "tick")
    BIG("«ЦЕЛЫЙ ДЕНЬ ДОМА»", 1150, 78.6, 80.1, 84)
    # 18 aha, home: no coffee, no toilet
    S(80.1, 86.4, "paper")
    E(Sprite(cut("IMG_6987.png", 900), 830, 1000, N(80.1), N(86.4), "up", dur=0.45), "swish")
    IT("ага, дома.", 360, 300, 80.26, 86.4, 70, INK, None)
    E(Fn(checklist_fn([("выпить кофе", N(81.78), N(82.9)), ("сходить в туалет", N(85.26), N(85.75))], N(80.5)), N(80.5), N(86.4)))
    for o in (82.9, 85.75): sfx.append((N(o), "error"))
    # 19 the strangest thing: I miss my old life (zoom-blur into the pregnancy mirror selfie)
    S(86.4, 91.4, "zoomblur", (dl("IMG_4023.MOV"), 3.0))
    IT("самое странное", CX, 700, 87.32, 91.4, 66); BIG("ПРЕЖНЕЙ", 860, 90.30, 91.4, 150); BIG("ЖИЗНИ", 1020, 90.62, 91.4, 150)
    # 20 the freedom to just go somewhere
    S(91.4, 97.1, "paper")
    E(Sprite(cut("IMG_7153.png", 1000), 790, 1000, N(91.4), N(97.1), "up", dur=0.45), "swish")
    E(Sprite(ticket_img(), 420, 520, N(93.36), N(97.1), "drop", -4), "ding")
    TY("ни под кого не подстраиваться", 90, 1260, 95.32, 97.1, 46)
    # 21 though I wanted to be a mom so much
    S(97.1, 99.5, "clip", V("IMG_4023.MOV", 5.0), mode="warm", whip=True)
    PL("очень хотела стать мамой", CX, 1180, 97.62, 99.5, "blush", -2, 46)
    # 22 such opposite feelings
    S(99.5, 108.8, "split", [V("1924850015330377605.MP4", 15.0, h=950), V("IMG_3113.MOV", 6.0, h=950)], whip=True)
    IT("противоположные", CX, 960, 105.46, 108.8, 76); PL("чувства", CX, 1080, 106.34, 108.8, "ink", 2, 48)
    PL("а потом начала обсуждать", CX, 1240, 107.3, 108.8, "blush", -2, 42)
    # 23 love her madly and get tired of motherhood
    S(108.8, 117.8, "clip", V("06b85b40-c32f-4a9a-a26e-e95049063abc.mp4", 30.0), mode="warm", whip=True)
    BIG("ЛЮБИТЬ", 1100, 114.22, 117.8, 170); PL("и уставать", CX, 1240, 116.58, 117.8, "ink", 2, 52)
    # 24 be alone and miss her
    S(117.8, 121.1, "clip", V("IMG_9888.MOV", 5.0), whip=True)
    PL("побыть одна", 360, 420, 118.50, 121.1, "blush", -3, 50); PL("скучать по ней", 560, 520, 119.80, 121.1, "ink", 3, 50)
    # 25 one doesn't cancel the other
    S(121.1, 125.6, "window", V("IMG_6740.MOV", 4.0, w=840, h=760))
    BIG("НЕ ОТМЕНЯЕТ", 1190, 122.34, 125.6, 104, MAUVE, 0.6, None); IT("одно другого", CX, 1290, 121.46, 125.6, 52, INK, None)
    # 26 moms, is it the same for you?
    S(125.6, 140.0, "photo", dl("72BC58DA-326D-4D9C-933E-BE9E71C75B77.png"))
    PL("у вас тоже так?", CX, 1305, 126.80, 140.0, "blush", -3, 46); BIG("ПРОТИВОРЕЧИВАЯ?", 330, 128.6, 140.0, 82)
    sc[-1]["t1"] = T.end
    for e in el:
        if e.t1 > T.end: e.t1 = T.end
    return sc, el, sfx, whips

def background(s, t, live, whips):
    lt = t - s["t0"]; i = int(round(lt * FPS)); k = s["kind"]; dur = s["t1"] - s["t0"]
    if k == "clip":
        if "st" not in live: live["st"] = s["spec"](dur)
        return treat(zoom(live["st"].frame(i), s.get("z", 1.0) + 0.03 * lt / dur), s.get("mode"))
    if k == "paper": return paper()
    if k == "window":
        cv = Image.fromarray(paper()).convert("RGBA")
        if "st" not in live: live["st"] = s["spec"](dur)
        fr = live["st"].frame(i); win, off = window_frame(fr.shape[1], fr.shape[0]); win.paste(Image.fromarray(fr), off)
        win = shadowed(win, 22, 0.25, (0, 14)); y = 300 + (1 - ease(lt / 0.3)) * 120
        cv.alpha_composite(win, (int(CX - win.width / 2), int(y - 44))); return np.asarray(cv.convert("RGB")).copy()
    if k == "split":
        if "a" not in live: live["a"] = s["spec"][0](dur); live["b"] = s["spec"][1](dur)
        out = np.empty((H, W, 3), np.uint8); out[:] = CREAM; out[:950] = live["a"].frame(i); out[970:] = live["b"].frame(i); return out
    if k == "photo":
        if "img" not in live:
            im = ImageOps.exif_transpose(Image.open(s["spec"])).convert("RGB"); sc_ = max(W / im.width, H / im.height)
            im = im.resize((int(im.width * sc_) + 1, int(im.height * sc_) + 1), Image.LANCZOS)
            live["img"] = np.asarray(im.crop(((im.width - W) // 2, (im.height - H) // 2, (im.width - W) // 2 + W, (im.height - H) // 2 + H)))
        return zoom(live["img"], 1.0 + 0.07 * lt / dur, 0.4)
    if k == "zoomblur":
        if "img" not in live:
            src, ss = s["spec"]; st = Stream(src, ss, 0.2); f0 = st.frame(0).copy(); st.close()
            g = treat(f0, "bw").astype(np.float32); live["img"] = np.clip(cv2.GaussianBlur(g, (0, 0), 22) * 0.62, 0, 255).astype(np.uint8); live["sharp"] = treat(f0, "bw")
        z = 1.0 + 0.25 * (1 - ease(lt / 0.45)); base = live["img"] if lt > 0.25 else live["sharp"]; a = zoom(base, z, 0.5)
        if lt < 0.45:
            acc = a.astype(np.float32)
            for q in (1.04, 1.08, 1.12): acc += zoom(base, z * q, 0.5)
            a = (acc / 4).astype(np.uint8)
        return a
    raise ValueError(k)

def whip(a, t, whips):
    for c in whips:
        d = t - c
        if abs(d) < 0.12:
            s = 1 - abs(d) / 0.12; a = cv2.blur(a, (int(1 + s * 150), 1)); a = np.roll(a, int(np.sign(d or 1) * s * 70), axis=1)
    return a

def sounds(sfx, whips):
    ev = [(t - 0.16, AU.whoosh(0.34, gain=0.42)) for t in whips]
    make = {"pop": AU.pop, "swell": AU.swell, "key": AU.key, "bloop": AU.bloop, "shutter": AU.shutter, "ding": AU.ding,
            "tick": lambda: AU.tick(0.18, 1700), "error": AU.error, "shimmer": lambda: AU.shimmer(0.12, 1.3),
            "swish": lambda: AU.whoosh(0.45, 200, 1800, 0.3), "soft": lambda: AU.whoosh(0.3, 400, 3000, 0.22)}
    return ev + [(t, make[k]()) for t, k in sfx]

def render(M, C, ST, words, out, preview=None):
    T = Timeline(speech_runs(os.path.join(M, VOICE))); cues = cues_of(words, T.N); sc, el, sfx, whips = build(M, C, ST, T)
    print(f"reel {T.end:.1f} s (voice {T.speech_end:.1f} s)")
    times = [T.N(o) for o in preview] if preview else [i / FPS for i in range(int(T.end * FPS))]
    enc = None if preview else subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                                                 "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", out + ".video.mp4"],
                                                stdin=subprocess.PIPE)
    live, cur, shots = {}, None, []
    for t in times:
        s = next((x for x in sc if x["t0"] <= t < x["t1"]), sc[-1])
        if s is not cur:
            for v in live.values():
                if isinstance(v, Stream): v.close()
            live, cur = {}, s
        cv = Image.fromarray(background(s, t, live, whips)).convert("RGBA")
        for e in el: e.draw(cv, t)
        subs_draw(cv, t, cues, ink=s["kind"] in ("paper", "window"))
        a = whip(np.asarray(cv.convert("RGB")), t, whips)
        if t > T.end - 0.6: k = (t - (T.end - 0.6)) / 0.6; a = (a * (1 - k) + np.array(CREAM) * k).astype(np.uint8)
        if preview: shots.append(Image.fromarray(a))
        else: enc.stdin.write(np.ascontiguousarray(a).tobytes())
    if preview:
        sheet = Image.new("RGB", (len(shots) * 360, 640))
        for i, im in enumerate(shots): sheet.paste(im.resize((360, 640)), (i * 360, 0))
        sheet.save(out); return
    enc.stdin.close(); enc.wait()
    v = out + ".v.wav"; AU.studio_voice(os.path.join(M, VOICE), T.runs, T.gaps, T.end, v)
    AU.mix(v, None, 0, sounds(sfx, whips), T.end, out + ".a.wav"); os.remove(v)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out + ".video.mp4", "-i", out + ".a.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out], check=True)
    os.remove(out + ".video.mp4"); os.remove(out + ".a.wav")

if __name__ == "__main__":
    a = sys.argv; pv = [float(x) for x in a[a.index("--preview") + 1].split(",")] if "--preview" in a else None
    render(a[1], a[2], a[3], a[4], a[5], pv)
