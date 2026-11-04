"""Animated Instagram carousel "Мне 54. Не поздно ли?" (8 slides, 1080x1350, 30 fps).

Text is animated on top of footage / photos in the surgeon's style (Bebas + Montserrat, graphite,
navy highlights, pastel pink accents); the surgeon's voice note is cut per slide and laid over a
soft synthesised piano bed (tools/music_soft_piano.py).

usage: python3 post_age54.py SRC_DIR VOICE.wav MUSIC.wav OUT_DIR [slide numbers...]
"""
import os, sys, subprocess, wave
import numpy as np, cv2

SRC, VOICE, MUSIC, OUT = sys.argv[1:5]
ONLY = {int(a) for a in sys.argv[5:]}
sys.argv = [sys.argv[0], SRC, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carousel_tvauri as ct  # noqa: E402
from PIL import Image, ImageDraw, ImageOps  # noqa: E402

W, H, FPS, SR = 1080, 1350, 30, 48000
PINK = (247, 207, 217)
NAVY = (16, 46, 70)
WHITE, INK, GRAPHITE = ct.WHITE, ct.INK, ct.GRAPHITE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ct.LILAC = PINK


# ---------------------------------------------------------------- layers
class El:
    """an RGBA layer cropped to its bbox, shown from t0 with an entrance animation"""
    def __init__(self, img, t0=0.0, kind="rise", d=0.5, t1=None):
        bb = img.getbbox() or (0, 0, 1, 1)
        self.x, self.y = bb[0], bb[1]
        a = np.asarray(img.crop(bb)).astype(np.float32) / 255
        self.rgb = a[..., :3] * a[..., 3:]; self.a = a[..., 3:]
        self.t0, self.kind, self.d, self.t1 = t0, kind, d, t1

    def draw(self, fr, t):
        p = (t - self.t0) / self.d
        if p <= 0: return
        e = 1 - (1 - min(p, 1)) ** 3
        rgb, a, x, y = self.rgb, self.a, self.x, self.y
        if self.kind == "rise":
            y += int(46 * (1 - e)); k = e
        elif self.kind == "fade":
            k = e
        elif self.kind == "pop":
            s = 0.86 + 0.14 * (1 + 2.2 * (min(p, 1) - 1) ** 3 + 1.2 * (min(p, 1) - 1) ** 2) if p < 1 else 1.0
            k = min(1, p * 2.5)
            if abs(s - 1) > 0.004:
                h0, w0 = a.shape[:2]; nw, nh = max(1, int(w0 * s)), max(1, int(h0 * s))
                rgb = cv2.resize(rgb, (nw, nh)); a = cv2.resize(a, (nw, nh))[..., None]
                x += (w0 - nw) // 2; y += (h0 - nh) // 2
        elif self.kind == "wipe":  # left-to-right reveal (strike-through line)
            w0 = a.shape[1]; cut = max(1, int(w0 * e)); rgb = rgb[:, :cut]; a = a[:, :cut]; k = 1
        else:
            k = 1
        if self.t1 is not None:  # fade out
            k *= 1 - min(max((t - self.t1) / 0.35, 0), 1)
            if k <= 0: return
        blend(fr, rgb, a, x, y, k)


def blend(fr, rgb, a, x, y, k=1.0):
    h, w = a.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    s = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    d = fr[y0:y1, x0:x1]
    d *= 1 - a[s] * k; d += rgb[s] * k


def layer(): return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def text_center(lines, font, y, color=WHITE, lh=None, box=None, shadow=True):
    im = layer(); d = ImageDraw.Draw(im); lh = lh or int(font.size * 1.02)
    for i, ln in enumerate(lines):
        tw = font.getlength(ln); x = (W - tw) / 2; yy = y + i * lh
        if box:
            d.rectangle((x - 16, yy + int(font.size * 0.06), x + tw + 16, yy + int(font.size * 1.0)), fill=box + (255,))
            d.text((x, yy), ln, font=font, fill=color)
        elif shadow: ct.shadow_text(im, (x, yy), ln, font, color, blur=7, a=150)
        else: d.text((x, yy), ln, font=font, fill=color)
    return im


def wrap(text, font, width): return ct.wrap(text, font, width)


def pill(text, font, cx, y, bg=PINK, fg=INK, pad=24, hgt=None):
    im = layer(); d = ImageDraw.Draw(im); tw = font.getlength(text); hgt = hgt or int(font.size * 1.7)
    d.rounded_rectangle((cx - tw / 2 - pad, y, cx + tw / 2 + pad, y + hgt), radius=hgt // 2, fill=bg + (255,))
    d.text((cx, y + hgt / 2 + 1), text, font=font, fill=fg, anchor="mm")
    return im


def logo(y=92):
    im = layer(); ct.put_logo(im, y=y, alpha=0.95); return im


def rule(y, w=150):
    im = layer(); ImageDraw.Draw(im).rectangle(((W - w) / 2, y, (W + w) / 2, y + 4), fill=PINK + (255,)); return im


# ---------------------------------------------------------------- backgrounds
def video_bg(path, start, dur, blur=6, dim=0.55, cy=0.5, zoom=1.0):
    """footage scaled to cover 4:5, softened and dimmed so the type reads; loops if the clip is short"""
    sw = int(W * zoom) // 2 * 2
    vf = (f"scale={sw}:{H * 2}:force_original_aspect_ratio=increase,crop={W}:{H}:(iw-{W})/2:(ih-{H})*{cy},"
          f"fps={FPS}" + (f",gblur=sigma={blur}" if blur else "") + f",eq=brightness={-dim * 0.35}:saturation=0.9,format=rgb24")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-stream_loop", "-1", "-ss", f"{start}", "-i", os.path.join(SRC, path),
                          "-t", f"{dur + 0.5}", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    shade = np.float32(GRAPHITE) / 255
    while True:
        b = p.stdout.read(W * H * 3)
        if len(b) < W * H * 3: break
        f = np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32) / 255
        yield f * (1 - dim) + shade * dim


def flat_bg(col=GRAPHITE, dark_bottom=0.5):
    g = np.linspace(0, dark_bottom, H, dtype=np.float32)[:, None, None]
    base = np.float32(col) / 255 * (1 - g * 0.6)
    fr = np.broadcast_to(base, (H, W, 3)).copy()
    while True: yield fr.copy()


def body_cell(fname, w, h):
    im = ImageOps.exif_transpose(Image.open(os.path.join(SRC, fname))).convert("RGB")
    return np.asarray(ct.body_crop(im, w / h).resize((w, h), Image.LANCZOS)).astype(np.float32) / 255


def ba_labels(cell, t_in, t_wipe, wipe_d, size=40):
    """ДО sits on the before photo and leaves as the line passes; ПОСЛЕ arrives with the after photo"""
    x, y, w, h = cell; out = []
    for txt, right in (("ДО", False), ("ПОСЛЕ", True)):
        im = layer(); d = ImageDraw.Draw(im); f = B(size); tw = f.getlength(txt)
        x0 = x + w - tw - 46 if right else x + 18
        d.rounded_rectangle((x0, y + 16, x0 + tw + 28, y + 16 + size + 10), radius=12, fill=NAVY + (235,))
        d.text((x0 + 14, y + 19), txt, font=f, fill=WHITE)
        out.append(El(im, t_wipe + 0.45 * wipe_d, "fade", 0.35) if right else El(im, t_in, "fade", 0.4, t1=t_wipe + 0.55 * wipe_d))
    return out


def wipe_bg(before, after, cell, t_wipe, wipe_d=1.4):
    """black ground (the photos' own backdrop), before photo, then a pink line sweeps in the after photo"""
    x, y, w, h = cell
    bi, ai = body_cell(before, w, h), body_cell(after, w, h)
    xs = np.arange(w, dtype=np.float32)[None, :, None]
    pink = np.float32(PINK) / 255
    t = 0.0
    while True:
        fr = np.zeros((H, W, 3), np.float32)
        p = min(max((t - t_wipe) / wipe_d, 0), 1); p = p * p * (3 - 2 * p)
        lx = w * (1 - p)
        m = np.clip((xs - lx) / 6 + 0.5, 0, 1)
        c = bi * (1 - m) + ai * m
        if 0 < p < 1:
            g = np.exp(-((xs - lx) / 5) ** 2); c = c * (1 - g) + pink * g
        fr[y:y + h, x:x + w] = c
        yield fr
        t += 1 / FPS


# ---------------------------------------------------------------- audio
def read_wav(path):
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        ch, sr = w.getnchannels(), w.getframerate()
    a = a.reshape(-1, ch).mean(1)
    if sr != SR: a = np.interp(np.arange(int(len(a) * SR / sr)) * sr / SR, np.arange(len(a)), a)
    return a


VOX = read_wav(VOICE); MUS = read_wav(MUSIC)
VOX /= np.sqrt(np.mean(VOX[np.abs(VOX) > 0.02] ** 2)) + 1e-9; VOX *= 0.16  # speech level
MUS /= np.sqrt(np.mean(MUS ** 2)) + 1e-9


def mix_audio(dur, voice=None, at=0.4, music_at=0.0):
    n = int(dur * SR); out = np.zeros(n, np.float32)
    m = MUS[int(music_at * SR):int(music_at * SR) + n]; m = np.pad(m, (0, n - len(m)))
    gain = np.full(n, 0.075 if voice else 0.11, np.float32)
    if voice:
        a, b = voice; seg = VOX[int(a * SR):int(b * SR)].copy()
        f = int(0.03 * SR); seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
        s = int(at * SR); seg = seg[:n - s]; out[s:s + len(seg)] += seg
        # duck the music under the voice a bit more
        env = np.zeros(n, np.float32); env[s:s + len(seg)] = 1
        env = np.convolve(env, np.ones(int(0.25 * SR)) / int(0.25 * SR), mode="same")
        gain *= 1 - 0.45 * env
    fi, fo = int(0.35 * SR), int(0.6 * SR)
    mm = m * gain; mm[:fi] *= np.linspace(0, 1, fi); mm[-fo:] *= np.linspace(1, 0, fo)
    out += mm
    return np.clip(out, -1, 1)


# ---------------------------------------------------------------- render
MUSIC_POS = [0.0]


def render(name, dur, bg, els, voice=None, at=0.4, post=None):
    path = os.path.join(OUT, name + ".mp4"); wav = os.path.join(OUT, f"_{name}.wav")
    au = mix_audio(dur, voice, at, MUSIC_POS[0]); MUSIC_POS[0] += dur
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((au * 32767).astype(np.int16).tobytes())
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-i", wav, "-map", "0:v", "-map", "1:a", "-shortest",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for i in range(int(dur * FPS)):
        t = i / FPS
        fr = next(bg)
        for e in els: e.draw(fr, t)
        if post: post(fr, t)
        enc.stdin.write((np.clip(fr, 0, 1) * 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait(); os.remove(wav)
    print("done", name, round(dur, 2), flush=True)


B = ct.bebas
M = lambda s, w=500: ct.mont(s, w)
MI = lambda s: ct.mont(s, 300, italic=True)


# ---------------------------------------------------------------- slides
def s1():
    # voice: "Сегодня получила такое сообщение... мне 54 года, не поздно ли" (0.0 - 8.0)
    dur = 9.4
    els = [El(logo(), 0, "fade"),
           El(text_center(["«МНЕ 54.", "НЕ ПОЗДНО ЛИ УЖЕ ДЕЛАТЬ", "ПЛАСТИЧЕСКУЮ ОПЕРАЦИЮ?»"], B(104), 240), 0.0, "fade", 0.01),
           El(rule(580), 0.0, "fade", 0.01),
           El(text_center(["вопрос, который я получила от пациентки"], MI(36), 606), 0.0, "fade", 0.01)]
    # the message itself, redrawn without the sender's name or avatar
    im = layer(); d = ImageDraw.Draw(im); f = M(50, 400)
    lines = wrap("Меня останавливает мой возраст)), мне 54, не поздно ли?", f, 760)
    bw, bh = 860, 60 + 64 * len(lines) + 44; x0, y0 = (W - bw) // 2, 850
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=34, fill=(38, 41, 48, 245))
    d.polygon([(x0 + 4, y0 + bh - 40), (x0 - 18, y0 + bh), (x0 + 40, y0 + bh - 8)], fill=(38, 41, 48, 245))
    for i, ln in enumerate(lines): d.text((x0 + 44, y0 + 34 + i * 64), ln, font=f, fill=WHITE)
    d.text((x0 + bw - 40, y0 + bh - 30), "20:08", font=M(24, 400), fill=(140, 146, 160), anchor="rm")
    els.append(El(im, 3.3, "pop", 0.55))
    els.append(El(pill("сообщение пациентки", M(26, 600), W // 2, y0 - 76), 3.1, "rise"))
    render("01_cover", dur, video_bg("copy_28DEC6FD-30C5-4A62-A30A-83288D9DE14E.mov", 0, dur, blur=5, dim=0.58), els, (0.0, 8.0), 0.35)


def s2():
    # voice: "Что значит поздно? Во-первых, 54 года... у нас и в 60 лет, и в 65 оперируются" (8.76 - 18.95)
    dur = 11.4
    cell = ((W - 560) // 2, 470, 560, 840)
    els = [El(logo(70), 0, "fade"),
           El(text_center(["ВОЗРАСТ САМ ПО СЕБЕ НЕ ЯВЛЯЕТСЯ", "ПРИЧИНОЙ ОТКАЗЫВАТЬСЯ", "ОТ ОПЕРАЦИИ"], B(68), 132), 0.2),
           El(text_center(["У меня оперируются пациентки и старше."], M(32), 360, color=PINK), 1.0)]
    els += ba_labels(cell, 1.4, 6.6, 1.6)
    render("02_age_is_not_a_reason", dur, wipe_bg("IMG_9377.JPG", "IMG_9386.JPG", cell, 6.6, 1.6), els, (8.76, 18.95), 0.35)


def s3():
    dur = 8.0
    els = [El(logo(), 0, "fade"),
           El(text_center(["ГЛАВНЫЙ ВОПРОС НЕ"], B(74), 300), 0.3),
           El(text_center(["«СКОЛЬКО МНЕ ЛЕТ?»"], B(116), 384), 0.8)]
    f = B(116); tw = f.getlength("«СКОЛЬКО МНЕ ЛЕТ?»")
    st = layer(); ImageDraw.Draw(st).rectangle(((W - tw) / 2 - 10, 446, (W + tw) / 2 + 10, 456), fill=PINK + (255,))
    els += [El(st, 1.9, "wipe", 0.55),
            El(text_center(["А:"], B(74), 560), 2.6),
            El(text_center(["«ПОЗВОЛЯЕТ ЛИ МНЕ ЗДОРОВЬЕ", "СДЕЛАТЬ ОПЕРАЦИЮ БЕЗОПАСНО?»"], B(80), 660, box=NAVY, lh=96), 3.0),
            El(text_center(["Потому что решение принимается", "не только по возрасту."], M(34), 930, lh=46), 4.6)]
    render("03_main_question", dur, video_bg("copy_20EBCAB3-DB96-4C28-AF34-BAC74DE4B0D8.mov", 1.0, dur, blur=7, dim=0.62), els)


def s4():
    # voice: "очень важно, чтобы вас не подвело здоровье" (22.44 - 25.5)
    dur = 9.5
    els = [El(logo(), 0, "fade"),
           El(text_center(["ПЕРЕД ОПЕРАЦИЕЙ", "МЫ ОЦЕНИВАЕМ:"], B(86), 170), 0.2)]
    items = ["состояние здоровья", "хронические заболевания", "результаты обследований", "риски анестезии",
             "объём предполагаемой операции"]
    f = M(38, 500); x0 = 190
    for i, it in enumerate(items):
        im = layer(); d = ImageDraw.Draw(im); y = 400 + i * 96
        S = 3; hi = Image.new("RGBA", (64 * S, 64 * S), (0, 0, 0, 0)); hd = ImageDraw.Draw(hi)
        hd.ellipse((0, 0, 60 * S, 60 * S), fill=PINK + (255,))
        hd.line([(16 * S, 31 * S), (26 * S, 41 * S), (45 * S, 21 * S)], fill=INK + (255,), width=6 * S, joint="curve")
        im.alpha_composite(hi.resize((64, 64), Image.LANCZOS), (x0, y - 4))
        ct.shadow_text(im, (x0 + 88, y), it, f, WHITE, blur=6, a=150)
        els.append(El(im, 1.3 + i * 0.55, "rise", 0.45))
    els.append(El(text_center(["И только после этого можно говорить,", "есть ли противопоказания."], MI(36), 920, lh=48), 4.5))
    render("04_we_assess", dur, video_bg("IMG_1252.MOV", 4.2, dur, blur=7, dim=0.64), els, (22.44, 25.55), 0.5)


def s5():
    # voice: "если нет противопоказаний ... анестезиологи всем довольны, оперирующий хирург доволен ... можно оперироваться" (26.18 - 40.9)
    dur = 15.6
    els = [El(logo(), 0, "fade"),
           El(text_center(["ЕСЛИ ПРОТИВОПОКАЗАНИЙ НЕТ,", "ВОЗРАСТ НЕ ОБЯЗАТЕЛЬНО", "СТАНОВИТСЯ ПРЕПЯТСТВИЕМ"], B(80), 170), 0.3)]
    def card(text, y, icon):
        im = layer(); d = ImageDraw.Draw(im); f = M(36, 500); lines = wrap(text, f, 680)
        hh = 60 + 48 * len(lines); x0, w = 110, W - 220
        d.rounded_rectangle((x0, y, x0 + w, y + hh), radius=26, fill=(255, 255, 255, 236))
        d.ellipse((x0 + 30, y + hh / 2 - 30, x0 + 90, y + hh / 2 + 30), fill=PINK)
        d.text((x0 + 60, y + hh / 2 + 2), icon, font=B(40), fill=INK, anchor="mm")
        for i, ln in enumerate(lines): d.text((x0 + 120, y + 30 + i * 48), ln, font=f, fill=INK)
        return im
    els += [El(card("Нужен допуск анестезиолога.", 560, "1"), 5.6, "pop", 0.5),
            El(card("Нужно, чтобы хирург понимал, что операция безопасна именно в вашем случае.", 730, "2"), 7.7, "pop", 0.5)]
    render("05_no_contraindications", dur, video_bg("copy_3AF02E7B-EC4D-4A5C-8FF7-65665E757D3C.mov", 4.0, dur, blur=6, dim=0.6),
           els, (26.18, 40.95), 0.4)


def s6():
    dur = 10.0
    lines = [("Я знаю,", 0.3), ("Сначала дети.", 1.1), ("Потом работа.", 1.9), ("Потом «сейчас не время».", 2.7),
             ("А потом появляется мысль:", 3.8)]
    els = [El(logo(), 0, "fade")]
    f = M(46, 500)
    for i, (ln, t) in enumerate(lines):
        im = layer(); ct.shadow_text(im, (110, 220 + i * 74), ln, f, WHITE, blur=7, a=170); els.append(El(im, t))
    els += [El(text_center(["«НАВЕРНОЕ, УЖЕ ПОЗДНО»"], B(104), 640, box=NAVY), 4.8, "pop", 0.55),
            El(text_center(["Хотя на самом деле вопрос может быть", "совсем не в возрасте."], MI(38), 860, lh=50), 6.3)]
    render("06_too_late_thought", dur, video_bg("IMG_1302.MOV", 20.0, dur, blur=3, dim=0.58), els)


def s7():
    # voice: "Если есть такое желание, это же очень и очень хорошо" (41.1 - 43.35)
    dur = 9.5
    cell = ((W - 470) // 2, 560, 470, 760)
    els = [El(logo(62), 0, "fade"),
           El(text_center(["НЕ РЕШАЙТЕ ЗА СЕБЯ ЗАРАНЕЕ,", "ЧТО ВАМ УЖЕ НЕЛЬЗЯ"], B(70), 112), 0.2)]
    f = M(30, 500)
    t1 = wrap("Если желание есть, начните с консультации и обследования.", f, 900)
    t2 = wrap("И уже после этого можно понять, подходит ли вам операция сейчас и какой объём вмешательства будет безопасным.", f, 900)
    els += [El(text_center(t1, f, 270, color=PINK, lh=40), 1.0),
            El(text_center(t2, f, 270 + 40 * len(t1) + 18, lh=40), 1.7)]
    els += ba_labels(cell, 2.4, 4.0, 1.6, size=38)
    render("07_dont_decide_for_yourself", dur, wipe_bg("IMG_9379.JPG", "IMG_9384.JPG", cell, 4.0, 1.6), els, (41.1, 43.35), 0.5)


def s8():
    dur = 6.5
    els = [El(logo(150), 0, "fade"),
           El(text_center(["ЕЩЁ БОЛЬШЕ", "ПОЛЕЗНОЙ ИНФОРМАЦИИ"], B(92), 240), 0.2),
           El(text_center(["вы найдёте у меня на странице"], M(38), 446, color=PINK), 0.8)]
    shot = Image.open(os.path.join(ROOT, "carousel_lipofilling", "src", "profile_header.jpg")).convert("RGB")
    sw = 900; sh = int(shot.height * sw / shot.width); x0, y0 = (W - sw) // 2, 560
    S = 2
    card = shot.resize((sw * S, sh * S), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", card.size, 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, card.width - 1, card.height - 1), radius=30 * S, fill=255)
    card.putalpha(m); card = card.resize((sw, sh), Image.LANCZOS)
    im = layer(); im.alpha_composite(card, (x0, y0))
    ImageDraw.Draw(im).rounded_rectangle((x0 - 2, y0 - 2, x0 + sw + 1, y0 + sh + 1), radius=30, outline=PINK + (170,), width=3)
    els.append(El(im, 1.2, "pop", 0.6))
    k = sw / shot.width
    hl = layer(); ImageDraw.Draw(hl).rounded_rectangle((x0 + 360 * k, y0 + 30 * k, x0 + 880 * k, y0 + 120 * k), radius=18, outline=PINK + (255,), width=5)
    els.append(El(hl, 2.4, "fade", 0.4))
    render("08_blog", dur, flat_bg(), els)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i, fn in enumerate([s1, s2, s3, s4, s5, s6, s7, s8], 1):
        if ONLY and i not in ONLY:
            MUSIC_POS[0] += {1: 9.4, 2: 11.4, 3: 8.0, 4: 9.5, 5: 15.6, 6: 10.0, 7: 9.5, 8: 6.5}[i]
            continue
        fn()
