"""Reels 1080x1920 (~40 s): «Не хочу импланты. Не хочу подтяжку. Что тогда можно сделать?» — the same patient as
stories_breast_fat_only: marking (before), the operating room, the first result.  Plus a cover (JPG).

Voice pieces rA..rG are cut from the client's videos with tools/clean_speech.py (pauses, «ааа», «ага» removed, the
patient's voice changed), the subtitle text is the exact phrase put on their word timings (<piece>.subs.json).
Old story design: Inter Bold caps headline + Montserrat, black fade at the bottom, one-word subtitles, lower case,
Montserrat Light on a navy pill.  Text sits above the Reels caption and left of the buttons.

usage: python3 reels_no_implants_no_lift.py SRC_DIR OUT_DIR [cover|reel]
"""
import json, os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, SR = 1080, 1920, 30, 48000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
BEBAS = os.path.join(ROOT, "skills/montage-svg-typography/Шрифты/Коллекция шрифтов/BebasNeueCyrillic.ttf")
WHITE, NAVY, GRAPHITE = (255, 255, 255), (2, 31, 71), (20, 23, 31)
SRC, OUT = sys.argv[1], sys.argv[2]
WHAT = sys.argv[3:] or ["cover", "reel"]
DRIVE = os.path.join(SRC, "drive")
os.makedirs(OUT, exist_ok=True)
MARGIN, TEXT_W, TEXT_BOTTOM, TEXT_TOP = 84, 830, 1500, 250   # right: the Reels buttons; below 1500: the caption


def mont(size, wght=300):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([wght]); return f


def bold(size): return ImageFont.truetype(BOLD, size)


def wrap(text, f, width):
    """greedy wrap; "\n" forces a break (headlines are broken by meaning, not by width)"""
    out = []
    for par in text.split("\n"):
        line = ""
        for w in par.split(" "):
            t = (line + " " + w).strip()
            if f.getlength(t) <= width or not line: line = t
            else: out.append(line); line = w
        out.append(line)
    return out


def ease(x): x = min(max(x, 0.0), 1.0); return 1 - (1 - x) ** 3


class Sprite:
    def __init__(self, img, x, y, t0=0.0, dur=0.01, dy=0):
        self.a = np.asarray(img, np.float32) / 255.0; self.x, self.y, self.t0, self.dur, self.dy = int(x), int(y), t0, dur, dy

    def draw(self, fr, t):
        k = ease((t - self.t0) / self.dur)
        if k <= 0: return
        y = self.y + int(round(self.dy * (1 - k))); h, w = self.a.shape[:2]
        s = self.a[max(0, -y):min(h, H - y)]; y = max(y, 0)
        al = s[..., 3:4] * k
        fr[y:y + s.shape[0], self.x:self.x + w] = fr[y:y + s.shape[0], self.x:self.x + w] * (1 - al) + s[..., :3] * al


def text_img(lines, f, lh, width):
    step = int(f.size * lh); pad = 24
    im = Image.new("RGBA", (width + 2 * pad, step * len(lines) + 2 * pad), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        d.text((pad, pad + i * step + step / 2), ln, font=f, fill=WHITE + (255,), anchor="lm")
    sh = Image.new("RGBA", im.size, GRAPHITE + (0,))
    sh.putalpha(im.getchannel("A").point(lambda v: v * 150 // 255))
    return Image.alpha_composite(sh.filter(ImageFilter.GaussianBlur(9)), im), pad


def navy_pill(text, f, padx=26, h=None):
    h = h or int(f.size * 1.75); w = int(f.getlength(text)) + 2 * padx
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), h // 2, fill=NAVY + (235,))
    d.text((w / 2, h / 2 + 1), text, font=f, fill=WHITE + (255,), anchor="mm"); return im


class Column:
    """left-aligned block, laid out top-down then moved so it ends at TEXT_BOTTOM"""
    def __init__(self): self.y = 0; self.items = []

    def _add(self, im, pad, t0, gap, dy=0):
        self.items.append(Sprite(im, MARGIN - pad, self.y - pad, t0, 0.5 if t0 > 0 else 0.01, dy))
        self.y += im.height - 2 * pad + gap

    def head(self, text, size=60, gap=14, t0=0.0):
        f = bold(size); im, pad = text_img(wrap(text, f, TEXT_W), f, 1.14, TEXT_W); self._add(im, pad, t0, gap, 22 if t0 else 0); return self

    def body(self, text, size=36, wght=400, gap=10, t0=0.0):
        f = mont(size, wght); im, pad = text_img(wrap(text, f, TEXT_W), f, 1.3, TEXT_W); self._add(im, pad, t0, gap, 16 if t0 else 0); return self

    def chip(self, text, size=34, gap=18, t0=0.0):
        im = navy_pill(text, bold(size)); self.items.append(Sprite(im, MARGIN, self.y, t0, 0.4 if t0 else 0.01, 14 if t0 else 0))
        self.y += im.height + gap; return self

    def done(self, at_top=False):
        """at the bottom (above the caption) or, where the result lies low in the frame, at the top"""
        d = TEXT_TOP if at_top else TEXT_BOTTOM - self.y
        for it in self.items: it.y += d
        self.at_top = at_top
        self.top = min(it.y for it in self.items) + 24; self.bottom = self.y + d; return self


def black_fade(top, length=420, a=0.8):
    ys = np.arange(H, dtype=np.float32); p = np.clip((ys - (top - length)) / (length + 80), 0, 1); p = p * p * (3 - 2 * p)
    return (a * p)[:, None, None]


def top_fade(bottom, length=380, a=0.8):
    return black_fade(H - bottom, length, a)[::-1]


class Video:
    """decoded 1080x1920 frames of a clip from `a`, slow push-in"""
    def __init__(self, path, a, dur, zoom=(1.0, 1.03), zc=(0.5, 0.4)):
        vf = f"scale={W}:{H}:flags=lanczos,fps={FPS}"
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(dur + 0.2), "-i", path, "-vf", vf,
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.n, self.zoom, self.zc, self.last = int(round(dur * FPS)), zoom, zc, None

    def frame(self, i):
        buf = self.p.stdout.read(W * H * 3)
        if len(buf) == W * H * 3: self.last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        z = self.zoom[0] + (self.zoom[1] - self.zoom[0]) * min(i / max(1, self.n), 1)
        M = cv2.getRotationMatrix2D((W * self.zc[0], H * self.zc[1]), 0, z)
        return cv2.warpAffine(self.last, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def dur_of(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True).stdout)


def pcm(path, a=0.0, dur=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(a)] + (["-t", str(dur)] if dur else []) + ["-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)


class PillSubs:
    """one word at a time, lower case, Montserrat Light on a navy pill, small pop-in"""
    def __init__(self, words, y, size=44):
        f = mont(size, 300); self.items = []
        for i, (a, b, t) in enumerate(words):
            t = t.lower().strip(".,!?«»…")
            e = min(words[i + 1][0] if i + 1 < len(words) else b + 0.5, b + 0.45)
            self.items.append((a, e, np.asarray(navy_pill(t, f, 22), np.float32) / 255.0, y))

    def draw(self, fr, t):
        for a, e, arr, y in self.items:
            if a - 0.03 <= t < e:
                k = min(1.0, (t - a + 0.03) / 0.12); sc = 0.85 + 0.15 * k
                a2 = cv2.resize(arr, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR) if sc < 0.999 else arr
                h, w = a2.shape[:2]; x, yy = (W - w) // 2, y - h // 2
                al = a2[..., 3:4] * k
                fr[yy:yy + h, x:x + w] = fr[yy:yy + h, x:x + w] * (1 - al) + a2[..., :3] * al
                return


# ---------------- scenes ----------------
# voice piece (picture + sound), optional other picture under the voice, silent extra footage / held frame after it
def scenes():
    out = []

    c = Column().chip("ДО", 38).head("ИМПЛАНТЫ НЕ ХОЧУ", 72).done()
    out.append(dict(piece="rA.mp4", col=c))

    c = Column().head("ПОДТЯЖКУ\nТОЖЕ НЕ ХОЧУ", 64, gap=26).head("ТОГДА ЧТО МОЖНО СДЕЛАТЬ?", 50, t0=3.0).done()
    out.append(dict(piece="rB.mp4", col=c, hold=0.7))

    c = Column().head("ВЫБРАЛИ ЛИПОФИЛИНГ", 66).body("собственная жировая ткань", 38).done()
    out.append(dict(piece="rC.mp4", col=c))

    c = Column().head("ДАЖЕ У СТРОЙНОЙ ПАЦИЕНТКИ\nМОЖНО НАЙТИ\nДОНОРСКИЕ ЗОНЫ", 54, gap=22)
    for i, t in enumerate(["внутренние бёдра", "галифе", "колени"]): c.chip(t, 34, gap=12, t0=0.5 + 0.45 * i)
    out.append(dict(piece="rD.mp4", col=c.done(), extra=("v1.mp4", 48.35, 1.8)))

    c = Column().head("ОБЪЁМ ДОБАВЛЯЕМ\nНЕ ВСЕГДА ОДИНАКОВО", 60).body("можно учитывать асимметрию", 38).done(at_top=True)
    out.append(dict(piece="rE.mp4", col=c, picture=("v5.mp4", 1.0)))          # first result on the table

    c = (Column().chip("ВАЖНО", 40).head("ЛИПОФИЛИНГ\nНЕ ЗАМЕНЯЕТ ПОДТЯЖКУ", 60)
         .body("но может помочь добавить наполненность, если пациентка не готова к более объёмной операции", 37).done())
    out.append(dict(piece="rF.mp4", col=c))

    c = Column().head("ИМЕННО ЭТУ ЗАДАЧУ\nИ РЕШАЛИ", 62, gap=22).chip("Подробнее в актуальном «Липофилинг»", 30, t0=3.6).done(at_top=True)
    out.append(dict(piece="rG.mp4", col=c, picture=("v5.mp4", 7.4), tail=2.0))
    return out


def reel():
    sc = scenes(); t = 0.0
    for s in sc:
        s["voice"] = dur_of(os.path.join(DRIVE, s["piece"]))
        s["after"] = s.get("hold", 0) + (s["extra"][2] if "extra" in s else 0) + s.get("tail", 0)
        s["t0"], s["dur"] = t, s["voice"] + s["after"]; t += s["dur"]
        words = json.load(open(os.path.join(DRIVE, s["piece"] + ".subs.json")))[0]["w"]
        col = s["col"]
        s["subs"] = PillSubs([(a, b, w) for a, b, w in words], y=col.bottom + 60 if col.at_top else col.top - 70)
        s["shade"] = top_fade(col.bottom) if col.at_top else black_fade(col.top)
    total = t; n = int(round(total * FPS)); print(f"reel {total:.1f}s", flush=True)
    tmp = os.path.join(OUT, "reel.v.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for s in sc:
        pic = s.get("picture")
        main = Video(os.path.join(DRIVE, pic[0]), pic[1], s["voice"] + s.get("tail", 0)) if pic else \
            Video(os.path.join(DRIVE, s["piece"]), 0.0, s["voice"])
        ex = Video(os.path.join(DRIVE, s["extra"][0]), s["extra"][1], s["extra"][2], zoom=(1.03, 1.05)) if "extra" in s else None
        nv = int(round((s["voice"] + s.get("tail", 0)) * FPS)); nh = int(round(s.get("hold", 0) * FPS))
        last = None
        for i in range(int(round(s["dur"] * FPS))):
            lt = i / FPS
            if i < nv: fr = main.frame(i); last = fr
            elif i < nv + nh: fr = last                                    # held frame
            else: fr = ex.frame(i - nv - nh)
            f = fr.astype(np.float32) / 255.0 * (1 - s["shade"])
            for it in s["col"].items: it.draw(f, lt)
            s["subs"].draw(f, lt)
            o = (np.clip(f, 0, 1) * 255).astype(np.uint8); enc.stdin.write(o.tobytes())
            if i == int(1.5 * FPS): Image.fromarray(o).save(os.path.join(OUT, f"kadr_{sc.index(s) + 1}.jpg"), quality=90)
        for v in (main, ex):
            if v: v.p.kill()
    enc.stdin.close(); enc.wait()
    # sound: the voice pieces, silence under the held / extra / tail footage
    aud = []
    for s in sc:
        a = pcm(os.path.join(DRIVE, s["piece"]))[:int(s["voice"] * SR)]
        aud += [a, np.zeros(int(s["dur"] * SR) - len(a), np.float32)]
    wav = os.path.join(OUT, "reel.f32"); np.concatenate(aud).tofile(wav)
    out = os.path.join(OUT, "reels.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", wav,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5", "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", "-shortest", "-movflags", "+faststart", out], check=True)
    os.remove(tmp); os.remove(wav); print("wrote", out, flush=True)


def cover():
    """the before body (marking video, face out of frame), Bebas title inside the central 3:4 zone (y 240..1680)"""
    fr = subprocess.run(["ffmpeg", "-v", "error", "-ss", "20.0", "-i", os.path.join(DRIVE, "v2.mp4"), "-frames:v", "1",
                         "-vf", f"scale={W}:{H}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    img = np.frombuffer(fr, np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0
    img = img * (1 - black_fade(1300, 520, 0.85))
    pil = Image.fromarray((img * 255).astype(np.uint8)); d = ImageDraw.Draw(pil)
    ft = ImageFont.truetype(BEBAS, 138); y = 1215
    for ln in ["НЕ ХОЧУ ИМПЛАНТЫ.", "НЕ ХОЧУ ПОДТЯЖКУ"]:
        d.text(((W - d.textlength(ln, font=ft)) / 2, y), ln, font=ft, fill=WHITE); y += 138
    fs = mont(44, 400); sub = "что можно сделать с грудью?"
    d.text(((W - d.textlength(sub, font=fs)) / 2, y + 26), sub, font=fs, fill=WHITE)
    out = os.path.join(OUT, "oblozhka.jpg"); pil.save(out, quality=95); print("wrote", out)


if "cover" in WHAT: cover()
if "reel" in WHAT: reel()
