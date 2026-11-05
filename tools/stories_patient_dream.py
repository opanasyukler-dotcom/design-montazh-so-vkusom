"""5 Instagram stories (1080x1920): a patient who dreamed of the operation almost all her life (abdominoplasty,
breast lift / reduction, lipofilling) -> «не могу на себя насмотреться».

The text lies right over the footage (white with a soft shadow, no dark haze, no black fields).  Faces blurred
(face_blur_track.py on the interview clips, a hand blur on the photos); the clinic's navy stickers on the nipples
(optical-flow tracking in the videos, sized by the distance between the two nipples).  The sound is the original,
continuous (whole phrases, no cutting inside them, the voice not changed); subtitles in thin Montserrat.
Story 2 is animated: the zones are outlined, the arrows drawn and the words come in one after another.

Sources (DRIVE, Drive folder 1XCqfpzWMBTqMFHwtDn9tVY4e0d-tEEXR) and CLIPS (blurred clips made with face_blur_track.py):
  v09 2024-03-18 23-01-39.mp4 (interview, «я уже очень давно хотела»), v06 …(28).mp4 («что не любим, всё показываем»),
  v05 …(2).mp4 («всё буду делать, кроме лица») -> f3.mp4 (4.4-7.9 s), …(12).mp4 before / marking with the pen,
  …(3).mp4 the finished marking, v15 2024-03-19 16-15-15 (1).mp4 -> f4.mp4 (5-18 s), v17 …16-15-15 (3).mp4 mirror ->
  f5.mp4 (14-19, 25.7-37.2 s), v14 2024-03-19 16-14-58 (3).mp4 the surgeon at the check-up; photos (23)/(24) before,
  IMG_2294 / IMG_2309 after; sticker_navy.png the clinic's sticker.
usage: python3 stories_patient_dream.py DRIVE_DIR CLIPS_DIR ASR_DIR OUT_DIR [story numbers...]
"""
import json, os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

W, H, FPS = 1080, 1920, 30
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
WHITE, GRAPHITE = (255, 255, 255), (20, 23, 31)
DRIVE, CLIPS, ASR, OUT = sys.argv[1:5]
ONLY = {int(a) for a in sys.argv[5:]}
os.makedirs(OUT, exist_ok=True)
MARGIN = 72
V09, V06, V05 = "2024-03-18 23-01-39.mp4", "2024-03-18 23-01-39 (28).mp4", "2024-03-18 23-01-39 (2).mp4"
V12, V03 = "2024-03-18 23-01-39 (12).mp4", "2024-03-18 23-01-39 (3).mp4"
V15, V17, V14 = "2024-03-19 16-15-15 (1).mp4", "2024-03-19 16-15-15 (3).mp4", "2024-03-19 16-14-58 (3).mp4"
P24, P23 = "2024-03-18 23-01-39 (24).jpeg", "2024-03-18 23-01-39 (23).jpeg"


def mont(size, wght=400):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([wght]); return f


def bold(size): return ImageFont.truetype(BOLD, size)


def wrap(text, f, width):
    out, line = [], ""
    for w in text.split(" "):
        t = (line + " " + w).strip()
        if f.getlength(t) <= width or not line: line = t
        else: out.append(line); line = w
    return out + [line]


class Block:
    """a static text block wrapped to a width"""
    def __init__(self, width=W - 2 * MARGIN): self.items, self.width = [], width

    def add(self, text, f, lh=1.2, gap=0):
        for l in wrap(text, f, self.width): self.items.append([l, f, int(f.size * lh)])
        self.items[-1][2] += gap; return self

    def head(self, text, size, gap=14): return self.add(text, bold(size), 1.14, gap)

    def body(self, text, size, wght=500, gap=10): return self.add(text, mont(size, wght), 1.3, gap)

    @property
    def h(self): return sum(s for _, _, s in self.items[:-1]) + int(self.items[-1][1].size * 1.15)

    def draw(self, lay, x, y):
        d = ImageDraw.Draw(lay)
        for t, f, s in self.items: d.text((x, y), t, font=f, fill=WHITE + (255,)); y += s
        return y


def with_shadow(lay, a=200, blur=9):
    """the soft shadow that keeps white text readable on any footage (no haze)"""
    al = lay.getchannel("A")
    sh = Image.new("RGBA", lay.size, GRAPHITE + (0,)); sh.putalpha(al.point(lambda v: v * a // 255).filter(ImageFilter.GaussianBlur(blur)))
    sh2 = Image.new("RGBA", lay.size, GRAPHITE + (0,)); sh2.putalpha(al.point(lambda v: v * 170 // 255).filter(ImageFilter.GaussianBlur(2.5)))
    return Image.alpha_composite(Image.alpha_composite(sh, sh2), lay)


def over(fr, rgba, x=0, y=0):
    h, w = rgba.shape[:2]; x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return fr
    a = rgba[y0 - y:y1 - y, x0 - x:x1 - x]; al = a[..., 3:4]
    fr[y0:y1, x0:x1] = fr[y0:y1, x0:x1] * (1 - al) + a[..., :3] * al
    return fr


# ---------------- sound: original phrases, continuous ----------------
def words(src, a, b, fixes=()):
    """ASR words of src between a and b (source seconds), with word-sequence fixes [(["или","по","филингу."], "и липофилинг.")]"""
    ws = [[w[0], w[1], w[2].strip()] for s in json.load(open(os.path.join(ASR, src + ".json"))) for w in s["w"]
          if w[0] >= a - 0.05 and w[1] <= b + 0.2]
    for seq, rep in fixes:
        n = len(seq)
        for i in range(len(ws) - n + 1):
            if [w[2] for w in ws[i:i + n]] == seq:
                ws[i:i + n] = [[ws[i][0], ws[i + n - 1][1], rep]]; break
    return ws


def cues(ws, src_a, t0, maxc=30):
    """subtitle cues on the story timeline (the clip src_a.. starts at t0)"""
    out, cur = [], []
    for i, w in enumerate(ws):
        cur.append(w); txt = " ".join(x[2] for x in cur); nxt = ws[i + 1] if i + 1 < len(ws) else None
        if nxt is None or len(txt) + 1 + len(nxt[2]) > maxc or w[2][-1] in ".?!…" or (w[2][-1] == "," and len(txt) > 16) \
                or nxt[0] - w[1] > 0.5:
            out.append(cur); cur = []
    res = []
    for i, c in enumerate(out):
        end = out[i + 1][0][0] if i + 1 < len(out) else c[-1][1] + 0.8
        end = min(end, c[-1][1] + 0.8)
        t = " ".join(x[2] for x in c); t = t[0].upper() + t[1:]
        res.append((t0 + c[0][0] - src_a - 0.05, t0 + end - src_a, t))
    return res


def sub_sprite(text, size=44):
    f = mont(size, 300); lines = wrap(text, f, W - 2 * MARGIN - 40)
    step = int(size * 1.3); lay = Image.new("RGBA", (W, step * len(lines) + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for i, l in enumerate(lines): d.text((W // 2, 20 + i * step), l, font=f, fill=WHITE + (255,), anchor="ma")
    return np.asarray(with_shadow(lay, 230, 6), np.float32) / 255


class Subs:
    def __init__(self, cl, y): self.c = [(a, b, sub_sprite(t)) for a, b, t in cl]; self.y = y

    def draw(self, fr, t):
        for a, b, sp in self.c:
            if a <= t < b: over(fr, sp, 0, self.y - sp.shape[0] // 2)


def mux(vid, out, dur, audio):
    """audio: [(src, a, b, story start)]: whole phrases of the original sound, 40 ms soft edges, no voice change"""
    ins, flt = [], []
    for k, (src, a, b, st) in enumerate(audio):
        ins += ["-i", os.path.join(DRIVE, src)]
        flt.append(f"[{k + 1}:a]atrim={a}:{b},asetpts=PTS-STARTPTS,afade=t=in:d=0.04,afade=t=out:st={b - a - 0.08:.2f}:d=0.08,"
                   f"adelay={int(st * 1000)}:all=1[a{k}]")
    flt.append("".join(f"[a{k}]" for k in range(len(audio))) + f"amix=inputs={len(audio)}:normalize=0,"
               f"loudnorm=I=-16:TP=-1.5,apad,atrim=0:{dur}[aout]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid] + ins + ["-filter_complex", ";".join(flt), "-map", "0:v", "-map", "[aout]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out], check=True)


# ---------------- backgrounds ----------------
class Video:
    """a clip covering 1080x1920 at 30 fps from `ss`; the last frame holds"""
    def __init__(self, path, ss=0.0, speed=1.0):
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(ss), "-i", path, "-an", "-vf",
                                   f"setpts=PTS/{speed},fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.last = None

    def frame8(self):
        b = self.p.stdout.read(W * H * 3)
        if len(b) == W * H * 3: self.last = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        return self.last

    def close(self): self.p.kill()


def photo(name):
    return ImageOps.exif_transpose(Image.open(os.path.join(DRIVE, name))).convert("RGB")


def blur_zone(im, box, r=40):
    m = Image.new("L", im.size, 0); ImageDraw.Draw(m).ellipse(box, fill=255); m = m.filter(ImageFilter.GaussianBlur(r // 2))
    return Image.composite(im.filter(ImageFilter.GaussianBlur(r)), im, m)


_ST = {}


def sticker(d):
    d = int(round(d))
    if d not in _ST:
        im = Image.open(os.path.join(DRIVE, "sticker_navy.png")).convert("RGBA").resize((d, d), Image.LANCZOS)
        _ST[d] = np.asarray(im, np.float32) / 255
    return _ST[d]


def stick(fr, pts, d):
    ds = np.broadcast_to(np.asarray(d, np.float32), (len(pts),))
    for (x, y), dd in zip(pts, ds):
        s = sticker(dd); over(fr, s, int(x - s.shape[1] / 2), int(y - s.shape[0] / 2))
    return fr


class Nipples:
    """two points followed by optical flow (median shift of the texture around each point); the sticker size follows
    the distance between them (the camera comes closer)"""
    def __init__(self, pts, d):
        self.p = np.array(pts, np.float32); self.d0 = d; self.dist0 = np.linalg.norm(self.p[0] - self.p[1]) if len(pts) > 1 else 1
        self.prev = None

    def update(self, rgb8):
        g = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
        if self.prev is not None:
            for k, (x, y) in enumerate(self.p):
                m = np.zeros_like(g); m[int(max(y - 110, 0)):int(y + 110), int(max(x - 110, 0)):int(x + 110)] = 255
                f = cv2.goodFeaturesToTrack(self.prev, 80, 0.01, 6, mask=m)
                if f is None: continue
                nf, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, f, None, winSize=(31, 31), maxLevel=3)
                ok = st[:, 0] == 1
                if ok.sum() >= 5: self.p[k] += np.median((nf - f)[ok, 0], axis=0)
        self.prev = g
        d = self.d0 * (np.linalg.norm(self.p[0] - self.p[1]) / self.dist0 if len(self.p) > 1 else 1)
        return self.p.copy(), d


def gray_frames(path, ss, dur, scale=0.5):
    w, h = int(W * scale), int(H * scale)
    b = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(ss), "-t", str(dur + 0.1), "-i", path, "-an", "-vf",
                        f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},scale={w}:{h}",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(b, np.uint8).reshape(-1, h, w)


def lk_step(g0, g1, pts, scale=0.5, r=60):
    out = pts.copy()
    for k, (x, y) in enumerate(pts * scale):
        m = np.zeros_like(g0); m[int(max(y - r, 0)):int(y + r), int(max(x - r, 0)):int(x + r)] = 255
        f = cv2.goodFeaturesToTrack(g0, 60, 0.01, 4, mask=m)
        if f is None: continue
        nf, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, f, None, winSize=(21, 21), maxLevel=3)
        ok = st[:, 0] == 1
        if ok.sum() >= 5: out[k] += np.median((nf - f)[ok, 0], axis=0) / scale
    return out


class KeyTrack:
    """nipples from hand-placed keyframes {t: [(x, y), ...]} (story px, t from the shot start) with sticker sizes
    {t: (d, ...)}; between two keyframes optical flow carries the points and its drift is spread out linearly so the
    track lands exactly on the next keyframe"""
    def __init__(self, path, ss, dur, keys, sizes):
        g = gray_frames(path, ss, dur); n = min(len(g), int(round(dur * FPS)) + 1)
        kt = sorted(keys); self.pos = np.zeros((n, len(keys[kt[0]]), 2), np.float32)
        for a, b in zip(kt, kt[1:] + [None]):
            i0 = int(round(a * FPS)); i1 = n - 1 if b is None else min(int(round(b * FPS)), n - 1)
            p = np.array(keys[a], np.float32); tr = [p.copy()]
            for i in range(i0, i1): p = lk_step(g[i], g[i + 1], p); tr.append(p.copy())
            tr = np.array(tr)
            if b is not None:
                err = np.array(keys[b], np.float32) - tr[-1]
                tr += err[None] * np.linspace(0, 1, len(tr))[:, None, None]
            self.pos[i0:i1 + 1] = tr
        st = sorted(sizes); self.sz = np.array([[np.interp(i / FPS, st, [sizes[t][k] for t in st]) for k in range(len(keys[kt[0]]))]
                                                for i in range(n)])
        self.i = 0

    def update(self, rgb8):
        i = min(self.i, len(self.pos) - 1); self.i += 1
        return self.pos[i], self.sz[i]


def render(name, dur, frame_at, overlay=None, subs=None, audio=(), extra=None):
    """frame_at(i, t) -> float RGB; overlay: static RGBA layer; extra(fr, t) -> animated layers"""
    out = os.path.join(OUT, name); tmp = out + ".v.mp4"
    ov = None if overlay is None else np.asarray(overlay, np.float32) / 255
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for i in range(int(round(dur * FPS))):
        t = i / FPS; fr = frame_at(i, t)
        if extra: fr = extra(fr, t)
        if ov is not None: over(fr, ov)
        if subs: subs.draw(fr, t)
        o = (np.clip(fr, 0, 1) * 255).astype(np.uint8); enc.stdin.write(o.tobytes())
    enc.stdin.close(); enc.wait()
    mux(tmp, out, dur, audio); os.remove(tmp); print("wrote", out, flush=True)


class Shots:
    """a sequence of shots [(Video, seconds, Nipples or None)] played back to back"""
    def __init__(self, shots): self.s = shots; self.t0 = np.cumsum([0] + [d for _, d, _ in shots])

    def __call__(self, i, t):
        k = min(int(np.searchsorted(self.t0, t, side="right") - 1), len(self.s) - 1)
        v, _, nip = self.s[k]; f8 = v.frame8(); fr = f8.astype(np.float32) / 255
        if nip: p, d = nip.update(f8); stick(fr, p, d)
        return fr


def text_layer(*blocks):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for b, x, y in blocks: b.draw(lay, x, y)
    return with_shadow(lay)


# ---------------- stories ----------------
def s1():
    """«я давно хочу что-то изменить»: the before video (standing, stickers on), her voice from the interview over it"""
    K = {0.0: [(370, 615), (785, 640)], 0.5: [(443, 696), (961, 724)], 1.0: [(378, 799), (1026, 800)],
         1.5: [(140, 866), (929, 930)], 2.0: [(32, 849), (854, 849)], 2.5: [(97, 903), (930, 773)],
         3.0: [(259, 931), (973, 870)], 3.5: [(249, 968), (940, 816)], 4.0: [(216, 968), (886, 751)],
         4.5: [(195, 968), (886, 730)], 5.0: [(205, 968), (897, 730)], 5.5: [(173, 1032), (865, 968)],
         6.0: [(130, 1022), (886, 1011)], 6.5: [(130, 1011), (876, 1011)], 7.0: [(151, 957), (843, 989)],
         7.5: [(173, 903), (876, 935)]}
    D = {0.0: (64, 68), 0.5: (82, 90), 1.0: (96, 102), 1.5: (102, 106), 3.0: (106, 110), 4.0: (110, 110), 6.0: (118, 118), 7.5: (116, 116)}
    path = os.path.join(DRIVE, V12)
    shots = Shots([(Video(path, ss=4.0), 9.0, KeyTrack(path, 4.0, 7.6, K, D))])
    b = (Block().head("«Я ДАВНО ХОЧУ ЧТО-ТО ИЗМЕНИТЬ. НО ВСЁ ОТКЛАДЫВАЮ…»", 60, gap=18)
         .body("У этой пациентки мысль об операции была с ней много лет.", 34, gap=0))
    ws = words(V09, 3.5, 9.3)
    render("01_davno_hochu.mp4", 7.6, shots, text_layer((b, MARGIN, 110)), Subs(cues(ws, 3.5, 0.4), 1010),
           audio=[(V09, 3.5, 9.3, 0.4)])
    # the poll sticker: low, over the legs (y 1350-1700)


def ease(x): x = min(max(x, 0.0), 1.0); return 1 - (1 - x) ** 3


def s2():
    """the zones: the before photo on the whole screen (stickers on), a slow push in; the zones are outlined one after
    another, the arrows drawn and the words come in; her live words from v06 over it"""
    src = blur_zone(photo(P24), (560, -160, 1060, 140), 44)
    s = H / src.height; ph = src.resize((int(src.width * s), H), Image.LANCZOS); cx0 = (ph.width - W) // 2
    ph = ph.crop((cx0, 0, cx0 + W, H)); base = np.asarray(ph, np.float32) / 255
    M = lambda x, y: (x * s - cx0, y * s)                                  # source px -> story px
    stick(base, [M(620, 728), M(990, 739)], 96 * s)
    P = lambda px, py: M(px * 2.844, py * 2.844)                            # preview (540x720) -> story
    ZONES = [  # label, label xy, anchor, arrow from, arrow to, ellipse (x0, y0, x1, y1) in preview coords, start time
        ("Грудь", (W - 56, P(0, 196)[1]), "rs", (W - 150, P(0, 204)[1]), P(380, 238), (178, 203, 382, 292), 0.7),
        ("Живот", (56, P(0, 398)[1]), "ls", (150, P(0, 406)[1]), P(222, 352), (203, 298, 357, 388), 2.3),
        ("Бока", (W - 56, P(0, 372)[1]), "rs", (W - 140, P(0, 380)[1]), P(374, 332), None, 3.9),
    ]
    b1 = Block().body("После рождения ребёнка её стали беспокоить сразу несколько зон.", 38, wght=600)
    b2 = (Block().body("И уже не хотелось «исправить что-то одно».", 34, gap=10)
          .body("Хотелось снова посмотреть на своё тело и почувствовать:", 34, gap=16)
          .head("«ВОТ ТЕПЕРЬ МНЕ НРАВИТСЯ»", 56, gap=0))
    static = text_layer((b1, MARGIN, 96), (b2, MARGIN, H - 110 - b2.h))
    dur = 10.0; C = np.array([540, 900], np.float32)

    def zoom(t): return 1 + 0.06 * ease(t / dur)

    def frame(i, t):
        z = zoom(t)
        Mx = np.float32([[z, 0, C[0] * (1 - z)], [0, z, C[1] * (1 - z)]])
        return cv2.warpAffine(base, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    SC = 2; fl = mont(36, 600)

    def zones(fr, t):
        z = zoom(t); tr = lambda p: (C + (np.array(p, np.float32) - C) * z) * SC
        big = Image.new("RGBA", (W * SC, H * SC), (0, 0, 0, 0)); d = ImageDraw.Draw(big)
        glow = np.zeros((H, W, 1), np.float32)
        for lab, lxy, anc, a0, a1, box, ts in ZONES:
            k_out, k_arr, k_lab = ease((t - ts) / 0.8), ease((t - ts - 0.5) / 0.45), ease((t - ts - 0.8) / 0.4)
            if box and k_out > 0:                                          # the outline draws itself round
                (x0, y0), (x1, y1) = P(box[0], box[1]), P(box[2], box[3]); cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
                n = 40; m = int(n * k_out)
                for j in range(m):
                    ang = np.linspace(2 * np.pi * j / n, 2 * np.pi * (j + 0.55) / n, 8) - np.pi / 2
                    d.line([tuple(tr((cx + rx * np.cos(q), cy + ry * np.sin(q)))) for q in ang], fill=WHITE + (240,), width=3 * SC)
                # a soft light inside the zone: rises when it is outlined, then settles
                pulse = 0.16 * np.exp(-((t - ts - 0.9) / 0.5) ** 2) + 0.05 * k_out
                yy, xx = np.ogrid[:H, :W]; ccx, ccy = tr((cx, cy)) / SC
                g = np.exp(-(((xx - ccx) / (rx * z)) ** 2 + ((yy - ccy) / (ry * z)) ** 2) * 1.6)[..., None]
                glow = np.maximum(glow, g * pulse)
            if k_arr > 0:                                                  # the arrow grows from the word to the zone
                p0, p1 = tr(a0), tr(a1); p = p0 + (p1 - p0) * k_arr
                d.line([tuple(p0), tuple(p)], fill=WHITE + (255,), width=3 * SC)
                if k_arr >= 1:
                    ang = np.arctan2(*(p1 - p0)[::-1]); hd = 18 * SC
                    for sgn in (-0.45, 0.45):
                        d.line([tuple(p1), (p1[0] - hd * np.cos(ang + sgn), p1[1] - hd * np.sin(ang + sgn))], fill=WHITE + (255,), width=3 * SC)
        fr = fr * (1 - glow) + glow
        lay = big.resize((W, H), Image.LANCZOS); dl = ImageDraw.Draw(lay)
        for lab, lxy, anc, a0, a1, box, ts in ZONES:
            k = ease((t - ts - 0.8) / 0.4)
            if k > 0:                                                      # the word slides in a little and fades up
                x, y = lxy; dx = (1 - k) * (24 if anc == "rs" else -24)
                tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tmp).text((x + dx, y), lab, font=fl, fill=WHITE + (255,), anchor=anc)
                tmp.putalpha(tmp.getchannel("A").point(lambda v: int(v * k))); lay = Image.alpha_composite(lay, tmp)
        return over(fr, np.asarray(with_shadow(lay, 170, 6), np.float32) / 255)

    ws = words(V06, 8.5, 14.4, fixes=[(["любовь,"], "любим,")])
    render("02_zony.mp4", dur, frame, static, Subs(cues(ws, 8.5, 0.5), 470), audio=[(V06, 8.5, 14.4, 0.5)], extra=zones)


def s3():
    """«всё буду делать, кроме лица»: her interview (face blurred), then the surgeon marking with the pen, then the
    finished marking (stickers on); her whole sentence under it"""
    shots = Shots([(Video(os.path.join(CLIPS, "f3.mp4"), ss=0.2), 3.1, None),
                   (Video(os.path.join(DRIVE, V12), ss=19.0), 3.2, KeyTrack(os.path.join(DRIVE, V12), 19.0, 3.2,
                    {0.0: [(329, 385), (696, 403)], 0.5: [(281, 312), (783, 333)], 1.0: [(249, 262), (1005, 219)],
                     1.5: [(130, 262), (984, 208)], 2.0: [(97, 305), (930, 305)], 2.5: [(108, 305), (930, 316)],
                     3.0: [(216, 295), (995, 284)]},
                    {0.0: (60, 70), 0.5: (80, 100), 1.0: (112, 172), 2.0: (112, 166), 3.0: (110, 158)})),
                   (Video(os.path.join(DRIVE, V03)), 5.8, Nipples([(252, 862), (852, 892)], 70))])
    b = (Block().head("«ВСЁ БУДУ ДЕЛАТЬ, КРОМЕ ЛИЦА. В ЭТОТ РАЗ»", 68, gap=20)
         .body("После осмотра составили план комплексной коррекции.", 34, gap=0))
    ws = words(V05, 4.6, 16.3, fixes=[(["Все"], "Всё"), (["или", "по", "филингу."], "и липофилинг.")])
    render("03_krome_lica.mp4", 12.1, shots, text_layer((b, MARGIN, H - 110 - b.h)), Subs(cues(ws, 4.6, 0.0), 200),
           audio=[(V05, 4.6, 16.3, 0.0)])


def s4():
    """a month later: the restrictions were there, but «очень рада… не жалею… это того стоит»"""
    shots = Shots([(Video(os.path.join(CLIPS, "f4.mp4"), ss=3.2), 10.0, None)])
    b = (Block().head("ПРОШЁЛ МЕСЯЦ", 64, gap=18)
         .body("Да, были бытовые ограничения.", 36, wght=600, gap=4)
         .body("Было неудобно наклоняться, двигаться, завязывать шнурки.", 34, gap=12)
         .body("Но сама пациентка говорит:", 34, gap=0))
    b2 = (Block().head("«ЭТО ТОГО СТОИТ»", 64, gap=14)
          .body("восстановление у каждой пациентки проходит индивидуально", 28, wght=400, gap=0))
    top = 1060
    cl = cues(words(V15, 8.4, 14.35), 8.4, 0.2) + cues(words(V15, 34.95, 37.8), 34.95, 6.4)
    render("04_ne_zhaleyu.mp4", 10.0, shots, text_layer((b, MARGIN, top), (b2, MARGIN, H - 100 - b2.h)),
           Subs(cl, top + b.h + 90), audio=[(V15, 8.4, 14.35, 0.2), (V15, 34.95, 37.8, 6.4)])


# ---------------- before / after collages: after on top, before below, neck to fingertips ----------------
def half(name, cx, top, bot, stickers=(), d=0, blur=None, walls=None):
    """one half (1080x960): the band top..bot (source px) centred on the body axis cx, black around the photo"""
    im = photo(name)
    if blur: im = blur_zone(im, blur, 40)
    a = np.asarray(im, np.float32) / 255
    if walls:                                                              # the white wall at the edge of the backdrop -> black
        xs = np.arange(a.shape[1])[None, :, None]; l, r = walls
        a = a * np.clip((xs - l) / 40, 0, 1) * np.clip((r - xs) / 40, 0, 1)
    if stickers: stick_src(a, stickers, d)
    h = (bot - top); w = h * W / 960
    x0, y0 = cx - w / 2, top
    Mx = np.float32([[W / w, 0, -x0 * W / w], [0, 960 / h, -y0 * 960 / h]])
    return cv2.warpAffine(a, Mx, (W, 960), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def stick_src(a, pts, d):
    for x, y in pts:
        st = sticker(d)
        h, w = st.shape[:2]; x0, y0 = int(x - w / 2), int(y - h / 2); al = st[..., 3:4]
        a[y0:y0 + h, x0:x0 + w] = a[y0:y0 + h, x0:x0 + w] * (1 - al) + st[..., :3] * al


def collages():
    """front, side and the front closer (neck to the hips); the after photos carry the clinic's stickers already"""
    fb = dict(stickers=[(620, 728), (990, 739)], d=96, blur=(560, -160, 1060, 140), walls=(95, 1450))
    sb = dict(stickers=[(1001, 722)], d=88, blur=(600, -200, 1150, 190), walls=(40, 1500))
    out = []
    for (an, acx, at, ab), (bn, bcx, bt, bb, kw) in [
            (("IMG_2294.JPG", 760, 150, 1790), (P24, 796, 130, 1460, fb)),
            (("IMG_2309.JPG", 790, 240, 1920), (P23, 725, 190, 1625, sb)),
            (("IMG_2294.JPG", 760, 150, 1560), (P24, 796, 130, 1310, fb))]:
        c = np.zeros((H, W, 3), np.float32)
        c[:960] = half(an, acx, at, ab); c[960:] = half(bn, bcx, bt, bb, **kw)
        c[958:962] = 1.0                                                   # a thin white seam
        out.append(c)
    return out


def s5():
    """«не могу на себя насмотреться»: her story at the mirror, the surgeon's comment at the check-up and her words,
    then three before / after collages (after on top, before below) with the stickers"""
    shots = Shots([(Video(os.path.join(CLIPS, "f5.mp4")), 16.5, None), (Video(os.path.join(DRIVE, V14), ss=20.6), 8.2, None)])
    cols = collages(); T1 = 16.5 + 8.2; dur = T1 + 3 * 3.0

    def frame(i, t):
        if t < T1: return shots(i, t)
        return cols[min(int((t - T1) / 3.0), 2)].copy()

    hb = Block().head("«НЕ МОГУ НА СЕБЯ НАСМОТРЕТЬСЯ»", 64, gap=0)
    b = (Block().body("Если вы тоже годами думаете об операции, но не понимаете, что именно нужно делать и можно ли "
                      "скорректировать несколько зон за один этап, начните с консультации.", 32, gap=14)
         .body("Для записи напишите «КОНСУЛЬТАЦИЯ» в Direct.", 32, wght=700, gap=0))
    top = H - 90 - b.h
    cl = cues(words(V17, 21.4, 37.95), 21.4, 0.0) + cues(words(V14, 20.6, 28.65), 20.6, 16.5)
    render("05_ne_mogu_nasmotretsya.mp4", dur, frame, text_layer((hb, MARGIN, 96), (b, MARGIN, top)), Subs(cl, top - 100),
           audio=[(V17, 21.4, 37.95, 0.0), (V14, 20.6, 28.7, 16.5)])
    for k, c in enumerate(cols):                                           # the collages also as photos
        Image.fromarray((c * 255).astype(np.uint8)).save(os.path.join(OUT, f"kollazh_{k + 1}.jpg"), quality=94)


for k, f in [(1, s1), (2, s2), (3, s3), (4, s4), (5, s5)]:
    if not ONLY or k in ONLY: f()
