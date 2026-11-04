# Reels: AliExpress unboxing (kids clothes). Vertical 4K talking head (2160x3840) -> 1080x1920,
# word-level edit (no repeats/stutters/pauses), product cards from AliExpress screenshots, brand palette.
# usage: python3 reel_ali_unboxing.py <src_dir> <work_dir> [info|stills t1,t2|render]
#   work_dir needs clean.wav (denoised voice 48k mono)
import sys, os, subprocess, numpy as np, cv2, soundfile as sf
from PIL import Image, ImageDraw, ImageFont
SRC, WK = sys.argv[1], sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "render"
W, H, FPS, SR = 1080, 1920, 25, 48000
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Montserrat-VariableFont_wght.ttf")
INK, MOCHA, ROSE, TAUPE, BLUSH = (0x2F, 0x06, 0x00), (0x82, 0x5D, 0x4D), (0xC5, 0xA2, 0x9C), (0xC5, 0xB0, 0xAD), (0xF7, 0xDF, 0xDD)

# ---- edit decision list: (src_in, src_out, tag); tag "K" = covered by the bodysuit photo ----
EDL = [(2.30, 4.40, ""), (8.36, 13.55, ""),                                        # intro
       (25.02, 28.40, ""),                                                         # 2 weeks delivery
       (37.70, 38.34, ""), (38.84, 42.80, ""), (44.40, 45.72, ""),                 # cheap & cute, she grows fast
       (55.46, 56.30, ""), (57.98, 59.44, ""), (59.96, 60.82, ""), (63.28, 64.55, ""),   # fleece suit
       (80.16, 80.76, ""), (81.86, 83.90, ""), (86.16, 87.44, ""), (90.84, 91.90, ""),   # boots
       (114.10, 115.74, ""), (118.90, 120.46, ""), (120.44, 122.86, ""),  # jacket
       (135.36, 136.24, ""), (136.44, 137.54, ""), (137.84, 139.62, ""), (139.68, 141.98, ""), (145.88, 148.80, ""),  # knit set
       (154.00, 155.28, ""), (160.46, 162.00, ""), (162.60, 164.70, ""), (165.82, 167.26, ""), (167.34, 168.48, ""),  # overall
       (183.06, 184.70, ""), (188.34, 191.60, ""),                                  # fleece NY
       (173.04, 177.60, "K"),                                                      # bodysuits 900
       (224.78, 225.95, "")]                                                       # outro

# ---- audio-driven tightening: trim silence at edges, squeeze inner pauses, drop lone clicks ----
voice, sr = sf.read(os.path.join(WK, "clean.wav")); assert sr == SR
HOP = 240
env = 20 * np.log10(np.sqrt(np.convolve(voice ** 2, np.ones(HOP * 2) / (HOP * 2), "same"))[::HOP] + 1e-9)
TH = -38.0
def pieces(a, b):
    i0, i1 = int(a * SR / HOP), int(b * SR / HOP); act = env[i0:i1] > TH
    runs = []; k = 0
    while k < len(act):
        if act[k]:
            j = k
            while j < len(act) and act[j]: j += 1
            runs.append([k, j]); k = j
        else: k += 1
    runs = [r for r in runs if (r[1] - r[0]) * HOP / SR > 0.12 or len(runs) == 1]    # lone clicks/breaths out
    merged = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]) * HOP / SR < 0.26: merged[-1][1] = r[1]
        else: merged.append(r)
    pad = 0.07
    return [(max(a, (i0 + r[0]) * HOP / SR - pad), min(b, (i0 + r[1]) * HOP / SR + pad)) for r in merged]
CLIPS = []   # (src_in, src_out, tag, out_start)
t = 0.0
for a, b, tag in EDL:
    for pa, pb in pieces(a, b):
        CLIPS.append((pa, pb, tag, t)); t += pb - pa
DUR = round(t + 0.6, 2)

# ---- text ----
_fc = {}
def font(size, wght):
    if (size, wght) not in _fc:
        f = ImageFont.truetype(FONT, size); f.set_variation_by_axes([wght]); _fc[(size, wght)] = f
    return _fc[(size, wght)]
def text_sprite(text, size, wght, color, track=0.0):
    f = font(size, wght); wd = int(sum(f.getlength(c) + track * size for c in text)) + 8
    im = Image.new("RGBA", (wd, int(size * 1.4)), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 2
    for c in text: d.text((x, 0), c, font=f, fill=color + (255,)); x += f.getlength(c) + track * size
    return np.asarray(im).astype(np.float32) / 255
def paste(fr, s, x, y, a=1.0):
    h, w = s.shape[:2]; X0, Y0, X1, Y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if X1 <= X0 or Y1 <= Y0 or a <= 0: return
    ss = s[Y0 - y:Y1 - y, X0 - x:X1 - x]; al = ss[..., 3:4] * a
    fr[Y0:Y1, X0:X1] = fr[Y0:Y1, X0:X1] * (1 - al) + ss[..., :3] * al
def rrect_mask(w, h, r):
    m = np.zeros((h, w), np.uint8); cv2.rectangle(m, (r, 0), (w - r, h), 255, -1); cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in [(r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)]: cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m.astype(np.float32) / 255
def ease(x): x = np.clip(x, 0, 1); return 1 - (1 - x) ** 3
def ease_io(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

def shot(name):  # product photo out of an AliExpress screenshot (1320x2868, or name:y0:y1:x0:x1 for other sizes)
    name, *box = name.split(":")
    im = cv2.cvtColor(cv2.imread(os.path.join(SRC, name)), cv2.COLOR_BGR2RGB)
    y0, y1, x0, x1 = map(int, box) if box else (420, 1460, 40, 1180)
    return im[y0:y1, x0:x1]

# product card: photo + CAPS name + thin note + price chip
class Card:
    def __init__(self, src_t, name, note, price, img):
        self.src_t, self.name, self.note, self.price = src_t, name, note, price
        self.photo = None if img is None else shot(img)
    def build(self):
        pad = 26
        t1 = text_sprite(self.name, 34, 550, INK, 0.12); t2 = text_sprite(self.note, 28, 400, INK)
        cw = max(470, t1.shape[1] + 2 * pad, t2.shape[1] + 2 * pad); ph = 0 if self.photo is None else 340
        pr = text_sprite(self.price, 36, 600, BLUSH, 0.02) if self.price else None
        ch = ph + pad + t1.shape[0] + t2.shape[0] + (64 if pr is not None else 0) + pad - 6
        im = np.zeros((ch, cw, 4), np.float32); im[..., :3] = np.array(BLUSH) / 255; im[..., 3] = rrect_mask(cw, ch, 28)
        if ph:
            p = cv2.resize(self.photo, (cw, int(self.photo.shape[0] * cw / self.photo.shape[1])), interpolation=cv2.INTER_AREA)
            y0 = (p.shape[0] - ph) // 2; im[:ph, :, :3] = p[y0:y0 + ph] / 255
        y = ph + pad - 4; paste_rgba(im, t1, pad, y); y += t1.shape[0] - 2; paste_rgba(im, t2, pad, y); y += t2.shape[0] + 6
        if pr is not None:
            chip = np.zeros((54, pr.shape[1] + 34, 4), np.float32); chip[..., :3] = np.array(MOCHA) / 255; chip[..., 3] = rrect_mask(chip.shape[1], 54, 27)
            paste_rgba(chip, pr, 17, 3); paste_rgba(im, chip, pad, y)
        sh = np.zeros((ch + 60, cw + 60), np.float32); sh[30:30 + ch, 30:30 + cw] = im[..., 3]; sh = cv2.GaussianBlur(sh, (0, 0), 14) * 0.35
        out = np.zeros((ch + 60, cw + 60, 4), np.float32); out[..., 3] = sh; out[..., :3] = np.array(INK) / 255
        paste_rgba(out, im, 30, 22); self.spr = out
def paste_rgba(dst, s, x, y):
    h, w = s.shape[:2]; h = min(h, dst.shape[0] - y); w = min(w, dst.shape[1] - x)
    if h <= 0 or w <= 0: return
    a = s[:h, :w, 3:4]; dst[y:y + h, x:x + w, :3] = dst[y:y + h, x:x + w, :3] * (1 - a) + s[:h, :w, :3] * a
    dst[y:y + h, x:x + w, 3:4] = a + dst[y:y + h, x:x + w, 3:4] * (1 - a)

CARDS = [Card(55.6, "СПОРТИВНЫЙ КОСТЮМ", "с ромашкой, на флисе", "1 159 ₽", "suit_daisy.png:480:1500:0:1175"),
         Card(82.0, "БОТИНОЧКИ", "тёплые, на липучке", "359 ₽", "IMG_9423.PNG"),
         Card(114.2, "КУРТКА", "с капюшоном", "1 096 ₽", "IMG_9421.PNG"),
         Card(135.4, "ВЯЗАНЫЙ КОСТЮМ", "свитер + штанишки, хлопок", "1 069 ₽", "IMG_9424.PNG"),
         Card(154.1, "КОМБИНЕЗОН", "вязаный, с ёжиком", "718 ₽", "IMG_9422.PNG"),
         Card(183.1, "ФЛИСКА NEW YORK", "тёплая толстовка", "461 ₽", "IMG_9420.PNG")]
def src2out(st):
    for a, b, tag, o in CLIPS:
        if a - 0.01 <= st <= b + 0.01: return o + max(0, st - a)
    best = min(CLIPS, key=lambda c: abs(c[0] - st)); return best[3]
def item_span(card):   # card visible from its clip until the next item starts
    t0 = src2out(card.src_t); nxt = [src2out(c.src_t) for c in CARDS if src2out(c.src_t) > t0 + 0.5]
    k_start = min([o for a, b, tag, o in CLIPS if tag == "K"] + [DUR])
    t1 = min(nxt + [k_start, t0 + 4.2]); return t0 + 0.15, t1 - 0.1

# ---- frames ----
BIG = (2160, 3840)   # displayed size (ffmpeg applies the rotation)
_cache = {}
def src_frames(a, b):
    k = (a, b)
    if k not in _cache:
        _cache.clear()
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, a - 0.04):.3f}", "-t", f"{b - a + 0.2:.3f}", "-i", os.path.join(SRC, "C0983.MP4"),
                              "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        n = len(raw) // (BIG[0] * BIG[1] * 3); _cache[k] = (max(0, a - 0.04), np.frombuffer(raw[:n * BIG[0] * BIG[1] * 3], np.uint8).reshape(n, BIG[1], BIG[0], 3))
    return _cache[k]
def vertical(img, z):
    ch = BIG[1] / z; cw = BIG[0] / z
    x0 = (BIG[0] - cw) / 2; y0 = (BIG[1] - ch) * 0.42
    M = np.float32([[W / cw, 0, -x0 * W / cw], [0, H / ch, -y0 * H / ch]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA if z < 1.3 else cv2.INTER_LINEAR)

BODY = None
def body_photo(u):
    global BODY
    if BODY is None:
        im = cv2.cvtColor(cv2.imread(os.path.join(SRC, "IMG_9419.PNG")), cv2.COLOR_BGR2RGB)[360:1490, 120:1200]
        BODY = im
    ph, pw = BODY.shape[:2]; z = 1.0 + 0.08 * u; ch = ph / z; cw = ch * 9 / 16
    if cw > pw: cw = pw; ch = cw * 16 / 9
    x0 = (pw - cw) / 2; y0 = (ph - ch) / 2
    M = np.float32([[W / cw, 0, -x0 * W / cw], [0, H / ch, -y0 * H / ch]])
    return cv2.warpAffine(BODY, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def clip_at(t):
    for i, (a, b, tag, o) in enumerate(CLIPS):
        if o <= t < o + (b - a): return i
    return len(CLIPS) - 1
# punch-in alternates on cuts between EDL blocks (hides jump cuts)
ZOOM = {}
zi = 0; prev_end = None
for i, (a, b, tag, o) in enumerate(CLIPS):
    if prev_end is not None and a - prev_end > 0.35: zi += 1
    ZOOM[i] = [1.0, 1.14][zi % 2]; prev_end = b

def frame(t):
    i = clip_at(t); a, b, tag, o = CLIPS[i]; st = a + (t - o)
    if tag == "K":
        ks = [c for c in CLIPS if c[2] == "K"]; k0 = ks[0][3]; k1 = ks[-1][3] + ks[-1][1] - ks[-1][0]
        f = body_photo((t - k0) / (k1 - k0)).astype(np.float32) / 255
    else:
        s0, fr = src_frames(a, b); k = int(np.clip(round((st - s0) * FPS), 0, len(fr) - 1))
        f = vertical(fr[k], ZOOM[i]).astype(np.float32) / 255
    return f

def grade(f):
    lum = f.mean(2, keepdims=True)
    f = f * 1.02 + (np.array(BLUSH, np.float32) / 255 - 0.93) * 0.12 * lum ** 2
    return f

TITLE = None
# Instagram Reels safe area (1080x1920): top 250 px = header, bottom 420 px = caption/buttons, right 140 px low = like/comment column
SAFE_TOP, SAFE_BOTTOM = 250, 1500

# ---- word subtitles: words from the source transcript (tr.json), mapped through the edit ----
import json
FIX = {"флеску": "флиску", "алиэкспресс": "AliExpress", "все": "всё", "еще": "ещё", "пришел": "пришёл", "прошел": "прошёл", "желтый": "жёлтый"}
def build_subs():
    words = []
    for seg in json.load(open(os.path.join(WK, "tr.json"))):
        for j, (ws, we, w) in enumerate(seg["w"]):
            w = w.strip()
            if j == len(seg["w"]) - 1 and w[-1] not in ".,!?": w += ","
            if w.startswith("-") and words and abs(words[-1][3] - ws) < 0.3:   # "хоп -хоп", "что -то"
                words[-1][2] += w; words[-1][1] = we; words[-1][3] = we; continue
            words.append([ws, we, w, we])
    out = []
    for ws, we, w, _ in words:
        core = w.strip(".,!?«»").lower(); core = FIX.get(core, core)
        if we - ws < 0.03:                      # zero-length tokens are alignment noise, except a word we hear in the edit
            if core != "через": continue
            we = ws + 0.2
        best = max(CLIPS, key=lambda c: min(c[1], we) - max(c[0], ws))
        a, b, tag, o = best
        if min(b, we) - max(a, ws) < min((we - ws) * 0.5, 0.2): continue
        out.append([o + max(0, ws - a), o + min(b, we) - a, core, w[-1] in ".,!?", ws])
    out.sort(key=lambda x: x[0])
    lines = []; cur = []
    for wd in out:
        if cur and (len(" ".join(x[2] for x in cur + [wd])) > 20 or wd[0] - cur[-1][1] > 0.5 or abs(wd[4] - cur[-1][4]) > 2.5): lines.append(cur); cur = []
        cur.append(wd)
        if wd[3] and len(" ".join(x[2] for x in cur)) > 8: lines.append(cur); cur = []
    if cur: lines.append(cur)
    for i, ln in enumerate(lines):
        nxt = lines[i + 1][0][0] if i + 1 < len(lines) else DUR
        ln[-1][1] = min(nxt, ln[-1][1] + 0.35)
    return lines
SUBS = None; _sub_spr = {}
SUB_SIZE, SUB_PAD = 50, 26
def sub_line_sprites(ln):
    k = id(ln)
    if k not in _sub_spr:
        f = font(SUB_SIZE, 500); sp = f.getlength(" ")
        xs = []; x = 0
        for w in ln: xs.append(x); x += f.getlength(w[2]) + sp
        tw = int(x - sp); ww = [text_sprite(w[2], SUB_SIZE, 500, INK) for w in ln]
        _sub_spr[k] = (tw, xs, ww)
    return _sub_spr[k]
def draw_subs(f, t):
    global SUBS
    if SUBS is None: SUBS = build_subs()
    for ln in SUBS:
        t0, t1 = ln[0][0], ln[-1][1]
        if not (t0 - 0.05 <= t <= t1): continue
        tw, xs, ww = sub_line_sprites(ln)
        a = ease((t - t0 + 0.05) / 0.15) * (1 - ease_io((t - t1 + 0.12) / 0.12))
        bw, bh = tw + 2 * SUB_PAD, int(SUB_SIZE * 1.55); x0 = (W - bw) // 2; y0 = SAFE_BOTTOM - 40 - bh
        m = rrect_mask(bw, bh, bh // 2)[..., None] * 0.9 * a
        f[y0:y0 + bh, x0:x0 + bw] = f[y0:y0 + bh, x0:x0 + bw] * (1 - m) + np.array(BLUSH, np.float32) / 255 * m
        for w, wx, spr in zip(ln, xs, ww):        # current word: rose underline pill
            if w[0] <= t < max(w[1], w[0] + 0.12):
                hw = spr.shape[1] - 4; hm = rrect_mask(hw + 16, 8, 4)[..., None] * a
                hx, hy = x0 + SUB_PAD + int(wx) - 8, y0 + bh - 20
                f[hy:hy + 8, hx:hx + hw + 16] = f[hy:hy + 8, hx:hx + hw + 16] * (1 - hm) + np.array(ROSE, np.float32) / 255 * hm
            paste(f, spr, x0 + SUB_PAD + int(wx) - 2, y0 + (bh - spr.shape[0]) // 2 + 4, a * (1.0 if w[0] <= t + 0.02 else 0.55))
def overlays(f, t):
    global TITLE
    if TITLE is None:
        TITLE = (text_sprite("РАСПАКОВКА", 76, 500, INK, 0.16), text_sprite("С ALIEXPRESS", 76, 500, INK, 0.16),
                 text_sprite("одежда для дочки", 36, 300, INK, 0.02))
    # intro title on a blush band (first ~3.6 s)
    a = ease(t / 0.4) * (1 - ease_io((t - 3.4) / 0.35))
    if a > 0:
        bh = 330; y0 = 250
        band = rrect_mask(W - 120, bh, 34)[..., None] * 0.9 * a
        f[y0:y0 + bh, 60:W - 60] = f[y0:y0 + bh, 60:W - 60] * (1 - band) + np.array(BLUSH, np.float32) / 255 * band
        for j, s in enumerate(TITLE[:2]):
            p = ease((t - 0.15 - j * 0.15) / 0.45); paste(f, s, 104 + int((1 - p) * 30), y0 + 40 + j * 96, p * a)
        p = ease((t - 0.6) / 0.5); f[y0 + 236:y0 + 238, 104:104 + int(110 * p)] = np.array(ROSE, np.float32) / 255
        paste(f, TITLE[2], 104, y0 + 256, p * a)
    for c in CARDS:
        t0, t1 = item_span(c)
        if not (t0 <= t <= t1): continue
        if not hasattr(c, "spr"): c.build()
        p = ease((t - t0) / 0.45); q = 1 - ease_io((t - (t1 - 0.3)) / 0.3)
        s = c.spr; x = W - s.shape[1] - 30 + int((1 - p) * 120); y = SAFE_TOP - 22
        paste(f, s, x, y, p * q)
    # bodysuit plate text
    ks = [c for c in CLIPS if c[2] == "K"]
    if ks:
        k0 = ks[0][3]; k1 = ks[-1][3] + ks[-1][1] - ks[-1][0]
        if k0 <= t <= k1 + 0.1:
            global KS
            if "KS" not in globals():
                KS = (text_sprite("НАБОР БОДИ", 70, 500, INK, 0.14), text_sprite("5 штук · хлопок 60%", 34, 300, INK, 0.02), text_sprite("900 ₽", 64, 600, BLUSH, 0.02))
            a = ease((t - k0) / 0.4) * (1 - ease_io((t - k1 + 0.15) / 0.25)); bh = 330; y0 = SAFE_TOP
            band = rrect_mask(W - 120, bh, 34)[..., None] * 0.92 * a
            f[y0:y0 + bh, 60:W - 60] = f[y0:y0 + bh, 60:W - 60] * (1 - band) + np.array(BLUSH, np.float32) / 255 * band
            paste(f, KS[0], 104, y0 + 40, a); paste(f, KS[1], 104, y0 + 136, a)
            p = ease((t - k0 - 1.8) / 0.4)
            if p > 0:
                cw = KS[2].shape[1] + 50; chip = np.zeros((92, cw, 4), np.float32); chip[..., :3] = np.array(MOCHA) / 255; chip[..., 3] = rrect_mask(cw, 92, 46)
                paste_rgba(chip, KS[2], 25, 4); paste(f, chip, 104, y0 + 200 + int((1 - p) * 20), p * a)
    draw_subs(f, t)
    # end fade
    e = ease_io((t - (DUR - 0.5)) / 0.5)
    if e > 0: f[:] = f * (1 - e) + np.array(INK, np.float32) / 255 * e

def render_frame(t):
    f = grade(frame(t)); overlays(f, t); return np.clip(f, 0, 1)

def audio():
    out = np.zeros(int(DUR * SR) + SR); fd = int(0.012 * SR)
    for a, b, tag, o in CLIPS:
        s = voice[int(a * SR):int(b * SR)].copy(); s[:fd] *= np.linspace(0, 1, fd); s[-fd:] *= np.linspace(1, 0, fd)
        i = int(o * SR); out[i:i + len(s)] += s
    out = out[:int(DUR * SR)]; sf.write(os.path.join(WK, "voice_edit.wav"), out, SR)

if __name__ == "__main__":
    if MODE == "info":
        print("DUR", DUR, "clips", len(CLIPS))
        for a, b, tag, o in CLIPS: print(f"{o:6.2f}  {a:7.2f}-{b:7.2f} {tag}")
        sys.exit()
    if not os.environ.get("VOICE"): audio()
    if MODE == "stills":
        for t in map(float, sys.argv[4].split(",")):
            cv2.imwrite(os.path.join(WK, f"st_{t:05.1f}.jpg"), cv2.cvtColor((render_frame(t) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        sys.exit()
    # env: VOICE (wav to mux, default voice_edit.wav), CRF (default 18), KEYS ("t1,t2" forced keyframes for lossless splitting), OUT
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-i", os.path.join(WK, os.environ.get("VOICE", "voice_edit.wav")), "-c:v", "libx264", "-preset", "slow",
                            "-crf", os.environ.get("CRF", "18"), *(["-maxrate", os.environ["MAXRATE"], "-bufsize", "20M"] if os.environ.get("MAXRATE") else []), "-profile:v", "high", "-pix_fmt", "yuv420p",
                            *(["-force_key_frames", os.environ["KEYS"]] if os.environ.get("KEYS") else []),
                            "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", os.path.join(WK, os.environ.get("OUT", "out.mp4"))], stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for k in range(n):
        enc.stdin.write((render_frame(k / FPS) * 255).astype(np.uint8).tobytes())
        if k % 250 == 0: print(f"{k / FPS:.1f}/{DUR}", flush=True)
    enc.stdin.close(); enc.wait()
