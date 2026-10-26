"""Frame-exact compositor for the posuda review (posuda_reel timeline), 3840x2160 @ 25 fps.
Reads the joined shots as raw frames and draws, per frame:
  • the warm-cream grade (luts/mcgva_warm_cream.cube) on the picture, before any type is drawn
  • whip transitions at section cuts (horizontal motion blur + soft flash)
  • «shrink» moments: the picture eases down into a rounded card over its own blurred copy, then back
  • a long black spray at the very bottom
  • subtitles one word at a time, lower case, thin Montserrat, a small pop; key words on a blush pill
  • the item pills (slide in from the left) and the opening title
"""
import json, os, re, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import posuda_reel as R
from stories_kit import font, INK

W, H, FPS = R.W, R.H, R.FPS
BLUSH = (247, 223, 221)
TRANSITIONS = [10.3, 17.25, 35.05, 53.4, 74.4, 99.0, 110.3]
SHRINKS = [(35.05, 39.2), (53.4, 56.55), (62.3, 65.15), (99.0, 102.15)]
KEY = ("прикорм", "заранее", "ребёнок", "красив", "посуд", "необходим", "ложечк", "вилочк", "секци", "глубок",
       "ниблер", "поильник", "присоск", "кормлени", "угодно", "продукт", "каш", "пюре", "пить", "цвет", "прекрасн",
       "сердечк", "кружечк", "подстилк", "коллекци", "хозяйк", "восторг", "покупок")
SUB_Y = H - 290
GRADE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "luts", "mcgva_warm_cream.cube")   # make_warm_lut.py

def ease(x): x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)

def rgba(img): return np.asarray(img, np.float32)

def blend(fr, patch, x, y, alpha=1.0):
    """alpha-composite an RGBA float patch onto a uint8 frame at (x, y), clipped to the frame"""
    h, w = patch.shape[:2]; x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    p = patch[y0 - y:y1 - y, x0 - x:x1 - x]; a = p[..., 3:4] / 255 * alpha
    reg = fr[y0:y1, x0:x1].astype(np.float32); fr[y0:y1, x0:x1] = (p[..., :3] * a + reg * (1 - a)).astype(np.uint8)

def text_patch(t, f, fill, pill=None, pad=(46, 22), sh=0.75):
    """RGBA patch: text with a soft dark glyph shadow, or on a rounded pill"""
    l, tp, r, b = f.getbbox(t); tw, th = r - l, b - tp; m = 40
    w, h = int(tw + 2 * (pad[0] if pill else 0) + 2 * m), int(th + 2 * (pad[1] if pill else 0) + 2 * m)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); cx, cy = w / 2, h / 2
    if pill:
        sm = Image.new("L", (w, h), 0); ImageDraw.Draw(sm).rounded_rectangle((m, m + 10, w - m, h - m + 10), (h - 2 * m) // 2, fill=90)
        s = Image.new("RGBA", (w, h), (40, 25, 25, 0)); s.putalpha(sm.filter(ImageFilter.GaussianBlur(18))); im = Image.alpha_composite(im, s)
        ImageDraw.Draw(im).rounded_rectangle((m, m, w - m, h - m), (h - 2 * m) // 2, fill=pill + (250,))
    else:
        sm = Image.new("L", (w, h), 0); ImageDraw.Draw(sm).text((cx, cy + 5), t, font=f, fill=int(255 * sh), anchor="mm")
        s = Image.new("RGBA", (w, h), (20, 14, 12, 0)); s.putalpha(sm.filter(ImageFilter.GaussianBlur(14))); im = Image.alpha_composite(im, s)
    ImageDraw.Draw(im).text((cx, cy), t, font=f, fill=fill + (255,), anchor="mm")
    return rgba(im)

def words(work, offs):
    """one word per cue (1–2 letter words ride with the next one), lower case, no punctuation"""
    wj = json.load(open(f"{work}/words.json")); toks = []
    for k, text in enumerate(R.SCRIPT):
        for t, w in zip(text.split(), wj[str(k + 1)]):
            toks.append([re.sub(r"[.,:;?!«»—]", "", t).lower(), offs[k] + w[1], offs[k] + w[2]])
    out, carry = [], None
    for t in toks:
        if not t[0]: continue
        if carry: t = [carry[0] + " " + t[0], carry[1], t[2]]; carry = None
        if len(t[0]) <= 2: carry = t; continue
        out.append(t)
    for a, b in zip(out, out[1:]): a[2] = b[1] if b[1] - a[2] < 0.5 else a[2] + 0.3
    out[-1][2] += 0.6
    return [(w, a - 0.04, b - 0.04) for w, a, b in out]

def build(work, offs):
    fs = font("second", 124); cues = []
    for w, a, b in words(work, offs):
        key = any(k in w for k in KEY)
        p = text_patch(w, fs, INK if key else (255, 255, 255), pill=BLUSH if key else None)
        cues.append((a, b, [cv2.resize(p, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for s in (0.84, 0.93, 1.0)]))
    pills = [(a, b, text_patch(t, font("detail", 66), INK, pill=(255, 255, 255), pad=(50, 26))) for a, b, t in R.PILLS]
    a, b, t1, t2 = R.TITLE
    title = (a, b, text_patch(t1, font("main", 124), INK, sh=0), text_patch(t2, font("detail", 94), INK, sh=0))
    y = np.linspace(0, 1, H, dtype=np.float32); r0 = int(0.6 * H)
    spray = (256 * (1 - np.clip((y[r0:] - 0.6) / 0.4, 0, 1) ** 1.6 * 0.6)).astype(np.uint16)
    return cues, pills, title, (r0, spray)

def shrink_frame(fr, s):
    """picture scaled to s inside a rounded card over its own blurred, slightly dimmed copy"""
    bg = cv2.resize(cv2.GaussianBlur(cv2.resize(fr, (480, 270), interpolation=cv2.INTER_AREA), (0, 0), 7), (W, H))
    bg = (bg.astype(np.uint16) * 200 >> 8).astype(np.uint8)
    w, h = int(W * s) // 2 * 2, int(H * s) // 2 * 2; x, y = (W - w) // 2, (H - h) // 2 - int(40 * (1 - s) / 0.2)
    small = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
    sh = np.zeros((H // 8, W // 8), np.uint8)
    cv2.rectangle(sh, ((x + 6) // 8, (y + 30) // 8), ((x + w + 6) // 8, (y + h + 30) // 8), 150, -1)
    sh = cv2.resize(cv2.GaussianBlur(sh, (0, 0), 5), (W, H)).astype(np.uint16)
    bg = (bg.astype(np.uint16) * (255 - sh[..., None]) >> 8).astype(np.uint8)
    m = np.zeros((h, w), np.uint8); rr = int(70 * s)
    cv2.rectangle(m, (rr, 0), (w - rr, h), 255, -1); cv2.rectangle(m, (0, rr), (w, h - rr), 255, -1)
    for cx, cy in ((rr, rr), (w - rr, rr), (rr, h - rr), (w - rr, h - rr)): cv2.circle(m, (cx, cy), rr, 255, -1, cv2.LINE_AA)
    a = (m.astype(np.uint16))[..., None]
    reg = bg[y:y + h, x:x + w].astype(np.uint16); bg[y:y + h, x:x + w] = ((small * a + reg * (255 - a)) >> 8).astype(np.uint8)
    return bg

def whip(fr, k):
    """k in 0..1: horizontal motion blur + flash, strongest at the cut"""
    if k <= 0: return fr
    sm = cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA); n = max(3, int(90 * k) | 1)
    bl = cv2.resize(cv2.blur(sm, (n, 1)), (W, H), interpolation=cv2.INTER_LINEAR)
    out = cv2.addWeighted(fr, 1 - min(1, 1.3 * k), bl, min(1, 1.3 * k), 0)
    return cv2.addWeighted(out, 1 - 0.35 * k, np.full_like(out, 255), 0.35 * k, 0)

def render(work, offs, out):
    cues, pills, title, (r0, spray) = build(work, offs)
    n = round(R.END * FPS)
    grade = ["-vf", f"lut3d=file={GRADE}:interp=tetrahedral"] if GRADE else []          # warm-cream grade on the picture only
    rd = subprocess.Popen(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", f"{work}/list.txt", "-frames:v", str(n)] + grade + [
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=W * H * 3 * 2)
    wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-i", f"{work}/voice.wav", "-map", "0:v", "-map", "1:a", "-t", str(R.END),
                           "-vf", f"fade=t=in:d=0.5,fade=t=out:st={R.END - 1.0}:d=1.0", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    sz = W * H * 3
    for i in range(n):
        buf = rd.stdout.read(sz)
        if len(buf) < sz: fr = last.copy()                               # pad if the joined shots run a frame short
        else: fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy(); last = fr
        t = i / FPS
        for a, b in SHRINKS:                                             # «уменьшение кадра»
            if a <= t < b:
                s = 1 - 0.2 * ease((t - a) / 0.4) * (1 - ease((t - (b - 0.4)) / 0.4))
                if s < 0.999: fr = shrink_frame(fr, s)
        for c in TRANSITIONS:                                            # whip at the cut
            dt = abs(t - c + 0.02)
            if dt < 0.2: fr = whip(fr, 1 - dt / 0.2)
        fr[r0:] = (fr[r0:].astype(np.uint16) * spray[:, None, None] >> 8).astype(np.uint8)
        a, b, t1, t2 = title
        if a <= t < b:
            for j, (p, y) in enumerate(((t1, 330), (t2, 480))):
                k = ease((t - a - 0.15 * j) / 0.45) * (1 - ease((t - b + 0.4) / 0.4))
                if k > 0: blend(fr, p, 240 - 40, int(y - p.shape[0] / 2 + 40 * (1 - k)), k)
        for a, b, p in pills:
            if a <= t < b:
                k = ease((t - a) / 0.3) * (1 - ease((t - b + 0.3) / 0.3))
                blend(fr, p, int(130 - 120 * (1 - k)), 130, k)
        for a, b, ps in cues:
            if a <= t < b:
                fi = int((t - a) * FPS); p = ps[min(fi, 2)]
                blend(fr, p, (W - p.shape[1]) // 2, SUB_Y - p.shape[0] // 2, min(1, (fi + 1) / 2))
        wr.stdin.write(fr.tobytes())
    wr.stdin.close(); wr.wait(); rd.kill()
