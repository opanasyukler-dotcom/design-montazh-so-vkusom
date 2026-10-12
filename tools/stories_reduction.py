"""Two 9:16 stories (1080x1920) about breast reduction over operating-room footage.
Fonts follow the reference story: Montserrat Light lowercase accent + Inter on a
slate caption plate + Inter Bold punchline. Story 2 leaves a clear zone for the poll sticker.

usage: python3 stories_reduction.py SRC_VIDEO OUT_DIR
"""
import os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 25
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
INTER = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
INTER_B = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
PLATE = (78, 86, 112, 225)     # slate caption plate from the reference
WHITE = (255, 255, 255, 255)

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
TMP = os.path.join(OUT, "_bg.mp4")


def mont(size, wght=350):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([wght]); return f


def wrap(text, f, width):
    out, line = [], ""
    for w in text.split(" "):
        t = (line + " " + w).strip()
        if f.getlength(t) <= width or not line: line = t
        else: out.append(line); line = w
    return out + [line]


def layer(): return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def shadowed(img, blur=10, a=150):
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sh.putalpha(img.getchannel("A").point(lambda v: v * a // 255))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    return Image.alpha_composite(sh, img)


def plate(lines, f, cy, pad=(30, 14), lh=1.22):
    """Centered caption block on one slate plate (like the reference subtitle)."""
    L = layer(); d = ImageDraw.Draw(L)
    asc, desc = f.getmetrics(); step = int(f.size * lh)
    wmax = max(f.getlength(s) for s in lines)
    h = step * len(lines) + pad[1] * 2
    y0 = cy - h // 2
    d.rectangle([W / 2 - wmax / 2 - pad[0], y0, W / 2 + wmax / 2 + pad[0], y0 + h], fill=PLATE)
    for i, s in enumerate(lines):
        d.text((W / 2, y0 + pad[1] + i * step + step / 2), s, font=f, fill=WHITE, anchor="mm")
    return L


def text(lines, f, cy, lh=1.18, blur=12, a=170):
    L = layer(); d = ImageDraw.Draw(L); step = int(f.size * lh)
    y0 = cy - step * len(lines) / 2
    for i, s in enumerate(lines):
        d.text((W / 2, y0 + i * step + step / 2), s, font=f, fill=WHITE, anchor="mm")
    return shadowed(L, blur, a)


def gradient(top, bottom, a0=0, a1=200):
    g = np.zeros((H, W), np.float32)
    ys = np.arange(H, dtype=np.float32)
    g[:] = np.clip((ys - top) / (bottom - top), 0, 1)[:, None]
    g = g * g * (3 - 2 * g)
    return (a0 + (a1 - a0) * g)[..., None] / 255.0


# ---------------- story content ----------------
def story1():
    items = ["Боль в спине", "Следы от лямок", "Потница", "Ограничения в спорте"]
    f = ImageFont.truetype(INTER, 60)
    els, t, y = [], 0.5, 905
    for s in items:
        els.append((plate([s], f, y), t)); t += 0.85; y += 122
    t += 0.5
    els.append((text(["поэтому женщины", "решаются"], mont(66), 1475), t))
    els.append((text(["на уменьшение груди"], ImageFont.truetype(INTER_B, 78), 1612), t + 0.45))
    return els, gradient(620, 1500, 0, 175)


def story2():
    els = []
    fb = ImageFont.truetype(INTER, 50)
    desc = "Пациентка будет иметь возможность заняться любым спортом, выбирать красивое белье, любые купальники"
    els.append((plate(wrap(desc, fb, 820), fb, 905), 0.5))
    els.append((text(["как считаете,", "стоило того,", "чтобы решиться?"], mont(70), 1200), 1.4))
    # 1300..1640 stays empty for the Instagram poll sticker
    return els, gradient(560, 1350, 0, 185)


# ---------------- render ----------------
def ease(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3


def render(name, els, grad, n):
    prepped = [(np.asarray(L, np.float32) / 255.0, t0) for L, t0 in els]
    out = os.path.join(OUT, name + ".mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "17", "-preset", "slow",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", TMP, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    still = None
    for i in range(n):
        fr = np.frombuffer(dec.stdout.read(W * H * 3), np.uint8).reshape(H, W, 3)
        t = i / FPS
        z = 1.0 + 0.06 * i / n                      # slow push-in
        M = cv2.getRotationMatrix2D((W / 2, H * 0.35), 0, z)
        img = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255
        img *= 1 - grad * 0.95
        for A, t0 in prepped:
            k = ease((t - t0) / 0.55)
            if k <= 0: continue
            dy = int(round(36 * (1 - k)))
            A2 = np.roll(A, dy, axis=0)
            a = A2[..., 3:4] * k
            img = img * (1 - a) + A2[..., :3] * a
        o = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        enc.stdin.write(o.tobytes())
        if i == n - 1: still = o
    enc.stdin.close(); enc.wait(); dec.stdout.close(); dec.wait()
    Image.fromarray(still).save(os.path.join(OUT, name + ".jpg"), quality=93)
    print("wrote", out)


# background: 4K source -> 1080x1920, played forward then backward for a seamless ~13.7 s loop, no audio
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-filter_complex",
                f"[0:v]scale={W}:{H}:flags=lanczos,fps={FPS},eq=contrast=1.04:saturation=0.92,split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[v]",
                "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "12", "-preset", "fast", TMP], check=True)
N = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", TMP], capture_output=True, text=True).stdout.strip())

render("story_1_reasons", *story1(), N)
render("story_2_poll", *story2(), N)
os.remove(TMP)
