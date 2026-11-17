"""Video slides (1080x1350, 30 fps, with the surgeon's voice) for the lipofilling carousel,
cut from the reel 'Увеличение груди без имплантов' phrase by phrase.

Layout = slide 6 of the photo carousel: navy, ANNA TVAURI mark, Bebas headline in powder,
thin Montserrat text, the footage in an arch card on the right with a dashed arrow.
Each shot is cropped clear of the reel's baked overlays (logo, word bubble, captions, pill,
Instagram tags); the top-right IG handle that falls inside some crops is removed with delogo.

Phrase boundaries come from pauses in the voice track; shot boundaries from scene detection.

usage: python3 post_lipofilling_reel_video.py REEL.mp4 OUT_DIR
"""
import os, sys, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw

REEL, OUT = sys.argv[1:3]
TMP = tempfile.mkdtemp()
first = os.path.join(TMP, "f.png")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "3.1", "-i", REEL, "-frames:v", "1", first], check=True)
sys.argv = [sys.argv[0], first, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surgeon_style as ss  # noqa: E402
from surgeon_style import Slide, bebas, mont, logo, vshade, POWDER, NAVY, DEEP, WHITE, S  # noqa: E402

PW, PH = 1080, 1350
CARD = (460, 560, 1000, 1235)            # arch card on the slide (w 540, h 675 = 4:5)
CW, CH = CARD[2] - CARD[0], CARD[3] - CARD[1]

# crops in reel pixels (720x1280), all 4:5
U = (270, 226, 391, 489)    # universal: clear of logo, word bubble, captions, pill, bottom IG tags
DONOR = (154, 460, 440, 550)
FACE = (120, 460, 440, 550)
BREAST = (140, 460, 440, 550)
IG_TR = "IG"  # top-right IG icon + handle on the OR shots: inpainted frame by frame (see clean_clip)
IG_BOXES = ((618, 278, 674, 338), (536, 340, 686, 372))

# shot = (start, end, crop, extra filters, duration on the slide or None for real time, hold first frame s)
SLIDES = [
    dict(n=2, a=(0.00, 2.58), name="02_donor",
         shots=[(1.0, 1.7, U, IG_TR, 1.40, 0), (1.7, 2.7, DONOR, "", 1.18, 0)],
         head=["ЖИР ИЗ ДОНОРСКИХ", "ЗОН"], text=["Забираем жир из ваших", "донорских зон"]),
    dict(n=3, a=(2.58, 4.44), name="03_transfer",
         shots=[(2.70, 3.60, FACE, "", None, 0.12), (3.60, 4.44, BREAST, "", None, 0)],
         head=["ПЕРЕСАЖИВАЕМ", "В ОБЛАСТЬ ГРУДИ"], text=["Собственная ткань", "вместо импланта"]),
    dict(n=4, a=(4.44, 9.84), name="04_syringes",
         shots=[(4.44, 5.00, U, IG_TR, None, 0), (5.00, 6.233, U, IG_TR, None, 0),
                (6.233, 8.633, U, IG_TR, None, 0), (8.633, 9.84, U, IG_TR, None, 0)],
         head=["ГОТОВИМ ЖИР", "К ВВЕДЕНИЮ"], text=["Собранный жир перекачиваем", "в специальные большие шприцы"]),
    dict(n=5, a=(9.84, 14.86), name="05_volume",
         shots=[(9.84, 14.86, U, "", None, 0)],
         head=["ОДИНАКОВЫЙ", "ОБЪЁМ"], text=["В каждую железу вводим", "точное количество миллилитров,", "чтобы объём был одинаковым"]),
    dict(n=6, a=(14.86, 17.56), name="06_asymmetry",
         shots=[(14.86, 17.56, U, "", None, 0)],
         head=["А ЕСЛИ ЕСТЬ", "АСИММЕТРИЯ?"], text=["Тогда в железы вводим", "разный объём и выравниваем", "форму груди"]),
]


def clean_clip(s0, s1):
    """re-encode a shot with the Instagram icon/handle inpainted on every frame (cleaner than delogo)"""
    import cv2
    out = os.path.join(TMP, f"clean_{s0}_{s1}.mp4")
    if os.path.exists(out): return out
    w, h = 720, 1280
    rd = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{s0}", "-t", f"{s1 - s0}", "-i", REEL, "-f", "rawvideo",
                           "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "30",
                           "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    m = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in IG_BOXES: m[y0:y1, x0:x1] = 255
    while True:
        b = rd.stdout.read(w * h * 3)
        if len(b) < w * h * 3: break
        f = np.frombuffer(b, np.uint8).reshape(h, w, 3)
        wr.stdin.write(cv2.inpaint(f, m, 7, cv2.INPAINT_TELEA).tobytes())
    wr.stdin.close(); wr.wait(); rd.wait()
    return out


def down(sl):
    return sl.im.resize((PW, PH), Image.LANCZOS)


def background(n):
    sl = Slide(PW, PH); sl.rrect((0, 0, PW, PH), 0, NAVY)
    vshade(sl, 0, PH, 0.0, 0.45, DEEP)
    logo(sl, 92)
    sl.shadow(CARD, 40, 22, 120)
    sl.text((PW - 80, 1286), f"{n:02d} / 07", mont(24, 400), WHITE, "rs", a=170, spacing=2)
    p = os.path.join(TMP, f"bg{n}.png"); down(sl).convert("RGB").save(p); return p


def text_layer(n, head, text):
    sl = Slide(PW, PH, Image.new("RGBA", (PW * S, PH * S), (0, 0, 0, 0)))
    y = sl.lines((80, 190), head, bebas(104), POWDER, lh=0.9)
    y = sl.lines((84, y + 26), text, mont(36, 300), WHITE, lh=1.22)
    a0 = (250, max(y + 60, 760)); pts = ss.bez(a0, (250, 980), (330, 1060), (440, 1020))
    sl.dashed(pts, POWDER, 2, 10, 9, 200); sl.head(pts, POWDER, 2, 14)
    p = os.path.join(TMP, f"tx{n}.png"); down(sl).save(p); return p


def arch_mask():
    m = Image.new("L", (CW * 4, CH * 4), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, CW * 4 - 1, CH * 4 - 1 + CW * 2), radius=CW * 2, fill=255)
    p = os.path.join(TMP, "mask.png"); m.resize((CW, CH), Image.LANCZOS).save(p); return p


def render(n, a, name, shots, head, text):
    bg, tx, mk = background(n), text_layer(n, head, text), arch_mask()
    dur = a[1] - a[0]
    args = ["ffmpeg", "-v", "error", "-y"]
    for s0, s1, _, extra, *_ in shots:
        args += ["-i", clean_clip(s0, s1)] if extra == IG_TR else ["-ss", f"{s0}", "-t", f"{s1 - s0}", "-i", REEL]
    k = len(shots)
    args += ["-loop", "1", "-i", bg, "-loop", "1", "-i", tx, "-loop", "1", "-i", mk, "-ss", f"{a[0]}", "-t", f"{dur}", "-i", REEL]
    fc = []
    for i, (s0, s1, (cx, cy, cw, ch), extra, d, hold) in enumerate(shots):
        f = f"[{i}:v]fps=30,setpts=PTS-STARTPTS"
        if d: f += f",setpts=PTS*{d / (s1 - s0):.4f}"
        if hold: f += f",tpad=start_duration={hold}:start_mode=clone"
        if extra and extra != IG_TR: f += "," + extra
        f += f",crop={cw}:{ch}:{cx}:{cy},scale={CW}:{CH}:flags=lanczos,setsar=1,format=rgba[c{i}]"
        fc.append(f)
    fc.append("".join(f"[c{i}]" for i in range(k)) + f"concat=n={k}:v=1:a=0,fps=30[cv]")
    fc.append(f"[{k + 2}:v]format=gray,scale={CW}:{CH}[m];[cv][m]alphamerge[ca]")
    fc.append(f"[{k}:v]format=rgba[b];[b][ca]overlay={CARD[0]}:{CARD[1]}:shortest=1[s]")
    fc.append(f"[{k + 1}:v]format=rgba,fade=in:st=0.1:d=0.45:alpha=1[t];[s][t]overlay=shortest=1,trim=duration={dur},format=yuv420p[o]")
    fc.append(f"[{k + 3}:a]afade=in:d=0.08,afade=out:st={dur - 0.15}:d=0.15[au]")
    args += ["-filter_complex", ";".join(fc), "-map", "[o]", "-map", "[au]", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
             "-c:a", "aac", "-b:a", "192k", "-t", f"{dur}", "-movflags", "+faststart", f"{OUT}/{name}.mp4"]
    subprocess.run(args, check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{min(1.0, dur / 2)}", "-i", f"{OUT}/{name}.mp4", "-frames:v", "1",
                    "-q:v", "2", f"{OUT}/{name}_frame.jpg"], check=True)
    print("done", name, round(dur, 2), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s in SLIDES: render(**s)
