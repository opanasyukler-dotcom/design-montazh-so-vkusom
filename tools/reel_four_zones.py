"""Reel 9:16 '4 зоны для коррекции. Можно ли сделать всё сразу?' (1080x1920, 30 fps).

Same look as the stories 'Хочу увеличить грудь…': three powder dots + bold caps Montserrat headline,
thin lines, static text straight on the footage with a soft dark halo and light gradients.

Sources: the surgeon talking at the desk (IMG_3155, also the voice) and the silent iPad screen
recording where she marks the patient's photo (abdomen ~54-72 s, breast ~68-88 s, arms ~94-112 s,
thighs ~128-146 s, everything marked from ~150 s). Voice quotes were located with Whisper (sherpa-onnx);
cuts sit in the pauses so no phrase is clipped.

usage: python3 reel_four_zones.py SURGEON.mp4 SCREEN.mp4 OUT.mp4
"""
import os, sys, subprocess, tempfile
import numpy as np
from PIL import Image, ImageFilter

SURG, SCREEN, OUT = sys.argv[1:4]
TMP = tempfile.mkdtemp()
first = os.path.join(TMP, "f.png")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "150", "-i", SCREEN, "-frames:v", "1", first], check=True)
sys.argv = [sys.argv[0], first, TMP]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surgeon_style import Slide, mont, POWDER, DEEP, WHITE, S  # noqa: E402

W, H, FPS, X = 1080, 1920, 30, 78
BOLD, THIN = 700, 300
PHOTO_BOX = (374, 60, 906, 826)     # the patient photo inside the iPad screen recording (drops status bar / title / toolbar)

# (voice ranges in IMG_3155), shots: ("scr"|"surg", src_from, src_to, duration on screen), texts
SEGMENTS = [
    dict(voice=[(26.2, 32.0)], shots=[("scr", 54, 152, 5.8)],
         head=["4 ЗОНЫ", "ДЛЯ КОРРЕКЦИИ.", "МОЖНО ЛИ СДЕЛАТЬ", "ВСЁ СРАЗУ?"], size=88),
    dict(voice=[(1.4, 10.0)], shots=[("scr", 10, 18, 4.3), ("surg", 5.7, 10.0, 4.3)],
         head=["ВЕС УШЁЛ,", "А ИЗБЫТОК КОЖИ", "ОСТАЛСЯ"]),
    dict(voice=[(32.1, 39.6)], shots=[("scr", 54, 72, 7.5)],
         head=["ПЕРВЫЙ ЭТАП"],
         lines=[("Абдоминопластика", BOLD), ("Коррекция талии", BOLD), ("Возможная липосакция 360", BOLD)],
         note="*ушивание диастаза — если он есть"),
    dict(voice=[(62.0, 74.45)], shots=[("surg", 62.0, 65.0, 3.0), ("scr", 68, 88, 9.45)],
         head=["ГРУДЬ МОЖНО", "ПОДТЯНУТЬ БЕЗ УВЕЛИЧЕНИЯ", "ИЛИ ДОБАВИТЬ ОБЪЁМ"],
         note="Вариант зависит от пожеланий пациентки и осмотра."),
    dict(voice=[(74.6, 80.15), (81.9, 85.8)], shots=[("scr", 94, 112, 5.55), ("scr", 128, 146, 3.9)],
         head=["РУКИ И БЁДРА"], lines=[("Возможная подтяжка кожи после снижения веса.", THIN)]),
    dict(voice=[(86.1, 90.35)], shots=[("surg", 86.1, 90.35, 4.25)],
         head=["А ПОЧЕМУ НЕ ВСЁ", "ЗА ОДИН РАЗ?"], size=92,
         lines=[("Потому что объём операции и восстановление должны быть безопасными для пациентки.", THIN)]),
    dict(voice=[(90.35, 92.95)], shots=[("scr", 152, 156, 4.0)], dur=4.0,
         head=["А ВАШИ ЗАПРОСЫ", "МОЖНО ОБЪЕДИНИТЬ", "В ОДНУ ОПЕРАЦИЮ?"],
         lines=[("Узнайте на консультации.", BOLD)]),
]


def wrap(text, f, width):
    out, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if f.getlength(t) / S <= width or not line: line = t
        else: out.append(line); line = w
    return out + [line]


def overlay(seg, path):
    """static text layer (dots + headline at the top, lines at the bottom) with halo and light gradients"""
    sl = Slide(W, H, Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0)))
    for i in range(3):
        cx = X + 6 + i * 24; sl.d.ellipse(((cx - 6) * S, (170 - 6) * S, (cx + 6) * S, (170 + 6) * S), fill=POWDER + (255,))
    size = seg.get("size", 76)
    while max(mont(size, BOLD).getlength(l) / S for l in seg["head"]) > W - 2 * X - 20: size -= 2
    lh = int(size * 1.1)
    for i, r in enumerate(seg["head"]): sl.text((X, 230 + i * lh), r, mont(size, BOLD), WHITE)
    # bottom block ends above the Reels caption / buttons zone
    rows = []
    for t, w in seg.get("lines", []):
        f = mont(42 if w == BOLD else 40, w if w == BOLD else 400)
        for r in wrap(t, f, W - 2 * X - 120): rows.append((r, f))
    note = seg.get("note"); fn = mont(34, 400)
    nrows = wrap(note, fn, W - 2 * X - 120) if note else []
    hgt = sum(f.size / S * 1.3 for _, f in rows) + (20 + len(nrows) * 40 if nrows else 0)
    y = 1500 - hgt
    if rows and seg["head"][0] == "ПЕРВЫЙ ЭТАП":            # the list gets powder bullets
        for r, f in rows:
            sl.dot((X + 8, y + f.size / S * 0.62), 7, POWDER); sl.text((X + 34, y), r, f, WHITE); y += f.size / S * 1.3
    else:
        for r, f in rows: sl.text((X, y), r, f, WHITE); y += f.size / S * 1.3
    if nrows:
        y += 20
        for r in nrows: sl.text((X, y), r, fn, POWDER); y += 44
    im = sl.im.resize((W, H), Image.LANCZOS)
    a = im.split()[3].filter(ImageFilter.GaussianBlur(9)).point(lambda v: int(min(255, v * 2.0) * 0.7))
    halo = Image.new("RGBA", im.size, DEEP + (0,)); halo.putalpha(a)
    ys = np.arange(H, dtype=np.float32)
    g = np.maximum(np.clip(1 - ys / 820, 0, 1) ** 1.4 * 0.6, np.clip((ys - 1000) / (1560 - 1000), 0, 1) ** 1.1 * 0.82)
    arr = np.zeros((H, W, 4), np.uint8); arr[..., :3] = DEEP; arr[..., 3] = (g * 255)[:, None]
    base = Image.fromarray(arr, "RGBA"); base.alpha_composite(halo); base.alpha_composite(im)
    base.save(path)


def shot_clip(kind, a, b, d, path):
    speed = (b - a) / d
    if kind == "surg":
        vf = f"[0:v]setpts=(PTS-STARTPTS)/{speed},scale={W}:{H}:flags=lanczos,fps={FPS},setsar=1[v]"
        src = SURG
    else:
        x0, y0, x1, y1 = PHOTO_BOX; pw, ph = x1 - x0, y1 - y0; fh = int(ph * W / pw) // 2 * 2
        vf = (f"[0:v]setpts=(PTS-STARTPTS)/{speed},fps={FPS},crop={pw}:{ph}:{x0}:{y0},split[p][q];"
              f"[q]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},gblur=sigma=30,eq=brightness=-0.12[bg];"
              f"[p]scale={W}:{fh}:flags=lanczos[fg];[bg][fg]overlay=0:(H-h)/2,setsar=1[v]")
        src = SCREEN
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a}", "-t", f"{b - a + 0.5}", "-i", src, "-filter_complex", vf,
                    "-map", "[v]", "-t", f"{d}", "-an", "-c:v", "libx264", "-crf", "14", "-preset", "fast", "-pix_fmt", "yuv420p", path],
                   check=True)


def main():
    seg_files = []
    for k, seg in enumerate(SEGMENTS):
        vd = sum(b - a for a, b in seg["voice"]); dur = max(vd, seg.get("dur", 0))
        shots = []
        for j, (kind, a, b, d) in enumerate(seg["shots"]):
            p = os.path.join(TMP, f"s{k}_{j}.mp4"); shot_clip(kind, a, b, d, p); shots.append(p)
        ov = os.path.join(TMP, f"ov{k}.png"); overlay(seg, ov)
        # voice: the quoted ranges back to back, short fades at each cut so nothing clicks
        ain, af = [], []
        for j, (a, b) in enumerate(seg["voice"]):
            ain += ["-ss", f"{a}", "-t", f"{b - a}", "-i", SURG]
        n = len(shots); m = len(seg["voice"])
        fc = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[cv];[cv][{n}:v]overlay=0:0,format=yuv420p,trim=duration={dur}[v];"
        for j, (a, b) in enumerate(seg["voice"]):
            fc += f"[{n + 1 + j}:a]afade=in:d=0.04,afade=out:st={b - a - 0.08}:d=0.08[a{j}];"
        fc += "".join(f"[a{j}]" for j in range(m)) + f"concat=n={m}:v=0:a=1,apad,atrim=duration={dur}[a]"
        out = os.path.join(TMP, f"seg{k}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y"] + sum((["-i", s] for s in shots), []) + ["-loop", "1", "-i", ov] + ain +
                       ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", f"{dur}", "-r", str(FPS),
                        "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out], check=True)
        seg_files.append(out); print("segment", k + 1, round(dur, 2), flush=True)
    lst = os.path.join(TMP, "list.txt")
    open(lst, "w").write("".join(f"file '{f}'\n" for f in seg_files))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", OUT], check=True)
    print("ok", OUT)


if __name__ == "__main__":
    main()
