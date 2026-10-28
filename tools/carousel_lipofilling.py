"""Instagram carousel (1080x1350) cut from the finished lipofilling reel (IMG_8380.MP4).

Each video slide is one block of the reel with its own headline and voice,
cropped to 4:5 so the headline, subtitles and body all stay in frame.
The last slide is a consultation card in the Tvauri style.

usage: python3 carousel_lipofilling.py REEL.MP4 SURGEON_PHOTO_DIR OUT_DIR
"""
import os, sys, subprocess

REEL, PHOTOS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], PHOTOS, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carousel_tvauri as ct  # noqa: E402  (shared fonts, palette, text blocks)

W, H = 1080, 1350
CROP_Y = 60  # keeps the reel's headline band and the body below it
# block boundaries = hard cuts in the reel (headline changes)
SLIDES = [
    ("01_implants_no", 0.00, 4.93),
    ("02_lift_no", 4.93, 10.67),
    ("03_chose_lipofilling", 10.67, 19.20),
    ("04_donor_zones", 19.20, 23.07),
    # headline band (top of the reel) stacked over the footage band (bottom of the reel): full 4:5, no side bars
    ("05_volume", 23.07, 29.20, "stack", (0.07, 0.36), (0.585, 1.0)),
    ("06_not_a_lift", 29.20, 36.60, "stack", (0.06, 0.335), (0.378, 1.0)),
    ("07_this_task", 36.60, None, "text"),  # footage cropped low, headline redrawn on top of it
]


G = "0x%02x%02x%02x" % ct.GRAPHITE


def stack(name, a, b, top, bot):
    """top band at native scale + bottom band filling the rest of the 4:5 frame, soft seam between them"""
    import numpy as np, cv2
    SH = 1920
    dur = (b or float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", REEL],
                                     capture_output=True, text=True).stdout)) - a
    t0, t1 = int(top[0] * SH), int(top[1] * SH); b0, b1 = int(bot[0] * SH), int(bot[1] * SH)
    ha = t1 - t0; hb = H - ha; seam = 60
    rd = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", REEL,
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    tmp = f"{OUT}/_{name}_v.mp4"
    wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30",
                           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", tmp],
                          stdin=subprocess.PIPE)
    k = np.linspace(0, 1, seam, dtype=np.float32)[:, None, None]; k = k * k * (3 - 2 * k)
    while True:
        buf = rd.stdout.read(W * SH * 3)
        if len(buf) < W * SH * 3: break
        f = np.frombuffer(buf, np.uint8).reshape(SH, W, 3)
        A = f[t0:t1].astype(np.float32)
        Bsrc = f[b0:b1]
        if b1 - b0 <= hb:  # native scale; the seam rows come from just above the band, so nothing repeats
            ext = f[max(b1 - hb - seam, 0):b1].astype(np.float32)
            B = ext[-hb:]; Bseam = ext[:seam] if len(ext) >= hb + seam else B[:seam] * 0 + A[-1:]
        else:  # taller than the room left: scale the footage down over its own (already blurred) background
            s = hb / (b1 - b0); fw = int(W * s) // 2 * 2
            fg = cv2.resize(Bsrc, (fw, hb), interpolation=cv2.INTER_AREA).astype(np.float32)
            bg = cv2.GaussianBlur(cv2.resize(Bsrc, (W, hb), interpolation=cv2.INTER_AREA), (0, 0), 25).astype(np.float32)
            m = np.ones((hb, fw, 1), np.float32); e = 70
            r = np.linspace(0, 1, e, dtype=np.float32)[None, :, None]; m[:, :e] *= r; m[:, -e:] *= r[:, ::-1]
            x = (W - fw) // 2; B = bg.copy(); B[:, x:x + fw] = bg[:, x:x + fw] * (1 - m) + fg * m
            Bseam = np.repeat(bg[:1], seam, 0)  # plain blurred background under the seam
        out = np.empty((H, W, 3), np.float32)
        out[:ha] = A; out[ha:] = B
        out[ha - seam:ha] = A[ha - seam:] * (1 - k) + Bseam * k  # soft seam between the two bands
        wr.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
    wr.stdin.close(); wr.wait(); rd.wait()
    fo = max(dur - 0.12, 0)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", REEL,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", f"afade=t=in:d=0.06,afade=t=out:st={fo:.3f}:d=0.12",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", f"{OUT}/{name}.mp4"], check=True)
    os.remove(tmp)


def headline_overlay(path):
    """the reel's last headline, redrawn over the footage (check mark, title, navy pill)"""
    from PIL import Image, ImageDraw
    S = 3
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov.alpha_composite(ct.vgrad(170, 380, 1.0, 0))  # the top of the crop is already dark: hides the reel's word pill
    hi = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0)); d = ImageDraw.Draw(hi)
    cx, cy, r = W // 2 * S, 92 * S, 40 * S
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=ct.WHITE + (255,), width=5 * S)
    d.line([(cx - 18 * S, cy + 1 * S), (cx - 5 * S, cy + 14 * S), (cx + 20 * S, cy - 13 * S)], fill=ct.WHITE + (255,), width=6 * S, joint="curve")
    ov.alpha_composite(hi.resize((W, H), Image.LANCZOS))
    ct.Block(W - 2 * ct.MARGIN, "center").head("Именно эту задачу\nи решали", 92, 0).draw(ov, ct.MARGIN, 152)
    f = ct.mont(27, 500); t = "Подробнее в актуальном «Липофилинг»"; tw = f.getlength(t)
    d = ImageDraw.Draw(ov); y = 352
    d.rounded_rectangle(((W - tw) / 2 - 22, y, (W + tw) / 2 + 22, y + 50), radius=25, fill=(16, 46, 70, 235))
    d.text((W / 2, y + 25), t, font=f, fill=ct.WHITE, anchor="mm")
    ov.save(path)


def cut(name, a, b, mode="crop", top=None, bot=None):
    if mode == "stack": return stack(name, a, b, top, bot)
    if mode == "text":
        ovp = f"{OUT}/_{name}_ov.png"; headline_overlay(ovp)
        dur = (b or float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", REEL],
                                         capture_output=True, text=True).stdout)) - a
        fo = max(dur - 0.12, 0)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", REEL, "-loop", "1", "-i", ovp,
                        "-filter_complex", f"[0:v]crop={W}:{H}:0:1920-{H},setsar=1[v];[v][1:v]overlay=0:0:shortest=1,format=yuv420p[o]",
                        "-map", "[o]", "-map", "0:a", "-t", f"{dur:.3f}",
                        "-af", f"afade=t=in:d=0.06,afade=t=out:st={fo:.3f}:d=0.12",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", f"{OUT}/{name}.mp4"], check=True)
        os.remove(ovp); return
    dur = (b or float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", REEL],
                                     capture_output=True, text=True).stdout)) - a
    fo = max(dur - 0.12, 0)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", REEL,
                    "-vf", (f"crop={W}:{H}:0:{CROP_Y},setsar=1" if mode == "crop" else
                            f"scale=-2:{H}:flags=lanczos,pad={W}:{H}:(ow-iw)/2:0:color={G},setsar=1"),  # whole frame: content sits low
                    "-af", f"afade=t=in:d=0.06,afade=t=out:st={fo:.3f}:d=0.12",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", f"{OUT}/{name}.mp4"], check=True)


def cta():
    """surgeon from the reel on top, consultation card below (same layout as the lift carousel)"""
    from PIL import Image, ImageDraw
    import numpy as np
    PH = 800
    im = ct.load_photo("DSC00888.JPG"); sc = W / im.width; sh = im.height * sc; off = 0.17 * sh
    ph = im.resize((W, round(sh)), Image.LANCZOS).crop((0, int(off), W, int(off) + PH)).convert("RGBA")
    c = Image.new("RGBA", (W, H), ct.WHITE + (255,)); c.alpha_composite(ph, (0, 0))
    ct.put_logo(c, y=64, alpha=0.95)
    arr = np.zeros((H, W, 4), np.uint8); arr[..., :3] = 255
    p = np.clip((np.arange(H)[:, None] - (PH - 150)) / 150, 0, 1); p = np.broadcast_to(p, (H, W))
    arr[..., 3] = (p * p * (3 - 2 * p) * 255).astype(np.uint8)
    c.alpha_composite(Image.fromarray(arr, "RGBA"))
    bw = 840; x = (W - bw) // 2
    b = ct.Block(bw, "center").head("На консультации разберём", 60, 10, color=ct.INK).rule(0.14, 22)
    y = b.draw(c, x, PH - 50)
    items = ["подойдёт ли вам липофилинг", "есть ли донорские зоны", "какой объём можно добавить",
             "нужна ли подтяжка или импланты"]
    f = ct.mont(28, 500); d = ImageDraw.Draw(c)
    for it in items:
        tw = f.getlength(it); x0 = (W - tw) / 2
        d.ellipse((x0 - 24, y + 12, x0 - 14, y + 22), fill=(214, 160, 222)); d.text((x0, y), it, font=f, fill=ct.INK); y += 42
    y += 22
    ct.Block(bw, "center").ital("Для записи напишите в Direct «КОНСУЛЬТАЦИЯ» или свяжитесь с нами по телефону +7 926 636 30 00 в WhatsApp, Telegram или MAX.", 27, 0, color=ct.MIST).draw(c, x, y)
    ct.save_jpg(c, f"{OUT}/08_cta.jpg")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for n, a, b, *m in SLIDES:
        cut(n, a, b, *m); print("done", n, flush=True)
    cta(); print("done cta")
