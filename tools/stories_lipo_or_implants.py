"""Stories 9:16 'Хочу увеличить грудь, но импланты не хочу…' — 4 stories (1080x1920, 30 fps).
Text sits straight on the footage (no black plate): light gradients + a soft dark halo under the type.

Story design approved earlier (stories_ptoz): three powder dots + big bold caps headline on black at
the top, footage fading in below, bold lead + thin line at the bottom over a dark gradient.
Classic Instagram-like type: Montserrat Bold for accents, Montserrat Light for the rest.
Text appears in sequence over the moving footage; the last 3 s are static.

usage: python3 stories_lipo_or_implants.py MATERIALS_DIR OUT_DIR [story numbers...]
"""
import os, sys, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter

SRC, OUT = sys.argv[1:3]
ONLY = {int(a) for a in sys.argv[3:]}
VID_HANDS = "C0816.MP4"; VID_OR = "copy_3AF02E7B-EC4D-4A5C-8FF7-65665E757D3C.mov"; PHOTO = "DSC01189.JPG"
sys.argv = [sys.argv[0], os.path.join(SRC, PHOTO), OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surgeon_style import Slide, mont, logo, icon_circle, POWDER, NAVY, DEEP, WHITE, S  # noqa: E402

W, H, FPS, X = 1080, 1920, 30, 78
HOLD = 0.0   # text is static, so no frozen tail: the footage plays to the end
STATIC_TEXT = True  # all text is on screen from the first frame, no entrance animation
BOLD, THIN = 700, 300


# ------------------------------------------------------------------ text
def wrap(text, f, width):
    out, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if f.getlength(t) / S <= width or not line: line = t
        else: out.append(line); line = w
    return out + [line]


def para(sl, x, y, text, size, w, col=WHITE, width=W - 2 * X, lh=1.25):
    f = mont(size, w); rows = text.split("\n") if "\n" in text else wrap(text, f, width)
    for i, r in enumerate(rows): sl.text((x, y + i * size * lh), r, f, col)
    return y + len(rows) * size * lh


def para_h(text, size, w, width=W - 2 * X, lh=1.25):
    rows = text.split("\n") if "\n" in text else wrap(text, mont(size, w), width)
    return len(rows) * size * lh


def heart(sl, cx, cy, r, col=POWDER):
    t = np.linspace(0, 2 * np.pi, 120)
    xs = 16 * np.sin(t) ** 3; ys = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    sl.d.polygon([((cx + x * r / 16) * S, (cy + y * r / 16) * S) for x, y in zip(xs, ys)], fill=col + (255,))


def text_heart(sl, x, y, text, size, w, col=WHITE, width=W - 2 * X, lh=1.25):
    """paragraph that ends with a powder heart (the ❤️ of the copy)"""
    f = mont(size, w); rows = text.split("\n") if "\n" in text else wrap(text, f, width - size)
    for i, r in enumerate(rows): sl.text((x, y + i * size * lh), r, f, col)
    last = rows[-1]; hx = x + f.getlength(last) / S + size * 0.62; hy = y + (len(rows) - 1) * size * lh + size * 0.58
    heart(sl, hx, hy, size * 0.42)
    return y + len(rows) * size * lh


def dots(sl, y=150):
    for i in range(3):
        cx = X + 6 + i * 24; sl.d.ellipse(((cx - 6) * S, (y - 6) * S, (cx + 6) * S, (y + 6) * S), fill=POWDER + (255,))


# ------------------------------------------------------------------ animation
class El:
    def __init__(self, rgba, t0=0.0, kind="rise", d=0.6):
        bb = rgba.getbbox() or (0, 0, 1, 1)
        self.x, self.y = bb[0], bb[1]
        a = np.asarray(rgba.crop(bb)).astype(np.float32) / 255
        self.rgb = a[..., :3] * a[..., 3:]; self.a = a[..., 3:]
        self.t0, self.kind, self.d = t0, kind, d

    def draw(self, fr, t):
        p = (t - self.t0) / self.d
        if p <= 0: return
        e = 1 - (1 - min(p, 1)) ** 3; rgb, a, x, y, k = self.rgb, self.a, self.x, self.y, 1.0
        if self.kind == "rise": y += int(40 * (1 - e)); k = e
        elif self.kind == "fade": k = e
        elif self.kind == "wipe":
            cut = max(1, int(a.shape[1] * e)); rgb, a = rgb[:, :cut], a[:, :cut]
        h, w = a.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x1 <= x0 or y1 <= y0: return
        s_ = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x)); d = fr[y0:y1, x0:x1]
        d *= 1 - a[s_] * k; d += rgb[s_] * k


def E(fn, t0=0.0, kind="rise", d=0.6, halo=True):
    if STATIC_TEXT: t0, kind, d = 0.0, "fade", 1e-3
    sl = Slide(W, H, Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))); fn(sl)
    im = sl.im.resize((W, H), Image.LANCZOS)
    if halo:  # soft dark halo under the type: legible straight on the footage, no black plate
        a = im.split()[3].filter(ImageFilter.GaussianBlur(9)).point(lambda v: int(min(255, v * 1.6) * 0.55))
        sh = Image.new("RGBA", im.size, DEEP + (0,)); sh.putalpha(a)
        sh.alpha_composite(im); im = sh
    return El(im, t0, kind, d)


def shade(top_solid, top_fade, bot_start, bot_a=0.88):
    """black above the footage with a soft fade into it, dark gradient towards the bottom captions"""
    ys = np.arange(H, dtype=np.float32)
    top = 1 - np.clip((ys - top_solid) / max(1, top_fade), 0, 1); top = top * top * (3 - 2 * top)
    bot = np.clip((ys - bot_start) / max(1, H - 120 - bot_start), 0, 1); bot = bot * bot * (3 - 2 * bot) * bot_a
    return np.maximum(top, bot)[:, None, None].astype(np.float32)


def footage(src, start, dur, cx=0.5, cy=0.5, zoom=1.0, slow=1.0):
    """frames of a clip (or a still with a slow push-in) covering 1080x1920"""
    path = os.path.join(SRC, src); n = int(dur * FPS)
    if src.lower().endswith((".jpg", ".jpeg", ".png")):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB"); im.thumbnail((1500, 2700), Image.LANCZOS)
        for i in range(n):
            z = zoom * (1 + 0.06 * i / n); sw, sh = int(W * z), int(H * z)
            s = max(sw / im.width, sh / im.height); r = im.resize((int(im.width * s) + 1, int(im.height * s) + 1), Image.BILINEAR)
            x = int((r.width - W) * cx); y = int((r.height - H) * cy)
            yield np.asarray(r.crop((x, y, x + W, y + H))).astype(np.float32) / 255
        return
    sw, sh = int(W * zoom) // 2 * 2, int(H * zoom) // 2 * 2
    vf = (f"setpts=PTS/{slow},scale={sw}:{sh}:force_original_aspect_ratio=increase,"
          f"crop={W}:{H}:(iw-{W})*{cx}:(ih-{H})*{cy},fps={FPS}")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-stream_loop", "-1", "-ss", f"{start}", "-i", path, "-t", f"{dur / slow + 1}",
                          "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    last = None
    for i in range(n):
        b = p.stdout.read(W * H * 3)
        if len(b) == W * H * 3: last = np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32) / 255
        yield last
    p.kill()


def grade(fr, k=0.25):
    """push the footage towards the brand navy so the stories sit together"""
    g = fr.mean(2, keepdims=True)
    return fr * (1 - k) + (np.float32(NAVY) / 255 * 0.6 + g * 0.55) * k


def render(n, bg, sh, els, dur):
    dur = max(dur, max(e.t0 + e.d for e in els)) + HOLD
    out = os.path.join(OUT, f"{n:02d}.mp4")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    frozen = None; t_freeze = dur - HOLD; last = None
    for i, fr in enumerate(bg(dur)):
        t = i / FPS
        if t >= t_freeze and frozen is not None:      # the end is fully static: footage stops too
            out_fr = frozen
        else:
            f = grade(fr) * (1 - sh)
            for e in els: e.draw(f, t)
            out_fr = (np.clip(f, 0, 1) * 255).astype(np.uint8)
            if t + 1 / FPS >= t_freeze: frozen = out_fr
        p.stdin.write(out_fr.tobytes()); last = out_fr
    p.stdin.close(); p.wait()
    Image.fromarray(last).save(os.path.join(OUT, f"{n:02d}_last.jpg"), quality=92)
    print("done", n, flush=True)


def head(lines, size=76, lh=84, y=210):
    while max(mont(size, BOLD).getlength(l) / S for l in lines) > W - 2 * X - 10:  # never past the right margin
        size -= 2; lh = int(size * 1.1)
    def f(sl):
        dots(sl)
        for i, ln in enumerate(lines): sl.text((X, y + i * lh), ln, mont(size, BOLD), WHITE)
    return f, y + len(lines) * lh


def bottom_block(lead, body, lead_size=42, body_size=38, gap=22, y_end=1680):
    hl = para_h(lead, lead_size, BOLD) if lead else 0; hb = para_h(body, body_size, THIN) if body else 0
    y0 = y_end - hl - (gap if lead and body else 0) - hb
    return y0, y0 + hl + (gap if lead and body else 0)


# ------------------------------------------------------------------ stories
def veil(top_end=760, bot_start=1150, top_a=0.6, bot_a=0.75, base=0.12):
    """light gradients only: the footage stays visible under the text"""
    ys = np.arange(H, dtype=np.float32)
    top = np.clip(1 - ys / top_end, 0, 1) ** 1.4 * top_a
    bot = np.clip((ys - bot_start) / (H - bot_start), 0, 1) ** 1.2 * bot_a
    return (base + (1 - base) * np.maximum(top, bot))[:, None, None].astype(np.float32)


def headline(rows, y=230, size=76):
    while max(mont(size, BOLD).getlength(l) / S for l in rows) > W - 2 * X - 60: size -= 2
    lh = int(size * 1.1)
    return size, lh, y + len(rows) * lh


def st1():
    rows = ["«ХОЧУ УВЕЛИЧИТЬ", "ГРУДЬ, НО ИМПЛАНТЫ", "НЕ ХОЧУ…»"]; size, lh, hy = headline(rows)
    t1 = "После родов грудь потеряла объём. Или после похудения стала совсем другой."
    t2 = "Хочется вернуть наполненность, но мысль об имплантах останавливает."
    y0, y1 = bottom_block(t1, t2, 40, 38)
    els = [E(lambda sl: (dots(sl, 170), [sl.text((X, 230 + i * lh), r, mont(size, BOLD), WHITE) for i, r in enumerate(rows)]), 0.3, "rise", 0.7),
           E(lambda sl: para(sl, X, hy + 30, "Знакомая мысль?", 44, THIN, POWDER), 1.6),
           E(lambda sl: para(sl, X, y0, t1, 40, BOLD), 3.0),
           E(lambda sl: para(sl, X, y1, t2, 38, THIN), 4.6)]
    render(1, lambda d: footage(VID_OR, 6.0, d, cx=0.55, cy=0.35), veil(820, 1080), els, 7.0)


def case_card(sl, im, box, r=20):
    x0, y0, x1, y1 = box; w, h = int((x1 - x0) * S), int((y1 - y0) * S)
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=r * S, fill=255)
    s_ = max(w / im.width, h / im.height); r_ = im.resize((int(im.width * s_) + 1, int(im.height * s_) + 1), Image.LANCZOS)
    cx, cy = (r_.width - w) // 2, (r_.height - h) // 2
    sl.im.paste(r_.crop((cx, cy, cx + w, cy + h)), (int(x0 * S), int(y0 * S)), m)


def pill(sl, x, y, text, bg, fg, size=24):
    f = mont(size, 600); tw = f.getlength(text) / S; ph = size + 20
    sl.rrect((x, y, x + tw + 30, y + ph), ph / 2, bg, 235); sl.text((x + 15, y + ph / 2), text, f, fg, "lm")


def st2():
    """the proof right after the claim: three real lipofilling cases, before / after"""
    rows = ["А ВЕДЬ УВЕЛИЧИТЬ", "ГРУДЬ МОЖНО И", "СОБСТВЕННЫМ ЖИРОМ"]; size, lh, hy = headline(rows)
    def hf(sl):
        dots(sl, 170)
        for i, r in enumerate(rows): sl.text((X, 230 + i * lh), r, mont(size, BOLD), WHITE)
        heart(sl, X + mont(size, BOLD).getlength(rows[-1]) / S + 40, 230 + 2 * lh + size * 0.58, size * 0.4)
    els = [E(hf)]
    gap = 18; cw = (W - 2 * 60 - gap) // 2; ch = int(cw * 355 / 490); y = int(hy + 70)
    for i in range(3):
        b = ImageOps.exif_transpose(Image.open(os.path.join(SRC, f"case{i + 1}_b.jpg"))).convert("RGB")
        a = ImageOps.exif_transpose(Image.open(os.path.join(SRC, f"case{i + 1}_a.jpg"))).convert("RGB")
        yy = y + i * (ch + 74)
        def row(sl, b=b, a=a, yy=yy, i=i):
            sl.text((60, yy), f"Кейс 0{i + 1}", mont(28, 600), POWDER)
            bb = (60, yy + 44, 60 + cw, yy + 44 + ch); ab = (60 + cw + gap, yy + 44, 60 + 2 * cw + gap, yy + 44 + ch)
            case_card(sl, b, bb); case_card(sl, a, ab)
            pill(sl, bb[0] + 12, bb[1] + 12, "до", NAVY, WHITE, 22); pill(sl, ab[0] + 12, ab[1] + 12, "после", POWDER, NAVY, 22)
        els.append(E(row, halo=False))
    render(2, lambda d: footage(VID_OR, 0.2, d, cx=0.5, cy=0.4), veil(H, H, 0.0, 0.0, 0.72), els, 7.0)


def st3():
    rows = ["НО ЗДЕСЬ ВОЗНИКАЕТ", "ДРУГОЙ ВОПРОС:"]; size, lh, hy = headline(rows)
    qs = ["А хватит ли своего жира?", "Какой объём получится?", "И что лучше выбрать именно мне?"]
    kinds = ["question", "search", "chart"]
    lead = "Ответ зависит от вашей анатомии и желаемого результата."
    els = [E(lambda sl: (dots(sl, 170), [sl.text((X, 230 + i * lh), r, mont(size, BOLD), WHITE) for i, r in enumerate(rows)]))]
    yq = hy + 110
    for i, (q, k) in enumerate(zip(qs, kinds)):
        yy = yq + i * 150
        els.append(E(lambda sl, k=k, yy=yy: (sl.circle((X + 44, yy + 44), 44, NAVY, 0, 255, fill=NAVY + (200,)),
                                             icon_circle(sl, (X + 44, yy + 44), k, 44)), halo=False))
        els.append(E(lambda sl, q=q, yy=yy: para(sl, X + 120, yy + 14, q, 40, BOLD, WHITE, W - 2 * X - 120)))
    y0, _ = bottom_block(lead, None, 42)
    els.append(E(lambda sl: (sl.poly([(X, y0 - 36), (X + 120, y0 - 36)], POWDER, 2.4), para(sl, X, y0, lead, 42, BOLD))))
    render(3, lambda d: footage(VID_HANDS, 1.0, d, cx=0.5, cy=0.5), veil(1100, 1250, 0.68, 0.8), els, 7.0)


def st4():
    rows = ["ЛИПОФИЛИНГ", "ИЛИ ИМПЛАНТЫ?"]; size, lh, hy = headline(rows, size=92)
    t1 = "Чтобы вам было проще разобраться, я подготовила небольшой тест"
    t2 = "Он поможет понять различия между методами и какой вариант стоит обсудить на консультации."
    y1 = 1080
    els = [E(lambda sl: (dots(sl, 170), [sl.text((X, 230 + i * lh), r, mont(size, BOLD), WHITE) for i, r in enumerate(rows)]), 0.3, "rise", 0.7),
           E(lambda sl: text_heart(sl, X, y1, t1, 42, BOLD), 1.5),
           E(lambda sl: para(sl, X, y1 + para_h(t1, 42, BOLD, W - 2 * X - 42) + 22, t2, 38, THIN), 2.8)]
    def cta(sl):
        f = mont(36, 400); sl.text((W / 2, 1500), "Пройти тест можно в актуальном", f, WHITE, "mm")
        fb = mont(46, BOLD); bt = "«ТЕСТ»"; bw = fb.getlength(bt) / S
        sl.rrect((W / 2 - bw / 2 - 60, 1550, W / 2 + bw / 2 + 60, 1650), 50, POWDER)
        sl.text((W / 2, 1600), bt, fb, NAVY, "mm")
    els.append(E(cta, 4.2, "rise", 0.7))
    els.append(E(lambda sl: logo(sl, 1760, POWDER, 0.9, 0.85), 4.6, "fade"))
    render(4, lambda d: footage(PHOTO, 0, d, cx=0.5, cy=0.12, zoom=1.05), veil(640, 980, 0.6, 0.85), els, 6.5)


STORIES = [st1, st2, st3, st4]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i, fn in enumerate(STORIES, 1):
        if not ONLY or i in ONLY: fn()
