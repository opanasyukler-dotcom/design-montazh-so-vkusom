# Epic acro rock'n'roll reel: beat-cut montage at 128 BPM, speed ramps into every throw,
# hero throw (flip, 44.2 s) = slow-ramp -> white flash + impact frames + freeze + 0.22x slow-mo on the drop.
# usage: python3 reel_acro_throw.py SRC.mp4 MUSIC.wav OUT.mp4
import numpy as np, cv2, subprocess, sys
from scipy.interpolate import PchipInterpolator
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC, MUS, OUT = sys.argv[1:4]
FONT = "/home/user/design-montazh-so-vkusom/fonts/Montserrat-VariableFont_wght.ttf"
W, H, FPS, SFPS = 1080, 1920, 30, 30.0
B = 60 / 128; NB = 74; NF = int(round(NB * B * FPS))

# (beat_in, beat_out, [(beat, src_sec)...], cx, cy, zoom)  -- keys form a PCHIP time-remap curve
EDL = [
    (0, 4,   [(0, 8.3), (4, 9.6)],                                   .50, .45, 1.45),
    (4, 8,   [(4, 10.0), (8, 11.3)],                                 .45, .45, 1.55),
    (8, 10,  [(8, 13.0), (10, 13.94)],                               .42, .45, 1.6),
    (10, 12, [(10, 14.6), (12, 15.54)],                              .50, .45, 1.6),
    (12, 14, [(12, 16.6), (14, 17.54)],                              .58, .45, 1.5),
    (14, 16, [(14, 21.0), (16, 21.94)],                              .48, .45, 1.3),
    (16, 18, [(16, 23.0), (18, 23.94)],                              .50, .47, 1.5),
    (18, 20, [(18, 27.9), (20, 28.84)],                              .50, .45, 1.45),
    (20, 24, [(20, 29.85), (22, 30.95), (23, 31.12), (24, 31.6)],    .50, .38, 1.3),   # throw A
    (24, 25, [(24, 33.0), (25, 33.47)],                              .60, .45, 1.5),
    (25, 26, [(25, 35.1), (26, 35.57)],                              .52, .45, 1.5),
    (26, 27, [(26, 38.7), (27, 39.17)],                              .55, .45, 1.4),
    (27, 28, [(27, 41.1), (28, 41.57)],                              .55, .45, 1.4),
    (28, 40, [(28, 42.75), (31, 44.05), (32, 44.25), (33, 44.2501), (37, 44.64), (39, 45.55), (40, 46.0)], .43, .37, 1.3),  # HERO
    (40, 42, [(40, 48.4), (42, 49.34)],                              .55, .45, 1.5),
    (42, 46, [(42, 55.15), (44, 56.05), (45, 56.25), (46, 56.75)],   .50, .38, 1.3),   # throw C (split)
    (46, 48, [(46, 59.9), (48, 60.84)],                              .45, .42, 1.4),
    (48, 50, [(48, 64.0), (50, 64.94)],                              .50, .45, 1.5),
    (50, 52, [(50, 70.9), (52, 71.84)],                              .48, .45, 1.5),
    (52, 56, [(52, 76.45), (54, 77.3), (55, 77.48), (56, 77.95)],    .46, .36, 1.3),   # throw D (flip)
    (56, 58, [(56, 81.2), (58, 82.14)],                              .55, .42, 1.45),
    (58, 60, [(58, 99.8), (60, 100.74)],                             .45, .50, 1.5),
    (60, 62, [(60, 104.3), (62, 105.24)],                            .42, .45, 1.4),
    (62, 64, [(62, 108.3), (64, 109.24)],                            .48, .45, 1.45),
    (64, 68, [(64, 112.0), (68, 113.4)],                             .52, .43, 1.4),
    (68, 74, [(68, 44.25), (74, 44.2501)],                            .43, .37, 1.3),   # callback freeze + title
]
HITS = {8: .35, 22: .7, 32: 1.0, 44: .7, 54: .7, 68: .9}          # beat: strength (flash/punch/shake/RGB split)
HERO, FREEZE_END = 32, 33

# ---------- source frames (decoded per clip, autorotated to 1080x1920) ----------
def load(a, b):
    a = max(0, a - 2 / SFPS); n = int((b - a) * SFPS) + 4
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-i", SRC, "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True)
    fr = np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3); return a, fr
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
def interp(f0, f1, a):                                              # optical-flow in-between for smooth slow-mo
    s = 0.5; g0 = cv2.cvtColor(cv2.resize(f0, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(cv2.resize(f1, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
    fw = cv2.resize(DIS.calc(g0, g1, None), (W, H)) / s; bw = cv2.resize(DIS.calc(g1, g0, None), (W, H)) / s
    a = np.float32(a); gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    w0 = cv2.remap(f0, gx - fw[..., 0] * a, gy - fw[..., 1] * a, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    w1 = cv2.remap(f1, gx - bw[..., 0] * (1 - a), gy - bw[..., 1] * (1 - a), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(w0, 1 - a, w1, a, 0)

# ---------- look ----------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.55 * np.clip(((xx / W - .5) ** 2 / .32 + (yy / H - .48) ** 2 / .42), 0, 1) ** 1.3)[..., None]
lut = np.clip(255 * (1 / (1 + np.exp(-(np.arange(256) / 255 - .5) * 6.5)) - 1 / (1 + np.exp(3.25))) / (1 / (1 + np.exp(-3.25)) - 1 / (1 + np.exp(3.25))), 0, 255).astype(np.uint8)
def grade(img, sat=1.25):
    img = cv2.LUT(img, lut); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255); img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    img[..., 0] = img[..., 0] * 1.04 + 4; img[..., 2] = img[..., 2] * 1.02                    # cool shadows, warm skin
    return img * VIG
def rgb_split(img, k):
    if k < 0.5: return img
    M = lambda dx, sc: cv2.warpAffine(img, np.float32([[sc, 0, (1 - sc) * W / 2 + dx], [0, sc, (1 - sc) * H / 2]]), (W, H), borderMode=cv2.BORDER_REFLECT)
    out = img.copy(); out[..., 2] = M(k, 1 + k / 900)[..., 2]; out[..., 0] = M(-k, 1 - k / 900)[..., 0]; return out
def zoom_blur(img, k):
    if k < 0.02: return img
    acc = img.copy(); n = 6
    for i in range(1, n):
        sc = 1 + k * 0.06 * i; acc += cv2.warpAffine(img, np.float32([[sc, 0, (1 - sc) * W / 2], [0, sc, (1 - sc) * H * .4]]), (W, H), borderMode=cv2.BORDER_REFLECT)
    return acc / n
def speed_lines(alpha, seed):
    r = np.random.RandomState(seed); im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im); cx, cy = W * .45, H * .3
    for _ in range(90):
        a = r.rand() * 2 * np.pi; r0 = 680 + r.rand() * 320; r1 = 1900; wd = r.rand() * 0.012 + 0.003
        pts = [(cx + np.cos(a) * r1, cy + np.sin(a) * r1), (cx + np.cos(a - wd) * r0, cy + np.sin(a - wd) * r0), (cx + np.cos(a + wd) * r1, cy + np.sin(a + wd) * r1)]
        d.polygon(pts, fill=255)
    return (np.asarray(im.filter(ImageFilter.GaussianBlur(2)), np.float32) / 255 * alpha)[..., None]

# ---------- typography ----------
def font(sz, w="Black"):
    f = ImageFont.truetype(FONT, sz)
    try: f.set_variation_by_name(w)
    except Exception: pass
    return f
def text_layer(lines, y0, tracking=6):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im); y = y0
    for txt, sz, wgt, col in lines:
        f = font(sz, wgt); cw = [d.textlength(c, font=f) + tracking for c in txt]; x = (W - sum(cw) + tracking) / 2
        for c, w_ in zip(txt, cw): d.text((x, y), c, font=f, fill=col); x += w_
        y += int(sz * 1.12)
    glow = im.filter(ImageFilter.GaussianBlur(18)); sh = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sh.paste((0, 0, 0, 170), mask=im.split()[3].filter(ImageFilter.GaussianBlur(10)))
    out = Image.alpha_composite(sh, glow); out = Image.alpha_composite(out, glow); out = Image.alpha_composite(out, im)
    return np.asarray(out, np.float32) / 255
SL = [speed_lines(0.45, k) for k in range(3)]
WHITE, MINT = (255, 255, 255, 255), (120, 255, 205, 255)
T_INTRO1 = text_layer([("ЧЕМПИОНАТ", 118, "Black", WHITE), ("МОСКВЫ", 150, "Black", WHITE)], 1180)
T_INTRO2 = text_layer([("АКРОБАТИЧЕСКИЙ", 74, "ExtraBold", MINT), ("РОК-Н-РОЛЛ", 132, "Black", WHITE)], 1220)
T_END = text_layer([("ВЫШЕ.", 170, "Black", WHITE), ("СМЕЛЕЕ.", 170, "Black", MINT), ("ЯРЧЕ.", 170, "Black", WHITE)], 1100)
def comp_text(img, layer, t, t0, t1, slam=True):
    if not (t0 <= t < t1): return img
    a = min(1, (t - t0) / 0.12) * min(1, (t1 - t) / 0.25); sc = 1 + 0.25 * max(0, 1 - (t - t0) / 0.18) ** 2 if slam else 1 + 0.03 * (t - t0)
    L = layer if abs(sc - 1) < 1e-3 else cv2.warpAffine(layer, np.float32([[sc, 0, (1 - sc) * W / 2], [0, sc, (1 - sc) * H * .68]]), (W, H))
    al = L[..., 3:4] * a; return img * (1 - al) + L[..., [2, 1, 0]] * 255 * al

# ---------- render ----------
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", MUS,
                       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-profile:v", "high",
                       "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
rng = np.random.RandomState(3); prev_ci = -1; cache = {}
for fi in range(NF):
    t = fi / FPS; bt = t / B
    ci = next(i for i, c in enumerate(EDL) if c[0] <= bt < c[1]); b0, b1, keys, cx, cy, z0 = EDL[ci]
    if ci != prev_ci:
        kb = np.array([k[0] for k in keys]) * B; ks = np.array([k[1] for k in keys])
        curve = PchipInterpolator(kb, ks); a0, frames = load(ks.min(), ks.max() + 0.1); cache.clear(); prev_ci = ci
    st = float(curve(t)); fpos = (st - a0) * SFPS; i0 = int(np.floor(fpos)); a = fpos - i0
    i0 = min(max(i0, 0), len(frames) - 2)
    speed = abs(float(curve(t + 0.5 / FPS)) - float(curve(t - 0.5 / FPS))) * FPS
    if speed < 0.85 and 0.08 < a < 0.92:
        key = (i0, round(a, 3)); src = cache.get(key)
        if src is None: src = cache[key] = interp(frames[i0], frames[i0 + 1], a)
    else: src = frames[min(i0 + (a > .5), len(frames) - 1)]

    # camera: base zoom + slow push + cut punch + hit punch, shake on hits
    lt = t - b0 * B; dur = (b1 - b0) * B; z = z0 * (1 + 0.06 * lt / dur) * (1 + 0.07 * np.exp(-lt / 0.12))
    hk = 0.0; shake = 0.0
    for hb, s in HITS.items():
        dt = t - hb * B
        if 0 <= dt < 1.5: hk = max(hk, s * np.exp(-dt / 0.18)); shake = max(shake, s * np.exp(-dt / 0.25))
    if b0 == 28 and HERO * B <= t < FREEZE_END * B: z *= 1 + 0.12 * (1 - np.exp(-(t - HERO * B) / 0.15))            # punch-in & hold on freeze
    if b0 == 28 and t >= FREEZE_END * B: z *= 1.12 - 0.08 * min(1, (t - FREEZE_END * B) / 1.5)
    if b0 == 68: z *= 1 + 0.10 * lt / dur
    z *= 1 + 0.10 * hk
    ox, oy = (rng.randn(2) * 26 * shake) if shake > 0.03 else (0, 0)
    sc = z; tx = W / 2 - cx * W * sc + ox; ty = H / 2 - cy * H * sc + oy
    tx = min(0, max(W - W * sc, tx)); ty = min(0, max(H - H * sc, ty))
    img = cv2.warpAffine(src, np.float32([[sc, 0, tx], [0, sc, ty]]), (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if sc > 1.25: img = cv2.addWeighted(img, 1.35, cv2.GaussianBlur(img, (0, 0), 2.2), -0.35, 0)                    # recover upscale softness
    hero_dt = t - HERO * B
    img = grade(img, 1.3 if b0 in (28, 68) else 1.22)
    if 0 <= hero_dt < FREEZE_END * B - HERO * B:                                                            # the WOW moment
        fk = int(hero_dt * FPS)
        if fk in (1, 2, 4):                                                                                 # impact frames: inverted B/W
            g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            g = 255 - np.clip((g - 90) * 2.2 + 128, 0, 255) if fk != 4 else np.clip((g - 90) * 2.2 + 128, 0, 255); img = np.repeat(g[..., None], 3, 2)
        img = img * (1 - SL[fk % 3] * np.exp(-hero_dt / 0.22)) + 255 * SL[fk % 3] * np.exp(-hero_dt / 0.22)
    if hk > 0.2: img = zoom_blur(img, (hk - 0.2) * (1.0 if abs(hero_dt) < 1 else 0.6))
    if hk > 0.08: img = rgb_split(img, 18 * hk)
    for hb, s in HITS.items():                                                                              # white flash
        dt = t - hb * B
        if 0 <= dt < 0.4: img = img + (255 - img) * (s * np.exp(-dt / 0.06))
    if 31 * B <= t < HERO * B: img *= 1 - 0.35 * (t / B - 31)                                               # breath: dip before the drop
    # text
    img = comp_text(img, T_INTRO1, t, 0.25, 4 * B - 0.05)
    img = comp_text(img, T_INTRO2, t, 4 * B, 8 * B)
    img = comp_text(img, T_END, t, 68 * B + 0.15, NB * B)
    img = img + rng.randn(H // 4, W // 4, 1).repeat(4, 0).repeat(4, 1) * 3                                  # light grain
    if t < 0.4: img *= t / 0.4
    if bt > 72: img *= max(0, 1 - (bt - 72) / 2)
    ff.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())
    if fi % 60 == 0: print(f"{fi}/{NF}", flush=True)
ff.stdin.close(); ff.wait(); print("done", OUT)
