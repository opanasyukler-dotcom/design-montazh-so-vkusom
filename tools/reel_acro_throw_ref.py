# Acro rock'n'roll reel in the style of a dark "car edit" reference: tight tilted close-ups, moody selective grade,
# camera-driven transitions (zoom / spin / whip) rendered with shutter motion blur, speed ramps into every throw.
# The hero flip (src 44.25 s) lands on the drop of the reference track (6.96 s): freeze + punch + shake + 0.35x slow-mo.
# usage: python3 reel_acro_throw_ref.py SRC.mp4 MUSIC.wav OUT.mp4
import numpy as np, cv2, subprocess, sys
from scipy.interpolate import PchipInterpolator

SRC, MUS, OUT = sys.argv[1:4]
W, H, FPS, SFPS = 1080, 1920, 60, 30.0
P, T0 = 0.462, 0.025                                   # beat grid of the track: kicks at T0 + P*b, drop at b = 15
DUR = 13.478; NF = int(DUR * FPS)
def tb(b): return T0 + P * b

# (beat_in, beat_out, [(beat, src_sec)...], cx, cy, zoom, tilt_deg, transition_in)
EDL = [
    (0, 4,     [(0, 10.0), (4, 11.4)],                                         .45, .45, 1.75, -6, None),
    (4, 6,     [(4, 13.0), (6, 13.92)],                                        .42, .45, 1.8,   5, "zoom"),
    (6, 8,     [(6, 16.6), (8, 17.52)],                                        .58, .45, 1.7,  -4, "spin"),
    (8, 10,    [(8, 21.0), (10, 21.92)],                                       .48, .45, 1.5,   3, "whip"),
    (10, 12,   [(10, 23.0), (12, 23.92)],                                      .50, .48, 1.7,  -5, "zoom"),
    (12, 13,   [(12, 28.0), (13, 28.46)],                                      .50, .45, 1.6,   4, "spin"),
    (13, 20,   [(13, 43.3), (14.3, 44.08), (15, 44.25), (15.6, 44.2501), (18.5, 44.72), (19.5, 45.2), (20, 45.45)],
                                                                               .43, .37, 1.38,  0, "zoom"),   # HERO
    (20, 22,   [(20, 55.25), (21, 56.05), (21.6, 56.2), (22, 56.55)],          .50, .38, 1.38, -3, "whip"),   # split
    (22, 24,   [(22, 76.5), (23, 77.3), (23.6, 77.45), (24, 77.85)],           .46, .36, 1.38,  3, "spin"),   # flip
    (24, 26,   [(24, 30.05), (25, 30.98), (25.6, 31.12), (26, 31.5)],          .50, .37, 1.38, -2, "zoom"),   # tall throw
    (26, 27,   [(26, 60.0), (27, 60.46)],                                      .45, .42, 1.55,  4, "whip"),
    (27, 28,   [(27, 64.1), (28, 64.56)],                                      .50, .45, 1.7,  -4, "spin"),
    (28, 29.3, [(28, 81.3), (29.3, 81.9)],                                     .55, .42, 1.55,  0, "zoom"),
]
HERO = 15
HITS = {15: 1.0, 21: .45, 23: .45, 25: .45}            # beat: punch strength (throw peaks)

def load(a, b):
    a = max(0, a - 2 / SFPS); n = int((b - a) * SFPS) + 5
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-i", SRC, "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True)
    return a, np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
GX, GY = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
flows = {}
def interp(fr, i, a):                                  # optical-flow in-between frame (smooth 60 fps + slow-mo)
    if i not in flows:
        s = 0.5; g0, g1 = (cv2.cvtColor(cv2.resize(fr[j], None, fx=s, fy=s), cv2.COLOR_BGR2GRAY) for j in (i, i + 1))
        flows.clear(); flows[i] = (cv2.resize(DIS.calc(g0, g1, None), (W, H)) / s, cv2.resize(DIS.calc(g1, g0, None), (W, H)) / s)
    fw, bw = flows[i]; a = np.float32(a)
    w0 = cv2.remap(fr[i], GX - fw[..., 0] * a, GY - fw[..., 1] * a, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    w1 = cv2.remap(fr[i + 1], GX - bw[..., 0] * (1 - a), GY - bw[..., 1] * (1 - a), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(w0, 1 - float(a), w1, float(a), 0)

# ---------- moody grade: crushed blacks, mint costumes + skin kept, everything else desaturated ----------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.72 * np.clip((xx / W - .5) ** 2 / .30 + (yy / H - .45) ** 2 / .38, 0, 1) ** 1.2)[..., None]
x_ = np.arange(256) / 255.0
lut = np.clip(255 * np.clip((x_ - 0.06) / 0.94, 0, 1) ** 1.25, 0, 255)
lut = np.clip(255 * (1 / (1 + np.exp(-(lut / 255 - .5) * 7)) - 1 / (1 + np.exp(3.5))) / (1 / (1 + np.exp(-3.5)) - 1 / (1 + np.exp(3.5))), 0, 255).astype(np.uint8)
hh = np.arange(180, dtype=np.float32)
MINT = np.exp(-((hh - 84) / 13) ** 2).astype(np.float32)                                 # costumes
SKIN = np.exp(-(np.minimum(np.abs(hh - 13), 180 - np.abs(hh - 13)) / 6) ** 2).astype(np.float32)
SKIN_S = np.clip((170 - np.arange(256, dtype=np.float32)) / 60, 0, 1)                      # skin is never as loud as the banner red
def grade(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); s = hsv[..., 1].astype(np.float32)
    s *= 0.34 + 0.95 * np.maximum(MINT[hsv[..., 0]], SKIN[hsv[..., 0]] * SKIN_S[hsv[..., 1]])
    hsv[..., 1] = np.clip(s, 0, 255).astype(np.uint8); img = cv2.LUT(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), lut).astype(np.float32)
    lum = img.mean(2, keepdims=True) / 255; sh = (1 - lum) ** 2
    img += sh * np.float32([10, 4, -6]) + lum ** 2 * np.float32([-6, 2, 8])                 # teal shadows, warm highlights
    return img * VIG

# ---------- camera with transitions ----------
def ease(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)
TIN, TOUT = 0.16, 0.12
def cam(t, ci):
    b0, b1, keys, cx, cy, z, rot, _ = EDL[ci]; a, e = tb(b0) if ci else 0.0, tb(b1) if ci < len(EDL) - 1 else DUR
    lt, dur = t - a, e - a; s, r, ox, oy = z * (1 + 0.05 * lt / dur), rot, 0.0, 0.0
    tr = EDL[ci][7]                                                                       # incoming transition
    if tr and lt < TIN:
        k = (1 - ease(lt / TIN)) ** 2
        if tr == "zoom": s *= 1 + 0.55 * k
        if tr == "spin": r += 28 * k; s *= 1 + 0.2 * k
        if tr == "whip": ox += W * 0.55 * k
    nxt = EDL[ci + 1][7] if ci < len(EDL) - 1 else "zoom"                                 # outgoing half of next transition
    if e - t < TOUT:
        k = ease(1 - (e - t) / TOUT) ** 2
        if nxt == "zoom": s *= 1 + 0.45 * k
        if nxt == "spin": r -= 22 * k; s *= 1 + 0.15 * k
        if nxt == "whip": ox -= W * 0.45 * k
    for hb, st in HITS.items():                                                           # punch-in on throw peaks
        dt = t - tb(hb)
        if 0 <= dt < 1.2: s *= 1 + 0.10 * st * np.exp(-dt / 0.2)
    if b0 == 13 and t >= tb(HERO): s *= 1 + 0.10 * ease((t - tb(HERO)) / 0.25) - 0.06 * ease((t - tb(HERO) - 0.3) / 1.6)
    return s, r, ox, oy
def matrix(s, r, ox, oy, cx, cy):
    th = np.radians(abs(r)); hx = (W / 2 * np.cos(th) + H / 2 * np.sin(th)) / s; hy = (W / 2 * np.sin(th) + H / 2 * np.cos(th)) / s
    px = min(max(cx * W, hx), W - hx) if hx < W / 2 else W / 2; py = min(max(cy * H, hy), H - hy) if hy < H / 2 else H / 2
    M = cv2.getRotationMatrix2D((px, py), r, s); M[0, 2] += W / 2 - px + ox; M[1, 2] += H / 2 - py + oy; return M

# ---------- render ----------
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", MUS,
                       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-profile:v", "high",
                       "-c:a", "aac", "-b:a", "256k", "-af", f"afade=t=out:st={DUR - 0.25}:d=0.25", "-shortest", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
rng = np.random.RandomState(5); prev_ci = -1
for fi in range(NF):
    t = fi / FPS; bt = (t - T0) / P
    ci = max(i for i, c in enumerate(EDL) if c[0] <= max(bt, 0)); b0, b1, keys, cx, cy, z, rot, _ = EDL[ci]
    if ci != prev_ci:
        kb = np.array([tb(k[0]) for k in keys]); ks = np.array([k[1] for k in keys]); curve = PchipInterpolator(kb, ks, extrapolate=True)
        a0, frames = load(ks.min() - 0.1, ks.max() + 0.15); flows.clear(); prev_ci = ci
    fpos = (float(curve(t)) - a0) * SFPS; i0 = min(max(int(np.floor(fpos)), 0), len(frames) - 2); a = fpos - i0
    src = interp(frames, i0, a) if 0.06 < a < 0.94 else frames[i0 + (a >= 0.5)]
    # shutter: average camera poses over 1.5 frame intervals -> zoom / spin / whip blur exactly where the camera moves
    poses = [cam(t - k * 1.5 / FPS / 8, ci) for k in range(8)]; d = abs(poses[0][0] - poses[-1][0]) * 900 + abs(poses[0][1] - poses[-1][1]) * 25 + abs(poses[0][2] - poses[-1][2])
    shake = 0.0
    for hb, st in HITS.items():
        dt = t - tb(hb)
        if 0 <= dt < 1.0: shake = max(shake, st * np.exp(-dt / 0.22))
    ox, oy = rng.randn(2) * 30 * shake if shake > 0.03 else (0, 0)
    if d < 3: poses = poses[:1]
    acc = None
    for s, r, px, py in poses:
        f = cv2.warpAffine(src, matrix(s, r, px + ox, py + oy, cx, cy), (W, H), flags=cv2.INTER_LINEAR if len(poses) > 1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        acc = f if acc is None else acc + f
    img = (acc / len(poses)).astype(np.uint8)
    img = cv2.addWeighted(img, 1.4, cv2.GaussianBlur(img, (0, 0), 2.0), -0.4, 0)                           # crisp detail after upscale
    img = grade(img)
    hd = t - tb(HERO)
    if 0 <= hd < 0.3: img = img + (255 - img) * 0.85 * np.exp(-hd / 0.05)                                   # drop flash
    if 0 <= hd < 0.5:                                                                                     # chromatic kick on the drop
        k = 14 * np.exp(-hd / 0.12)
        img[..., 2] = cv2.warpAffine(img[..., 2], np.float32([[1 + k / 700, 0, -W * k / 1400 + k], [0, 1 + k / 700, -H * k / 1400]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        img[..., 0] = cv2.warpAffine(img[..., 0], np.float32([[1 - k / 700, 0, W * k / 1400 - k], [0, 1 - k / 700, H * k / 1400]]), (W, H), borderMode=cv2.BORDER_REFLECT)
    if tb(14) <= t < tb(HERO): img *= 1 - 0.25 * ease((t - tb(14)) / P)                                    # sink into the drop
    img = img + rng.randn(H // 3, W // 3, 1).repeat(3, 0).repeat(3, 1) * 4
    if t < 0.2: img *= t / 0.2
    if t > DUR - 0.3: img *= max(0, (DUR - t) / 0.3)
    ff.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())
    if fi % 60 == 0: print(f"{fi}/{NF}", flush=True)
ff.stdin.close(); ff.wait(); print("done", OUT)
