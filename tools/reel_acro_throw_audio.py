# Synth score for the acro rock'n'roll reel: 128 BPM, F minor, hybrid trailer/EDM.
# Grid shared with reel_acro_throw.py: drop + mega impact on the hero throw (beat 32).
import numpy as np, wave, sys
from scipy.signal import butter, sosfilt, fftconvolve

SR = 48000; BPM = 128; B = 60 / BPM; NB = 74; DUR = NB * B + 2.5; N = int(DUR * SR)
HITS = {22: 0.6, 32: 1.0, 44: 0.6, 54: 0.6, 68: 0.9}             # beat: impact strength (throws + final)
rng = np.random.RandomState(7)
T = lambda b: int(round(b * B * SR))
def lp(x, f, o=2): return sosfilt(butter(o, f, "low", fs=SR, output="sos"), x)
def hp(x, f, o=2): return sosfilt(butter(o, f, "high", fs=SR, output="sos"), x)
def bp(x, a, b, o=2): return sosfilt(butter(o, [a, b], "band", fs=SR, output="sos"), x)
def tt(d): return np.arange(int(d * SR)) / SR
def saw(f, d, ph=0.0): return 2 * ((f * tt(d) + ph) % 1) - 1
def hz(m): return 440 * 2 ** ((m - 69) / 12)
def put(buf, s, b, g=1.0):
    i = T(b); s = s[: max(0, len(buf) - i)]; buf[i:i + len(s)] += s * g

# ---------- instruments ----------
def kick(big=False):
    k = tt(0.6 if big else 0.35); f = 45 + 120 * np.exp(-k * 28); ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-k * (5 if big else 9)); s[:200] += rng.randn(200) * np.linspace(0.6, 0, 200)
    return np.tanh(s * 1.8)
def clap():
    k = tt(0.35); nz = bp(rng.randn(len(k)), 900, 5000); env = np.zeros(len(k))
    for o in (0, 0.011, 0.022): i = int(o * SR); env[i:] += np.exp(-(k[: len(k) - i]) * 30)
    return nz * env * 0.8 + np.sin(2 * np.pi * 190 * k) * np.exp(-k * 25) * 0.4
def snare():
    k = tt(0.22); return bp(rng.randn(len(k)), 1500, 8000) * np.exp(-k * 22) + np.sin(2 * np.pi * 200 * k) * np.exp(-k * 30) * 0.5
def hat(o=False):
    k = tt(0.25 if o else 0.05); return hp(rng.randn(len(k)), 7000, 4) * np.exp(-k * (12 if o else 80))
def tom(f=90):
    k = tt(0.7); ph = 2 * np.pi * np.cumsum(f * (1 + 0.6 * np.exp(-k * 20))) / SR
    return np.tanh(2.5 * np.sin(ph) * np.exp(-k * 4)) + bp(rng.randn(len(k)), 300, 3000) * np.exp(-k * 18) * 0.3
def boom():                                    # cinematic impact: sub drop + body + noise blast
    k = tt(4.0); f = 32 + 90 * np.exp(-k * 6); s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-k * 1.1)
    s += lp(rng.randn(len(k)), 1800) * np.exp(-k * 5) * 0.9 + hp(rng.randn(len(k)), 3000) * np.exp(-k * 3) * 0.25
    return np.tanh(s * 1.6)
def crash():
    k = tt(3.0); return hp(rng.randn(len(k)), 4000, 2) * np.exp(-k * 1.6) * 0.6
def braam(notes, d, bright=1.0):              # massive brass-like saw stack with opening filter
    k = tt(d); s = np.zeros(len(k))
    for m in notes:
        for dt in (-0.12, 0, 0.12): s += saw(hz(m + dt), d, rng.rand())
    s /= len(notes) * 3; env = np.minimum(1, k / 0.03) * np.exp(-k * 0.7)
    cut = 200 + 2600 * bright * np.minimum(1, k / 0.25) * np.exp(-k * 0.8)
    out = np.zeros(len(k)); blk = 512                                   # block-wise time-varying lowpass
    for i in range(0, len(k), blk):
        out[i:i + blk] = lp(s[max(0, i - 4096):i + blk], max(60, cut[i]))[-len(s[i:i + blk]):]
    return np.tanh(out * env * 2.5)
def supersaw(notes, d, cut):
    k = tt(d); s = np.zeros(len(k))
    for m in notes:
        for dt in np.linspace(-0.18, 0.18, 7): s += saw(hz(m + dt), d, rng.rand())
    return lp(s / (len(notes) * 7), cut, 2) * np.minimum(1, k / 0.01) * np.minimum(1, (d - k) / 0.03 + 1e-3)
def pad(notes, d):
    k = tt(d); s = sum(np.sin(2 * np.pi * hz(m) * k) + 0.3 * np.sin(2 * np.pi * hz(m + 12.07) * k) for m in notes)
    return s / len(notes) * np.minimum(1, k / 0.8) * np.minimum(1, (d - k) / 0.4)
def bass(m, d):
    k = tt(d); f = hz(m); s = np.sin(2 * np.pi * f * k) + 0.5 * lp(saw(f, d), 400)
    return np.tanh(1.6 * s) * np.minimum(1, k / 0.005) * np.exp(-k * 2.5)
def riser(d, f0=200, f1=6000):
    k = tt(d); x = k / d; nz = rng.randn(len(k)); out = np.zeros(len(k)); blk = 1024
    for i in range(0, len(k), blk):
        c = f0 * (f1 / f0) ** x[i]; out[i:i + blk] = bp(nz[max(0, i - 4096):i + blk], c * 0.7, min(c * 1.4, 20000))[-len(nz[i:i + blk]):]
    tone = np.sin(2 * np.pi * np.cumsum(110 * 2 ** (3 * x)) / SR) * 0.3
    return (out * 1.5 + tone) * x ** 2
def whoosh(d=0.45, rev=False):
    k = tt(d); x = k / d; e = np.sin(np.pi * x) ** 2; nz = bp(rng.randn(len(k)), 400, 4000) * e
    return nz[::-1] if rev else nz
def revcym(d):
    c = crash(); c = c[: int(d * SR)]; return c[::-1] * 1.4

CH = [[53, 56, 60], [49, 53, 56], [56, 60, 63], [51, 55, 58]]      # Fm  Db  Ab  Eb (one bar each)
RT = [41, 37, 44, 39]
def bar_of(b): return int(b // 4) % 4

# ---------- arrangement ----------
dr = np.zeros(N); ba = np.zeros(N); sy = np.zeros(N); fx = np.zeros(N); kicks = []
def K(b, big=False, g=1.0): put(dr, kick(big), b, g); kicks.append(b)

# intro 0-8: pad + ticking + riser
for c in range(2): put(sy, pad([m - 12 for m in CH[c]] + [RT[c]], 4 * B), c * 4, 0.35)
for b in np.arange(0, 8, 0.5): put(dr, hat(), b, 0.12 + 0.1 * (b / 8))
put(fx, riser(7 * B, 150, 5000), 1, 0.35); put(fx, revcym(B * 1.5), 6.5, 0.35)
put(fx, boom(), 0, 0.25)

def groove(b0, b1, full=False, gap=()):
    for b in np.arange(b0, b1, 1):
        if any(g0 <= b < g1 for g0, g1 in gap): continue
        K(b); put(dr, hat(True), b + 0.5, 0.18)
        for s in (0.25, 0.75): put(dr, hat(), b + s, 0.09)
        if b % 2 == 1: put(dr, clap(), b, 0.45)
        c = bar_of(b); put(ba, bass(RT[c], B * 0.45), b + 0.5, 0.55)
        if full: put(ba, bass(RT[c] + 12, B * 0.2), b + 0.75, 0.25)
        put(sy, supersaw(CH[c], B * 0.42, 5200 if full else 1800), b + 0.5, 0.42 if full else 0.3)
        if full and b % 4 == 3: put(sy, supersaw([m + 12 for m in CH[c]], B * 0.2, 6000), b + 0.25, 0.2)

groove(8, 24, gap=[(22, 23)])
# build 24-31: kicks + accelerating snare roll, riser, filter lifts; 31-32 breath
for b in range(24, 31): K(b, g=0.9); put(sy, supersaw(CH[bar_of(b)], B * 0.9, 900 + (b - 24) * 450), b, 0.25)
roll = list(np.arange(24, 27, 0.5)) + list(np.arange(27, 29, 0.25)) + list(np.arange(29, 31, 0.125))
for i, b in enumerate(roll): put(dr, snare(), b, 0.15 + 0.4 * i / len(roll))
put(fx, riser(7 * B, 120, 9000), 24, 0.6); put(fx, revcym(B * 1.0), 31, 0.5); put(fx, whoosh(B, rev=True), 31, 0.4)

# hero drop 32: mega impact, half-time trailer section 32-37, fill 37-40
put(fx, boom(), 32, 1.0); put(fx, crash(), 32, 0.7); put(sy, braam([29, 41, 44, 48], 5 * B), 32, 0.9); K(32, big=True, g=1.2)
K(33.5, g=0.7); K(34, big=True); K(36.5, g=0.8); put(dr, snare() * 1.4, 35, 0.7); put(dr, clap(), 35, 0.7); put(sy, braam([25, 37, 41, 44], 3 * B, 0.8), 35, 0.6)
for i, b in enumerate(np.arange(37, 40, 0.25)): put(dr, tom(110 - 6 * i), b, 0.25 + 0.03 * i)
put(fx, riser(3 * B, 300, 8000), 37, 0.4)

# main drop 40-64 (hits on throws 44 and 54)
groove(40, 64, full=True)
for b in (40, 48, 56): put(fx, crash(), b, 0.4)
for h in (44, 54): put(fx, boom(), h, 0.7); put(fx, crash(), h, 0.5); put(sy, braam([29, 41, 48], 2 * B, 0.7), h, 0.45); put(fx, whoosh(0.5, True), h - 1.07, 0.35)
put(fx, boom(), 22, 0.6); put(fx, crash(), 22, 0.4); put(sy, braam([29, 41, 48], 2 * B, 0.6), 22, 0.45); put(fx, whoosh(0.5, True), 22 - 1.07, 0.3)
put(fx, boom(), 8, 0.4); put(fx, crash(), 8, 0.4)

# outro 64-68 filtered down, 68 final hit + sustained braam tail
for b in range(64, 68): K(b, g=0.8); put(sy, supersaw(CH[bar_of(b)], B * 0.9, 2500 - (b - 64) * 450), b, 0.25)
for i, b in enumerate(np.arange(66, 68, 0.25)): put(dr, snare(), b, 0.2 + 0.05 * i)
put(fx, revcym(B * 2), 66, 0.4)
put(fx, boom(), 68, 0.95); put(fx, crash(), 68, 0.6); K(68, big=True, g=1.2); put(sy, braam([29, 41, 44, 48, 53], 6 * B, 1.0), 68, 0.8)
put(sy, pad([53, 56, 60, 65], 6 * B), 68, 0.3)

# whooshes on the fast cuts of the build
for b in (24, 25, 26, 27): put(fx, whoosh(0.3), b - 0.3 / B, 0.18)

# ---------- mix ----------
sc = np.ones(N)                                            # sidechain from every kick
for b in kicks:
    i = T(b); L = int(0.3 * SR); e = 1 - 0.75 * np.exp(-np.arange(L) / (0.09 * SR)); sc[i:i + L] = np.minimum(sc[i:i + L], e[: len(sc[i:i + L])])
music = (sy + ba * 0.9) * sc
def verb(x, d=1.8, mix=0.25):
    k = tt(d); out = []
    for ch in range(2):
        ir = rng.randn(len(k)) * np.exp(-k * 3.2); ir = lp(ir, 6000); ir /= np.sqrt(np.sum(ir ** 2))
        out.append(x + fftconvolve(x, ir)[: len(x)] * mix)
    return np.stack(out, 1)
gate = np.ones(N); i0, i1 = T(31), T(32); gate[i0:i1] = np.linspace(0.25, 0.05, i1 - i0)   # breath before the drop
mix = (verb(music, mix=0.35) * 0.9 + verb(dr, 1.0, 0.12)) * gate[:, None] + verb(fx, 2.5, 0.3) * 0.9
# stereo width on synths
mix += np.stack([np.roll(sy * sc, 240), sy * sc], 1) * 0.12
tail = np.clip((DUR - np.arange(N) / SR) / 2.0, 0, 1); mix *= tail[:, None]
mix = hp(mix.T, 25).T
mix /= np.percentile(np.abs(mix), 99.9) + 1e-9; mix = np.tanh(mix * 1.1) / np.tanh(1.1) * 0.95
out = sys.argv[1] if len(sys.argv) > 1 else "acro_mix.wav"
with wave.open(out, "wb") as f:
    f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR); f.writeframes((mix * 32767).astype(np.int16).tobytes())
print("wrote", out, DUR)
