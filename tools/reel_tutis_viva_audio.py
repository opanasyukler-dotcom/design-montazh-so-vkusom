# audio for reel_tutis_viva: edited voice + synthesized soft pad/e-piano bed (ducked) + light SFX
# usage: python3 reel_tutis_viva_audio.py <work_dir>   (needs voice_clean.wav; writes mix.wav)
import sys, json, subprocess, numpy as np, soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt
WK = sys.argv[1]; SR = 48000
info = json.loads(subprocess.run([sys.executable, __file__.replace("_audio", ""), "-", WK, "info"], capture_output=True, text=True).stdout)
DUR, CUTS, CAPS, END = info["DUR"], info["cuts"], info["caps"], info["end"]
n = int(DUR * SR); rng = np.random.default_rng(7)

# ---- voice ----
v, sr = sf.read(f"{WK}/voice_clean.wav"); assert sr == SR
vo = np.zeros(n)
for a, b, t0 in [(1.45, 39.70, 0.40), (39.90, 41.60, 38.65), (41.85, 74.10, 40.35)]:
    s = v[int(a * SR):int(b * SR)].copy(); f = int(0.015 * SR); s[:f] *= np.linspace(0, 1, f); s[-f:] *= np.linspace(1, 0, f)
    i = int(t0 * SR); vo[i:i + len(s)] += s[:n - i]
# ---- music bed ----
def lp(x, fc): return sosfilt(butter(2, fc, fs=SR, output="sos"), x)
def hp(x, fc): return sosfilt(butter(2, fc, "high", fs=SR, output="sos"), x)
def hz(m): return 440 * 2 ** ((m - 69) / 12)
BPM = 84; beat = 60 / BPM; bar = 4 * beat
CH = [[53, 57, 60, 64], [50, 53, 57, 60, 64], [46, 50, 53, 57], [48, 52, 55, 57]]   # Fmaj7 Dm9 Bbmaj7 C6
t = np.arange(n) / SR
pad = np.zeros(n); keys = np.zeros(n)
cl = 2 * bar; k = 0
while k * cl < DUR:
    ch = CH[k % 4]; i0 = int(k * cl * SR); i1 = min(n, int((k + 1) * cl * SR + 1.5 * SR)); tt = t[i0:i1] - k * cl
    env = np.minimum(1, tt / 1.2) * np.clip((cl + 1.5 - tt) / 1.5, 0, 1)
    for m in ch:
        for det in (-0.07, 0.0, 0.07):
            pad[i0:i1] += env * np.sin(2 * np.pi * hz(m + det) * tt + rng.uniform(0, 6)) / len(ch)
    pad[i0:i1] += 0.6 * env * np.sin(2 * np.pi * hz(ch[0] - 12) * tt)
    arp = [ch[j % len(ch)] + 12 for j in (0, 2, 1, 3, 2, 1, 3, 0)]
    for j in range(16):                                   # gentle e-piano eighths
        if rng.random() < 0.25: continue
        st = k * cl + j * beat / 2; i = int(st * SR)
        if i >= n: break
        L = min(n - i, int(1.6 * SR)); x = np.arange(L) / SR; f0 = hz(arp[j % 8])
        nt = (np.sin(2 * np.pi * f0 * x) + 0.25 * np.sin(2 * np.pi * 2 * f0 * x) * np.exp(-x * 6)) * np.exp(-x * 2.6) * np.minimum(1, x / 0.004)
        keys[i:i + L] += nt * rng.uniform(0.5, 0.8)
    k += 1
mus = lp(pad, 1400) * 0.5 + lp(keys, 3500) * 0.35
ir = rng.standard_normal(int(2.2 * SR)) * np.exp(-np.arange(int(2.2 * SR)) / SR * 2.8); ir = lp(ir, 5000)
mus = mus + 0.25 * fftconvolve(mus, ir)[:n] / np.abs(ir).sum() * 40
mus /= np.abs(mus).max()
mus *= np.minimum(1, t / 1.0) * np.clip((DUR - t) / 1.2, 0, 1)
act = (np.abs(vo) > 0).astype(float)
act = fftconvolve(act, np.ones(int(0.3 * SR)) / int(0.3 * SR), "same"); act = np.clip(act * 3, 0, 1)
act = lp(act, 3)
mus *= 0.16 - 0.10 * np.clip(act, 0, 1)                 # ~-16 dB bed, ~-24 dB under the voice
# ---- sfx ----
sfx = np.zeros(n + 2 * SR)
def put(s, at, g): i = int(max(0, at) * SR); seg = sfx[i:i + len(s)]; seg += s[:len(seg)] * g
def whoosh(d=0.5):
    L = int(d * SR); x = np.linspace(0, 1, L); nz = rng.standard_normal(L)
    out = np.zeros(L); env = np.sin(np.pi * x ** 0.7) ** 2
    for j, f0 in enumerate((500, 1100, 2300, 4500)):    # band sweep: low bands first, highs swell in
        bnd = hp(lp(nz, f0 * 1.6), f0 * 0.6); g = np.exp(-((x - (0.25 + 0.17 * j)) / 0.28) ** 2); out += bnd * g
    return out * env / np.abs(out).max()
def pop():
    L = int(0.12 * SR); x = np.arange(L) / SR; f = 900 * np.exp(-x * 30) + 500
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 45); return s / np.abs(s).max()
def tick():
    L = int(0.05 * SR); x = np.arange(L) / SR; s = hp(rng.standard_normal(L), 2500) * np.exp(-x * 160) + 0.4 * np.sin(2 * np.pi * 2100 * x) * np.exp(-x * 90)
    return s / np.abs(s).max()
def chime(notes, sp=0.09, d=2.5):
    L = int((d + sp * len(notes)) * SR); s = np.zeros(L)
    for j, m in enumerate(notes):
        i = int(j * sp * SR); x = np.arange(L - i) / SR; f0 = hz(m)
        s[i:] += (np.sin(2 * np.pi * f0 * x) + 0.3 * np.sin(2 * np.pi * 3.01 * f0 * x) * np.exp(-x * 4)) * np.exp(-x * 2.2)
    return s / np.abs(s).max()
WH = [whoosh(0.45) for _ in range(4)]; WS = whoosh(0.3)
cap_t = [c[0] for c in CAPS]
for c in CUTS[1:]:
    big = any(abs(c - ct) < 0.05 for ct in cap_t)
    put(WH[rng.integers(4)] if big else WS, c - 0.25, 0.20 if big else 0.08)
for c0, lines in CAPS:
    put(pop(), c0 + 0.1, 0.10)
    for a in lines: put(tick(), a, 0.07)
put(chime([72, 76, 79, 84]), 0.35, 0.10)
put(chime([77, 81, 84, 88, 91], 0.11, 3.0), END + 0.25, 0.14)
# ---- mix ----
mix = vo * 1.0 + mus + sfx[:n] * 0.9
st = np.stack([mix, mix], 1)
st[:, 0] += 0.02 * np.roll(mus, 240); st[:, 1] += 0.02 * np.roll(mus, -240)
st /= max(1, np.abs(st).max() / 0.95)
sf.write(f"{WK}/mix_raw.wav", st, SR)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{WK}/mix_raw.wav", "-af", "loudnorm=I=-14:TP=-1.0:LRA=9", "-ar", "48000", f"{WK}/mix.wav"], check=True)
print("ok", DUR)
