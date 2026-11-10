"""Sound for the СДВГ reels: synthesized sound effects (whoosh, pop, swell, boom, bloop, shutter, ding, tick, shimmer, error),
CC0 keyboard clicks from sfx/keyboard, a «studio» voice chain and the final mix (music ducked under the voice).
"""
import glob, os, subprocess
import numpy as np
from scipy.signal import butter, sosfilt

SR = 48000
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

def env(n, a, d):
    t = np.arange(n) / SR; return np.minimum(t / max(a, 1e-4), 1) * np.exp(-np.maximum(t - a, 0) / d)

def bp(x, lo, hi):
    return sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), x)

def whoosh(d=0.38, lo=300, hi=5000, gain=0.5):
    n = int(d * SR); x = rng.normal(0, 1, n); out = np.zeros(n); k = 8
    for i in range(k):                                     # band sweeping up then down, crossfaded slices
        u = i / (k - 1); f = lo * (hi / lo) ** (np.sin(np.pi * u))
        y = bp(x, max(f * 0.5, 60), min(f * 1.6, 20000)); w = np.exp(-((np.arange(n) / n - u) ** 2) / 0.02); out += y * w
    e = np.sin(np.pi * np.arange(n) / n) ** 2
    return out / (np.abs(out).max() + 1e-9) * e * gain

def pop(gain=0.35, f0=1100, f1=320):
    n = int(0.09 * SR); t = np.arange(n) / SR; f = f1 + (f0 - f1) * np.exp(-t / 0.012)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.002, 0.025) * gain

def swell(gain=0.28):
    n = int(0.7 * SR); t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 110 * t) * 0.6 + np.sin(2 * np.pi * 220 * t) * 0.25 + bp(rng.normal(0, 0.3, n), 200, 2500)
    return x * env(n, 0.06, 0.28) * gain

def boom(gain=0.7):
    n = int(1.2 * SR); t = np.arange(n) / SR; f = 42 + 50 * np.exp(-t / 0.08)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) + bp(rng.normal(0, 0.5, n), 60, 900) * np.exp(-t / 0.05)
    return x * env(n, 0.003, 0.45) * gain

def bloop(gain=0.3):
    a = np.sin(2 * np.pi * 660 * np.arange(int(0.06 * SR)) / SR) * env(int(0.06 * SR), 0.003, 0.03)
    b = np.sin(2 * np.pi * 990 * np.arange(int(0.09 * SR)) / SR) * env(int(0.09 * SR), 0.003, 0.04)
    return np.concatenate([a, np.zeros(int(0.015 * SR)), b]) * gain

def shutter(gain=0.4):
    c = lambda d: bp(rng.normal(0, 1, int(d * SR)), 1500, 9000) * env(int(d * SR), 0.001, 0.012)
    return np.concatenate([c(0.04), np.zeros(int(0.05 * SR)), c(0.06) * 0.8]) * gain

def ding(gain=0.3):
    n = int(1.4 * SR); t = np.arange(n) / SR
    x = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / d) for f, a, d in ((1318.5, 1, 0.6), (2637, 0.5, 0.35), (3951, 0.25, 0.2), (1975.5, 0.3, 0.45)))
    return x * env(n, 0.002, 10) * gain

def tick(gain=0.18, f=2200):
    n = int(0.03 * SR); return np.sin(2 * np.pi * f * np.arange(n) / SR) * env(n, 0.001, 0.008) * gain

def shimmer(gain=0.12, d=1.2):
    n = int(d * SR); t = np.arange(n) / SR; x = np.zeros(n)
    for f in rng.uniform(1800, 4200, 7): x += np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(3, 7) * t))
    return x / 7 * np.sin(np.pi * t / d) ** 2 * gain

def error(gain=0.22):
    a = lambda f, d: np.sign(np.sin(2 * np.pi * f * np.arange(int(d * SR)) / SR)) * 0.4 * env(int(d * SR), 0.003, 0.05)
    return np.concatenate([a(330, 0.09), np.zeros(int(0.02 * SR)), a(247, 0.14)]) * gain

_keys = None
def key(gain=0.35):
    global _keys
    if _keys is None:
        _keys = []
        for f in sorted(glob.glob(os.path.join(HERE, "..", "sfx", "keyboard", "keypress-*.wav"))):
            _keys.append(np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                                                      capture_output=True, check=True).stdout, np.float32))
    k = _keys[rng.integers(len(_keys))]; return k / (np.abs(k).max() + 1e-9) * gain * rng.uniform(0.7, 1.0)

def place(track, t, x):
    i = int(t * SR)
    if i >= len(track) or i < 0: return
    j = min(len(track), i + len(x)); track[i:j] += x[:j - i]

def read(path, ss=0.0, dur=None, ch=2):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(ss)] + (["-t", str(dur)] if dur else []) + ["-i", path, "-ac", str(ch), "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, ch).copy()

def write(path, x):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", str(x.shape[1]), "-i", "-", path],
                   input=np.ascontiguousarray(x, np.float32).tobytes(), check=True)

STUDIO = ("highpass=f=85,lowpass=f=15000,afftdn=nr=14:nf=-48:tn=1,"
          "equalizer=f=240:t=q:w=1.1:g=-2.5,equalizer=f=3400:t=q:w=1.2:g=2.5,equalizer=f=10000:t=q:w=0.9:g=1.5,"
          "deesser=i=0.35,acompressor=threshold=-21dB:ratio=3:attack=6:release=70:makeup=3,"
          "loudnorm=I=-14:TP=-1.5:LRA=8")

def studio_voice(src, runs, gaps, total, out):
    """speech runs (orig times) glued with short silences, then the studio chain; returns nothing, writes out (stereo wav)"""
    v = read(src, ch=1)[:, 0]; parts = []
    for (a, b), g in zip(runs, gaps):
        x = v[int(a * SR):int(b * SR)].copy(); f = int(0.012 * SR); r = np.linspace(0, 1, f)
        x[:f] *= r; x[-f:] *= r[::-1]; parts += [x, np.zeros(int(g * SR), np.float32)]
    y = np.concatenate(parts); y = np.pad(y, (0, max(0, int(total * SR) - len(y))))[:int(total * SR)]
    tmp = out + ".raw.wav"; write(tmp, y[:, None])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", STUDIO + ",aresample=48000", "-ac", "2", "-t", str(total), out], check=True)
    os.remove(tmp)

def mix(voice_wav, music_src, music_ss, sfx, total, out_wav, music_db=-21.0):
    """voice + music (ducked under the voice, faded in/out) + sound effects"""
    v = read(voice_wav); n = len(v)
    if music_src: m = read(music_src, music_ss, total + 1)[:n]; m = np.pad(m, ((0, n - len(m)), (0, 0)))
    else: m = np.zeros_like(v)
    g = 10 ** (music_db / 20)
    e = np.abs(v).mean(1); w = int(0.05 * SR); e = np.convolve(e, np.ones(w) / w, "same")       # voice envelope → ducking
    duck = 1 - 0.45 * np.clip(e / 0.05, 0, 1); k = int(0.25 * SR); duck = np.convolve(duck, np.ones(k) / k, "same")
    t = np.arange(n) / SR; fade = np.clip(t / 0.6, 0, 1) * np.clip((total - t) / 1.6, 0, 1)
    m = m * (g * duck * fade)[:, None]
    s = np.zeros(n); [place(s, tt, x) for tt, x in sfx]
    y = v + m + s[:, None] * 0.9
    peak = np.abs(y).max()
    if peak > 0.89: y = y / peak * 0.89
    tmp = out_wav + ".pre.wav"; write(tmp, y)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", "alimiter=limit=0.84:level=false", out_wav], check=True); os.remove(tmp)
