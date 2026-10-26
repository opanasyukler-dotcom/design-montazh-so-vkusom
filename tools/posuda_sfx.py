"""Sound design for the posuda review (posuda_reel.py timeline).
  foley  — the camera's own sound of each shot (box, film, silicone, lids), cleaned, quiet, ducked under the voice
  sfx    — soft synthesized accents: whooshes on section changes, a pop for every pill, a shutter for every gift photo,
           bell sparkles on the «wow» lines
  python3 posuda_sfx.py SRC_DIR voice.wav master.mp4 out.mp4 [--mute-foley i,j]
"""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import posuda_reel as R

SR = 48000
N = int(R.END * SR)

def db(x): return 10 ** (x / 20)

def read(path, ss=None, t=None, af=None):
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", str(ss)] if ss is not None else []) + ["-i", path] + \
          (["-t", str(t)] if t else []) + (["-af", af] if af else []) + ["-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()

def env(n, a, r):
    """raised-cosine attack/release envelope"""
    e = np.ones(n, np.float32); a, r = min(a, n // 2), min(r, n // 2)
    if a: e[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    if r: e[-r:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r))
    return e

def limit(x, ceil_db, rel=0.08):
    """peak limiter: 5 ms look-ahead block peaks, instant attack, smooth release — tames knocks on the table"""
    c = db(ceil_db); b = 240; n = len(x) // b * b
    pk = np.abs(x[:n]).max(1).reshape(-1, b).max(1); g = np.minimum(1, c / (pk + 1e-9))
    a = np.exp(-b / (rel * SR)); sm = np.empty_like(g); cur = 1.0
    for i, v in enumerate(g): cur = v if v < cur else a * cur + (1 - a) * v; sm[i] = cur
    sm = np.minimum(sm, np.r_[sm[1:], sm[-1]])                    # look one block ahead
    x[:n] *= np.repeat(sm, b)[:, None]; return x

def foley(src, mute):
    out = np.zeros((N, 2), np.float32)
    for i, (t, d, c, a, sp, _) in enumerate(R.pieces(src)):         # same jump cuts and speed as the picture
        if c.startswith("F:") or i in mute: continue
        x = read(f"{src}/{c}.MP4", a, d * sp + 0.05, f"highpass=f=90,lowpass=f=14000,afftdn=nf=-42,atempo={sp:.4f}")  # no rumble, no hiss
        rms = np.sqrt(np.mean(x ** 2)) + 1e-9
        x *= min(db(-30) / rms, db(6))                          # level every shot to ~-30 dBFS rms, max +6 dB
        n = min(len(x), N - int(t * SR)); x = x[:n] * env(n, int(0.04 * SR), int(0.06 * SR))[:, None]
        out[int(t * SR):int(t * SR) + n] += x
    return limit(out, -16)

def noise_sweep(dur, f0, f1, gain):
    """whoosh: noise through a moving band-pass (FFT, per 20 ms block) with a swell"""
    n = int(dur * SR); rng = np.random.default_rng(7); x = rng.standard_normal(n).astype(np.float32)
    blk = 960; y = np.zeros(n, np.float32); win = np.hanning(blk * 2).astype(np.float32)
    for k in range(0, n - blk * 2, blk):
        f = f0 * (f1 / f0) ** (k / n); seg = np.fft.rfft(x[k:k + blk * 2] * win)
        fr = np.fft.rfftfreq(blk * 2, 1 / SR); seg *= np.exp(-0.5 * (np.log(fr + 1) - np.log(f)) ** 2 / 0.35)
        y[k:k + blk * 2] += np.fft.irfft(seg).astype(np.float32)
    sw = np.sin(np.linspace(0, np.pi, n)) ** 1.5
    y *= sw / (np.abs(y).max() + 1e-9) * gain
    return np.stack([y * np.linspace(1.15, 0.85, n), y * np.linspace(0.85, 1.15, n)], 1)

def pop(gain):
    n = int(0.09 * SR); t = np.arange(n) / SR
    f = 620 * np.exp(-t * 18) + 380; ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t * 45) * gain
    return np.stack([y, y], 1).astype(np.float32)

def bells(notes, gain, step=0.07):
    """soft sparkle: a few sine bells with a touch of inharmonic partial, staggered and panned"""
    n = int((step * len(notes) + 1.6) * SR); out = np.zeros((n, 2), np.float32)
    for k, f in enumerate(notes):
        m = n - int(k * step * SR); t = np.arange(m) / SR
        y = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * f * 2.76 * t)) * np.exp(-t * 3.2) * env(m, 120, 2000)
        p = 0.5 + 0.35 * np.sin(k * 1.9)
        out[-m:, 0] += y * (1 - p) * gain; out[-m:, 1] += y * p * gain
    return out

def shutter(gain):
    """camera «щелк»: a bright click + body thump, the mirror's second click 70 ms later"""
    rng = np.random.default_rng(3); n = int(0.16 * SR); y = np.zeros(n, np.float32); t = np.arange(n) / SR
    for t0, g in ((0.0, 1.0), (0.07, 0.6)):
        i = int(t0 * SR); m = int(0.012 * SR); nz = np.diff(rng.standard_normal(m + 1)).astype(np.float32)
        y[i:i + m] += nz * np.exp(-np.arange(m) / (0.0025 * SR)) * g * 0.5
        tt = t[i:] - t0; y[i:] += np.sin(2 * np.pi * 140 * tt) * np.exp(-tt * 60) * g * 0.6
    y *= gain / np.abs(y).max()
    return np.stack([y, y], 1)

def place(bus, x, t):
    i = int(t * SR); n = min(len(x), N - i); bus[i:i + n] += x[:n]

def main(src, voice_wav, master, out, mute=()):
    v = read(voice_wav)[:N]; v = np.pad(v, ((0, N - len(v)), (0, 0)))
    fol = foley(src, set(mute))
    # duck the foley under the voice: envelope of the voice, 30 ms attack / 400 ms release, -9 dB while speaking
    lv = np.abs(v).mean(1); k = int(0.02 * SR); lv = np.convolve(lv, np.ones(k) / k, "same")
    talk = (lv > db(-38)).astype(np.float32); sm = np.zeros_like(talk); g = 0.0
    up, dn = 1 - np.exp(-1 / (0.03 * SR)), 1 - np.exp(-1 / (0.4 * SR))
    for i in range(0, N, 48):                                     # 1 ms steps
        g += (talk[i] - g) * (up if talk[i] > g else dn) * 48; sm[i:i + 48] = g
    fol *= (db(-8) * sm + (1 - sm))[:, None] * db(5)
    sfx = np.zeros((N, 2), np.float32)
    place(sfx, noise_sweep(0.9, 400, 5000, db(-20)), 0.0)                        # intro air
    place(sfx, bells([1318.5, 1568, 1975.5, 2637], db(-24)), 0.35)              # title sparkle
    for t in (17.25 - 0.45, 35.05 - 0.45, 53.4 - 0.45, R.FREEZE[0] - 0.5, R.FREEZE[1] - 0.45, 99.0 - 0.45):  # section changes
        place(sfx, noise_sweep(0.9, 300, 3500, db(-22)), t)
    for a, b, _ in R.PILLS: place(sfx, pop(db(-17)), a + 0.02)                   # pill pops
    place(sfx, bells([1568, 1975.5, 2349, 3136], db(-23)), 66.5)                # «радовал цвет»
    for t in R.SNAPS: place(sfx, shutter(db(-12)), t - 0.01)                      # щелк, щелк, щелк
    place(sfx, pop(db(-19)), R.GIFTS[0][3])                                      # the «подстилка» label
    place(sfx, bells([1046.5, 1318.5, 1568, 2093, 2637], db(-21), 0.09), 112.2) # «в полном восторге»
    mix = v + fol + sfx
    fade = np.ones(N, np.float32); fade[-SR:] = np.linspace(1, 0, SR); mix *= fade[:, None]
    peak = np.abs(mix).max()
    if peak > db(-1): mix *= db(-1) / peak
    wav = out + ".wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", wav],
                   input=mix.astype(np.float32).tobytes(), check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", master, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out], check=True)

if __name__ == "__main__":
    m = sys.argv[sys.argv.index("--mute-foley") + 1].split(",") if "--mute-foley" in sys.argv else []
    main(*sys.argv[1:5], mute=[int(x) for x in m if x])
