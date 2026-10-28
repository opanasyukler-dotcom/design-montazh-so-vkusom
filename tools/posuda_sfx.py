"""Sound design for the posuda review (posuda_reel.py timeline).
  music  — a light, upbeat original bed (F major, 112 bpm: pad, pluck arp, kick/clap/hats, bouncy bass; breakdown on the freeze),
           synthesized here so there are no rights issues, ducked under the voice
  foley  — the camera's own sound of each shot (off by default: the blog asked for no background sound)
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

def reverb(x, sec=2.2, mix=0.28):
    rng = np.random.default_rng(11); n = int(sec * SR); t = np.arange(n) / SR
    out = x * (1 - mix)
    for ch in range(2):
        ir = rng.standard_normal(n).astype(np.float32) * np.exp(-t * 6.9 / sec); ir /= np.sqrt((ir ** 2).sum())
        L = len(x) + n; F = 1 << (L - 1).bit_length()
        wet = np.fft.irfft(np.fft.rfft(x[:, ch], F) * np.fft.rfft(ir, F), F)[:len(x)]
        out[:, ch] += wet.astype(np.float32) * mix * 1.4
    return out

def music(dur):
    """light, upbeat bed — 112 bpm, F – C/E – Dm – Bb / F – Am – Gm – C.
    intro (pad, arp, then hats and kick) → full groove from «В наборе…» (kick, clap, hats, bouncy bass, 16th arp)
    → breakdown on the gift freeze-frame (pad + arp, a riser into the drop) → groove again → ringing out"""
    bpm = 112; beat = 60 / bpm; bar = 4 * beat; n = int(dur * SR); out = np.zeros((n, 2), np.float32)
    hz = lambda m: 440 * 2 ** ((m - 69) / 12); rng = np.random.default_rng(5)
    progA = [[53, 57, 60], [52, 55, 60], [50, 53, 57], [46, 50, 53]]
    progB = [[53, 57, 60], [45, 52, 57], [43, 50, 55], [48, 52, 55]]
    groove = lambda t: (17.25 - beat <= t < R.FREEZE[0]) or (R.FREEZE[1] <= t < dur - 3.0)
    def add(y, start, pan=0.5):
        i = int(start * SR); m = min(len(y), n - i)
        if m > 0: out[i:i + m, 0] += y[:m] * (1 - pan) * 1.4; out[i:i + m, 1] += y[:m] * pan * 1.4
    def tone(f, d, att, dec, harm, g):
        m = int(d * SR); t = np.arange(m) / SR; y = np.zeros(m, np.float32)
        for k, a in harm: y += a * np.sin(2 * np.pi * f * k * t * (1 + 0.0004 * rng.standard_normal()))
        return y * np.minimum(1, t / att) * np.exp(-t * dec) * env(m, 0, int(0.03 * SR)) * g
    def noise(d, dec, g, hp=2):
        m = int(d * SR); z = rng.standard_normal(m + hp).astype(np.float32)
        for _ in range(hp): z = np.diff(z)
        return z[:m] * np.exp(-np.arange(m) / (dec * SR)) * g
    kick = (lambda m: np.sin(2 * np.pi * np.cumsum(45 + 80 * np.exp(-np.arange(m) / SR * 30)) / SR) * np.exp(-np.arange(m) / SR * 9))(int(0.35 * SR)) * 0.5
    clap = sum(np.pad(noise(0.14, 0.035, 0.16, 1), (int(k * 0.011 * SR), int(0.04 * SR)))[:int(0.15 * SR)] for k in range(3))
    for b in range(int(dur / bar) + 1):
        t0 = b * bar; ch = (progA if (b // 4) % 2 == 0 else progB)[b % 4]; g = groove(t0 + 0.1)
        for m in ch: add(tone(hz(m + 12), bar + 0.3, 0.4, 0.4, [(1, 1), (2, 0.2), (3, 0.06)], 0.028), t0)        # pad
        arp = [ch[0] + 24, ch[1] + 24, ch[2] + 24, ch[1] + 36, ch[2] + 24, ch[1] + 24, ch[0] + 36, ch[2] + 24]
        steps = 16 if g else 8
        for k in range(steps):                                                                                # pluck arp
            m = arp[(k if steps == 8 else k // 2 + (k % 2) * 3) % 8]
            add(tone(hz(m), 0.9, 0.003, 6.0, [(1, 1), (3.01, 0.12), (5.4, 0.04)], (0.045 if k % 2 == 0 else 0.03) * (0.8 if g else 1)),
                t0 + k * bar / steps, 0.5 + 0.3 * np.sin(k * 1.3 + b))
        if t0 >= 4.6 and (g or t0 < 17.25):                                                                  # hats
            for k in range(8): add(noise(0.05, 0.012 if k % 2 else 0.02, 0.05 if k % 2 else 0.032), t0 + k * beat / 2, 0.62)
        if t0 >= 10.3 - bar and (g or t0 < 17.25):                                                           # kick
            for k in (0, 2) if not g else (0, 1.5, 2, 3.5)[:4 if b % 2 else 3]: add(kick, t0 + k * beat)
        if g:
            for k in (1, 3): add(clap, t0 + k * beat, 0.45)                                                  # clap
            for off, oc in ((0, 0), (0.75, 0), (1.5, 12), (2, 0), (2.75, 0), (3.5, 12)):                      # bouncy bass
                add(tone(hz(ch[0] - 12 + oc), 0.4, 0.005, 7.0, [(1, 1), (2, 0.3), (3, 0.08)], 0.11), t0 + off * beat)
    r0, r1 = R.FREEZE[1] - 2.0, R.FREEZE[1]                                                                  # riser into the drop
    m = int((r1 - r0) * SR); z = rng.standard_normal(m).astype(np.float32); z = np.diff(np.r_[0, z])
    add(z * np.linspace(0, 1, m) ** 2 * 0.05, r0)
    out = reverb(out, 1.6, 0.2)
    out[-int(3.0 * SR):] *= np.linspace(1, 0, int(3.0 * SR))[:, None] ** 1.5
    out[:int(0.5 * SR)] *= np.linspace(0, 1, int(0.5 * SR))[:, None]
    return out / (np.sqrt(np.mean(out ** 2)) + 1e-9) * db(-19)         # bed at -19 dBFS rms before ducking

def place(bus, x, t):
    i = int(t * SR); n = min(len(x), N - i); bus[i:i + n] += x[:n]

def main(src, voice_wav, master, out, mute=(), with_foley=False):
    v = read(voice_wav)[:N]; v = np.pad(v, ((0, N - len(v)), (0, 0)))
    fol = foley(src, set(mute)) if with_foley else np.zeros((N, 2), np.float32)
    mus = limit(music(R.END), -9)                                # kicks never poke above the voice
    # duck the foley under the voice: envelope of the voice, 30 ms attack / 400 ms release, -9 dB while speaking
    lv = np.abs(v).mean(1); k = int(0.02 * SR); lv = np.convolve(lv, np.ones(k) / k, "same")
    talk = (lv > db(-38)).astype(np.float32); sm = np.zeros_like(talk); g = 0.0
    up, dn = 1 - np.exp(-1 / (0.03 * SR)), 1 - np.exp(-1 / (0.4 * SR))
    for i in range(0, N, 48):                                     # 1 ms steps
        g += (talk[i] - g) * (up if talk[i] > g else dn) * 48; sm[i:i + 48] = g
    fol *= (db(-8) * sm + (1 - sm))[:, None] * db(5)
    mus *= (db(-10) * sm + (1 - sm))[:, None]                     # music dips ~10 dB under the voice
    sfx = np.zeros((N, 2), np.float32)
    place(sfx, noise_sweep(0.9, 400, 5000, db(-20)), 0.0)                        # intro air
    place(sfx, bells([1318.5, 1568, 1975.5, 2637], db(-24)), 0.35)              # title sparkle
    for t in (17.25 - 0.45, 35.05 - 0.45, 53.4 - 0.45, R.FREEZE[0] - 0.5, R.FREEZE[1] - 0.45, 99.0 - 0.45):  # section changes
        place(sfx, noise_sweep(0.9, 300, 3500, db(-22)), t)
    for a, b, _ in R.PILLS: place(sfx, pop(db(-17)), a + 0.02)                   # pill pops
    place(sfx, bells([1568, 1975.5, 2349, 3136], db(-23)), 66.5)                # «радовал цвет»
    for t in R.SNAPS: place(sfx, shutter(db(-6)), t - 0.01)                      # щелк, щелк, щелк
    place(sfx, pop(db(-19)), R.GIFTS[0][3])                                      # the «подстилка» label
    place(sfx, bells([1046.5, 1318.5, 1568, 2093, 2637], db(-21), 0.09), 112.2) # «в полном восторге»
    mix = v + fol + sfx + mus
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
