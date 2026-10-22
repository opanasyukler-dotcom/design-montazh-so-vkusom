# voice (source minus the first OFF seconds) + a soft plucked bed + quiet ticks on cuts / keywords -> mix.wav
import numpy as np, wave, subprocess, sys
SR = 48000; OFF = 1.2; DUR = 25.9; n = int(DUR * SR)
src = sys.argv[1] if len(sys.argv) > 1 else "v.mov"
raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(OFF), "-t", str(DUR), "-i", src, "-ac", "1", "-ar", str(SR),
                      "-af", "highpass=f=80,afftdn=nf=-28,acompressor=threshold=-20dB:ratio=3:attack=5:release=120", "-f", "s16le", "-"],
                     capture_output=True, check=True).stdout
vo = np.frombuffer(raw, np.int16).astype(np.float32) / 32768; vo = np.pad(vo, (0, max(0, n - len(vo))))[:n]
vo *= 0.5 / (np.sqrt(np.mean(vo[np.abs(vo) > 0.01] ** 2)) + 1e-6) * 0.16
rng = np.random.RandomState(2)
def pluck(f, dur=1.6, damp=0.996):
    N = int(SR / f); buf = rng.uniform(-1, 1, N); out = np.zeros(int(dur * SR))
    for i in range(len(out)): out[i] = buf[i % N]; buf[i % N] = damp * 0.5 * (buf[i % N] + buf[(i + 1) % N])
    return out * np.exp(-np.arange(len(out)) / SR * 2.0)
def tick(f, dur=0.35):
    k = np.arange(int(dur * SR)) / SR; return (np.sin(2 * np.pi * f * k) + 0.3 * np.sin(2 * np.pi * f * 3 * k) * np.exp(-k * 50)) * np.exp(-k * 14)
bed = np.zeros(n + 3 * SR)
def put(buf, s, t, g): i = int(max(0, t) * SR); seg = buf[i:i + len(s)]; seg += s[:len(seg)] / (np.abs(s).max() + 1e-9) * g
# light major arpeggio, 100 bpm eighths: Fmaj7 - Am7 - Cmaj7 - G
CH = [[349, 440, 523, 659], [220, 330, 392, 523], [262, 330, 392, 494], [196, 294, 392, 494]]
NOTE = {}
step = 0.3; t = 0.0; i = 0
while t < DUR + 1:
    ch = CH[(i // 8) % 4]; f = ch[[0, 1, 2, 3, 2, 1, 2, 3][i % 8]] * 2
    if f not in NOTE: NOTE[f] = pluck(f, 1.4)
    put(bed, NOTE[f], t, 0.5 if i % 2 == 0 else 0.35)
    if i % 8 == 0: put(bed, pluck(ch[0] / 2, 2.4, 0.998), t, 0.6)
    t += step; i += 1
bed = bed[:n]; bed /= np.abs(bed).max()
tt = np.arange(n) / SR
act = (np.abs(vo) > 0.02).astype(np.float32); kern = np.ones(int(0.25 * SR)) / int(0.25 * SR); act = np.convolve(act, kern, "same")
bed *= (0.11 - 0.06 * np.clip(act * 4, 0, 1)) * np.minimum(1, tt / 0.4) * np.clip((DUR - tt) / 0.8, 0, 1)
sfx = np.zeros(n + SR)
for c, f in [(3.15, 880), (7.75, 784), (13.85, 880), (19.55, 784)]: put(sfx, tick(f), c - 0.2, 0.08)                   # wave transitions
for c in (14.3, 14.9, 15.95): put(sfx, tick(1175, 0.25), c, 0.06)                                                    # zones light up
put(sfx, tick(1318, 0.2), 9.7, 0.06)                                                                                 # bait lands
k = np.arange(int(0.12 * SR)) / SR; put(sfx, rng.randn(len(k)) * np.exp(-k * 60), 13.05, 0.12)                       # trap snaps
for j in range(6): put(sfx, tick(1568, 0.25), 18.9 + 0.12 * j, 0.03)                                                 # sparkles
mix = vo + bed + sfx[:n]; mix /= max(1, np.abs(mix).max() / 0.95)
with wave.open("mix.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
