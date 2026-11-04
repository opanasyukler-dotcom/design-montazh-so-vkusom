"""Soft piano + pad bed, synthesised from scratch (no third-party audio, no licensing questions).

usage: python3 music_soft_piano.py OUT.wav [seconds]
Fmaj7 - Am7 - Dm7 - Bbmaj7 at 66 bpm, gentle arpeggios over a warm pad, long room reverb.
"""
import sys, wave
import numpy as np

SR = 48000
OUT = sys.argv[1]
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
BPM = 66.0
BEAT = 60.0 / BPM
rng = np.random.default_rng(7)


def midi(n): return 440.0 * 2 ** ((n - 69) / 12)


def piano(f, dur, vel=0.5):
    t = np.arange(int(dur * SR)) / SR
    tone = np.zeros_like(t)
    for k, a in enumerate([1.0, 0.45, 0.22, 0.12, 0.06, 0.03], 1):
        fk = f * k * (1 + 0.0004 * k * k)  # slight string inharmonicity
        tone += a * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.28)) * np.exp(-t * (1.2 + 0.9 * k))
    att = np.minimum(t / 0.006, 1)
    return tone * att * vel


def pad(freqs, dur):
    t = np.arange(int(dur * SR)) / SR
    s = np.zeros_like(t)
    for f in freqs:
        for det in (-0.12, 0.0, 0.12):
            s += np.sin(2 * np.pi * f * (1 + det / 100) * t + rng.uniform(0, 6.28))
            s += 0.25 * np.sin(2 * np.pi * 2 * f * (1 + det / 100) * t)
    env = np.minimum(t / 1.2, 1) * np.minimum((dur - t) / 1.2, 1)
    return s * np.clip(env, 0, 1) / (len(freqs) * 3)


CHORDS = [  # (bass, chord tones) in MIDI
    (41, [53, 57, 60, 64]),   # Fmaj7
    (45, [57, 60, 64, 67]),   # Am7
    (38, [50, 53, 57, 60]),   # Dm7
    (46, [50, 53, 57, 58]),   # Bbmaj7
]
bar = 4 * BEAT
n = int(DUR * SR) + SR * 4
mix = np.zeros(n)
t0 = 0.0
i = 0
while t0 < DUR:
    bass, tones = CHORDS[i % 4]
    s = int(t0 * SR)
    p = pad([midi(x) for x in tones[:3]], bar + 0.8)
    mix[s:s + len(p)] += 0.10 * p
    b = piano(midi(bass), 3.5, 0.32); mix[s:s + len(b)] += b
    # arpeggio in eighths, a little humanised
    pattern = [0, 1, 2, 3, 2, 1, 2, 3] if (i // 4) % 2 == 0 else [0, 2, 1, 3, 2, 3, 1, 2]
    for k, idx in enumerate(pattern):
        tt = t0 + k * BEAT / 2 + rng.normal(0, 0.008)
        if tt >= DUR: break
        v = 0.20 + 0.06 * (k % 2 == 0) + rng.normal(0, 0.015)
        nt = piano(midi(tones[idx] + 12), 2.6, v)
        ss = int(tt * SR); mix[ss:ss + len(nt)] += nt
    # a soft top melody note every other bar
    if i % 2 == 1:
        nt = piano(midi(tones[-1] + 24), 3.0, 0.10); mix[s:s + len(nt)] += nt
    t0 += bar; i += 1

# room reverb: decaying filtered noise impulse
ir_len = int(2.8 * SR)
tt = np.arange(ir_len) / SR
ir = rng.normal(0, 1, ir_len) * np.exp(-tt * 2.2)
ir = np.convolve(ir, np.ones(24) / 24, mode="same")  # darker tail
ir[0] = 0
wet = np.fft.irfft(np.fft.rfft(mix, len(mix) + ir_len) * np.fft.rfft(ir, len(mix) + ir_len))[:len(mix)]
wet *= np.std(mix) / (np.std(wet) + 1e-9) * 0.55
out = mix + wet
out = out[:int(DUR * SR)]
fade = int(2.0 * SR)
out[:int(0.5 * SR)] *= np.linspace(0, 1, int(0.5 * SR))
out[-fade:] *= np.linspace(1, 0, fade)
out /= np.max(np.abs(out)) + 1e-9
out *= 10 ** (-3 / 20)
st = np.stack([out, np.roll(out, int(0.011 * SR)) * 0.96], 1)  # a touch of width
with wave.open(OUT, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((st * 32767).astype(np.int16).tobytes())
print("ok", OUT, DUR)
