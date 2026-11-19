#!/usr/bin/env python3
# Re-voices the patient lines of IMG_2831 with kNN-VC (tools/voice_knnvc.py): a 32-min Piper ru_RU-irina reference pool, k=6, 0.5 s padding
# around every line (clean edges), 30 ms crossfades. Run from the folder with the ONNX models and 16_ref*.wav: S=<scratch> python3 voice_age52_convert.py
import numpy as np, wave, glob, subprocess, sys, os
sys.argv = ["x", "."]; exec(open("knnvc.py").read().split('if __name__')[0])
if os.path.exists("pool.npy"): pool = np.load("pool.npy")
else: pool = np.concatenate([feats(rd(p)) for p in sorted(glob.glob("16_ref*.wav"))]).astype(np.float32); np.save("pool.npy", pool)
print("pool", pool.shape)
S = os.environ["S"]
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", S + "/src17/wav/IMG_2831.wav", "-af", "highpass=f=80", "src_dn.wav"], check=True)
src = rd("src_dn.wav")
w = wave.open(S + "/src17/_work/IMG_2831.wav"); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768; sr = w.getframerate()
P = 0.5
for a, b in [(11.30, 14.30), (22.48, 31.85), (70.84, 76.50), (214.36, 223.88), (231.80, 235.45)]:
    seg = src[int((a - P) * 16000):int((b + P) * 16000)]; seg = seg / (np.abs(seg).max() + 1e-9) * 0.9
    q = feats(seg); y = match(q, pool, k=6)
    n = len(y); m = int(round(n * 320 / 256)); yi = np.stack([np.interp(np.linspace(0, n - 1, m), np.arange(n), y[:, c]) for c in range(y.shape[1])], 1)
    wav = voc.run(None, {"features": yi.T[None].astype(np.float32)})[0][0, 0]
    wv = wave.open("tmp16.wav", "wb"); wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(16000); wv.writeframes((np.clip(wav / (np.abs(wav).max() + 1e-9) * 0.9, -1, 1) * 32767).astype(np.int16).tobytes()); wv.close()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "tmp16.wav", "-ar", str(sr), "-af", "highpass=f=70,lowpass=f=7600", "tmp48.wav"], check=True)
    v = np.frombuffer(wave.open("tmp48.wav").readframes(10 ** 9), np.int16).astype(np.float64) / 32768
    v = v[int(P * sr):int(P * sr) + int((b - a) * sr)]                      # trim the padding: the edges are clean
    i = int(a * sr); n = min(len(v), len(x) - i); orig = x[i:i + n]
    g = np.sqrt((orig ** 2).mean()) / (np.sqrt((v[:n] ** 2).mean()) + 1e-9); v = v[:n] * g
    f = int(0.03 * sr); r = np.linspace(0, 1, f); v[:f] = orig[:f] * (1 - r) + v[:f] * r; v[-f:] = v[-f:] * (1 - r) + orig[-f:] * r
    x[i:i + n] = v; print(a, b, "done")
o = wave.open(S + "/src17/_work/IMG_2831_alt.wav", "wb"); o.setnchannels(1); o.setsampwidth(2); o.setframerate(sr); o.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
