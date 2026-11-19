#!/usr/bin/env python3
# Re-voices the client-edited clips IMG_0580/0581/0584/0585 with the same kNN-VC voice as the age-52 reel (only the patient; the question in 0584 kept).
# Run from the folder with the ONNX models and pool.npy: S=<scratch> python3 voice_age52_clips.py
import numpy as np, wave, subprocess, sys, os
sys.argv = ["x", "."]; exec(open("knnvc.py").read().split('if __name__')[0])
pool = np.load("pool.npy"); S = os.environ["S"]; P = 0.5
MUTE = {"IMG_0585": [(9.40, 10.04)]}
INTV = rd(f"{S}/src17/wav/IMG_2831.wav"); CTX = {"IMG_0580": 22.50, "IMG_0581": 70.86}
JOBS = {"IMG_0580": [(0.0, None)], "IMG_0581": [(0.0, None)], "IMG_0584": [(4.75, None)], "IMG_0585": [(0.0, None)]}   # the patient's speech (0584: after the question)
for n, ranges in JOBS.items():
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{S}/src18/wav/{n}.wav", "-af", "highpass=f=80", "s16.wav"], check=True)
    src = rd("s16.wav"); w = wave.open(f"{S}/src18/_work/{n}.wav"); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    for a, b in ranges:
        b = b or len(src) / 16000
        core = src[int(a * 16000):int(b * 16000)]; p_ = int(P * 16000)
        if n in CTX:                                                     # the clip starts on a word: lead in with the interview's own audio just before it
            pre = INTV[int((CTX[n] - P) * 16000):int(CTX[n] * 16000)]; pre = pre * (np.sqrt((core[:16000] ** 2).mean()) / (np.sqrt((INTV[int(CTX[n] * 16000):int((CTX[n] + 1) * 16000)] ** 2).mean()) + 1e-9))
        else: pre = np.zeros(p_, np.float32)
        seg = np.concatenate([pre, core, np.zeros(p_, np.float32)]); seg = seg / (np.abs(seg).max() + 1e-9) * 0.9
        y = match(feats(seg), pool, k=6); k_ = len(y); m = int(round(k_ * 1.25))
        yi = np.stack([np.interp(np.linspace(0, k_ - 1, m), np.arange(k_), y[:, c]) for c in range(1024)], 1)
        wav = voc.run(None, {"features": yi.T[None].astype(np.float32)})[0][0, 0]
        wv = wave.open("t16.wav", "wb"); wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(16000); wv.writeframes((np.clip(wav / (np.abs(wav).max() + 1e-9) * 0.9, -1, 1) * 32767).astype(np.int16).tobytes()); wv.close()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "t16.wav", "-ar", str(sr), "-af", "highpass=f=70,lowpass=f=7600", "t48.wav"], check=True)
        v = np.frombuffer(wave.open("t48.wav").readframes(10 ** 9), np.int16).astype(np.float64) / 32768
        v = v[int(P * sr):int(P * sr) + int((b - a) * sr)]; i = int(a * sr); k = min(len(v), len(x) - i); orig = x[i:i + k]
        v = v[:k] * np.sqrt((orig ** 2).mean()) / (np.sqrt((v[:k] ** 2).mean()) + 1e-9)
        for m0, m1 in MUTE.get(n, []):                                    # a real pause in her speech stays silent (no invented syllable)
            g = np.ones(k); j0, j1 = int((m0 - a) * sr), int((m1 - a) * sr); f2 = int(0.04 * sr)
            g[j0:j1] = 0; g[j0 - f2:j0] = np.linspace(1, 0, f2); g[j1:j1 + f2] = np.linspace(0, 1, f2); v = v * g
        f = int(0.03 * sr); r = np.linspace(0, 1, f)
        if a > 0: v[:f] = orig[:f] * (1 - r) + v[:f] * r
        x[i:i + k] = v
    o = wave.open(f"{S}/src18/_work/{n}_vc.wav", "wb"); o.setnchannels(1); o.setsampwidth(2); o.setframerate(sr); o.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()); o.close()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{S}/src18/{n}.mov", "-i", f"{S}/src18/_work/{n}_vc.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", f"{S}/out/{n}_voice.mov"], check=True)
    print(n, "done", flush=True)
