#!/usr/bin/env python3
# kNN-VC voice conversion (ONNX export TigreGotico/voiceclonnx-knn-vc on Hugging Face); the reference pool is synthesised with Piper ru_RU-irina-medium.
# usage: python3 tools/voice_knnvc.py <dir with wavlm_layer6.onnx, hifigan_knnvc.onnx> ref1.wav,ref2.wav,... src16k.wav out16k.wav
"""kNN-VC (ONNX): src.wav (16 kHz) spoken in the timbre of the reference pool -> out.wav (16 kHz)"""
import numpy as np, onnxruntime as ort, wave, sys, glob
D = sys.argv[1]; enc = ort.InferenceSession(f"{D}/wavlm_layer6.onnx"); voc = ort.InferenceSession(f"{D}/hifigan_knnvc.onnx")
def rd(p):
    w = wave.open(p); return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
def feats(x, chunk=16000 * 15):
    out = []
    for i in range(0, len(x), chunk):
        c = x[i:i + chunk]
        if len(c) < 1600: break
        out.append(enc.run(None, {"input_values": c[None]})[0][0])
    return np.concatenate(out)
def match(q, pool, k=4):
    qn = q / np.linalg.norm(q, axis=1, keepdims=True); pn = pool / np.linalg.norm(pool, axis=1, keepdims=True)
    out = np.empty_like(q)
    for i in range(0, len(q), 500):
        sim = qn[i:i + 500] @ pn.T; idx = np.argpartition(-sim, k, axis=1)[:, :k]; out[i:i + 500] = pool[idx].mean(1)
    return out
if __name__ == "__main__":
    pool_files, src, dst = sys.argv[2].split(","), sys.argv[3], sys.argv[4]
    pool = np.concatenate([feats(rd(p)) for p in pool_files]); print("pool frames", len(pool))
    x = rd(src); x = x / (np.abs(x).max() + 1e-9) * 0.9; q = feats(x); y = match(q, pool)
    n = len(y); m = int(round(n * 320 / 256))                          # this vocoder export has a 256-sample hop: resample the frames so 20 ms stays 20 ms
    yi = np.stack([np.interp(np.linspace(0, n - 1, m), np.arange(n), y[:, c]) for c in range(y.shape[1])], 1)
    wav = voc.run(None, {"features": yi.T[None].astype(np.float32)})[0][0, 0]
    wav = wav / (np.abs(wav).max() + 1e-9) * 0.9
    w = wave.open(dst, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes((wav * 32767).astype(np.int16).tobytes()); print(len(x) / 16000, len(wav) / 16000)
