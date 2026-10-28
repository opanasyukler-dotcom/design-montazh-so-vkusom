"""Lay a music track under a finished edit, ducked under the voice, and remux (picture is copied untouched).
  python3 add_music.py video.mp4 voice.wav music.(wav|mp3|m4a|mp4) out.mp4 [--offset SEC] [--bed DB] [--duck DB]
  --offset  music time that plays at video 0 (skip an intro / land a drop on a cue), default 0
  --bed     music level in the pauses, dBFS rms (default -22); --duck: extra dip while she speaks (default 9)
"""
import subprocess, sys
import numpy as np

SR = 48000

def read(path, ss=0.0):
    return np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-ss", str(ss), "-i", path, "-vn", "-ac", "2", "-ar", str(SR),
                                         "-f", "f32le", "-"], capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()

def db(x): return 10 ** (x / 20)

def main(video, voice, music, out, offset=0.0, bed=-22.0, duck=9.0):
    mix = read(video); v = read(voice)[:len(mix)]; v = np.pad(v, ((0, len(mix) - len(v)), (0, 0))); N = len(mix)
    m = read(music, offset)[:N]; m = np.pad(m, ((0, N - len(m)), (0, 0)))
    # level the music on its loud part, then a slow envelope so quiet intros stay quiet
    rms = np.sqrt(np.mean(m[np.abs(m).mean(1) > 1e-4] ** 2)) + 1e-9; m *= db(bed) / rms
    lv = np.convolve(np.abs(v).mean(1), np.ones(960) / 960, "same"); talk = (lv > db(-40)).astype(np.float32)
    g = np.zeros(N, np.float32); cur = 0.0; up, dn = 1 - np.exp(-1 / (0.06 * SR)), 1 - np.exp(-1 / (0.5 * SR))
    for i in range(0, N, 48):
        tgt = talk[i]; cur += (tgt - cur) * (1 - (1 - (up if tgt > cur else dn)) ** 48); g[i:i + 48] = cur
    m *= (db(-duck) * g + (1 - g))[:, None]
    f = int(0.6 * SR); m[:f] *= np.linspace(0, 1, f)[:, None]; f = int(2.5 * SR); m[-f:] *= np.linspace(1, 0, f)[:, None] ** 1.5
    y = mix + m; y *= min(1, db(-1) / (np.abs(y).max() + 1e-9))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-i", video,
                    "-map", "1:v", "-map", "0:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out],
                   input=y.astype(np.float32).tobytes(), check=True)

if __name__ == "__main__":
    a = sys.argv; opt = lambda k, d: float(a[a.index(k) + 1]) if k in a else d
    main(*a[1:5], offset=opt("--offset", 0.0), bed=opt("--bed", -22.0), duck=opt("--duck", 9.0))
