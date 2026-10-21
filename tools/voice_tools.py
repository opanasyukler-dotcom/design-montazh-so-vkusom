"""Speech helpers (numpy only): MFCC, a two-speaker classifier trained on reference stretches, and a detector of
drawn-out hesitation vowels («ааа», «эээ») — long stretches where the spectrum stays still while the voice keeps going.
"""
import subprocess
import numpy as np

SR, HOP, WIN = 16000, 160, 400            # 10 ms hop, 25 ms window


def load(path):
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.float32)


def _mel_fb(n_fft=512, n_mels=40):
    hz = lambda m: 700 * (10 ** (m / 2595) - 1); mel = lambda f: 2595 * np.log10(1 + f / 700)
    pts = hz(np.linspace(mel(60), mel(7600), n_mels + 2)); bins = np.floor((n_fft + 1) * pts / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        fb[i - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1); fb[i - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


FB = _mel_fb()
DCT = np.cos(np.pi / 40 * (np.arange(40)[None] + 0.5) * np.arange(1, 14)[:, None])    # 13 cepstra (no c0)


def frames(x):
    n = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None] + HOP * np.arange(n)[:, None]
    return x[idx] * np.hanning(WIN)


def features(x):
    """per 10 ms: log-mel (40), mfcc (13), energy dB"""
    spec = np.abs(np.fft.rfft(frames(x), 512)) ** 2
    lm = np.log(spec @ FB.T + 1e-10)
    mf = lm @ DCT.T
    en = 10 * np.log10(spec.sum(1) + 1e-12)
    return lm, mf, en


def voiced(en):
    return en > np.percentile(en, 15) + 12


class Speakers:
    """diagonal-Gaussian mixture per speaker (k-means init), trained on reference ranges"""
    def __init__(self, refs, k=8):
        self.models = {}
        for name, chunks in refs.items():
            X = np.concatenate(chunks); X = (X - 0)                                      # cepstral frames
            rng = np.random.default_rng(0); C = X[rng.choice(len(X), k, replace=False)]
            for _ in range(20):
                lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
                C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            V = np.stack([X[lab == j].var(0) + 1e-2 if (lab == j).sum() > 2 else X.var(0) for j in range(k)])
            Wt = np.bincount(lab, minlength=k) / len(lab) + 1e-6
            self.models[name] = (C, V, Wt)

    def loglik(self, name, X):
        C, V, Wt = self.models[name]
        ll = -0.5 * (((X[:, None] - C[None]) ** 2) / V[None] + np.log(2 * np.pi * V[None])).sum(-1) + np.log(Wt)[None]
        m = ll.max(1); return m + np.log(np.exp(ll - m[:, None]).sum(1))


def cmn(mf):
    return mf - mf.mean(0)


def drawn_vowels(lm, en, min_len=0.30, keep=0.10):
    """(start, end) seconds of over-long steady voiced stretches; the first `keep` s of each is kept"""
    v = voiced(en)
    flux = np.r_[0, np.sqrt(((np.diff(lm, axis=0)) ** 2).mean(1))]
    flux = np.convolve(flux, np.ones(5) / 5, "same")
    thr = np.percentile(flux[v], 35)
    steady = v & (flux < thr)
    out, j, n = [], 0, len(steady)
    while j < n:
        if steady[j]:
            k = j
            while k < n and steady[k]: k += 1
            if (k - j) / 100 >= min_len: out.append((j / 100 + keep, k / 100))
            j = k
        else: j += 1
    return out
