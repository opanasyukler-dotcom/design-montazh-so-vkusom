"""Hand-drawn line doodles for the Reels: each icon is a list of strokes in a unit box (x → right, y → down).
draw_icon() draws them on in order up to `prog` (the pencil tip position is returned), then keeps them «boiling»
(re-jittered every few frames like hand-drawn animation) with small per-icon motions (a clapper that claps, a wand
star that twinkles, flames that flicker).
"""
import numpy as np
import cv2

def _rect(x0, y0, x1, y1): return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]

def _arc(cx, cy, r, a0, a1, n=24, ry=None):
    a = np.deg2rad(np.linspace(a0, a1, n)); ry = r if ry is None else ry
    return list(zip(cx + r * np.cos(a), cy + ry * np.sin(a)))

def _star(cx, cy, r, k=5, inner=0.45, rot=-90):
    pts = []
    for i in range(2 * k + 1):
        a = np.deg2rad(rot + i * 180 / k); rr = r if i % 2 == 0 else r * inner
        pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
    return pts

def _heart(cx, cy, s, n=60):
    t = np.linspace(0, 2 * np.pi, n); x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return list(zip(cx + x * s / 34, cy + y * s / 34))

def _flame(cx, by, w, h):
    left = [(cx - w / 2 * np.sin(np.pi * u) * (1 - u * 0.2), by - h * u) for u in np.linspace(0, 1, 14)]
    right = [(cx + w / 2 * np.sin(np.pi * u) * (1 - u * 0.2) + w * 0.12 * u, by - h * u) for u in np.linspace(1, 0, 14)]
    return left + right

ICONS = {
    "qmark": [[(0.30, 0.30), *_arc(0.48, 0.30, 0.18, 180, 360, 16)[1:], (0.62, 0.42), (0.50, 0.52), (0.48, 0.66)],
              _arc(0.48, 0.80, 0.03, 0, 360, 10), [(0.78, 0.18), (0.74, 0.62)], _arc(0.735, 0.78, 0.03, 0, 360, 10)],
    "hospital": [_rect(0.12, 0.38, 0.88, 0.94), _rect(0.36, 0.12, 0.64, 0.38), [(0.50, 0.17), (0.50, 0.33)], [(0.42, 0.25), (0.58, 0.25)],
                 _rect(0.20, 0.48, 0.34, 0.62), _rect(0.66, 0.48, 0.80, 0.62), [(0.42, 0.94), (0.42, 0.72), (0.58, 0.72), (0.58, 0.94)]],
    "clapper": [_rect(0.10, 0.42, 0.90, 0.92), [(0.10, 0.56), (0.90, 0.56)], [(0.24, 0.42), (0.16, 0.56)], [(0.44, 0.42), (0.36, 0.56)],
                [(0.64, 0.42), (0.56, 0.56)], [(0.84, 0.42), (0.76, 0.56)],
                "BAR", [(0.10, 0.40), (0.90, 0.40), (0.90, 0.28), (0.10, 0.28), (0.10, 0.40)], [(0.28, 0.28), (0.20, 0.40)],
                [(0.48, 0.28), (0.40, 0.40)], [(0.68, 0.28), (0.60, 0.40)]],
    "cloud": [_arc(0.30, 0.58, 0.14, 100, 270, 18) + _arc(0.48, 0.44, 0.18, 200, 340, 18) + _arc(0.70, 0.54, 0.15, 250, 420, 18)
              + [(0.74, 0.72), (0.28, 0.72), (0.30, 0.72)], [(0.86, 0.22), (0.86, 0.34)], [(0.80, 0.28), (0.92, 0.28)],
              [(0.16, 0.24), (0.16, 0.32)], [(0.12, 0.28), (0.20, 0.28)]],
    "pin": [[(0.50, 0.92), *_arc(0.50, 0.38, 0.26, 150, 390, 28), (0.50, 0.92)], _arc(0.50, 0.38, 0.09, 0, 360, 16),
            _arc(0.50, 0.93, 0.22, 0, 360, 24, ry=0.05)],
    "ruble": [_arc(0.50, 0.50, 0.40, 0, 360, 40), [(0.42, 0.78), (0.42, 0.24), (0.56, 0.24), *_arc(0.56, 0.36, 0.12, 270, 450, 12)[1:], (0.42, 0.48)],
              [(0.32, 0.48), (0.42, 0.48)], [(0.32, 0.62), (0.58, 0.62)]],
    "wand": [[(0.14, 0.88), (0.62, 0.40)], [(0.18, 0.92), (0.66, 0.44)], "STAR", _star(0.72, 0.30, 0.16),
             [(0.40, 0.18), (0.40, 0.28)], [(0.35, 0.23), (0.45, 0.23)], [(0.90, 0.62), (0.90, 0.70)], [(0.86, 0.66), (0.94, 0.66)]],
    "flames": ["FLAME", _flame(0.30, 0.92, 0.22, 0.55), _flame(0.52, 0.92, 0.28, 0.78), _flame(0.74, 0.92, 0.2, 0.48)],
    "heart": [_heart(0.5, 0.52, 0.42)],
}

def _strokes(name):
    """→ list of (points, tag) where tag marks strokes that move (BAR / STAR / FLAME)"""
    out, tag = [], None
    for s in ICONS[name]:
        if isinstance(s, str): tag = s; continue
        out.append((np.asarray(s, np.float32), tag))
    return out

def _len(p): return float(np.hypot(*np.diff(p, axis=0).T).sum())

def draw_icon(fr, name, x, y, size, prog, t, col, th=8, motion=0.0, seed=0):
    """draw icon `name` into box (x, y, size) up to prog (0..1 over all strokes); returns pencil tip or None.
    motion: seconds since the drawing finished (drives the clap / twinkle / flicker)"""
    st = _strokes(name); lens = [_len(p) for p, _ in st]; tot = sum(lens); d = tot * min(max(prog, 0), 1)
    rng = np.random.default_rng(seed * 1000 + int(t * 25) // 3)          # boil: new jitter every 3 frames
    tip = None
    for (p, tag), L in zip(st, lens):
        if d <= 0: break
        q = p.copy()
        if tag == "BAR" and motion > 0:                                   # the clapper bar opens and claps shut
            a = np.deg2rad(-24 * max(0.0, np.sin(min(motion, 0.6) / 0.6 * np.pi)) if motion < 0.6 else 0.0)
            h = np.array([0.10, 0.40]); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]); q = (q - h) @ R.T + h
        if tag == "STAR" and motion > 0:
            c = q.mean(0); q = (q - c) * (1 + 0.12 * np.sin(motion * 9)) + c
        if tag == "FLAME":
            b = q[:, 1].max(); q[:, 0] += 0.02 * np.sin(t * 13 + q[:, 1] * 20) * (b - q[:, 1]) * 2
        q = q * size + (x, y) + rng.normal(0, 1.6, q.shape)
        if d >= L: part = q
        else:
            seg = np.hypot(*np.diff(q, axis=0).T); cum = np.r_[0, np.cumsum(seg)]; k = int(np.searchsorted(cum, d * _len(q) / max(L, 1e-6)))
            k = min(max(k, 1), len(q) - 1); f = (d * _len(q) / L - cum[k - 1]) / max(seg[k - 1], 1e-6)
            end = q[k - 1] + (q[k] - q[k - 1]) * min(max(f, 0), 1); part = np.vstack([q[:k], end]); tip = (float(end[0]), float(end[1]))
        cv2.polylines(fr, [part.astype(np.int32)], False, col, th, cv2.LINE_AA)
        d -= L
    return tip
