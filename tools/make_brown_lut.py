"""«Brown moody» grade from the blog's reference palette (#6B4A38 · #2B1F1A · #8B6B5C · #ADA69E · #DCE1E7):
muted colour, warm brown shadows and mids, pale cool-grey highlights, softly lifted warm blacks. Built in Lab as a
.cube 3D LUT; the per-brightness a*/b* targets were measured on the reference photo.
  python3 make_brown_lut.py out.cube [size] [strength]
"""
import sys
import numpy as np
import cv2

#        L     a-offset  b-offset   chroma kept
BANDS = [(8,    1.2,      2.0,      0.70),       # warm, almost-black brown
         (35,   2.2,      5.5,      0.62),       # brown shadows / hair
         (52,   1.6,      4.5,      0.62),       # skin and mids, browner and calmer
         (68,   0.2,      1.5,      0.66),
         (82,  -0.7,     -1.8,      0.70),       # pale cool-grey light (the sky in the reference)
         (96,  -0.6,     -1.8,      0.70)]

def grade_lab(L, a, b):
    Ls, ao, bo, ck = (np.array(c, np.float32) for c in zip(*BANDS))
    A, B, K = np.interp(L, Ls, ao), np.interp(L, Ls, bo), np.interp(L, Ls, ck)
    x = L / 100
    y = 0.035 + 0.95 * x                                   # lifted, warm blacks; no clipped whites
    y = y + 0.04 * np.sin(np.pi * (y - 0.5)) * (y > 0.2)  # a gentle S so it does not look flat
    return np.clip(y, 0, 1) * 100, a * K + A, b * K + B

def main(out, n=33, strength=1.0):
    g = np.linspace(0, 1, n, dtype=np.float32)
    b_, g_, r_ = np.meshgrid(g, g, g, indexing="ij")       # .cube order: red fastest, then green, then blue
    rgb = np.stack([r_, g_, b_], -1).reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = grade_lab(lab[..., 0], lab[..., 1], lab[..., 2])
    res = np.clip(cv2.cvtColor(np.stack([L, a, b], -1).astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1).reshape(-1, 3)
    res = rgb.reshape(-1, 3) + strength * (res - rgb.reshape(-1, 3))
    with open(out, "w") as f:
        f.write(f'TITLE "mcg.va brown moody"\nLUT_3D_SIZE {n}\n')
        for r, gg, bb in res: f.write(f"{r:.6f} {gg:.6f} {bb:.6f}\n")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 33, float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
