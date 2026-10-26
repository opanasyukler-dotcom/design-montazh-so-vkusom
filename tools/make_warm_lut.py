"""Warm-cream grade like the blog's linen photos (light, low contrast, creamy whites, no blue cast) as a .cube 3D LUT.
Built in Lab: brighter mids with lifted shadows and a soft shoulder, gentler contrast, neutrals pushed to cream
(a +2, b +7.5, blue and green-cyan tamed), a touch less saturation.  python3 make_warm_lut.py out.cube [size]
"""
import sys
import numpy as np
import cv2

def grade_lab(L, a, b):
    x = L / 100
    y = x ** 0.68                                          # brighter mids, lifted shadows
    y = 0.62 + (y - 0.62) * 0.86                           # softer contrast around the light mids
    y = np.where(y > 0.9, 0.9 + (1 - np.exp(-(y - 0.9) / 0.1)) * 0.1, y)   # soft shoulder, no clipped whites
    w = np.clip((x - 0.15) / 0.5, 0, 1)                    # warm the mids/highlights more than the deep shadows
    a = np.where(a < 0, a * 0.6, a * 0.9)                  # tame green-cyan, keep the pinks
    b = np.where(b < 0, b * 0.5, b * 0.88)                 # the blue cast goes first
    return y * 100, a + 2.0 * w, b + 7.5 * w

def main(out, n=33):
    g = np.linspace(0, 1, n, dtype=np.float32)
    b_, g_, r_ = np.meshgrid(g, g, g, indexing="ij")       # .cube order: red fastest, then green, then blue
    rgb = np.stack([r_, g_, b_], -1).reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)             # float input: L 0..100, a/b signed
    L, a, b = grade_lab(lab[..., 0], lab[..., 1], lab[..., 2])
    res = np.clip(cv2.cvtColor(np.stack([L, a, b], -1).astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1).reshape(-1, 3)
    with open(out, "w") as f:
        f.write(f'TITLE "mcg.va warm cream"\nLUT_3D_SIZE {n}\n')
        for r, gg, bb in res: f.write(f"{r:.6f} {gg:.6f} {bb:.6f}\n")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 33)
