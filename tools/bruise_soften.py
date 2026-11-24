#!/usr/bin/env python3
"""Soften post-op bruises on skin: inside a feathered box, pull purple/yellow-green blotches toward the surrounding
skin tone (low frequencies only, so the skin texture stays). Stickers (blue), tapes (white) are left alone.

usage: python3 tools/bruise_soften.py in.jpg out.jpg x0,y0,x1,y1 [strength 0..1]
"""
import sys, numpy as np, cv2
src, dst, box = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3].split(",")]; K = float(sys.argv[4]) if len(sys.argv) > 4 else 0.85
img = cv2.imread(src); lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32); L, A, B = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
x0, y0, x1, y1 = box; H, W = L.shape
roi = np.zeros((H, W), np.float32); roi[y0:y1, x0:x1] = 1; roi = cv2.GaussianBlur(roi, (0, 0), 12)
chroma = np.hypot(A, B)
skin = (B > 8) & (A > 4) & (L > 35) & (L < 92) & (chroma > 12)                   # plain skin (not sticker, tape, shadow)
# skin baseline around the bruise: skin pixels that are not bruised, spread by normalised convolution
hue_ok = skin & (B > A * 0.9)                                                      # warm skin: yellow ≥ red; purple bruises fail this
def nconv(ch, m, s): return cv2.GaussianBlur(ch * m, (0, 0), s) / (cv2.GaussianBlur(m, (0, 0), s) + 1e-4)
m0 = hue_ok.astype(np.float32)
bl, ba, bb = nconv(L, m0, 40), nconv(A, m0, 40), nconv(B, m0, 40)
# bruise score: purple (less yellow, more red/blue than the skin around), darker, or a greenish-yellow cast
dp = np.clip((bb - B - 1.5) / 6, 0, 1) + np.clip((A - ba - 2) / 7, 0, 1) * 0.5
dd = np.clip((bl - L - 3) / 12, 0, 1)
dy = np.clip((ba - A - 1.5) / 6, 0, 1) * np.clip((B - bb + 4) / 6, 0, 1)
tape = cv2.dilate(((L > 66) & (chroma < 17)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0; sticker = cv2.dilate((B < -5).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
score = np.clip(np.maximum(dp, dy) + 0.6 * dd * (dp > 0.1), 0, 1) * (~tape) * (~sticker) * (L > 25)
score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), 5) * roi
# correct only the low frequencies: texture (high-pass) is kept
low = [cv2.GaussianBlur(c, (0, 0), 6) for c in (L, A, B)]
k = np.clip(score * 2.2, 0, 1) * K
Ln = L + k * np.maximum(0, bl - low[0]) * 0.9; An = A + k * (ba - low[1]); Bn = B + k * (bb - low[2])
out = cv2.cvtColor(np.dstack([np.clip(Ln * 255 / 100, 0, 255), np.clip(An + 128, 0, 255), np.clip(Bn + 128, 0, 255)]).astype(np.uint8), cv2.COLOR_LAB2BGR)
cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 96])
if len(sys.argv) > 5: cv2.imwrite(sys.argv[5], (np.clip(k, 0, 1) * 255).astype(np.uint8))
