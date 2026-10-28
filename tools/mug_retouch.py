"""Photo retouch for the coffee-mug shot: clean coffee stains off the mug's inner wall and lip, put a light milk foam on
the coffee, then a gentle colour grade.
  python3 -I mug_retouch.py photo.jpg out.jpg [debug_dir]
Geometry (full-res px): liquid ellipse and outer-lip ellipse, measured on the 1932x2576 original.
Stain cleaning works in polar coordinates around the mug (angle x radial position between liquid and outer lip): along
the angle the clean glaze changes only slowly (shading), so a rolling robust envelope gives the clean colour, and pixels
that are more orange/darker than it are replaced by it while keeping the fine grain; the printed text (neutral-green)
is not orange, so it survives.
"""
import sys
import numpy as np, cv2

LIQ = ((890.0, 1047.0), (392.0, 377.0))
OUT = ((891.0, 1020.0), (511.0, 508.0))

def ell(e, th):
    (cx, cy), (rx, ry) = e
    return cx + rx * np.cos(th), cy + ry * np.sin(th)

def polar_maps(NT=8192, NR=360, r_in=5.0, r_out=-3.0):
    th = np.linspace(0, 2 * np.pi, NT, endpoint=False)[None, :]
    t = np.linspace(0, 1, NR)[:, None]
    x0, y0 = ell(((LIQ[0]), (LIQ[1][0] + r_in, LIQ[1][1] + r_in)), th)
    x1, y1 = ell(((OUT[0]), (OUT[1][0] + r_out, OUT[1][1] + r_out)), th)
    return (x0 + (x1 - x0) * t).astype(np.float32), (y0 + (y1 - y0) * t).astype(np.float32)

def rolling(p, win, q):
    """rolling percentile along axis 1 (angle, wraps) with window `win` columns, on a decimated grid then upsampled"""
    NR, NT = p.shape; step = max(1, win // 16)
    pad = np.concatenate([p[:, -win // 2:], p, p[:, :win // 2]], 1)
    cols = np.arange(0, NT, step)
    out = np.stack([np.percentile(pad[:, c:c + win], q, axis=1) for c in cols], 1)
    return cv2.resize(out.astype(np.float32), (NT, NR), interpolation=cv2.INTER_LINEAR)

def clean_stains(img, dbg=None):
    mx, my = polar_maps()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    P = cv2.remap(lab, mx, my, cv2.INTER_LINEAR)
    L, A, B = P[..., 0], P[..., 1] - 128, P[..., 2] - 128
    # Coffee stains are always darker and more orange than the glaze around them, never brighter. So the glaze is rebuilt:
    # lightness = max(original, clean-glaze envelope) (keeps highlights and clean texture, lifts every stain), and colour
    # = the clean glaze's own a*/b* for that lightness (low percentiles per lightness bin: stains only push them up).
    NT = L.shape[1]; pad = 1024
    def wrap_blur(x, sx, sy):
        xp = np.concatenate([x[:, -pad:], x, x[:, :pad]], 1)
        return cv2.GaussianBlur(xp, (0, 0), sigmaX=sx, sigmaY=sy)[:, pad:pad + NT]
    env = np.maximum(wrap_blur(rolling(L, NT * 10 // 360, 88), 30, 1.5), wrap_blur(rolling(L, NT * 24 // 360, 93), 80, 1.5) - 4)
    Ln = np.maximum(L, env)
    bins = np.arange(0, 256, 8); ca, cb = [], []
    for b0 in bins:
        sel = (L >= b0 - 8) & (L < b0 + 16)
        ca.append(np.percentile(A[sel], 20) if sel.sum() > 500 else np.nan)
        cb.append(np.percentile(B[sel], 25) if sel.sum() > 500 else np.nan)
    ok = ~np.isnan(ca); ca = np.interp(bins + 4, (bins + 4)[ok], np.array(ca)[ok]); cb = np.interp(bins + 4, (bins + 4)[ok], np.array(cb)[ok])
    An, Bn = np.interp(Ln, bins + 4, ca), np.interp(Ln, bins + 4, cb)
    grain = cv2.GaussianBlur(np.random.default_rng(3).standard_normal(L.shape).astype(np.float32), (0, 0), 0.8) * 2.0
    lifted = np.clip((env - L) / 6, 0, 1)                              # where a stain was lifted, add a little grain back
    newP = np.stack([Ln + grain * lifted, An, Bn], 2)
    # printed text: thin dark strokes inside the lettering arc stay exactly as photographed
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 9)))
    zone = np.zeros(L.shape, np.uint8); zone[205:315, 4400:6100] = 1
    text = cv2.dilate(((bh > 10) & (zone > 0)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    keep = cv2.GaussianBlur(text, (0, 0), 1.0)
    s = 1 - keep
    res = np.stack([L, A, B], 2) * keep[..., None] + newP * s[..., None]
    res[..., 1] += 128; res[..., 2] += 128
    if dbg:
        cv2.imwrite(f"{dbg}/polar_before.jpg", cv2.cvtColor(np.clip(P, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)[:, :4096])
        cv2.imwrite(f"{dbg}/polar_mask.jpg", (np.clip((env - L) / 30, 0, 1)[:, :4096] * 255).astype(np.uint8))
        cv2.imwrite(f"{dbg}/polar_after.jpg", cv2.cvtColor(np.clip(res, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)[:, :4096])
    # back to the image: splat polar samples (dense enough to cover every pixel), fill pinholes
    H, W = img.shape[:2]
    acc = np.zeros((H, W, 3), np.float32); wt = np.zeros((H, W), np.float32); sm = np.zeros((H, W), np.float32)
    xi, yi = np.rint(mx).astype(int).ravel(), np.rint(my).astype(int).ravel()
    np.add.at(acc, (yi, xi), res.reshape(-1, 3)); np.add.at(wt, (yi, xi), 1); np.add.at(sm, (yi, xi), s.ravel())
    ok = wt > 0
    acc[ok] /= wt[ok, None]; sm[ok] /= wt[ok]
    fill = cv2.dilate(ok.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    accb = cv2.blur(acc * ok[..., None], (3, 3)); wb = cv2.blur(ok.astype(np.float32), (3, 3))
    smb = cv2.blur(sm * ok, (3, 3))
    hole = fill & ~ok
    acc[hole] = accb[hole] / wb[hole, None]; sm[hole] = smb[hole] / wb[hole]
    out_lab = lab.copy(); m = fill
    out_lab[m] = acc[m]
    out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    # soft edges of the annulus so nothing steps at the liquid line or the lip edge
    feather = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)[..., None]
    return (out * feather + img * (1 - feather)).astype(np.uint8)

def foam(img, seed=7):
    """light milk foam: the coffee surface is lifted to a latte tone in Lab (same hue, a bit less chroma), its own light
    falloff kept, with a very fine microfoam grain and soft clouds; a thin darker coffee ring stays at the wall"""
    H, W = img.shape[:2]; rng = np.random.default_rng(seed)
    (cx, cy), (rx, ry) = LIQ
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)       # 1.0 on the liquid edge
    inside = np.clip((1.0 - d) / 0.010, 0, 1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    m = d < 0.95
    Ls = cv2.GaussianBlur(L, (0, 0), 45)                             # light falloff across the surface
    Am, Bm = np.median(A[m]), np.median(B[m])
    def noise(sig):
        n = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), sig); return n / (n.std() + 1e-6)
    cloud = noise(30) * 0.6 + noise(10) * 0.4
    micro = noise(1.0) * 0.5 + noise(2.0) * 0.3 + noise(4.0) * 0.2
    # foam covers the surface and thins into a narrow coffee meniscus at the wall; slightly darker toward the edge,
    # fine microfoam everywhere and a few larger soft bubbles near the rim, like steamed-milk foam
    cover = np.clip((0.975 - d) / 0.03, 0, 1) ** 1.3 * np.clip(0.95 + 0.04 * cloud, 0, 1)
    bub = np.zeros((H, W), np.float32)
    nb = 900; rr = rng.uniform(0.80, 0.97, nb); aa = rng.uniform(0, 2 * np.pi, nb)
    for x, y, r in zip(cx + rr * rx * np.cos(aa), cy + rr * ry * np.sin(aa), rng.uniform(1.5, 4.5, nb)):
        cv2.circle(bub, (int(x), int(y)), int(round(r)), 1.0, -1, cv2.LINE_AA)
    bub = cv2.GaussianBlur(bub, (0, 0), 0.8)
    bub_rim = np.clip(cv2.GaussianBlur(bub, (0, 0), 1.5) - bub, 0, 1)
    hp = L - cv2.GaussianBlur(L, (0, 0), 6)                          # the surface's own fine pattern, kept faintly
    fL = (Ls - Ls[m].mean()) * 0.6 + 163 + 1.2 * cloud + 2.6 * micro + 0.3 * hp - 14 * np.clip((d - 0.75) / 0.25, 0, 1) ** 2
    fL = fL + 5 * bub - 9 * bub_rim
    fL = np.where(L > 175, np.maximum(fL, L), fL)                    # keep the real reflection on the meniscus
    fA, fB = 3.2 + 0.4 * cloud, 20.5 + 0.8 * cloud
    nL = L + (fL - L) * cover
    nA = A + (fA - A) * cover; nB = B + (fB - B) * cover
    ring = np.exp(-((d - 0.992) / 0.006) ** 2) * 0.15                # thin meniscus line at the wall
    nL = nL * (1 - ring)
    out = np.stack([nL, nA + 128, nB + 128], 2)
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    return (img * (1 - inside[..., None]) + out * inside[..., None]).astype(np.uint8)

def grade(img):
    """gentle: neutral-warm white balance from the floor tiles, a touch of exposure, soft S-curve, mild vibrance"""
    f = img.astype(np.float32) / 255
    floor = f[300:700, 1300:1800].reshape(-1, 3).mean(0)             # grey tiles, BGR
    wb = np.ones(3)                                                   # the photo's own white balance is kept
    f = np.clip(f * wb * 1.05, 0, 1)
    s = 1 / (1 + np.exp(-(f - 0.5) * 5.5)); lo, hi = 1 / (1 + np.exp(2.75)), 1 / (1 + np.exp(-2.75))
    f = f * 0.75 + ((s - lo) / (hi - lo)) * 0.25                      # soft contrast
    hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255; hsv[..., 1] = np.clip(sat + 0.06 * sat * (1 - sat), 0, 1) * 255
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 2.0)
    return out

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]; dbg = sys.argv[3] if len(sys.argv) > 3 else None
    img = cv2.imread(src)
    img = clean_stains(img, dbg)
    img = foam(img)
    img = grade(img)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 96])
