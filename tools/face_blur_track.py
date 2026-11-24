import json, numpy as np, sys
R = json.load(open(sys.argv[1])); n = len(R); B = np.full((n, 4), np.nan)
for i, fs in enumerate(R):
    c = [r for r in fs if r[2] < 420 and r[3] < 460 and r[0] + r[2] / 2 < 620 and r[4] > 0.5]
    if c: r = max(c, key=lambda r: r[4]); B[i] = [r[0] + r[2] / 2, r[1] + r[3] / 2, r[2], r[3]]
# split into runs at jumps, fill short gaps inside a run
ok = ~np.isnan(B[:, 0]); idx = np.where(ok)[0]
for a, b in zip(idx[:-1], idx[1:]):
    if 1 < b - a <= 36 and np.hypot(*(B[b, :2] - B[a, :2])) < 220:
        for j in range(a + 1, b): B[j] = B[a] + (B[b] - B[a]) * (j - a) / (b - a)
# smooth each continuous run
ok = ~np.isnan(B[:, 0]); out = B.copy(); i = 0
while i < n:
    if not ok[i]: i += 1; continue
    j = i
    while j + 1 < n and ok[j + 1] and np.hypot(*(B[j + 1, :2] - B[j, :2])) < 120: j += 1
    seg = B[i:j + 1]; k = np.exp(-0.5 * (np.arange(-12, 13) / 5.0) ** 2); k /= k.sum()
    pad = np.pad(seg, ((12, 12), (0, 0)), mode='edge'); out[i:j + 1] = np.stack([np.convolve(pad[:, c], k, 'valid') for c in range(4)], 1)
    out[i:j + 1, 2:] = np.maximum(out[i:j + 1, 2:], np.median(seg[:, 2:], 0) * 0.9)
    i = j + 1
json.dump([None if np.isnan(r[0]) else [float(v) for v in r] for r in out], open(sys.argv[2], 'w'))
runs = []; i = 0; okk = ~np.isnan(out[:, 0])
while i < n:
    if okk[i]:
        j = i
        while j + 1 < n and okk[j + 1]: j += 1
        runs.append((round(i / 30, 2), round(j / 30, 2))); i = j + 1
    else: i += 1
print(runs)
