"""put a known text onto the word timings clean_speech.py mapped (difflib match, gaps by letter count) -> CLIP.subs.json
usage: python3 subs_retext.py CLIP.mp4 "exact text"
"""
import sys, json, difflib, re, numpy as np
clip, text = sys.argv[1:3]
ww = json.load(open(clip + '.words.json'))[0]['w']
norm = lambda t: re.sub(r'[^а-яёa-z0-9]', '', t.lower()).replace('ё', 'е')
toks = text.split(); tim = [None] * len(toks)
sm = difflib.SequenceMatcher(None, [norm(t) for t in toks], [norm(w[2]) for w in ww], autojunk=False)
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op == 'equal' or (op == 'replace' and i2 - i1 == j2 - j1):
        for k in range(i2 - i1): tim[i1 + k] = tuple(ww[j1 + k][:2])
    elif op == 'replace':                                   # spread i-tokens over the j-span
        a, b = ww[j1][0], ww[j2 - 1][1]; L = [max(len(norm(t)), 1) for t in toks[i1:i2]]; c = np.cumsum([0] + L) / sum(L)
        for k in range(i1, i2): tim[k] = (a + (b - a) * c[k - i1], a + (b - a) * c[k - i1 + 1])
i = 0
while i < len(toks):
    if tim[i] is None:
        j = i
        while j < len(toks) and tim[j] is None: j += 1
        t0 = tim[i - 1][1] if i > 0 else max(0.0, tim[j][0] - 0.3 * (j - i))
        t1 = tim[j][0] if j < len(toks) else t0 + 0.3 * (j - i)
        L = [max(len(norm(t)), 1) for t in toks[i:j]]; c = np.cumsum([0] + L) / sum(L)
        for k in range(i, j): tim[k] = (t0 + (t1 - t0) * c[k - i], t0 + (t1 - t0) * c[k - i + 1])
        i = j
    else: i += 1
json.dump([{'w': [[a, b, t] for (a, b), t in zip(tim, toks)]}], open(clip + '.subs.json', 'w'), ensure_ascii=False)
print(clip.split('/')[-1], ' '.join(f'{t}[{a:.2f}]' for (a, b), t in zip(tim, toks)))
