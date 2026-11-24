import cv2, numpy as np, json, sys
W = sys.argv[1]; x0,y0,x1,y1 = 352,30,928,840
cap = cv2.VideoCapture(W+'/SR.mp4'); fr = []
while True:
    ok,f = cap.read()
    if not ok: break
    fr.append(f[y0:y1,x0:x1])
print(len(fr))
clean = fr[90].astype(np.int16); fin = fr[int(77*30)].astype(np.int16)
hsv = cv2.cvtColor(fin.astype(np.uint8), cv2.COLOR_BGR2HSV)
d = np.abs(fin-clean).max(2)
col = (hsv[...,1]>50) & (((hsv[...,0]>=35)&(hsv[...,0]<=130))) & (d>40)
n,lab,st,_ = cv2.connectedComponentsWithStats(col.astype(np.uint8),8)
keep = np.zeros_like(col)
for i in range(1,n):
    if st[i,4]>60: keep |= lab==i
# stability over time: check the photo doesn't move (compare the face region)
first = np.full(col.shape, -1, np.int32)
ys,xs = np.where(keep)
for i in range(90, int(77*30)+1):
    f = fr[i].astype(np.int16)
    on = (np.abs(f[ys,xs]-fin[ys,xs]).max(1) < 45) & (np.abs(f[ys,xs]-clean[ys,xs]).max(1) > 35)
    new = on & (first[ys,xs] < 0); first[ys[new], xs[new]] = i
    off = ~on & (first[ys,xs] >= 0) & (i - first[ys,xs] < 15); first[ys[off],xs[off]] = -1
print('unset', (first[ys,xs]<0).sum(), 'of', len(ys))
first[keep & (first<0)] = int(77*30)
n2,lab2,st2,_ = cv2.connectedComponentsWithStats(keep.astype(np.uint8),8)
info = {}
for i in range(1,n2):
    m = lab2==i; ts = first[m]
    cy,cx = np.argwhere(m).mean(0)
    info[i] = dict(n=int(m.sum()), cx=float(cx), cy=float(cy), t0=float(ts.min()/30), t1=float(np.percentile(ts,98)/30))
for k,v in sorted(info.items(), key=lambda kv: kv[1]['t0']): print(k, v)
cv2.imwrite(W+'/SR_clean.png', fr[90]); cv2.imwrite(W+'/SR_final.png', fr[int(77*30)])
cv2.imwrite(W+'/SR_mask.png', keep.astype(np.uint8)*255)
np.save(W+'/SR_first.npy', first)
np.save(W+'/SR_lab.npy', lab2.astype(np.int16))
