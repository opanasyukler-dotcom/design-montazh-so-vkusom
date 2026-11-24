import cv2, numpy as np, json, sys
src, model, out = sys.argv[1:4]
cap = cv2.VideoCapture(src); W_, H_ = int(cap.get(3)), int(cap.get(4))
sc = 0.5; det = cv2.FaceDetectorYN.create(model, '', (int(W_*sc), int(H_*sc)), 0.5, 0.3, 50)
res = []; i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    _, fs = det.detect(cv2.resize(f, None, fx=sc, fy=sc))
    res.append([] if fs is None else [[float(v)/sc for v in r[:4]] + [float(r[14])] for r in fs])
    i += 1
json.dump(res, open(out, 'w')); print(i)
