"""Where people are, for text placement: face boxes (FaceLandmarker, unioned over the frames) and a coarse person mask
(selfie segmenter, 20 px cells, max over the frames). MediaPipe, run in the mpenv.
  mpenv/bin/python face_boxes.py MODELS_DIR frame1.png [frame2.png ...]  → JSON {"faces": [[x0, y0, x1, y1], ...], "person": [[0..1]]}
"""
import json, os, sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

def main(models, frames):
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(models, "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_faces=4, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5))
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(models, "selfie_segmenter.tflite")), output_confidence_masks=True))
    boxes, person = [], None
    for f in frames:
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); h, w = im.shape[:2]
        m = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(im))).confidence_masks[0].numpy_view()
        g = cv2.resize((m > 0.5).astype(np.float32), (w // 20, h // 20), interpolation=cv2.INTER_AREA)
        person = g if person is None else np.maximum(person, g)
        found = []                                                         # the full frame, then overlapping upscaled tiles for small faces
        tiles = [(0, 0, w, h)] + [(x, y, t, t) for t in (w // 2, w // 3) for y in range(0, h - t + 1, t // 2) for x in range(0, w - t + 1, t // 2)]
        for x, y, tw, th in tiles:
            src = cv2.resize(im[y:y + th, x:x + tw], (768, 768)) if tw < w else im
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(src)))
            found += [np.array([(q.x * tw + x, q.y * th + y) for q in fl]) for fl in r.face_landmarks]
            if tw == w and r.face_landmarks: break
        for p in found:
            x0, y0 = p.min(0); x1, y1 = p.max(0)
            if g[int(y0) // 20:int(y1) // 20 + 1, int(x0) // 20:int(x1) // 20 + 1].mean() < 0.25: continue   # not on a person: a toy, a print
            b =[float(x0), float(y0 - 0.25 * (y1 - y0)), float(x1), float(y1)]       # + the forehead / hair line
            for u in boxes:                                                # same face in another frame → grow its box
                if not (b[2] < u[0] or b[0] > u[2] or b[3] < u[1] or b[1] > u[3]):
                    u[:] = [min(u[0], b[0]), min(u[1], b[1]), max(u[2], b[2]), max(u[3], b[3])]; break
            else: boxes.append(b)
    print(json.dumps({"faces": [[round(v) for v in b] for b in boxes], "person": np.round(person, 2).tolist()}))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
