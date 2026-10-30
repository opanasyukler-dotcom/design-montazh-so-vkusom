"""Privacy blur over the chest, tracked per frame (MediaPipe, mpenv): raw RGB frames in → raw RGB frames out.
  ffmpeg ... -f rawvideo -pix_fmt rgb24 - | mpenv/bin/python chest_blur.py MODELS_DIR W H FPS | ffmpeg -f rawvideo ...
The chest is found from the face (FaceLandmarker) — the band from just under the chin down about two face heights — cut to
the person's own silhouette (selfie segmenter), so only her body there is blurred, with a soft edge. Smoothed in time; when the
face is lost, the last place is kept.
"""
import os, sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

def main(models, w, h, fps):
    w, h, fps = int(w), int(h), float(fps)
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(models, "face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, min_face_detection_confidence=0.3, min_tracking_confidence=0.3))
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(models, "selfie_segmenter.tflite")), output_confidence_masks=True))
    yy = np.arange(h, dtype=np.float32)[:, None]; state = None; n = w * h * 3; fi = 0
    while True:
        buf = sys.stdin.buffer.read(n)
        if len(buf) < n: break
        fr = np.frombuffer(buf, np.uint8).reshape(h, w, 3); img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr))
        r = lm.detect_for_video(img, int(fi * 1000 / fps))
        if r.face_landmarks:
            p = np.array([(q.x * w, q.y * h) for q in r.face_landmarks[0]]); fh = p[:, 1].max() - p[:, 1].min()
            new = np.array([p[152, 1], fh])                                # chin y, face height
            state = new if state is None else 0.7 * state + 0.3 * new
        out = fr
        if state is not None:
            chin, fh = state; y0, y1 = chin + 0.35 * fh, chin + 2.6 * fh
            band = np.clip(np.minimum((yy - y0) / (0.25 * fh), (y1 - yy) / (0.35 * fh)), 0, 1)
            body = seg.segment(img).confidence_masks[0].numpy_view()
            body = cv2.GaussianBlur(cv2.dilate((body > 0.4).astype(np.float32), np.ones((41, 41), np.uint8)), (0, 0), 14)
            m = (band * body)[..., None]
            if m.max() > 0.01:
                bl = cv2.GaussianBlur(fr, (0, 0), 28)
                out = (fr * (1 - m) + bl * m).astype(np.uint8)
        sys.stdout.buffer.write(out.tobytes()); fi += 1

if __name__ == "__main__":
    main(*sys.argv[1:5])
