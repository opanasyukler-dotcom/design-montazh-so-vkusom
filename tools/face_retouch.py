"""Natural face retouch + soft «no-makeup» makeup for video frames (RGB uint8, any size), MediaPipe based.
  • skin: edge-preserving smoothing on face skin only, fine texture kept (pores stay), redness evened out
  • under-eyes: a touch lighter, less purple
  • makeup: soft rose blush on the cheek apples, nude-rose lip tint, brows a shade deeper, eyes a bit clearer
Face landmarks (FaceLandmarker) are found on a downscaled frame and lightly smoothed in time; the skin mask is the face
oval ∩ the selfie-multiclass «face skin» class, minus eyes / brows / lips, all feathered so nothing has an edge.
Needs the mpenv (mediapipe) and models: face_landmarker.task, selfie_multiclass.tflite.
"""
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
EYE_L = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_R = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LOW_L = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LOW_R = [263, 249, 390, 373, 374, 380, 381, 382, 362]
BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_R = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
NOSTRIL = [98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64]
CHEEK_L, CHEEK_R = 205, 425

class Retoucher:
    def __init__(self, models, strength=1.0):
        self.lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(models, "face_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_faces=1))
        self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(models, "selfie_multiclass.tflite")), output_confidence_masks=True))
        self.prev, self.ts, self.k = None, 0, strength

    def landmarks(self, fr):
        h, w = fr.shape[:2]; s = 960 / w; small = cv2.resize(fr, (960, int(h * s)), interpolation=cv2.INTER_AREA)
        self.ts += 40
        r = self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)), self.ts)
        if not r.face_landmarks: return self.prev
        p = np.array([(q.x * w, q.y * h) for q in r.face_landmarks[0]], np.float32)
        if self.prev is not None and np.abs(p - self.prev).mean() < 0.01 * w: p = 0.65 * p + 0.35 * self.prev   # calm jitter, no lag on moves
        self.prev = p; return p

    def apply(self, fr):
        P = self.landmarks(fr)
        if P is None: return fr
        h, w = fr.shape[:2]; x0, y0 = P[OVAL].min(0); x1, y1 = P[OVAL].max(0); fw = x1 - x0
        pad = int(0.25 * fw); bx0, by0 = int(max(x0 - pad, 0)), int(max(y0 - pad, 0)); bx1, by1 = int(min(x1 + pad, w)), int(min(y1 + pad, h))
        crop = fr[by0:by1, bx0:bx1].copy(); ch, cw = crop.shape[:2]; Q = P - (bx0, by0)
        poly = lambda idx: Q[idx].astype(np.int32)
        def fill(idx, grow=0):
            m = np.zeros((ch, cw), np.uint8); cv2.fillPoly(m, [poly(idx)], 255)
            if grow: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1)))
            return m
        # skin mask: oval ∩ «face skin», minus features; feathered
        seg = self.seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.resize(crop, (256, 256)))))
        face_skin = cv2.resize(seg.confidence_masks[3].numpy_view(), (cw, ch))
        g = max(3, int(fw * 0.018))
        feats = cv2.max(cv2.max(fill(EYE_L, g), fill(EYE_R, g)), cv2.max(fill(BROW_L, g), fill(BROW_R, g)))
        feats = cv2.max(feats, fill(LIPS, g))
        skin = (fill(OVAL).astype(np.float32) / 255) * np.clip(face_skin * 1.3, 0, 1) * (1 - feats.astype(np.float32) / 255)
        skin = cv2.GaussianBlur(skin, (0, 0), fw * 0.012)
        out = crop.astype(np.float32)
        # 1. skin: bilateral base + the finest texture back (pores stay), redness evened
        base = cv2.bilateralFilter(crop, 0, 20, max(3, fw * 0.012)).astype(np.float32)
        fine = out - cv2.GaussianBlur(out, (0, 0), max(1.0, fw * 0.0022))
        smooth = base + 0.65 * fine
        lab = cv2.cvtColor(np.clip(smooth, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        a_lo = cv2.GaussianBlur(lab[..., 1], (0, 0), fw * 0.05); lab[..., 1] = a_lo + (lab[..., 1] - a_lo) * 0.6   # blotchy red evened
        smooth = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        m = (skin * 0.75 * self.k)[..., None]; out = out * (1 - m) + smooth * m
        lab = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        # 2. under-eyes: the lower lid band, shifted down, a bit lighter and less purple
        ue = np.zeros((ch, cw), np.float32)
        for low, eye in ((LOW_L, EYE_L), (LOW_R, EYE_R)):
            e = Q[eye]; eh = e[:, 1].max() - e[:, 1].min(); band = Q[low].copy()
            pts = np.vstack([band, band[::-1] + (0, 1.6 * eh)]).astype(np.int32); cv2.fillPoly(ue, [pts], 1.0)
        ue = cv2.GaussianBlur(ue, (0, 0), fw * 0.03) * skin.clip(0.3, 1) * self.k
        lab[..., 0] += 5.0 * ue; lab[..., 2] += (128 + 4 - lab[..., 2]) * 0.25 * ue
        # 3. blush on the cheek apples
        bl = np.zeros((ch, cw), np.float32); r = fw * 0.13
        for c in (CHEEK_L, CHEEK_R):
            cv2.circle(bl, tuple(Q[c].astype(int)), int(r), 1.0, -1)
        bl = cv2.GaussianBlur(bl, (0, 0), r * 0.6) * skin * self.k
        lab[..., 1] += 5.5 * bl; lab[..., 2] += 1.5 * bl; lab[..., 0] -= 1.0 * bl
        # 4. lips: nude-rose tint, a little fuller colour, mouth opening left alone
        lp = (fill(LIPS).astype(np.float32) - fill(MOUTH).astype(np.float32)).clip(0, 255) / 255
        lp = cv2.GaussianBlur(lp, (0, 0), max(1.5, fw * 0.006)) * self.k
        lab[..., 1] += (150 - lab[..., 1]) * 0.22 * lp; lab[..., 2] += (138 - lab[..., 2]) * 0.15 * lp; lab[..., 0] -= 2.0 * lp
        # 5. brows a shade deeper
        br = cv2.GaussianBlur(cv2.max(fill(BROW_L), fill(BROW_R)).astype(np.float32) / 255, (0, 0), max(1.5, fw * 0.006)) * self.k
        lab[..., 0] -= 5.0 * br * (lab[..., 0] < 150)
        res = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        # 6. eyes a bit clearer: gentle local contrast inside the eye shapes
        ey = cv2.GaussianBlur(cv2.max(fill(EYE_L, 2), fill(EYE_R, 2)).astype(np.float32) / 255, (0, 0), 2)[..., None] * self.k
        res = res + (res - cv2.GaussianBlur(res, (0, 0), 3)) * 0.6 * ey + 3 * ey
        out = fr.copy(); out[by0:by1, bx0:bx1] = np.clip(res, 0, 255).astype(np.uint8)
        return out
