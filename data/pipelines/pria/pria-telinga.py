"""
Pipeline: pria-telinga
Extracts the MOST VISIBLE ear (left or right) from approved male face images.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_pipeline import BasePipeline, MIN_CROP

LEFT_EDGE = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
RIGHT_EDGE = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_JAW = 172
RIGHT_JAW = 397
FOREHEAD = 10
CHIN = 152


class EarPipeline(BasePipeline):
    def crop_part(self, image, landmarks):
        h, w = image.shape[:2]
        nose_x = landmarks[NOSE_TIP][0]
        face_center_x = (landmarks[LEFT_EDGE].mean(axis=0)[0] +
                         landmarks[RIGHT_EDGE].mean(axis=0)[0]) / 2

        left_pts = landmarks[LEFT_EDGE]
        right_pts = landmarks[RIGHT_EDGE]

        if nose_x > face_center_x:
            ear_pts = left_pts
            eye_y = landmarks[LEFT_EYE_OUTER][1]
            jaw_y = landmarks[LEFT_JAW][1]
            is_left = True
        else:
            ear_pts = right_pts
            eye_y = landmarks[RIGHT_EYE_OUTER][1]
            jaw_y = landmarks[RIGHT_JAW][1]
            is_left = False

        face_h = landmarks[CHIN][1] - landmarks[FOREHEAD][1]
        ear_top = int(eye_y - face_h * 0.08)
        ear_bot = int(jaw_y + face_h * 0.05)
        ear_cy = (ear_top + ear_bot) // 2
        ear_h_half = (ear_bot - ear_top) // 2

        if is_left:
            ear_cx = int(ear_pts[:, 0].min())
        else:
            ear_cx = int(ear_pts[:, 0].max())

        ear_w_half = int(ear_h_half * 0.6)

        x1 = max(0, ear_cx - ear_w_half)
        y1 = max(0, ear_cy - ear_h_half)
        x2 = min(w, ear_cx + ear_w_half)
        y2 = min(h, ear_cy + ear_h_half)

        crop = image[y1:y2, x1:x2]
        if crop.shape[0] < MIN_CROP or crop.shape[1] < MIN_CROP:
            return None
        return crop


ALL_LANDMARKS = (LEFT_EDGE + RIGHT_EDGE +
                 [NOSE_TIP, LEFT_EYE_OUTER, RIGHT_EYE_OUTER,
                  LEFT_JAW, RIGHT_JAW, FOREHEAD, CHIN])

pipeline = EarPipeline(
    gender="pria",
    part="telinga",
    landmarks=ALL_LANDMARKS,
    pad=0.30,
    port=6106,
)

if __name__ == "__main__":
    pipeline.run()
