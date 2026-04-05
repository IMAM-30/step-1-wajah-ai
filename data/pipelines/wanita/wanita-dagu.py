"""
Pipeline: wanita-dagu
Extracts chin region centered below the lips.
Uses lip bottom + chin tip as vertical reference.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_pipeline import BasePipeline, MIN_CROP

# Reference landmarks
CHIN_TIP = 152          # bottom of chin
LIP_BOTTOM = 17         # bottom center of lower lip
LEFT_JAW = 172           # left jawline
RIGHT_JAW = 397          # right jawline
FOREHEAD = 10
NOSE_TIP = 1

# Jawline contour for width
JAW_LEFT = [172, 136, 150, 149, 176, 148]
JAW_RIGHT = [397, 365, 379, 378, 400, 377]


class ChinPipeline(BasePipeline):
    def crop_part(self, image, landmarks):
        h, w = image.shape[:2]

        # Vertical: from just below lips to below chin
        lip_bot_y = landmarks[LIP_BOTTOM][1]
        chin_y = landmarks[CHIN_TIP][1]
        face_h = landmarks[CHIN_TIP][1] - landmarks[FOREHEAD][1]

        # Top: at lower lip level
        crop_top = int(lip_bot_y - face_h * 0.05)
        # Bottom: below chin tip with more margin
        crop_bot = int(chin_y + face_h * 0.18)

        # Vertical center
        cy = (crop_top + crop_bot) // 2

        # Horizontal: center between jaw sides
        left_jaw_x = int(landmarks[JAW_LEFT].mean(axis=0)[0])
        right_jaw_x = int(landmarks[JAW_RIGHT].mean(axis=0)[0])
        cx = (left_jaw_x + right_jaw_x) // 2

        # Width: full jaw width + margin
        jaw_width = right_jaw_x - left_jaw_x
        half_w = int(jaw_width * 0.65)

        # Height
        half_h = (crop_bot - crop_top) // 2

        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(w, cx + half_w)
        y2 = min(h, cy + half_h)

        crop = image[y1:y2, x1:x2]
        if crop.shape[0] < MIN_CROP or crop.shape[1] < MIN_CROP:
            return None
        return crop


ALL_LANDMARKS = JAW_LEFT + JAW_RIGHT + [CHIN_TIP, LIP_BOTTOM, FOREHEAD, NOSE_TIP]

pipeline = ChinPipeline(
    gender="wanita",
    part="dagu",
    landmarks=ALL_LANDMARKS,
    pad=0.25,
    port=6004,
)

if __name__ == "__main__":
    pipeline.run()
