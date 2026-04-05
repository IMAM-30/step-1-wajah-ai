"""
Pipeline: pria-baju
Extracts neck + collar/clothing region from approved male face images.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_pipeline import BasePipeline, MIN_CROP

CHIN_TIP = 152
LIP_BOTTOM = 17
FOREHEAD = 10
JAW_LEFT = [172, 136, 150, 149, 176, 148]
JAW_RIGHT = [397, 365, 379, 378, 400, 377]


class BajuPipeline(BasePipeline):
    def crop_part(self, image, landmarks):
        h, w = image.shape[:2]
        face_h = landmarks[CHIN_TIP][1] - landmarks[FOREHEAD][1]
        chin_y = landmarks[CHIN_TIP][1]

        crop_top = int(chin_y - face_h * 0.05)
        crop_bot = h

        x1 = 0
        y1 = max(0, crop_top)
        x2 = w
        y2 = crop_bot

        crop = image[y1:y2, x1:x2]
        if crop.shape[0] < MIN_CROP or crop.shape[1] < MIN_CROP:
            return None
        return crop


ALL_LANDMARKS = JAW_LEFT + JAW_RIGHT + [CHIN_TIP, LIP_BOTTOM, FOREHEAD]

pipeline = BajuPipeline(
    gender="pria",
    part="baju",
    landmarks=ALL_LANDMARKS,
    pad=0.50,
    port=6107,
)

if __name__ == "__main__":
    pipeline.run()
