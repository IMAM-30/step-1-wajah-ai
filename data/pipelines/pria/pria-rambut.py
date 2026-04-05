"""
Pipeline: pria-rambut
Extracts full head + hair with face blurred precisely along face contour.
"""

import sys, os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_pipeline import BasePipeline, MIN_CROP

FACE_OVAL_ORDERED = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109,
]

FACE_INNER = [
    10, 151, 9, 8, 168, 6, 197, 195, 5,
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
    33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    1, 2, 3, 4, 5, 6, 19, 94, 125, 141, 168, 188, 195, 196, 197, 198,
    0, 11, 12, 13, 14, 15, 16, 17, 37, 39, 40, 61, 78, 80, 81, 82,
    84, 87, 88, 91, 95, 146, 178, 181, 185, 191,
    267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321,
    116, 117, 118, 119, 120, 100, 142, 203, 206, 207,
    345, 346, 347, 348, 349, 329, 371, 423, 426, 427,
]

CHIN_TIP = 152
FOREHEAD = 10


class RambutPipeline(BasePipeline):
    def crop_part(self, image, landmarks):
        h, w = image.shape[:2]
        face_h = landmarks[CHIN_TIP][1] - landmarks[FOREHEAD][1]

        crop_top = 0
        crop_bot = int(landmarks[CHIN_TIP][1] + face_h * 0.3)
        crop_bot = min(h, crop_bot)

        crop = image[crop_top:crop_bot, 0:w].copy()
        ch = crop.shape[0]

        oval_pts = []
        for idx in FACE_OVAL_ORDERED:
            px = landmarks[idx][0]
            py = landmarks[idx][1] - crop_top
            py = max(0, min(ch - 1, py))
            oval_pts.append([px, py])
        oval_pts = np.array(oval_pts, dtype=np.int32)

        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [oval_pts], 255)

        blurred = cv2.GaussianBlur(crop, (99, 99), 60)
        crop = np.where(mask[:, :, None] == 255, blurred, crop)

        if crop.shape[0] < MIN_CROP or crop.shape[1] < MIN_CROP:
            return None
        return crop


ALL_LANDMARKS = list(set(FACE_OVAL_ORDERED + FACE_INNER + [CHIN_TIP, FOREHEAD]))

pipeline = RambutPipeline(
    gender="pria",
    part="rambut",
    landmarks=ALL_LANDMARKS,
    pad=0.0,
    port=6105,
)

if __name__ == "__main__":
    pipeline.run()
