"""
Pipeline: wanita-mata
Extracts the MOST VISIBLE eye (left or right) close-up.
Picks the eye that is more facing the camera.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_pipeline import BasePipeline, MIN_CROP

# Left eye + brow
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

# Right eye + brow
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152


class MataPipeline(BasePipeline):
    def crop_part(self, image, landmarks):
        h, w = image.shape[:2]

        # Determine which eye is more visible based on eye width (spread)
        left_pts = landmarks[LEFT_EYE]
        right_pts = landmarks[RIGHT_EYE]
        left_width = np.ptp(left_pts[:, 0])
        right_width = np.ptp(right_pts[:, 0])

        # Pick the wider eye (more open/facing camera)
        if left_width >= right_width:
            eye_pts = left_pts
            brow_pts = landmarks[LEFT_BROW]
        else:
            eye_pts = right_pts
            brow_pts = landmarks[RIGHT_BROW]

        all_pts = np.vstack([eye_pts, brow_pts])
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)
        bw, bh = x_max - x_min, y_max - y_min

        pad_x = int(bw * 0.40)
        pad_y = int(bh * 0.50)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + int(pad_y * 0.7))

        crop = image[y1:y2, x1:x2]
        if crop.shape[0] < MIN_CROP or crop.shape[1] < MIN_CROP:
            return None
        return crop


ALL_LANDMARKS = LEFT_EYE + LEFT_BROW + RIGHT_EYE + RIGHT_BROW + [NOSE_TIP, FOREHEAD, CHIN]

pipeline = MataPipeline(
    gender="wanita",
    part="mata",
    landmarks=ALL_LANDMARKS,
    pad=0.30,
    port=6002,
)

if __name__ == "__main__":
    pipeline.run()
