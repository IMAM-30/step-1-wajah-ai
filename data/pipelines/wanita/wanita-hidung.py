"""
Pipeline: wanita-hidung
Extracts nose region from approved female face images.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_pipeline import BasePipeline

NOSE_INDICES = [
    1, 2, 3, 4, 5, 6, 19, 94, 125, 141, 168,
    188, 195, 196, 197, 198, 209, 217,
    236, 237, 238, 239, 241, 242,
    354, 370, 392, 412, 419, 420,
    456, 457, 458, 459, 461, 462,
]

pipeline = BasePipeline(
    gender="wanita",
    part="hidung",
    landmarks=NOSE_INDICES,
    pad=0.25,
    port=6001,
)

if __name__ == "__main__":
    pipeline.run()
