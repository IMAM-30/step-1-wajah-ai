"""
Pipeline: wanita-bibir
Extracts lip region from approved female face images.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_pipeline import BasePipeline

LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402, 317,
]

pipeline = BasePipeline(
    gender="wanita",
    part="bibir",
    landmarks=LIP_INDICES,
    pad=0.30,
    port=6003,
)

if __name__ == "__main__":
    pipeline.run()
