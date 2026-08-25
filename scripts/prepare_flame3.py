import os
import sys
import json
import logging
from app.datasets.flame3 import FLAME3DatasetAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PrepareFLAME3")

def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "data/flame3"
    logger.info(f"Scanning FLAME 3 dataset in directory: {root_dir}")
    
    adapter = FLAME3DatasetAdapter(root_dir)
    samples = adapter.scan()

    out_file = "data/flame3_normalized.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    with open(out_file, "w") as f:
        json.dump(samples, f, indent=2)

    logger.info(f"Discovered {len(samples)} aligned RGB+Thermal pairs. Saved index to {out_file}")

if __name__ == "__main__":
    main()
