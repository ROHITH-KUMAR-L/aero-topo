import os
import sys
import json
import logging
from app.datasets.ignite import IGNITEDatasetAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PrepareIGNITE")

def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "data/ignite"
    logger.info(f"Scanning IGNITE dataset in directory: {root_dir}")

    adapter = IGNITEDatasetAdapter(root_dir)
    samples = adapter.scan()

    out_file = "data/ignite_normalized.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, "w") as f:
        json.dump(samples, f, indent=2)

    logger.info(f"Discovered {len(samples)} aligned RGB+Thermal pairs. Saved index to {out_file}")

if __name__ == "__main__":
    main()
