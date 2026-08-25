import os
import glob
from typing import List, Dict, Any
from app.datasets.base import BaseDatasetAdapter

class IGNITEDatasetAdapter(BaseDatasetAdapter):
    """
    Adapter for IGNITE Paired Wildfire Dataset.
    Expected structure:
    Root/
        RGB/
        Thermal/
    """
    def scan(self) -> List[Dict[str, Any]]:
        samples = []
        if not os.path.exists(self.root_dir):
            return samples

        rgb_dir = os.path.join(self.root_dir, "RGB")
        thermal_dir = os.path.join(self.root_dir, "Thermal")

        if not os.path.exists(rgb_dir):
            rgb_dir = self.root_dir

        rgb_files = glob.glob(os.path.join(rgb_dir, "*.jpg")) + glob.glob(os.path.join(rgb_dir, "*.png"))

        for rgb_path in rgb_files:
            filename = os.path.basename(rgb_path)
            basename = os.path.splitext(filename)[0]

            thermal_match = os.path.join(thermal_dir, filename)
            if not os.path.exists(thermal_match):
                thermal_candidates = glob.glob(os.path.join(thermal_dir, f"*{basename}*"))
                if thermal_candidates:
                    thermal_match = thermal_candidates[0]
                else:
                    thermal_match = None

            if thermal_match and os.path.exists(thermal_match):
                samples.append({
                    "sample_id": f"ignite_{basename}",
                    "rgb_path": rgb_path,
                    "thermal_path": thermal_match,
                    "source": "IGNITE",
                    "category": "Wildfire",
                    "metadata": {
                        "rgb_filename": filename,
                        "thermal_filename": os.path.basename(thermal_match)
                    }
                })

        return samples
