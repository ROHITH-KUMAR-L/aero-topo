import os
import glob
from typing import List, Dict, Any
from app.datasets.base import BaseDatasetAdapter

class FLAME3DatasetAdapter(BaseDatasetAdapter):
    """
    Adapter for FLAME 3 Wildfire Dataset.
    Expected structure:
    Root/
        Fire/
            RGB/
            Thermal/
        No Fire/
            RGB/
            Thermal/
    """
    def scan(self) -> List[Dict[str, Any]]:
        samples = []
        if not os.path.exists(self.root_dir):
            return samples

        categories = ["Fire", "No Fire", "fire", "no_fire"]
        for cat in categories:
            cat_dir = os.path.join(self.root_dir, cat)
            if not os.path.exists(cat_dir):
                continue

            # Search RGB files
            rgb_files = glob.glob(os.path.join(cat_dir, "**", "*.jpg"), recursive=True) + \
                        glob.glob(os.path.join(cat_dir, "**", "*.png"), recursive=True)

            for rgb_path in rgb_files:
                basename = os.path.splitext(os.path.basename(rgb_path))[0]
                
                # Match paired thermal TIFF/PNG file
                thermal_dir = os.path.join(os.path.dirname(os.path.dirname(rgb_path)), "Thermal")
                if not os.path.exists(thermal_dir):
                    thermal_dir = os.path.join(cat_dir, "Thermal")

                thermal_candidates = glob.glob(os.path.join(thermal_dir, "**", f"*{basename}*"), recursive=True)
                if thermal_candidates:
                    samples.append({
                        "sample_id": f"flame3_{cat.replace(' ', '_')}_{basename}",
                        "rgb_path": rgb_path,
                        "thermal_path": thermal_candidates[0],
                        "source": "FLAME3",
                        "category": cat,
                        "metadata": {
                            "rgb_filename": os.path.basename(rgb_path),
                            "thermal_filename": os.path.basename(thermal_candidates[0])
                        }
                    })

        return samples
