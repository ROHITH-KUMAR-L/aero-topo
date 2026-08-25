import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseDatasetAdapter(ABC):
    """
    Abstract Base Class for Aero-Topo Paired RGB+Thermal Dataset Adapters.
    Normalizes dataset samples into a standard internal metadata schema:
    [
        {
            "sample_id": "flame3_fire_001",
            "rgb_path": "/path/to/rgb.png",
            "thermal_path": "/path/to/thermal.tif",
            "source": "FLAME3",
            "category": "Fire",
            "metadata": {}
        },
        ...
    ]
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    @abstractmethod
    def scan(self) -> List[Dict[str, Any]]:
        """
        Scan directory and discover paired RGB + Thermal samples.
        """
        pass
