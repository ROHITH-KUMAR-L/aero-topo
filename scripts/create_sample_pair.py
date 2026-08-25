import os
import numpy as np
import cv2

os.makedirs("data/sample_pair", exist_ok=True)

# Generate synthetic 640x512 RGB test image
rgb = np.zeros((512, 640, 3), dtype=np.uint8)
cv2.rectangle(rgb, (50, 50), (300, 300), (34, 139, 34), -1)   # Forest
cv2.rectangle(rgb, (320, 100), (600, 450), (139, 69, 19), -1)  # Terrain hill
cv2.circle(rgb, (450, 250), 40, (255, 69, 0), -1)             # Fire region
# Add smoke haze simulation
smoke_overlay = np.full((512, 640, 3), 200, dtype=np.uint8)
rgb = cv2.addWeighted(rgb, 0.6, smoke_overlay, 0.4, 0)
cv2.imwrite("data/sample_pair/rgb_sample.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# Generate synthetic 640x512 16-bit Thermal TIFF test image
thermal_16bit = np.full((512, 640), 2000, dtype=np.uint16)
# High heat at fire location
rr, cc = np.ogrid[:512, :640]
dist_from_fire = np.sqrt((rr - 250)**2 + (cc - 450)**2)
thermal_16bit[dist_from_fire < 60] = 8500  # Hot fire heat signature
cv2.imwrite("data/sample_pair/thermal_sample.tif", thermal_16bit)

print("Created sample RGB and 16-bit Thermal TIFF pair in data/sample_pair/")
