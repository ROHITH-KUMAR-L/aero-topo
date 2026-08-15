import unittest
import numpy as np

class TestAeroTopoPipeline(unittest.TestCase):

    def test_preprocessing_resize(self):
        from utils.preprocessing import resize_and_normalize
        dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        resized = resize_and_normalize(dummy_img, target_size=(256, 256))
        self.assertEqual(resized.shape, (256, 256, 3))
        self.assertEqual(resized.dtype, np.uint8)

    def test_base64_encoding(self):
        from utils.preprocessing import numpy_to_base64_data_url
        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        b64_url = numpy_to_base64_data_url(dummy_img)
        self.assertTrue(b64_url.startswith("data:image/png;base64,"))

    def test_canny_failsafe(self):
        from pipeline.failsafe import run_canny_failsafe
        dummy_thermal = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        edges = run_canny_failsafe(dummy_thermal)
        self.assertEqual(edges.shape, (256, 256, 3))
        self.assertEqual(edges.dtype, np.uint8)

    def test_projection_math(self):
        from utils.projection import project_depth_to_3d_points
        dummy_zmap = np.ones((256, 256), dtype=np.float32) * 50.0
        pts = project_depth_to_3d_points(dummy_zmap, subsample_stride=4)
        self.assertIn("x", pts)
        self.assertIn("y", pts)
        self.assertIn("z", pts)
        self.assertEqual(pts["x"].shape, (64, 64))
        self.assertEqual(pts["z"].shape, (64, 64))

    def test_cgan_fallback(self):
        from models.cgan import ThermalToRGBEngine
        engine = ThermalToRGBEngine()
        dummy_thermal = np.ones((256, 256, 3), dtype=np.uint8) * 100
        translated = engine.translate(dummy_thermal)
        self.assertEqual(translated.shape, (256, 256, 3))
        self.assertEqual(translated.dtype, np.uint8)

    def test_depth_fallback(self):
        from models.depth_engine import DepthEstimationEngine
        engine = DepthEstimationEngine()
        dummy_rgb = np.ones((256, 256, 3), dtype=np.uint8) * 150
        depth_map = engine.estimate_depth(dummy_rgb)
        self.assertEqual(depth_map.shape, (256, 256))
        self.assertEqual(depth_map.dtype, np.uint8)

if __name__ == "__main__":
    unittest.main()
