from pathlib import Path
import tempfile
import unittest

import numpy as np

from datasets.frames import list_frames
from datasets.epic_kitchens import build_clip_tiers, load_visor_annotations, rasterize_object
from simplification.hands import infer_hand_circle, intersection_percentage
from simplification.priority import PriorityConfig, object_brightness
from simplification.temporal import WeightedAverageConfig, weighted_average


class SmokeTests(unittest.TestCase):
    def test_import_pipeline_modules(self):
        import pipelines.han_baseline  # noqa: F401
        import pipelines.temporal_priority  # noqa: F401
        import models.segmentation.deva  # noqa: F401
        import models.depth.cached  # noqa: F401
        import models.saliency.cached  # noqa: F401

    def test_natural_frame_sorting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["frame_10.png", "frame_2.png", "frame_1.png"]:
                (root / name).touch()

            self.assertEqual(
                [p.name for p in list_frames(root)],
                ["frame_1.png", "frame_2.png", "frame_10.png"],
            )

    def test_weighted_average_keeps_consistent_pixels(self):
        frames = [
            np.array([[255, 0]], dtype=np.uint8),
            np.array([[255, 0]], dtype=np.uint8),
            np.array([[255, 255]], dtype=np.uint8),
        ]

        result = weighted_average(frames, WeightedAverageConfig(window=3, decay=1.0, threshold=0.7))

        self.assertGreater(result[0, 0], 0)
        self.assertEqual(result[0, 1], 0)

    def test_hand_circle_and_intersection(self):
        arm = np.zeros((100, 100), dtype=np.uint8)
        arm[10:80, 45:55] = 1
        circle = infer_hand_circle(arm)

        self.assertIsNotNone(circle)
        self.assertEqual(circle.mask.shape, arm.shape)
        self.assertEqual(intersection_percentage(circle.mask, circle.mask), 100.0)

    def test_priority_brightness_policy(self):
        config = PriorityConfig()

        self.assertEqual(object_brightness(near_gaze=True, config=config), config.gaze_brightness)
        self.assertEqual(object_brightness(near_hand_percent=75, config=config), config.primary_brightness)
        self.assertEqual(
            object_brightness(hand_recently_seen=True, config=config),
            config.secondary_with_hand_brightness,
        )

    def test_epic_visor_loader_and_tiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            path.write_text(
                """
                {
                  "video_annotations": [
                    {
                      "image": {"name": "P01_01_frame_0000000001.jpg", "image_path": "P01_01/frame_0000000001.jpg", "video": "P01_01"},
                      "annotations": [
                        {"name": "left hand", "segments": [[[0, 0], [4, 0], [4, 4], [0, 4]]]},
                        {"name": "cup", "key": "cup-1", "relation": "in_hand", "segments": [[[10, 10], [14, 10], [14, 14], [10, 14]]]}
                      ]
                    },
                    {
                      "image": {"name": "P01_01_frame_0000000002.jpg", "image_path": "P01_01/frame_0000000002.jpg", "video": "P01_01"},
                      "annotations": [
                        {"name": "left hand", "segments": [[[1, 1], [5, 1], [5, 5], [1, 5]]]},
                        {"name": "cup", "key": "cup-1", "segments": [[[11, 11], [15, 11], [15, 15], [11, 15]]]}
                      ]
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            frames = load_visor_annotations(path)
            tiers = build_clip_tiers(frames)
            cup_mask = rasterize_object(frames[0].annotations[1], (20, 20))

            self.assertEqual(len(frames), 2)
            self.assertIn("cup-1", tiers[0].foreground_ids)
            self.assertIn("cup-1", tiers[1].background_interactable_ids)
            self.assertGreater(cup_mask.sum(), 0)


if __name__ == "__main__":
    unittest.main()
