from pathlib import Path
import tempfile
import unittest

import numpy as np

from datasets.frames import list_frames
from datasets.epic_kitchens import build_clip_tiers, load_visor_annotations, rasterize_object
from datasets.epic_kitchens.annotations import EpicFrame, VisorObject
from models.segmentation.deva.visor_split import DevaSplitConfig, split_frame_ids
from simplification.hands import infer_hand_circle, intersection_percentage
from simplification.fusion import baseline_fusion
from simplification.priority import PriorityConfig, object_brightness
from simplification.temporal import WeightedAverageConfig, weighted_average


class SmokeTests(unittest.TestCase):
    def test_import_pipeline_modules(self):
        import pipelines.han_baseline  # noqa: F401
        import pipelines.combination1  # noqa: F401
        import pipelines.temporal_priority  # noqa: F401
        import models.segmentation.deva  # noqa: F401
        import models.depth.cached  # noqa: F401
        import models.depth.tc_monodepth  # noqa: F401
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

    def test_baseline_fusion_uses_han_saliency_threshold_rule(self):
        saliency = np.array([[0, 50], [90, 100]], dtype=np.uint8)
        segmentation = np.array([[0, 255], [0, 0]], dtype=np.uint8)
        depth = np.array([[0, 64], [128, 255]], dtype=np.uint8)

        fused = baseline_fusion(
            segmentation=segmentation,
            saliency=saliency,
            depth=depth,
            saliency_threshold_fraction=0.90,
        )

        self.assertEqual(fused[0, 0], 0)
        self.assertGreater(fused[0, 1], 0)
        self.assertEqual(fused[1, 0], 0)
        self.assertEqual(fused[1, 1], 255)

    def test_deva_split_uses_annotation_coverage_for_large_instances(self):
        raw_ids = np.ones((30, 30), dtype=np.uint32)
        frame = EpicFrame(
            video_id="P01_01",
            frame_name="P01_01_frame_0000000001.jpg",
            frame_index=1,
            image_path="P01_01/frame_0000000001.jpg",
            annotations=(
                VisorObject(name="right hand", track_id="hand", segments=(((2, 2), (8, 2), (8, 8), (2, 8)),)),
                VisorObject(name="cup", track_id="cup", segments=(((20, 20), (25, 20), (25, 25), (20, 25)),)),
            ),
        )

        arms, objects, scenes = split_frame_ids(raw_ids, frame, config=DevaSplitConfig())

        self.assertGreater(np.count_nonzero(arms), 0)
        self.assertGreater(np.count_nonzero(objects), 0)
        self.assertGreater(np.count_nonzero(scenes), 0)
        self.assertEqual(arms[4, 4], 255)
        self.assertEqual(objects[22, 22], 255)
        self.assertEqual(scenes[15, 15], 255)

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
                        {"name": "left hand", "id": "hand-1", "in_contact_object": "cup-1", "segments": [[[0, 0], [4, 0], [4, 4], [0, 4]]]},
                        {"name": "cup", "id": "cup-1", "segments": [[[10, 10], [14, 10], [14, 14], [10, 14]]]}
                      ]
                    },
                    {
                      "image": {"name": "P01_01_frame_0000000002.jpg", "image_path": "P01_01/frame_0000000002.jpg", "video": "P01_01"},
                      "annotations": [
                        {"name": "left hand", "id": "hand-1", "segments": [[[1, 1], [5, 1], [5, 5], [1, 5]]]},
                        {"name": "cup", "id": "cup-1", "segments": [[[11, 11], [15, 11], [15, 15], [11, 15]]]}
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

    def test_visor_rasterizer_scales_annotation_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            path.write_text(
                """
                {
                  "info": {"description": "All annotations generated in 10x5 resolution"},
                  "video_annotations": [
                    {
                      "image": {"name": "P01_01_frame_0000000001.jpg", "image_path": "P01_01/frame_0000000001.jpg", "video": "P01_01"},
                      "annotations": [
                        {"name": "right hand", "id": "hand-1", "segments": [[[8, 1], [9, 1], [9, 2], [8, 2]]]}
                      ]
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            frame = load_visor_annotations(path)[0]
            mask = rasterize_object(frame.annotations[0], (10, 20))
            ys, xs = np.nonzero(mask)

            self.assertEqual(frame.annotation_size, (10, 5))
            self.assertGreater(xs.min(), 14)
            self.assertLess(ys.min(), 3)


if __name__ == "__main__":
    unittest.main()
