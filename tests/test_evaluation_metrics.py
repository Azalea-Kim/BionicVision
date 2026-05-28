from __future__ import annotations

import unittest

import numpy as np

from datasets.epic_kitchens.annotations import EpicFrame, VisorObject
from evaluation.metrics import (
    EvaluationConfig,
    evaluate_clip,
    evaluate_clip_variants,
    flow_compensated_flicker,
    object_representation_score,
    recall_auc,
    spv_load,
    track_fragmentation_rate,
    track_dropout_rate,
)


def _box(
    name: str,
    oid: str,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    relation: str | None = None,
    mask_type: int | None = None,
) -> VisorObject:
    return VisorObject(
        name=name,
        track_id=oid,
        relation=relation,
        mask_type=mask_type,
        segments=(((x0, y0), (x1, y0), (x1, y1), (x0, y1)),),
    )


def _frame(index: int, annotations: tuple[VisorObject, ...]) -> EpicFrame:
    return EpicFrame(
        video_id="P01_01",
        frame_name=f"P01_01_frame_{index:010d}.jpg",
        frame_index=index,
        image_path=f"P01_01/frame_{index:010d}.jpg",
        annotations=annotations,
    )


class EvaluationMetricTests(unittest.TestCase):
    def test_object_representation_accepts_filled_mask_overlap(self):
        obj = np.zeros((30, 30), dtype=np.uint8)
        obj[5:15, 5:15] = 255
        pred = np.zeros_like(obj)
        pred[7:12, 7:12] = 255

        score = object_representation_score(obj, pred, config=EvaluationConfig(min_overlap_fraction=0.05))

        self.assertGreater(score, 0)

    def test_object_representation_accepts_outline_overlap(self):
        obj = np.zeros((40, 40), dtype=np.uint8)
        obj[10:30, 10:30] = 255
        pred = np.zeros_like(obj)
        pred[10, 10:30] = 255

        score = object_representation_score(obj, pred, config=EvaluationConfig(min_overlap_fraction=0.01))

        self.assertGreater(score, 0)

    def test_evaluate_clip_tracks_foreground_background_and_load(self):
        hand = _box("left hand", "hand-left", 0, 0, 5, 5)
        cup_active = _box("cup", "cup-1", 20, 20, 25, 25, relation="in_hand")
        cup_visible = _box("cup", "cup-1", 20, 20, 25, 25)
        frames = [
            _frame(1, (hand, cup_active)),
            _frame(2, (hand, cup_visible)),
        ]
        pred1 = np.zeros((30, 30), dtype=np.uint8)
        pred1[0:5, 0:5] = 255
        pred1[20:25, 20:25] = 255
        pred2 = np.zeros((30, 30), dtype=np.uint8)
        pred2[20:25, 20:25] = 255

        result = evaluate_clip(frames, [pred1, pred2])

        self.assertEqual(result.frames[0].foreground_total, 2)
        self.assertEqual(result.frames[0].foreground_represented, 2)
        self.assertEqual(result.frames[1].background_total, 1)
        self.assertEqual(result.frames[1].background_represented, 1)
        self.assertEqual(result.foreground_represented, 2)
        self.assertEqual(result.foreground_total, 3)
        self.assertEqual(result.background_represented, 1)
        self.assertEqual(result.background_total, 1)
        self.assertGreater(result.output_load, 0)
        self.assertEqual(result.activity_load, result.output_active_area)

    def test_evaluate_clip_separates_gold_and_pseudo_visor_masks(self):
        gold_hand = _box("left hand", "hand-left", 0, 0, 5, 5, mask_type=1)
        pseudo_cup = _box("cup", "cup-1", 20, 20, 25, 25, relation="in_hand", mask_type=0)
        frames = [_frame(1, (gold_hand, pseudo_cup))]
        pred = np.zeros((30, 30), dtype=np.uint8)
        pred[0:5, 0:5] = 255

        gold = evaluate_clip(frames, [pred], config=EvaluationConfig(annotation_quality="gold"))
        pseudo = evaluate_clip(frames, [pred], config=EvaluationConfig(annotation_quality="pseudo"))

        self.assertEqual(gold.foreground_total, 1)
        self.assertEqual(gold.foreground_represented, 1)
        self.assertEqual(pseudo.foreground_total, 1)
        self.assertEqual(pseudo.foreground_represented, 0)

    def test_track_dropout_rate_counts_visible_target_misses(self):
        obj = _box("cup", "cup-1", 5, 5, 15, 15, relation="in_hand")
        frames = [_frame(1, (obj,)), _frame(2, (obj,)), _frame(3, (obj,))]
        pred1 = np.zeros((20, 20), dtype=np.uint8)
        pred1[5:15, 5:15] = 255
        pred2 = np.zeros((20, 20), dtype=np.uint8)
        pred3 = pred1.copy()

        result = evaluate_clip(frames, [pred1, pred2, pred3])

        self.assertEqual(track_dropout_rate(result.frames), 1 / 3)

    def test_track_fragmentation_rate_counts_internal_gaps(self):
        obj = _box("cup", "cup-1", 5, 5, 15, 15, relation="in_hand")
        frames = [_frame(1, (obj,)), _frame(2, (obj,)), _frame(3, (obj,))]
        pred1 = np.zeros((20, 20), dtype=np.uint8)
        pred1[5:15, 5:15] = 255
        pred2 = np.zeros((20, 20), dtype=np.uint8)
        pred3 = pred1.copy()

        result = evaluate_clip(frames, [pred1, pred2, pred3])

        self.assertEqual(track_fragmentation_rate(result.frames), 1.0)

    def test_evaluate_clip_variants_reports_activity_auc(self):
        obj = _box("cup", "cup-1", 5, 5, 15, 15, relation="in_hand")
        frames = [_frame(1, (obj,))]
        empty = np.zeros((20, 20), dtype=np.uint8)
        hit = empty.copy()
        hit[5:15, 5:15] = 255

        curve = evaluate_clip_variants(frames, [("empty", [empty]), ("hit", [hit])], load_cap=0.5)

        self.assertEqual(len(curve.variants), 2)
        self.assertIsNotNone(curve.foreground_auc)
        self.assertGreater(curve.foreground_auc, 0)

    def test_recall_auc_uses_monotonic_envelope(self):
        auc = recall_auc([(0.2, 0.5), (0.1, 1.0)], load_cap=0.2)

        self.assertAlmostEqual(auc, 0.75)

    def test_spv_load_uses_fixed_normalization(self):
        frames = [np.array([[0, 255]], dtype=np.uint8), np.array([[0, 0]], dtype=np.uint8)]

        self.assertEqual(spv_load(frames), 0.25)

    def test_flow_compensated_flicker_is_zero_for_static_frames(self):
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        simplified = np.zeros((20, 20), dtype=np.uint8)
        simplified[5:10, 5:10] = 255

        flicker = flow_compensated_flicker([rgb, rgb], [simplified, simplified])

        self.assertAlmostEqual(flicker, 0.0)


if __name__ == "__main__":
    unittest.main()
