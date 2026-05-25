from __future__ import annotations

import os
from pathlib import Path
import unittest

from tests.named_model_proofs import assert_expected_outputs, run_named_model_proofs


class NamedModelProofTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("RUN_NAMED_MODEL_PROOFS") == "1",
        "set RUN_NAMED_MODEL_PROOFS=1 to run the CUDA model proof test",
    )
    def test_named_models_produce_visual_outputs(self):
        root = Path(__file__).resolve().parents[1]
        max_frames = int(os.environ.get("MODEL_PROOF_MAX_FRAMES", "6"))
        output_root = root / "outputs" / "named_model_proofs"

        result = run_named_model_proofs(
            clip_dir=root / "data" / "epic_kitchens" / "clips_10s",
            output_root=output_root,
            max_frames=max_frames,
            device=os.environ.get("MODEL_PROOF_DEVICE", "cuda"),
            clean=True,
        )

        self.assertEqual(result["frames"], max_frames)
        counts = assert_expected_outputs(output_root, max_frames)
        for model_name, count in counts.items():
            with self.subTest(model=model_name):
                self.assertGreaterEqual(count, max_frames)


if __name__ == "__main__":
    unittest.main()
