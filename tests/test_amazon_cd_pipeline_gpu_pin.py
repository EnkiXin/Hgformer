"""Static safety contract for the legacy Amazon-CD shell launcher."""

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class AmazonCDPipelineGpuPinTest(unittest.TestCase):
    def test_parent_and_recbole_child_use_the_same_physical_gpu_token(self):
        script = (
            REPO_ROOT / "slrec_experiments" / "run_amazon_cd_pipeline.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('CUDA_VISIBLE_DEVICES="${GPU_ID}"', script)
        self.assertIn('--gpu_id="${GPU_ID}"', script)
        self.assertNotIn("--gpu_id=0", script)


if __name__ == "__main__":
    unittest.main()
