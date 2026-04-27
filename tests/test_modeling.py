from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from clothing_mlops.modeling import _rebased_local_registry_model_uri
from clothing_mlops.vertex_pricing import HeuristicPricingBackend, _normalize_price_ladder


class ModelingTestCase(unittest.TestCase):
    def test_rebases_local_registry_source_to_tracking_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        source_tracking_root = repo_root / "mlruns"
        model_name = "clothing-value-model"
        required_run_dir = source_tracking_root / "404966465621661048"

        if not required_run_dir.exists():
            self.skipTest(
                f"Missing MLflow fixture directory: {required_run_dir}. "
                "This test requires local mlruns registry fixtures."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_root = Path(tmpdir) / "mlruns"
            shutil.copytree(source_tracking_root / "models", tracking_root / "models")
            shutil.copytree(
                source_tracking_root / "404966465621661048",
                tracking_root / "404966465621661048",
            )

            version_meta = tracking_root / "models" / model_name / "version-2" / "meta.yaml"
            metadata = yaml.safe_load(version_meta.read_text(encoding="utf-8"))
            metadata["source"] = (
                "file:///unmounted-host/mlruns/"
                "404966465621661048/8f8ce80bec5147a3912ad67a50df7407/artifacts/model"
            )
            version_meta.write_text(yaml.safe_dump(metadata), encoding="utf-8")

            rebased_uri = _rebased_local_registry_model_uri(
                f"models:/{model_name}/latest",
                tracking_root,
            )

            expected = (
                tracking_root
                / "404966465621661048"
                / "8f8ce80bec5147a3912ad67a50df7407"
                / "artifacts"
                / "model"
            ).as_uri()
            self.assertEqual(rebased_uri, expected)

    def test_normalize_price_ladder_monotonicity(self) -> None:
        like_new, good, used = _normalize_price_ladder(
            retail_price=200.0,
            like_new=90.0,
            good=130.0,
            used=150.0,
        )
        self.assertGreater(like_new, good)
        self.assertGreater(good, used)
        self.assertLessEqual(like_new, 200.0)
        self.assertLessEqual(good, 200.0)
        self.assertLessEqual(used, 200.0)

    def test_brand_feature_adjustment_effect(self) -> None:
        backend = HeuristicPricingBackend()
        retail_price = 120.0

        baseline = backend.estimate(
            description="Basic tee in cotton, size medium, lightly worn.",
            retail_price=retail_price,
        )
        premium = backend.estimate(
            description="Supreme archive tee in leather, size medium, lightly worn.",
            retail_price=retail_price,
        )

        self.assertGreater(premium.good, baseline.good)
        self.assertGreater(premium.like_new, baseline.like_new)
        self.assertGreater(premium.used, baseline.used)

    def test_heuristic_backend_price_bounds(self) -> None:
        backend = HeuristicPricingBackend()
        result = backend.estimate(
            description="Vintage jacket in wool with moderate wear.",
            retail_price=5.0,
        )

        self.assertGreaterEqual(result.retail_price, 15.0)
        self.assertGreaterEqual(result.like_new, 12.0)
        self.assertGreaterEqual(result.good, 10.0)
        self.assertGreaterEqual(result.used, 8.0)
        self.assertGreater(result.like_new, result.good)
        self.assertGreater(result.good, result.used)
        self.assertLessEqual(result.like_new, result.retail_price)
        self.assertLessEqual(result.good, result.retail_price)
        self.assertLessEqual(result.used, result.retail_price)


if __name__ == "__main__":
    unittest.main()
