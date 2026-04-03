from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from clothing_mlops.modeling import _rebased_local_registry_model_uri


class ModelingTestCase(unittest.TestCase):
    def test_rebases_local_registry_source_to_tracking_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        source_tracking_root = repo_root / "mlruns"
        model_name = "clothing-value-model"

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


if __name__ == "__main__":
    unittest.main()
