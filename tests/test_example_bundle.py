"""The public example bundle contains measured inputs, never licensed CIFs."""

import importlib.util
import tarfile
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "package_example_data", Path(__file__).resolve().parents[1] / "scripts/package_example_data.py"
)
bundle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bundle)


class ExampleBundleTests(unittest.TestCase):
    def test_only_required_inputs_are_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            for dataset in bundle.DATASETS:
                target = data / dataset
                (target / "mixed_patterns").mkdir(parents=True)
                (target / "samples.csv").write_text(
                    "sample_id,pattern_path\nsample,mixed_patterns/sample.ASC\n"
                )
                for name in ("ground_truth.csv", "basis_order.csv", "experimental.instprm"):
                    (target / name).write_text("input\n")
                (target / "mixed_patterns/sample.ASC").write_text("15 1\n16 2\n")
                (target / "mixed_patterns/unused.cif").write_text("licensed structure")
                (target / "raw_cifs").mkdir()
                (target / "raw_cifs/secret.cif").write_text("licensed structure")
                (target / "old_results.csv").write_text("results")
            (data / "open_world_phase_discovery/candidate_library.instprm").write_text("input")
            output = root / "example-data.tar.gz"
            bundle.package(data, output)
            with tarfile.open(output) as archive:
                names = archive.getnames()
                self.assertEqual(len(names), 12)
                self.assertFalse(any("cif" in name or "results" in name for name in names))
                self.assertEqual(
                    archive.extractfile("data/.example-data-version").read().decode().strip(),
                    bundle.DATA_RELEASE,
                )
                self.assertEqual(bundle.DATA_RELEASE, "example-data-v5")
            pattern = data / bundle.DATASETS[0] / "mixed_patterns/sample.ASC"
            pattern.unlink()
            pattern.symlink_to(data / bundle.DATASETS[0] / "raw_cifs/secret.cif")
            with self.assertRaisesRegex(ValueError, "must be regular files"):
                bundle.input_files(data)
            pattern.unlink()
            pattern.parent.rename(pattern.parent.with_name("saved_patterns"))
            pattern.parent.symlink_to(data / bundle.DATASETS[0] / "raw_cifs")
            (pattern.parent / "sample.ASC").write_text("licensed structure")
            with self.assertRaisesRegex(ValueError, "must be regular files"):
                bundle.input_files(data)
            (data / bundle.DATASETS[0] / "samples.csv").write_text(
                "sample_id,pattern_path\nsample,raw_cifs/secret.cif\n"
            )
            with self.assertRaisesRegex(ValueError, "Unexpected measured-pattern"):
                bundle.input_files(data)

    def test_staging_script_uses_the_bundle_release_version(self):
        script = Path(__file__).resolve().parents[1] / "scripts/stage_data.sh"
        self.assertIn(f'data_release="{bundle.DATA_RELEASE}"', script.read_text())


if __name__ == "__main__":
    unittest.main()
