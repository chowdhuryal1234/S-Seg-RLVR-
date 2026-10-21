"""Databricks notebooks build the same runner arguments as the Harvard supervisor."""
import ast
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/dbx"))
import dbx_common as dbx  # noqa: E402
from nucleus_rl import grpo, sft  # noqa: E402
from scripts.run_sft_experiments import command  # noqa: E402

class FakeFiles:
    """Local-disk stand-in for WorkspaceClient().files, mapping /Volumes/... below `root`."""

    def __init__(self, root):
        self.root = Path(root)

    def _local(self, path):
        return self.root / path.lstrip("/")

    def get_metadata(self, path):
        if not self._local(path).is_file():
            raise FileNotFoundError(path)

    def get_directory_metadata(self, path):
        if not self._local(path).is_dir():
            raise FileNotFoundError(path)

    def create_directory(self, path):
        self._local(path).mkdir(parents=True, exist_ok=True)

    def list_directory_contents(self, path):
        for child in sorted(self._local(path).iterdir()):
            yield SimpleNamespace(path=f"{path.rstrip('/')}/{child.name}", is_directory=child.is_dir(),
                                  file_size=None if child.is_dir() else child.stat().st_size)

    def upload(self, path, stream, overwrite=False):
        target = self._local(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(stream.read())

    def download(self, path):
        return SimpleNamespace(contents=io.BytesIO(self._local(path).read_bytes()))


PATH_FLAGS = {"--manifest", "--eval-manifest", "--output", "--sam-checkpoint", "--cohort-protocol", "--init-adapter"}


def flag_values(argv):
    """Map --flag -> value after the module name, skipping path-valued flags."""
    items = argv[argv.index("-m") + 2:]
    values, index = {}, 0
    while index < len(items):
        key = items[index]
        if index + 1 < len(items) and not items[index + 1].startswith("--"):
            values[key], index = items[index + 1], index + 2
        else:
            values[key], index = True, index + 1
    return {k: v for k, v in values.items() if k not in PATH_FLAGS}


class DatabricksNotebookTests(unittest.TestCase):
    def test_notebooks_are_valid_python_sources(self):
        notebooks = sorted((ROOT / "scripts/dbx").glob("0*.py"))
        self.assertEqual(len(notebooks), 5)
        for path in notebooks:
            text = path.read_text()
            self.assertTrue(text.startswith("# Databricks notebook source"), path.name)
            ast.parse(text, filename=str(path))

    def test_sft_arguments_match_harvard_supervisor(self):
        config = dbx.sft_config()
        for stage, augmentation in (("canary", "d4"), ("main", "none"), ("main", "d4")):
            ours = dbx.sft_argv("7b", stage, augmentation, "/tmp/out", config)
            args = SimpleNamespace(model_key="7b", stage=stage, python=Path(sys.executable), code=ROOT,
                                   sam_checkpoint=Path("/x/sam.pt"))
            harvard = command(args, config, augmentation, Path("/tmp/out"), 3600)
            expected = flag_values(["python", *harvard[1:]])
            expected.pop("--wall-time-minutes")
            self.assertEqual(flag_values(ours), expected)
            parsed = sft.build_parser().parse_args(ours[2:])
            sft.validate_args(parsed)

    def test_grpo_arguments_parse_and_keep_sft_input_contract(self):
        config = dbx.sft_config()
        for canary in (False, True):
            argv = dbx.grpo_argv("7b", "grpo", "/tmp/out", "/tmp/adapter", canary=canary, config=config)
            parsed = grpo.build_parser().parse_args(argv[2:])
            grpo.validate_args(parsed)
            self.assertEqual((parsed.min_pixels, parsed.max_pixels),
                             (config["shared"]["min_pixels"], config["shared"]["max_pixels"]))
            self.assertEqual(parsed.coordinate_frame, config["models"]["7b"]["coordinate_frame"])
            self.assertEqual(parsed.skip_before_eval, canary)
            self.assertEqual(parsed.max_steps, 10 if canary else 600)

    def test_upload_skips_in_progress_writes_and_is_incremental(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = dbx.Volume(FakeFiles(Path(tmp, "volume")))
            source = Path(tmp, "run")
            (source / "checkpoints/.writing-checkpoint-4").mkdir(parents=True)
            (source / "checkpoints/.writing-checkpoint-4/a.bin").write_text("partial")
            (source / "summary.json").write_text("{}")
            state = {}
            self.assertEqual(store.upload_tree(source, "/Volumes/c/s/v/run", state), 1)
            self.assertFalse(store.is_dir("/Volumes/c/s/v/run/checkpoints/.writing-checkpoint-4"))
            self.assertEqual(store.upload_tree(source, "/Volumes/c/s/v/run", state), 0)
            self.assertEqual(store.read_text("/Volumes/c/s/v/run/summary.json"), "{}")

    def test_run_cli_uploads_outputs_and_log_even_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(dbx, "_VOLUME", dbx.Volume(FakeFiles(Path(tmp, "volume")))), \
                patch.object(dbx, "local_root", return_value=Path(tmp, "local")), \
                patch("builtins.print"):
            local, destination = Path(tmp, "local/runs/r"), "/Volumes/c/s/processed/runs/r"
            script = f"import pathlib,sys; p=pathlib.Path({str(local)!r}); p.mkdir(parents=True); " \
                     "(p/'summary.json').write_text('{}'); print('step 1'); [print('noisy') for _ in range(9)]; sys.exit(3)"
            with patch("builtins.print") as shown, self.assertRaises(RuntimeError):
                dbx.run_cli(["-c", script], local, destination)
            self.assertEqual(sum(call.args[0] == "noisy\n" for call in shown.call_args_list), 3)
            self.assertEqual(dbx.volume().read_text(destination + "/summary.json"), "{}")
            self.assertIn("step 1", dbx.volume().read_text(destination + ".log"))

    def test_download_tree_restores_nested_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = dbx.Volume(FakeFiles(Path(tmp, "volume")))
            store.write_text("/Volumes/c/s/raw/train/images/a.tif", "img")
            store.write_text("/Volumes/c/s/raw/train/annotations/a.xml", "xml")
            local = Path(tmp, "local")
            self.assertEqual(store.download_tree("/Volumes/c/s/raw/train", local), 2)
            self.assertEqual((local / "images/a.tif").read_text(), "img")
            self.assertFalse(store.is_file("/Volumes/c/s/raw/missing.txt"))

    def test_model_comes_from_volume_copy_without_network(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(dbx, "_VOLUME", dbx.Volume(FakeFiles(Path(tmp, "volume")))), \
                patch.dict(dbx.PATHS, artifacts="/Volumes/c/s/processed/segrlvr"), \
                patch.object(dbx, "local_root", return_value=Path(tmp, "local")), \
                patch.dict("os.environ", {}, clear=False):
            snapshot = dbx.remote("cache/huggingface/hub/models--org--m/snapshots/abc")
            dbx.volume().write_text(snapshot + "/config.json", "{}")
            result = dbx.ensure_model("org/m", "abc")
            self.assertEqual(result["source"], "volume_copy")
            local = Path(tmp, "local/huggingface/hub/models--org--m/snapshots/abc")
            self.assertEqual((local / "config.json").read_text(), "{}")
            self.assertEqual(dbx.ensure_model("org/m", "abc")["source"], "local_cache")

    def test_manifest_comparison_ignores_archive_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, archive in (("a", "/local/x.zip"), ("b", "/harvard/y.zip")):
                Path(tmp, name).mkdir()
                row = {"id": "c1", "image_pixels_sha256": "p", "source_archive": archive}
                Path(tmp, name, "train.jsonl").write_text(json.dumps(row) + "\n")
            report = dbx.compare_manifests(Path(tmp, "a"), Path(tmp, "b"))
            self.assertTrue(report["train"].startswith("identical"))
            self.assertEqual(report["validation"], "reference missing")


if __name__ == "__main__":
    unittest.main()
