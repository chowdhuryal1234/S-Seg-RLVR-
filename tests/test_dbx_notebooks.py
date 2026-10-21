"""Databricks notebooks build the same runner arguments as the Harvard supervisor."""
import ast
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

    def test_mirror_skips_in_progress_writes_and_is_incremental(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp, "run"), Path(tmp, "volume")
            (source / "checkpoints/.writing-checkpoint-4").mkdir(parents=True)
            (source / "checkpoints/.writing-checkpoint-4/a.bin").write_text("partial")
            (source / "summary.json").write_text("{}")
            self.assertEqual(dbx.mirror_tree(source, target), 1)
            self.assertFalse((target / "checkpoints/.writing-checkpoint-4").exists())
            self.assertEqual(dbx.mirror_tree(source, target), 0)

    def test_model_comes_from_volume_mirror_without_network(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(dbx.PATHS, artifacts=Path(tmp, "volume")), \
                patch.object(dbx, "local_root", return_value=Path(tmp, "local")), \
                patch.dict("os.environ", {}, clear=False):
            snapshot = Path(tmp, "volume/cache/huggingface/hub/models--org--m/snapshots/abc")
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "s1.safetensors"}}))
            self.assertFalse(dbx._snapshot_complete(snapshot))  # Shard still missing: upload incomplete.
            (snapshot / "s1.safetensors").write_text("weights")
            result = dbx.ensure_model("org/m", "abc", mirror=False)
            self.assertEqual(result["source"], "volume_mirror")
            local = Path(tmp, "local/huggingface/hub/models--org--m/snapshots/abc")
            self.assertEqual((local / "s1.safetensors").read_text(), "weights")
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
