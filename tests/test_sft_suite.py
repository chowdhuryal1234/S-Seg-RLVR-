"""CPU checks for paired SFT schedules, canary gates, and bounded orchestration."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import run_sft_experiments as suite


CONFIG = Path(__file__).resolve().parents[1] / "configs/sft_experiments.json"


def canary_result(config, key="7b"):
    model = config["models"][key]
    return {
        "status": "verified_sft_run", "scope": "canary", "model": model["model"],
        "resolved_revision": model["revision"], "optimizer_steps": 20, "completed_requested_steps": True,
        "train_examples": 4, "validation_examples": 8, "adapter_changed": True,
        "frozen_base_versions_unchanged": True, "vision_sha256_before": "a" * 64,
        "vision_sha256_after": "a" * 64, "sam_freeze_verification": {"unchanged": True},
        "d4_collator_audit": {"passed": True, "transforms": list(range(8))},
        "before": {"completions": 8}, "after": {"completions": 8},
        "fixed_training_loss_before": {"mean_loss": 2.0},
        "fixed_training_loss_after": {"mean_loss": 1.0},
        "mean_optimizer_step_seconds": 5.0, "training_elapsed_seconds": 100.0,
        "before_evaluation_seconds": 20.0, "after_evaluation_seconds": 30.0,
        "peak_gpu_reserved_bytes": 20_000_000_000,
    }


class SuiteContracts(unittest.TestCase):
    def setUp(self):
        self.config = suite.load_config(CONFIG)

    def test_eight_arms_use_immutable_pins_equal_exposures_and_fresh_bases(self):
        self.assertEqual(len(self.config["models"]) * len(self.config["main"]["augmentations"]), 8)
        self.assertEqual(self.config["main"]["max_steps"] * self.config["shared"]["gradient_accumulation"], 2048)
        self.assertEqual(self.config["shared"]["lora_rank"], 16)
        for model in self.config["models"].values():
            self.assertRegex(model["revision"], r"^[a-f0-9]{40}$")
        args = suite.parser().parse_args(["--stage", "main", "--model-key", "7b", "--root", "/tmp/test", "--sam-checkpoint", "/tmp/sam.pt"])
        commands = [suite.command(args, self.config, arm, Path("/tmp/test") / arm, 3600) for arm in ("none", "d4")]
        for command in commands:
            self.assertNotIn("--resume-from-checkpoint", command)
            self.assertEqual(command[command.index("--max-steps") + 1], "512")
            self.assertEqual(command[command.index("--gradient-accumulation") + 1], "4")
            self.assertEqual(command[command.index("--seed") + 1], "42")
        # The only scientific difference is the transform, alongside its output.
        stripped = []
        for command in commands:
            command = command.copy()
            for option in ("--augmentation", "--output"):
                position = command.index(option)
                command[position + 1] = "PAIRED"
            stripped.append(command)
        self.assertEqual(stripped[0], stripped[1])

    def test_changed_schedule_or_mutable_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for mutation in (lambda c: c["models"]["7b"].update(revision="main"),
                             lambda c: c["main"].update(max_steps=1024),
                             lambda c: c["canary"].update(augmentation="none")):
                config = copy.deepcopy(self.config)
                mutation(config)
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    suite.load_config(path)

    def test_canary_checks_training_freeze_geometry_audit_and_loss(self):
        result = canary_result(self.config)
        model = self.config["models"]["7b"]
        self.assertEqual(suite.canary_errors(result, {"gpu": "NVIDIA A100 80GB"}, model["model"], model["revision"]), [])
        for key, value in (("optimizer_steps", 19), ("adapter_changed", False),
                           ("vision_sha256_after", "b" * 64), ("resolved_revision", "main"),
                           ("d4_collator_audit", {"passed": True, "transforms": [0]}),
                           ("fixed_training_loss_after", {"mean_loss": 3.0}),
                           ("mean_optimizer_step_seconds", float("nan"))):
            bad = {**result, key: value}
            with self.subTest(key=key):
                self.assertTrue(suite.canary_errors(bad, {"gpu": "A100"}, model["model"], model["revision"]))

    def test_canary_file_integrity_and_same_input_identity_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "7b"
            (directory / "d4").mkdir(parents=True)
            result = canary_result(self.config)
            for name, value in (("summary.json", result), ("metadata.json", {"gpu": "A100"})):
                (directory / "d4" / name).write_text(json.dumps(value))
            record = {"status": "passed", "input_identity_sha256": "same",
                      "files": {name: suite.file_digest(directory / "d4" / name) for name in ("summary.json", "metadata.json")}}
            (directory / "canary_result.json").write_text(json.dumps(record))
            suite.canary_record(root, "7b", self.config, {"input_identity_sha256": "same"})
            with self.assertRaisesRegex(ValueError, "exact pinned inputs"):
                suite.canary_record(root, "7b", self.config, {"input_identity_sha256": "different"})
            (directory / "d4/summary.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed after"):
                suite.canary_record(root, "7b", self.config, {"input_identity_sha256": "same"})

    def test_timing_accounts_for_two_arms_and_all_five_evaluations_per_arm(self):
        timing = suite.timing_from_canary({"elapsed_seconds": 300}, canary_result(self.config), self.config)
        self.assertEqual(timing["training_seconds"], 512 * 5 * 2)
        self.assertEqual(timing["periodic_evaluations"], 3)
        self.assertEqual(timing["seconds_per_evaluation64"],
                         max(30 * 8 * 2, self.config["budget"]["minimum_evaluation64_seconds"]))
        self.assertEqual(timing["before_after_evaluation_seconds"], 2 * timing["seconds_per_evaluation64"])
        self.assertEqual(timing["periodic_evaluation_seconds"], 3 * timing["seconds_per_evaluation64"])
        self.assertEqual(timing["paired_hard_seconds"], 2 * timing["hard_seconds_per_arm"])
        self.assertGreater(timing["cold_load_and_audit_seconds"], 0)

    def test_deadline_is_required_and_can_be_read_from_existing_allocation(self):
        expected = time.time() + 3600
        self.assertEqual(suite.resolve_deadline(environment={"SLURM_JOB_END_TIME": str(expected)}), expected)
        self.assertEqual(suite.resolve_deadline(expected, environment={}), expected)
        for value in (None, "not-a-time", float("nan"), time.time() - 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                suite.resolve_deadline(value, environment={})


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = suite.load_config(CONFIG)
        self.pins = {"files": {}, "input_identity_sha256": "same"}
        suite.STOP_REQUESTED = False

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, stage="canary", extra=()):
        args = suite.parser().parse_args(["--stage", stage, "--model-key", "7b", "--root", str(self.root / "outputs"),
                                         "--sam-checkpoint", str(self.root / "sam.pt"), *extra])
        args.config = CONFIG
        return args

    def test_cpu_dry_run_has_no_gpu_import_subprocess_or_allocation(self):
        args = self.args(extra=("--dry-run",))
        with (patch.object(suite, "pin_inputs", return_value=self.pins), patch.object(suite, "gpu_identity") as gpu,
              patch.object(suite, "execute") as execute):
            state = suite.run(args)
        self.assertEqual(state["status"], "dry_run_verified_without_gpu_loading_or_training")
        gpu.assert_not_called()
        execute.assert_not_called()
        self.assertEqual(len(state["commands"]), 1)
        self.assertIn("d4", state["commands"][0])
        status = json.loads((args.root / "7b/suite_status.json").read_text())
        self.assertEqual(status["status"], state["status"])

    def test_old_run_directory_is_not_overwritten(self):
        args = self.args(extra=("--dry-run",))
        directory = args.root / "7b"
        directory.mkdir(parents=True)
        marker = directory / "suite_status.json"
        marker.write_text("ORIGINAL")
        with self.assertRaises(FileExistsError):
            suite.run(args)
        self.assertEqual(marker.read_text(), "ORIGINAL")

    def test_input_failure_is_preserved_atomically_in_status(self):
        args = self.args(extra=("--dry-run",))
        with patch.object(suite, "pin_inputs", side_effect=ValueError("changed crop")):
            with self.assertRaisesRegex(ValueError, "changed crop"):
                suite.run(args)
        state = json.loads((args.root / "7b/suite_status.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error_type"], "ValueError")
        self.assertEqual(state["error"], "changed crop")

    def test_main_requires_all_four_canaries_before_loading_gpu(self):
        args = self.args("main", extra=("--canary-root", str(self.root / "canaries")))
        with patch.object(suite, "pin_inputs", return_value=self.pins), patch.object(suite, "gpu_identity") as gpu:
            with self.assertRaises(FileNotFoundError):
                suite.run(args)
        gpu.assert_not_called()

    def test_insufficient_pair_budget_starts_no_arm_and_keeps_fixed_steps(self):
        args = self.args("main", extra=("--canary-root", str(self.root / "canaries"), "--deadline-epoch", str(time.time() + 1000)))
        record = {"elapsed_seconds": 300}
        result = canary_result(self.config)
        with (patch.object(suite, "pin_inputs", return_value=self.pins),
              patch.object(suite, "canary_record", return_value=(record, result)) as canary,
              patch.object(suite, "gpu_identity", return_value={"gpu": "H200"}),
              patch.object(suite, "execute") as execute):
            with self.assertRaisesRegex(RuntimeError, "cannot fit"):
                suite.run(args)
        self.assertEqual(canary.call_count, 4)
        execute.assert_not_called()
        self.assertEqual(self.config["main"]["max_steps"], 512)


if __name__ == "__main__":
    unittest.main()
