"""No-allocation tests for the durable, gated H200 submission receipt."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import submit_sft_jobs as submission


TRACKED_BATCH = Path(__file__).resolve().parents[1] / "scripts/run_sft_h200.sh"


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve() / "sseg-sft-20261017"
        self.script = self.root / "code/scripts/run_sft_h200.sh"
        self.script.parent.mkdir(parents=True)
        self.script.write_text(TRACKED_BATCH.read_text())
        config = self.root / "code/configs/sft_experiments.json"
        config.parent.mkdir()
        config.write_text("{}")
        self.pins = {"input_identity_sha256": "frozen"}
        record = {"status": "passed", "files": {"summary.json": "abc", "metadata.json": "def"},
                  "input_identity_sha256": "frozen", "elapsed_seconds": 120.0}
        self.api = SimpleNamespace(
            MODEL_KEYS=("7b", "32b", "9b", "27b"),
            load_config=Mock(return_value={"budget": {"allocation_reserve_seconds": 600}}),
            pin_inputs=Mock(return_value=self.pins), verify_pins=Mock(),
            canary_record=Mock(return_value=(record, {"summary": "verified"})),
            timing_from_canary=Mock(return_value={"paired_hard_seconds": 36000}),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_preflight_requires_four_canaries_and_records_script_helper_hashes(self):
        with patch.object(submission, "_suite_api", return_value=self.api):
            plan = submission.build_plan(self.root)
        self.assertEqual(self.api.canary_record.call_count, 4)
        self.assertEqual(self.api.verify_pins.call_count, 2)
        self.assertEqual(set(plan["canaries"]), {"7b", "32b", "9b", "27b"})
        self.assertEqual(len(plan["files"]["batch_script"]["sha256"]), 64)
        self.assertEqual(len(plan["files"]["submission_helper"]["sha256"]), 64)
        self.assertEqual(plan["task_forecast_window_seconds"], 23 * 3600)

    def test_missing_canary_changed_pins_and_unaffordable_pair_each_block(self):
        for function, error in ((self.api.canary_record, ValueError("missing canary")),
                                (self.api.verify_pins, ValueError("changed input"))):
            with self.subTest(error=error), patch.object(submission, "_suite_api", return_value=self.api):
                function.side_effect = error
                with self.assertRaises(ValueError):
                    submission.build_plan(self.root)
                function.side_effect = None
        self.api.timing_from_canary.return_value = {"paired_hard_seconds": 23 * 3600 - 599}
        with patch.object(submission, "_suite_api", return_value=self.api):
            with self.assertRaisesRegex(ValueError, "cannot fit"):
                submission.build_plan(self.root)

    def test_batch_limits_and_shell_syntax_are_checked_without_sbatch(self):
        directives = submission.validate_batch_script(self.script)
        self.assertEqual(directives["--array"], "0-3%4")
        self.assertEqual(directives["--gres"], "gpu:nvidia_h200:1")
        self.script.write_text(TRACKED_BATCH.read_text().replace("#SBATCH --mem=128G", "#SBATCH --mem=256G"))
        with self.assertRaisesRegex(ValueError, "allocation differs"):
            submission.validate_batch_script(self.script)
        self.script.write_text(TRACKED_BATCH.read_text() + "\nif then\n")
        with self.assertRaisesRegex(ValueError, "shell syntax"):
            submission.validate_batch_script(self.script)

    def test_command_does_not_override_tracked_resources(self):
        command = submission.submission_command(self.root)
        self.assertEqual(command[0:2], ["sbatch", "--parsable"])
        self.assertIn(f"--export=ALL,SFT_ROOT={self.root}", command)
        self.assertIn(f"--output={self.root}/logs/h200-%A_%a.out", command)
        self.assertIn("--job-name=sseg-sft-20261017", command)
        self.assertEqual(command[-1], str(self.script))
        self.assertFalse(any(argument.startswith(("--gres=", "--array=", "--time=", "--mem=")) for argument in command))
        with self.assertRaisesRegex(ValueError, "export separators"):
            submission.submission_command(self.root.parent / "bad,path")

    def test_squeue_checks_user_and_exact_job_name(self):
        response = subprocess.CompletedProcess([], 0,
            stdout="123|sseg-sft-20261017-old|RUNNING\n124|sseg-sft-20261017|PENDING\n", stderr="")
        with patch.object(submission.subprocess, "run", return_value=response) as runner:
            matches = submission.active_matching_jobs("sseg-sft-20261017")
        self.assertEqual([job["job_id"] for job in matches], ["124"])
        command = runner.call_args.args[0]
        self.assertEqual(command[0], "squeue")
        self.assertIn("--user", command)
        with patch.object(submission.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "unavailable")):
            with self.assertRaises(RuntimeError):
                submission.active_matching_jobs("sseg-sft-20261017")

    def test_only_actual_parsable_job_ids_are_accepted(self):
        self.assertEqual(submission.parse_submission_id("53589999\n"), ("53589999", None))
        self.assertEqual(submission.parse_submission_id("53589999;cannon\n"), ("53589999", "cannon"))
        for value in ("Submitted batch job 53589999", "0", "53589999\n53590000\n", "", "would start 53589999"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                submission.parse_submission_id(value)


class IdempotenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve() / "sseg-sft-20261017"
        self.root.mkdir()
        self.plan = {"job_name": self.root.name, "command": ["sbatch", "--parsable", "script.sh"], "status": "ready"}

    def tearDown(self):
        self.temporary.cleanup()

    @property
    def receipt(self):
        return self.root / submission.RECEIPT_NAME

    def test_default_dry_run_creates_no_receipt_and_never_calls_sbatch(self):
        with (patch.object(submission, "build_plan", return_value=self.plan),
              patch.object(submission, "active_matching_jobs", return_value=[]),
              patch.object(submission.subprocess, "run") as runner):
            result = submission.run(self.root)
        self.assertEqual(result["status"], "dry_run_ready")
        self.assertFalse(self.receipt.exists())
        runner.assert_not_called()
        self.assertTrue(Path(result["audit_path"]).is_file())

    def test_submitting_receipt_is_durable_before_one_actual_sbatch_call(self):
        def submit(command, **kwargs):
            receipt = json.loads(self.receipt.read_text())
            self.assertEqual(receipt["status"], "SUBMITTING")
            self.assertEqual(receipt["plan"]["command"], command)
            self.assertFalse(any(key.startswith("SBATCH_") for key in kwargs["env"]))
            return subprocess.CompletedProcess(command, 0, "53589999\n", "")
        with (patch.object(submission, "build_plan", return_value=self.plan) as preflight,
              patch.object(submission, "active_matching_jobs", return_value=[]),
              patch.object(submission.subprocess, "run", side_effect=submit) as runner):
            first = submission.run(self.root, submit=True)
            second = submission.run(self.root, submit=True)
        self.assertEqual(first["status"], "submitted")
        self.assertEqual(first["receipt"]["job_id"], "53589999")
        self.assertEqual(second["status"], "already_submitted")
        runner.assert_called_once()
        preflight.assert_called_once()

    def test_nonzero_unparsable_and_timeout_results_each_block_every_retry(self):
        responses = (subprocess.CompletedProcess([], 1, "53589999\n", "error"),
                     subprocess.CompletedProcess([], 0, "unexpected output", ""),
                     subprocess.TimeoutExpired("sbatch", 120, output=b"uncertain", stderr=b""))
        for index, response in enumerate(responses):
            with self.subTest(response=response):
                root = self.root / f"attempt-{index}"
                root.mkdir()
                with (patch.object(submission, "build_plan", return_value=self.plan),
                      patch.object(submission, "active_matching_jobs", return_value=[]),
                      patch.object(submission.subprocess, "run") as runner):
                    if isinstance(response, BaseException):
                        runner.side_effect = response
                    else:
                        runner.return_value = response
                    with self.assertRaises(submission.SubmissionBlocked):
                        submission.run(root, submit=True)
                    with self.assertRaises(submission.SubmissionBlocked):
                        submission.run(root, submit=True)
                self.assertEqual(runner.call_count, 1)
                receipt = json.loads((root / submission.RECEIPT_NAME).read_text())
                self.assertEqual(receipt["status"], "AMBIGUOUS")
                self.assertNotIn("job_id", receipt)

    def test_validated_workspace_overrides_wrong_inherited_export_and_sbatch_options(self):
        def submit(command, **kwargs):
            environment = kwargs["env"]
            self.assertEqual(environment["SFT_ROOT"], str(self.root))
            self.assertNotIn("SBATCH_GRES", environment)
            self.assertNotIn("SBATCH_TIME", environment)
            return subprocess.CompletedProcess(command, 0, "53589999\n", "")
        with (patch.dict(submission.os.environ, {"SFT_ROOT": "/wrong/workspace", "SBATCH_GRES": "gpu:8", "SBATCH_TIME": "7-00:00:00"}),
              patch.object(submission, "build_plan", return_value=self.plan),
              patch.object(submission, "active_matching_jobs", return_value=[]),
              patch.object(submission.subprocess, "run", side_effect=submit)):
            result = submission.run(self.root, submit=True)
        self.assertEqual(result["status"], "submitted")

    def test_previous_crash_receipt_and_unreadable_receipt_block_without_preflight(self):
        for content in ('{"status":"SUBMITTING"}', 'not-json'):
            with self.subTest(content=content):
                self.receipt.write_text(content)
                with patch.object(submission, "build_plan") as preflight:
                    with self.assertRaises(submission.SubmissionBlocked):
                        submission.run(self.root, submit=True)
                preflight.assert_not_called()
                self.assertEqual(self.receipt.read_text(), content)

    def test_scheduler_name_guard_blocks_before_any_submission_receipt(self):
        with (patch.object(submission, "build_plan", return_value=self.plan),
              patch.object(submission, "active_matching_jobs", return_value=[{"job_id": "old"}]),
              patch.object(submission.subprocess, "run") as runner):
            with self.assertRaises(submission.SubmissionBlocked):
                submission.run(self.root, submit=True)
        self.assertFalse(self.receipt.exists())
        runner.assert_not_called()
        audit = list((self.root / "logs/submission-checks").glob("*.json"))
        self.assertEqual(len(audit), 1)
        self.assertEqual(json.loads(audit[0].read_text())["status"], "failed")

    def test_batch_change_during_scheduler_check_blocks_submission(self):
        source = self.root / "batch.sh"
        source.write_text("reviewed")
        plan = {**self.plan, "files": {"batch_script": {"path": str(source), "sha256": submission.file_sha256(source)}}}
        def check_jobs(name):
            source.write_text("changed")
            return []
        with (patch.object(submission, "build_plan", return_value=plan),
              patch.object(submission, "active_matching_jobs", side_effect=check_jobs),
              patch.object(submission.subprocess, "run") as runner):
            with self.assertRaisesRegex(ValueError, "source changed"):
                submission.run(self.root, submit=True)
        self.assertFalse(self.receipt.exists())
        runner.assert_not_called()

    def test_concurrent_helper_is_blocked_by_flock(self):
        with submission.submission_lock(self.root):
            with self.assertRaisesRegex(submission.SubmissionBlocked, "holds the workspace lock"):
                submission.run(self.root, submit=True)
        self.assertFalse(self.receipt.exists())

    def test_durable_json_rejects_nonfinite_payloads(self):
        with self.assertRaises(ValueError):
            submission.durable_json(self.receipt, {"bad": float("nan")})
        self.assertFalse(self.receipt.exists())


if __name__ == "__main__":
    unittest.main()
