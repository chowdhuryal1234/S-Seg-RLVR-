"""CPU-only tests of deadline decisions, immutable inputs and child failures."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("h200_supervisor", Path(__file__).parents[1] / "scripts/run_h200_experiments.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def benchmark(**updates):
    result = {"status": "verified_training_pilot", "adapter_changed": True, "nonzero_gradient_steps": 20,
              "sam": {"unchanged": True}, "completed_requested_steps": True,
              "optimizer_steps_this_run": 20, "training_elapsed_seconds": 100,
              "elapsed_seconds": 180, "before_evaluation_seconds": 15, "after_evaluation_seconds": 20,
              "before": {"completions": 8}, "after": {"completions": 8}}
    result.update(updates)
    return {"exit_code": 0, "result": result}


class SupervisorTests(unittest.TestCase):
    def args(self, root):
        return argparse.Namespace(root=str(root), python=sys.executable, code=str(Path(__file__).parents[1]),
                                  sam_checkpoint="/tmp/sam.pt", deadline_epoch=runner.time.time() + 28800,
                                  dry_run=True)

    def test_benchmark_requires_actual_complete_update_and_frozen_sam(self):
        for change in ({"status": "no_useful_update"}, {"completed_requested_steps": False},
                       {"sam": {"unchanged": False}}, {"nonzero_gradient_steps": 0},
                       {"after": {"completions": 7}}):
            with self.subTest(change=change):
                self.assertIsNone(runner.timing_from_benchmark(benchmark(**change)))
        self.assertEqual(runner.timing_from_benchmark(benchmark())["seconds_per_step"], 5)

    def test_forecast_includes_full_validation_periodic_load_and_save(self):
        timing = runner.timing_from_benchmark(benchmark())
        cost = runner.forecast(timing, 1000)
        self.assertEqual(cost["periodic_evaluations"], 9)
        self.assertEqual(cost["before_after_evaluation_seconds"], 480)
        self.assertEqual(cost["training_seconds"], 6250)
        self.assertEqual(cost["checkpoint_seconds"], 300)
        self.assertGreater(cost["hard_seconds"], cost["soft_training_seconds"])

    def test_equal_budget_pair_and_32b_are_fitted_together(self):
        timing = runner.timing_from_benchmark(benchmark())
        plan = runner.plan_main({"7b": timing, "32b": timing}, 18000)
        n, large = plan["7b_equal_steps"], plan["32b_steps"]
        self.assertGreater(n, 0)
        self.assertGreaterEqual(large, 500)
        total = 2 * runner.forecast(timing, n)["hard_seconds"] + runner.forecast(timing, large)["hard_seconds"]
        self.assertLessEqual(total, 18000)
        self.assertLessEqual(runner.forecast(timing, large)["hard_seconds"], 7200)
        self.assertLessEqual(n, 1000)

    def test_failed_32b_does_not_prevent_7b_pair_or_retry_32b(self):
        timing = runner.timing_from_benchmark(benchmark())
        plan = runner.plan_main({"7b": timing, "32b": None}, 18000)
        self.assertEqual(plan["32b_steps"], 0)
        self.assertGreater(plan["7b_equal_steps"], 0)

    def test_insufficient_time_never_invents_short_32b_or_negative_budget(self):
        timing = runner.timing_from_benchmark(benchmark())
        self.assertEqual(runner.plan_main({"7b": timing, "32b": timing}, -1)["7b_equal_steps"], 0)
        self.assertEqual(runner.affordable_steps(timing, 500, minimum=500), 0)

    def test_lr_pair_changes_only_lr_and_output(self):
        args = self.args(Path("/tmp/suite"))
        one = runner.command(args, "7b", "baseline", 1000, 1e-5, 42, 5000)
        two = runner.command(args, "7b", "lr_arm", 1000, 5e-5, 42, 5000)
        diff = [(one[i-1], one[i], two[i]) for i in range(1, len(one)) if one[i] != two[i]]
        self.assertEqual({item[0] for item in diff}, {"--output", "--learning-rate"})
        self.assertNotIn("--adapter", one)
        self.assertNotIn("--resume-from-checkpoint", one)
        from nucleus_rl.train import build_parser, validate_args
        parsed = build_parser().parse_args(one[3:])
        validate_args(parsed)
        self.assertEqual(parsed.revision, runner.MODELS["7b"][1])
        self.assertEqual(parsed.eval_every, 100)

    def test_selection_is_full_validation_pq_then_recall_not_reward(self):
        one, two = benchmark(), benchmark()
        one["result"]["after"] = {"completions": 64, "mean_pq": .2, "mean_recall": .9, "mean_total_reward": 2}
        two["result"]["after"] = {"completions": 64, "mean_pq": .3, "mean_recall": .1, "mean_total_reward": 0}
        self.assertGreater(runner.validation_key(two), runner.validation_key(one))
        two["result"]["after"]["completions"] = 8
        self.assertIsNone(runner.validation_key(two))

    def test_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_text("original")
            pins = {"files": {str(path): runner.digest(path)}}
            runner.verify_pins(pins)
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Pinned"):
                runner.verify_pins(pins)

    def test_child_failure_preserves_log_and_actual_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "failure.log"
            result = runner.run_child([sys.executable, "-c", "print('simulated OOM', flush=True); raise SystemExit(7)"], directory, log, 5)
            self.assertEqual(result["exit_code"], 7)
            self.assertFalse(result["timed_out"])
            self.assertIn("simulated OOM", log.read_text())

    def test_hard_timeout_retains_checkpoint_written_by_child(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "timeout.log"
            code = "from pathlib import Path; import time; Path('checkpoint.json').write_text('saved'); time.sleep(5)"
            result = runner.run_child([sys.executable, "-c", code], directory, log, .2)
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["exit_code"], 0)
            self.assertEqual((Path(directory) / "checkpoint.json").read_text(), "saved")

    def test_dry_run_records_commands_without_launch_and_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = runner.Suite(self.args(directory), {"files": {}})
            with patch.object(runner, "run_child", side_effect=AssertionError("Must not launch")):
                suite.execute()
            self.assertEqual(len(suite.summary["runs"]), 5)
            self.assertTrue(all(r["status"] == "dry_run_not_launched" for r in suite.summary["runs"]))
            self.assertTrue((Path(directory) / "plans/benchmark_7b_20.json").is_file())
            (Path(directory) / "already_used").mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                suite.launch("7b", "already_used", 20, 1e-5, 42, {"soft_training_seconds": 100, "hard_seconds": 200})

    def test_launch_failure_is_recorded_and_next_run_can_proceed(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            args.dry_run = False
            suite = runner.Suite(args, {"files": {}})
            def fail(cmd, cwd, log, hard):
                self.assertTrue((Path(directory) / "plans/failed32.json").is_file())
                raise OSError("simulated executable failure")
            with patch.object(runner, "run_child", side_effect=fail):
                failed = suite.launch("32b", "failed32", 20, 1e-5, 42,
                                      {"soft_training_seconds": 100, "hard_seconds": 200}, benchmark=True)
            self.assertEqual(failed["status"], "launch_or_artifact_failure")
            self.assertIsNone(failed["exit_code"])
            args.dry_run = True
            following = suite.launch("7b", "following7", 20, 1e-5, 42,
                                     {"soft_training_seconds": 100, "hard_seconds": 200}, benchmark=True)
            self.assertEqual(following["status"], "dry_run_not_launched")

    def test_successful_wall_stop_is_labeled_partial_not_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            args.dry_run = False
            suite = runner.Suite(args, {"files": {}})
            partial = benchmark(completed_requested_steps=False, stop_reason="wall_time_budget")
            with patch.object(runner, "run_child", return_value={"exit_code": 0, "timed_out": False}), \
                    patch.object(runner, "collect_outputs", return_value={"result": partial["result"]}):
                record = suite.launch("7b", "partial7", 1000, 1e-5, 42,
                                      {"soft_training_seconds": 100, "hard_seconds": 200})
            self.assertEqual(record["status"], "partial_budget_stop")
            self.assertTrue(runner.useful(record))
            self.assertFalse(runner.useful(record, complete=True))
            self.assertIsNone(runner.validation_key(record))

    def test_main_preserves_virtualenv_python_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            interpreter = base / "venv_python"
            interpreter.symlink_to(sys.executable)
            captured = []
            def execute(suite):
                captured.append(suite.args.python)
            with patch.object(runner, "pin_inputs", return_value={"files": {}}), patch.object(runner.Suite, "execute", execute):
                code = runner.main(["--root", str(base / "suite"), "--python", str(interpreter),
                                    "--code", str(base), "--sam-checkpoint", str(base / "sam"),
                                    "--deadline-epoch", str(runner.time.time() + 3600), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertEqual(captured, [str(interpreter)])


if __name__ == "__main__":
    unittest.main()
