"""Exercise checkpoint/data/budget safeguards on CPU, without importing Torch."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from nucleus_rl.train import build_parser, restore_resume_revision, validate_args
from nucleus_rl.training_budget import (
    CHECKPOINT_FILES, check_cohort, contract_digest, dataset_identity, pack_tree,
    preserve_evaluation_state, seal_checkpoint, unpack_tree, verify_checkpoint,
    wall_budget_reached,
)


class ExtendedTrainingTests(unittest.TestCase):
    def args(self, *extra):
        return build_parser().parse_args([
            "--mode", "train", "--manifest", "train.jsonl", "--eval-manifest", "validation.jsonl",
            "--output", "new-run", "--sam-checkpoint", "sam.pt", *extra,
        ])

    def checkpoint(self, root, step=10):
        path = root / f"checkpoint-{step}"
        path.mkdir()
        for name in CHECKPOINT_FILES:
            (path / name).write_bytes(b"synthetic payload, never loaded as a model")
        (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        contract = {"config": {"model": "Qwen/Qwen2.5-VL-3B-Instruct", "revision": "a" * 40,
                               "max_steps": 100}, "train_data": [{"image_sha256": "image-a"}],
                    "validation_data": [{"image_sha256": "image-b"}], "sam": {"checkpoint_sha256": "sam-a"}}
        seal_checkpoint(path, contract, {"finite_gradient_steps": step})
        return path, contract

    def test_bounded_extended_and_legacy_step_counts(self):
        for count in (10, 20, 100, 500, 1500, 5000):
            args = self.args("--max-steps", str(count), "--direct-gpu-load")
            validate_args(args)
            self.assertTrue(args.direct_gpu_load)
        for flag, value in (("--max-steps", "0"), ("--max-steps", "5001"),
                            ("--eval-every", "-1"), ("--wall-time-minutes", "0"),
                            ("--wall-time-minutes", "nan"), ("--wall-time-minutes", "inf")):
            with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                validate_args(self.args(flag, value))
        validate_args(self.args("--eval-every", "0", "--wall-time-minutes", ".5"))

    def test_resume_only_training_and_revision_pinned_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, contract = self.checkpoint(Path(directory))
            args = self.args("--max-steps", "100", "--resume-from-checkpoint", str(checkpoint))
            restore_resume_revision(args)
            self.assertEqual(args.revision, contract["config"]["revision"])
            for extra in (("--model", "another-model"), ("--revision", "b" * 40), ("--max-steps", "10")):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    restore_resume_revision(self.args("--max-steps", "100", "--resume-from-checkpoint", str(checkpoint), *extra))
            with self.assertRaises(ValueError):
                validate_args(self.args("--mode", "evaluate", "--resume-from-checkpoint", str(checkpoint)))

    def test_checkpoint_integrity_and_contract_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, contract = self.checkpoint(Path(directory))
            self.assertEqual(verify_checkpoint(checkpoint, contract)["global_step"], 10)
            for key in ("config", "train_data", "validation_data", "sam"):
                changed = deepcopy(contract)
                changed[key] = {"changed": True}
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contract mismatch"):
                    verify_checkpoint(checkpoint, changed)
            (checkpoint / "adapter_model.safetensors").write_bytes(b"changed model")
            with self.assertRaisesRegex(ValueError, "content changed"):
                verify_checkpoint(checkpoint)

    def test_partial_checkpoint_and_step_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, contract = self.checkpoint(Path(directory))
            marker = checkpoint / "resume_contract.json"
            marker.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_checkpoint(checkpoint)
            (checkpoint / "trainer_state.json").write_text('{"global_step":11}')
            with self.assertRaisesRegex(ValueError, "step disagree"):
                seal_checkpoint(checkpoint, contract, {})

    def test_data_identity_is_portable_but_detects_content_order_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i in range(2):
                image, mask = root / f"image{i}.png", root / f"mask{i}.png"
                image.write_bytes(f"image{i}".encode())
                mask.write_bytes(f"mask{i}".encode())
                rows.append({"id": f"crop{i}", "image_path": str(image), "mask_path": str(mask),
                             "prompt_width": 224, "prompt_height": 224})
            original = dataset_identity(rows)
            copies = []
            for i, row in enumerate(rows):
                image, mask = root / f"copyimage{i}.png", root / f"copymask{i}.png"
                image.write_bytes(Path(row["image_path"]).read_bytes())
                mask.write_bytes(Path(row["mask_path"]).read_bytes())
                copies.append({**row, "image_path": str(image), "mask_path": str(mask)})
            self.assertEqual(original, dataset_identity(copies))
            self.assertNotEqual(contract_digest(original), contract_digest(dataset_identity(rows[::-1])))
            copies[0]["prompt_width"] = 128
            self.assertNotEqual(original, dataset_identity(copies))
            Path(rows[0]["mask_path"]).write_bytes(b"different ground truth")
            self.assertNotEqual(original, dataset_identity(rows))

    def test_official_cohort_and_disjoint_patients_required(self):
        source1, source2 = "TCGA-AA-0001-01Z-00-DX1", "TCGA-AA-0002-01Z-00-DX1"
        cohort = {"image_ids": [source1, source2]}
        train = [{"id": "train1", "split": "train", "patient_id": "TCGA-AA-0001", "source_image_id": source1}]
        val = [{"id": "val1", "split": "validation", "patient_id": "TCGA-AA-0002", "source_image_id": source2}]
        check_cohort(train, val, cohort)
        for invalid in ([], [{**val[0], "source_image_id": "extra"}],
                        [{**val[0], "id": "train1"}], [{**val[0], "split": "train"}],
                        [{**val[0], "source_image_id": source1, "patient_id": "TCGA-AA-0001"}]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                check_cohort(train, invalid, cohort)

    def test_safe_state_tree_preserves_optimizer_keys_tuples_and_tensor_identity(self):
        class Tensor:
            pass
        tensor = Tensor()
        state = {"state": {0: {"step": tensor}}, "param_groups": [{"params": [0], "betas": (.9, .999)}],
                 "rng": (None, True, 3, .25)}
        tensors = {}
        encoded = pack_tree(state, tensors, lambda value: isinstance(value, Tensor))
        decoded = unpack_tree(json.loads(json.dumps(encoded)), tensors)
        self.assertEqual(decoded, state)
        self.assertIs(decoded["state"][0]["step"], tensor)
        self.assertIsInstance(decoded["param_groups"][0]["betas"], tuple)
        for invalid in (object(), float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                pack_tree(invalid, {}, lambda value: False)
        with self.assertRaises(ValueError):
            unpack_tree({"kind": "execute", "code": "not permitted"}, {})

    def test_validation_restores_rng_and_mixed_module_modes_even_on_failure(self):
        class Module:
            def __init__(self, mode):
                self.training = mode
        modules = [Module(True), Module(False), Module(True)]
        class Model:
            def modules(self):
                return iter(modules)
        rng = {"state": 42}
        def restore(state):
            rng["state"] = state
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            with preserve_evaluation_state(Model(), lambda: rng["state"], restore):
                rng["state"] = 999
                for module in modules:
                    module.training = False
                raise RuntimeError("evaluation failed")
        self.assertEqual([module.training for module in modules], [True, False, True])
        self.assertEqual(rng["state"], 42)

    def test_wall_budget_exact_boundary_and_disabled(self):
        self.assertFalse(wall_budget_reached(29.99, .5))
        self.assertTrue(wall_budget_reached(30, .5))
        self.assertTrue(wall_budget_reached(31, .5))
        self.assertFalse(wall_budget_reached(100000, None))


if __name__ == "__main__":
    unittest.main()
