"""CPU checks of assistant masking, paired sampling, freeze and resume contracts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nucleus_rl.sft import (assistant_label_mask, build_parser, checkpoint_manifest, encode_example,
                            exposure_for_index, frozen_versions, assert_frozen, language_lora_targets,
                            main, save_checkpoint, trainable_language_adapters, validate_args,
                            verify_sft_checkpoint)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class SFTContractTests(unittest.TestCase):
    def args(self, *flags):
        return build_parser().parse_args(["--manifest", "train.jsonl", "--eval-manifest", "val.jsonl",
                                          "--output", "new-run", "--model", "model", "--sam-checkpoint", "sam.pt", *flags])

    def test_bounds_and_canary_options(self):
        validate_args(self.args())
        validate_args(self.args("--max-steps", "20", "--train-limit", "4", "--eval-limit", "8", "--warmup-steps", "1"))
        for flags in (("--max-steps", "0"), ("--max-steps", "8193"), ("--warmup-steps", "512"),
                      ("--learning-rate", "nan"), ("--lora-dropout", "1"), ("--gradient-accumulation", "0"),
                      ("--train-limit", "0"), ("--eval-every", "-1"), ("--wall-time-minutes", "inf")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate_args(self.args(*flags))

    def test_eight_epochs_pair_base_exposures_and_cover_d4_once(self):
        noaug = [exposure_for_index(index, 7, 42, "none") for index in range(56)]
        augmented = [exposure_for_index(index, 7, 42, "d4") for index in range(56)]
        self.assertEqual([i for i, _ in noaug], [i for i, _ in augmented])
        self.assertEqual({transform for _, transform in noaug}, {0})
        for row in range(7):
            self.assertEqual([t for i, t in augmented if i == row], list(range(8)))
        self.assertEqual(augmented, [exposure_for_index(index, 7, 42, "d4") for index in range(56)])

    def test_only_answer_tokens_are_supervised_and_truncation_is_rejected(self):
        self.assertEqual(assistant_label_mask([1, 2], [1, 2, 7, 9], 4), [-100, -100, 7, 9])
        for prefix, full, limit in (([1, 2], [1, 3, 7], 8), ([1], [1], 8), ([1], [1, 2, 3], 2)):
            with self.assertRaises(ValueError):
                assistant_label_mask(prefix, full, limit)

    def checkpoint(self, root):
        path = root / "checkpoint-7"
        path.mkdir()
        for name in ("adapter_config.json", "adapter_model.safetensors", "safe_training_state.json",
                     "safe_training_state.safetensors", "coordinate_config.json"):
            (path / name).write_text("synthetic, never executed")
        (path / "trainer_state.json").write_text(json.dumps({"global_step": 7}))
        contract = {"config": {"model": "model", "revision": "a" * 40}, "train": ["sample"]}
        (path / "sft_checkpoint.json").write_text(json.dumps(checkpoint_manifest(path, contract, 7)))
        return path, contract

    def test_complete_checkpoints_are_hashed_and_bound_to_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path, contract = self.checkpoint(Path(directory))
            self.assertEqual(verify_sft_checkpoint(path, contract)["global_step"], 7)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                verify_sft_checkpoint(path, {**contract, "train": ["different"]})
            (path / "adapter_model.safetensors").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "content changed"):
                verify_sft_checkpoint(path)

    def test_unsealed_checkpoint_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                verify_sft_checkpoint(directory)

    def test_bad_manifest_fails_before_gpu_or_output_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            (output / "evidence.txt").write_text("preserve")
            with patch("nucleus_rl.evaluate.read_manifest", side_effect=ValueError("patient overlap")), \
                 patch("nucleus_rl.sft.load_model") as load, self.assertRaises(ValueError):
                main(["--manifest", "train", "--eval-manifest", "val", "--output", str(output),
                      "--model", "model", "--sam-checkpoint", "sam"])
            load.assert_not_called()
            self.assertEqual([path.name for path in output.iterdir()], ["evidence.txt"])
            self.assertEqual((output / "evidence.txt").read_text(), "preserve")


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is optional for CPU-only repository checks")
class SFTTorchTests(unittest.TestCase):
    def test_encoder_uses_same_inference_prefix_and_preserves_eos(self):
        import torch
        class Processor:
            tokenizer = SimpleNamespace(eos_token="<end>", eos_token_id=99)
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs, self.messages = kwargs, messages
                return "inference-prefix"
            def __call__(self, text, **kwargs):
                self.last_text = text[0]
                tokens = [1, 2, 3] if text == ["inference-prefix"] else [1, 2, 3, 7, 99]
                return {"input_ids": torch.tensor([tokens]), "pixel_values": torch.tensor([[.5]]),
                        "image_grid_thw": torch.tensor([[1, 2, 2]])}
        processor = Processor()
        example = {"messages": [{"role": "user", "content": "Find nuclei"}], "image": object(),
                   "completion": '<answer>{"objects":[]}</answer>'}
        encoded, counts = encode_example(processor, example, 8)
        self.assertEqual(encoded["labels"].tolist(), [[-100, -100, -100, 7, 99]])
        self.assertFalse(processor.kwargs["enable_thinking"])
        self.assertEqual(processor.last_text, "inference-prefix" + example["completion"] + "<end>")
        self.assertEqual(example["messages"][-1]["content"], "Find nuclei")
        self.assertEqual(counts, {"prompt_tokens": 3, "assistant_tokens": 2, "total_tokens": 5})

    def test_language_linear_targets_include_mlp_and_exclude_vision(self):
        import torch
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = torch.nn.Module()
                self.language_model.layers = torch.nn.ModuleList([torch.nn.ModuleDict({
                    "q_proj": torch.nn.Linear(2, 2), "gate_proj": torch.nn.Linear(2, 2),
                    "in_proj_qkv": torch.nn.Linear(2, 2)})])
                self.visual = torch.nn.Module()
                self.visual.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
                self.lm_head = torch.nn.Linear(2, 2)
        model = Net()
        self.assertEqual(set(language_lora_targets(model)), {
            "language_model.layers.0.q_proj", "language_model.layers.0.gate_proj", "language_model.layers.0.in_proj_qkv"})

    def test_freeze_checks_detect_in_place_changes_and_reject_non_adapter_training(self):
        import torch
        model = torch.nn.Module()
        model.register_parameter("frozen", torch.nn.Parameter(torch.ones(2), requires_grad=False))
        model.register_parameter("lora_A", torch.nn.Parameter(torch.zeros(2)))
        self.assertEqual(trainable_language_adapters(model), ["lora_A"])
        versions = frozen_versions(model)
        assert_frozen(model, versions)
        with torch.no_grad():
            model.frozen.add_(1)
        with self.assertRaises(RuntimeError):
            assert_frozen(model, versions)
        model.frozen.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            trainable_language_adapters(model)

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "safetensors is required for resume")
    def test_atomic_safe_optimizer_checkpoint_roundtrip(self):
        import torch
        from safetensors.torch import save_file
        from nucleus_rl.training_budget import load_training_state, restore_rng
        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.Parameter(torch.tensor([.5]))
            def save_pretrained(self, path, **kwargs):
                (Path(path) / "adapter_config.json").write_text("{}")
                save_file({"lora_A": self.lora_A.detach().clone()}, str(Path(path) / "adapter_model.safetensors"))
        class Processor:
            def save_pretrained(self, path):
                (Path(path) / "processor_config.json").write_text("{}")
        model = Adapter()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        def step(model, optimizer, scheduler):
            optimizer.zero_grad()
            (model.lora_A ** 2).sum().backward()
            optimizer.step()
            scheduler.step()
        step(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint-1"
            save_checkpoint(path, model, Processor(), optimizer, scheduler, {"configuration": "fixed"}, 1, {})
            self.assertTrue((path / "sft_checkpoint.json").is_file())
            self.assertFalse(path.with_name(".writing-checkpoint-1").exists())
            state = load_training_state(path)
            restored = Adapter()
            restored.load_state_dict(deepcopy(model.state_dict()))
            opt2 = torch.optim.AdamW(restored.parameters(), lr=.1)
            sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda _: 1.)
            opt2.load_state_dict(state["optimizer"])
            sch2.load_state_dict(state["scheduler"])
            restore_rng(state["rng"])
            step(model, optimizer, scheduler)
            step(restored, opt2, sch2)
            self.assertTrue(torch.equal(model.lora_A, restored.lora_A))
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, Processor(), optimizer, scheduler, {}, 1, {})


if __name__ == "__main__":
    unittest.main()
