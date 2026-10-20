"""CPU checks of the GRPO runner's arguments, schedule, adapter contract and batching."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from nucleus_rl.grpo import build_parser, check_init_adapter, exposure, validate_args

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


class GRPOArgumentTests(unittest.TestCase):
    def args(self, *flags):
        return build_parser().parse_args([
            "--manifest", "train.jsonl", "--eval-manifest", "val.jsonl", "--output", "run",
            "--model", "Qwen/Qwen3.5-9B", "--revision", REVISION, "--init-adapter", "sft",
            "--sam-checkpoint", "sam.pt", "--coordinate-frame", "normalized1000",
            "--min-pixels", "50176", "--max-pixels", "200704", *flags])

    def test_bounds(self):
        args = self.args()
        validate_args(args)
        self.assertEqual(args.reward_weights, {"format": 1.0, "pq": 1.0})
        validate_args(self.args("--objective", "sft", "--group-size", "1"))
        for flags in (("--group-size", "1"), ("--revision", "main"), ("--reward", "dice=1"),
                      ("--beta", "-0.1"), ("--temperature", "0"), ("--warmup-steps", "600"),
                      ("--min-pixels", "300000"), ("--prompts-per-step", "0")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate_args(self.args(*flags))

    def test_exposure_is_seeded_and_covers_each_epoch(self):
        rows = [exposure(index, 5, 42, "d4") for index in range(10)]
        self.assertEqual(rows, [exposure(index, 5, 42, "d4") for index in range(10)])
        self.assertEqual(sorted(row for row, _ in rows[:5]), list(range(5)))
        self.assertEqual(sorted(row for row, _ in rows[5:]), list(range(5)))
        many = {exposure(index, 5, 42, "d4")[1] for index in range(200)}
        self.assertEqual(many, set(range(8)))
        self.assertEqual({exposure(index, 5, 42, "none")[1] for index in range(20)}, {0})
        self.assertEqual([r for r, _ in rows], [exposure(index, 5, 42, "none")[0] for index in range(10)])

    def test_init_adapter_must_match_model_and_input_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "adapter_model.safetensors").write_bytes(b"weights")
            (path / "adapter_config.json").write_text(json.dumps(
                {"base_model_name_or_path": "Qwen/Qwen3.5-9B", "revision": REVISION, "r": 16}))
            (path / "coordinate_config.json").write_text(json.dumps(
                {"coordinate_frame": "normalized1000", "min_pixels": 50176, "max_pixels": 200704}))
            args = self.args("--init-adapter", directory)
            self.assertEqual(check_init_adapter(args)["lora_rank"], 16)
            for flags in (("--model", "Qwen/Qwen3.6-27B"), ("--revision", "0" * 40),
                          ("--coordinate-frame", "processed"), ("--min-pixels", "200704")):
                with self.subTest(flags=flags), self.assertRaises(ValueError):
                    check_init_adapter(self.args("--init-adapter", directory, *flags))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class GRPOBatchTests(unittest.TestCase):
    def test_group_batch_replicates_prompt_and_pads_answers(self):
        import torch
        from nucleus_rl.grpo import completion_batch, repeat_inputs
        inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
                  "pixel_values": torch.arange(8.).reshape(4, 2), "image_grid_thw": torch.tensor([[1, 2, 2]]),
                  "mm_token_type_ids": torch.tensor([[0, 1, 0]])}
        repeated = repeat_inputs(inputs, 3)
        self.assertEqual(tuple(repeated["pixel_values"].shape), (12, 2))
        self.assertEqual(tuple(repeated["image_grid_thw"].shape), (3, 3))
        batch, mask, longest = completion_batch(inputs, [[7, 8], [9], [4, 5, 6]], pad_id=0)
        self.assertEqual(longest, 3)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3, 7, 8, 0], [1, 2, 3, 9, 0, 0], [1, 2, 3, 4, 5, 6]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0], [1] * 6])
        self.assertEqual(mask.tolist(), [[1, 1, 0], [1, 0, 0], [1, 1, 1]])
        self.assertEqual(batch["mm_token_type_ids"].tolist(), [[0, 1, 0, 0, 0, 0]] * 3)

    def test_logprobs_align_answer_tokens_with_preceding_logits(self):
        import torch
        from nucleus_rl.grpo import token_logprobs

        class Fixed(torch.nn.Module):
            """Position t puts probability 0.5 on token t+1 and spreads the rest."""
            def forward(self, input_ids, use_cache=False, logits_to_keep=None, **_):
                length, vocab = input_ids.shape[1], 10
                logits = torch.zeros(input_ids.shape[0], length, vocab)
                for t in range(length - 1):
                    logits[:, t, input_ids[0, t + 1]] = 2.0
                if logits_to_keep:
                    logits = logits[:, -logits_to_keep:]
                return type("Output", (), {"logits": logits})()

        batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        for keep in (False, True):
            values = token_logprobs(Fixed(), batch, 2, keep)
            expected = 2.0 - torch.logsumexp(torch.tensor([2.0] + [0.0] * 9), 0)
            self.assertTrue(torch.allclose(values, expected.expand(1, 2)), values)


if __name__ == "__main__":
    unittest.main()
