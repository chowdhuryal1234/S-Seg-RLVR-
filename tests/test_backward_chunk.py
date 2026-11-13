import unittest
from contextlib import contextmanager
from types import SimpleNamespace

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "needs torch")
class BackwardChunkTest(unittest.TestCase):
    """Chunked GRPO backward must give the same loss, KL and gradient as the single pass."""

    def make_policy(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                self.embed = torch.nn.Embedding(20, 8)
                self.head = torch.nn.Linear(8, 20)
                self.delta = torch.nn.Parameter(torch.randn(20, 8) * 0.1)  # the "adapter"
                self.adapter_on = True
                self.device = torch.device("cpu")

            def forward(self, input_ids, attention_mask=None, use_cache=False, **_):
                weight = self.embed.weight + (self.delta if self.adapter_on else 0)
                hidden = torch.nn.functional.embedding(input_ids, weight).cumsum(1)
                return SimpleNamespace(logits=self.head(hidden))

            @contextmanager
            def disable_adapter(self):
                self.adapter_on = False
                try:
                    yield
                finally:
                    self.adapter_on = True
        return Toy()

    def run_group(self, chunk):
        from nucleus_rl.grpo import group_backward
        policy = self.make_policy()
        inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        rows = [[4, 5, 6], [7], [8, 9], [10, 11, 12, 13], [5], [6, 7, 8]]
        advantages = [1.2, -0.4, 0.3, -1.1, 0.0, 0.5]
        args = SimpleNamespace(temperature=1.0, beta=0.04, prompts_per_step=2, backward_chunk=chunk)
        loss, kl = group_backward(policy, inputs, rows, advantages, 0, False, args)
        grads = {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}
        return loss, kl, grads

    def test_chunks_match_single_pass(self):
        loss, kl, grads = self.run_group(None)
        for chunk in (1, 2, 4, 6, 10):
            c_loss, c_kl, c_grads = self.run_group(chunk)
            self.assertAlmostEqual(loss, c_loss, places=5)
            self.assertAlmostEqual(kl, c_kl, places=6)
            self.assertEqual(set(grads), set(c_grads))
            for name in grads:
                self.assertTrue(torch.allclose(grads[name], c_grads[name], atol=1e-6), (chunk, name))

    def test_flag_validation(self):
        from nucleus_rl.grpo import build_parser, validate_args
        base = ["--manifest", "m", "--eval-manifest", "e", "--output", "o", "--init-adapter", "a",
                "--model", "x", "--revision", "0" * 40, "--sam-checkpoint", "s",
                "--coordinate-frame", "normalized1000", "--min-pixels", "200704", "--max-pixels", "200704"]
        args = build_parser().parse_args(base + ["--backward-chunk", "0"])
        with self.assertRaises(ValueError):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
