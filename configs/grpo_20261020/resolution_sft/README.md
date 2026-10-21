# High-resolution SFT (H1, H2, H3)

**Question.** Does giving Qwen a larger input image improve SFT? A 128 px crop resized to 224 px is only 7×7 visual tokens for Qwen3.x (about 18 px of the crop per token); 448 px gives 14×14 and 672 px gives 21×21.

**Compared with.** The 224 px D4 arms of the 2026-10-17 SFT suite (PR #25): same code, 512 steps, 4 images per step, seed 42 and D4 augmentation. Only `--min-pixels/--max-pixels` change. Checkpoints 128/256/384 are evaluated during training; H1's and H2's checkpoint 128 are the starting points of the 448 px GRPO runs.

| Run | Script | Model | Input | Start | Training | LR | Max steps | Time limit | Launched as |
|---|---|---|---|---|---|---|---|---|---|
| H1 | `h1-9b-sft-448.sh` | 9B | 448 px | base model | SFT from base | 1e-4 (SFT default) | 512 | 540 min | sbatch 54047894 |
| H2 | `h2-27b-sft-448.sh` | 27B | 448 px | base model | SFT from base | 1e-4 (SFT default) | 512 | 540 min | sbatch 54047898 |
| H3 | `h3-9b-sft-672.sh` | 9B | 672 px | base model | SFT from base | 1e-4 (SFT default) | 512 | 540 min | sbatch 54055461 |

Scripts are verbatim copies of the FASRC files (`slurm/<name>.sbatch`, renamed to `.sh`). Paths, run names and settings are described in `../README.md`. Outputs stay on FASRC under `runs/<output name>`.
