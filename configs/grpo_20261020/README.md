# GRPO-from-SFT experiments (2026-10-19 → 10-21)

Launch scripts for the high-resolution SFT and GRPO-from-SFT runs on FASRC. Each experiment
folder holds the scripts exactly as they ran. The only change is that `.sbatch` files are renamed
to `.sh` so the repository content check accepts them. Code for all runs is `nucleus_rl/sft.py`
(SFT, PR #25) and `nucleus_rl/grpo.py` (GRPO and continued SFT, PR #26, commit `5b24264`).

## Run names

| Prefix | Meaning |
|---|---|
| `H` | High-resolution SFT from the base model (448 or 672 px input instead of 224) |
| `C` | Control: continued SFT from the same start, steps and images as the matching GRPO run |
| `G` | GRPO from a finished or partial SFT adapter |
| `x` | 448 px run that starts from H1's step-128 checkpoint (e.g. `g6x-9b448c128-pq`) |
| `r` | Repeat of a run with another seed (`g8r` = G8x with seed 43) |

Folder names follow `<id>-<model><px>[c<checkpoint>]-<reward>`. For example, `g7-27b448c128-pq`
is Qwen3.6-27B at 448 px, starting from H2's checkpoint 128, with reward `format=1,pq=1`.

## Folders

| Folder | Question | Runs |
|---|---|---|
| `resolution_sft/` | Does a larger input (more visual tokens) improve SFT? | H1, H2, H3 |
| `grpo_vs_sft_224/` | Does GRPO beat continued SFT from the same 224 px SFT adapter, and which reward? | C1, G1, G2, G3, G5 |
| `grpo_27b_224/` | Does GRPO also help the 27B model at 224 px? | G4 |
| `grpo_vs_sft_448/` | Does GRPO still beat continued SFT at 448 px, and does a count term help? | C2x, G6x, G8x, g8r |
| `grpo_27b_448/` | GRPO on the strongest SFT start (27B, 448 px) | G7 |
| `round2/` | Follow-ups: longer G7, count term at 27B, start point, exploration | G7b, G9, G10, G11, G12 |

## Shared settings

- Data: `monuseg_official30_grpo_v1` (official 30 MoNuSeg training images, 128 px crops, patient-disjoint
  256 train / 64 validation, 384 validation nuclei). The official test set is not used.
- Qwen language LoRA only; the vision encoder and SAM2.1-tiny stay frozen. `--coordinate-frame normalized1000`.
- Input size is set by `--min-pixels/--max-pixels`: 50176 → 224 px (7×7 visual tokens for Qwen3.x),
  200704 → 448 px (14×14), 451584 → 672 px (21×21).
- GRPO: group size 8, 2 images per step, temperature 1.0, k3 KL with β = 0.04, lr 2e-5, D4 views.
  Rewards are weighted sums of `format`, `fg_iou`, `pq`, `count` and `separation` (see `nucleus_rl/grpo_rewards.py`).
- Runs stop at `--max-steps` or at `--wall-time-minutes`, whichever comes first, and then run the
  final evaluation. Most 448 px and 27B runs hit the time limit.

## Files here

- `env.sh`: paths, offline Hugging Face cache and pinned model revisions on FASRC. The SFT workspace
  (`$R`) holds the virtual environment, data, SAM checkpoint and SFT adapters.
- `make_grpo.sh`: writes one GRPO or continued-SFT job script. It refuses to overwrite an existing
  output directory or script. Since 2026-10-20 it requests 2 CPUs / 32 GB, because jobstats showed about
  1–2 busy cores and under 5 GB RAM per job. Runs launched before that used 8 CPUs / 128 GB.
- `canary_grpo_9b.sh`, `canary_hires_9b.sh`: 10- and 20-step smoke runs that gated the full jobs.

Some runs were started inside an existing allocation (`srun --jobid=<host job> --overlap bash <script>`)
rather than with `sbatch`, because the H200 queue was full. An H200 has room for two 9B jobs. Each experiment
README lists how its runs were launched.
