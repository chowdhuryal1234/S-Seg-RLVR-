# Round 2 follow-ups (G7b, G9, G10, G11, G12)

**Question.** (a) Does G7 keep improving past step 429 (G7b)? (b) Does the count term help at 27B 448 px (G9 vs G7)? (c) Does GRPO gain more from the fully trained 512-step SFT than from the 128-step warm-up (G10 vs G9, G11 vs G8x/g8r)? (d) Does more exploration (temperature 1.3) raise recall (G12 vs G8x/g8r)?

**Compared with.** Submitted 2026-10-20 with 2 CPUs / 32 GB per GPU job (4 CPUs / 64 GB for the G11+G12 pair). As of 2026-10-21 all four jobs are still queued (Priority).

| Run | Script | Model | Input | Start | Training | LR | Max steps | Time limit | Launched as |
|---|---|---|---|---|---|---|---|---|---|
| G7b | `g7b-27b448c128-pq-resume429.sh` | 27B | 448 px | H2 checkpoint 128 | GRPO, reward `format=1,pq=1` (resumes g7-27b448c128-pq/checkpoints/checkpoint-429) | 2e-5 | 800 | 600 min | sbatch 54197038 (completed) |
| G9 | `g9-27b448c128-pqcount.sh` | 27B | 448 px | H2 checkpoint 128 | GRPO, reward `format=1,pq=1,count=1` | 2e-5 | 800 | 600 min | sbatch 54197039 (failed at start; see below) |
| G10 | `g10-27b448final-pqcount.sh` | 27B | 448 px | H2 final (512 steps) | GRPO, reward `format=1,pq=1,count=1` | 2e-5 | 800 | 600 min | sbatch 54197040 (failed at start; see below) |
| G11 | `g11-9b448final-pqcount.sh` | 9B | 448 px | H1 final (512 steps) | GRPO, reward `format=1,pq=1,count=1` | 2e-5 | 800 | 600 min | pair job 54197042 (failed at start; see below) |
| G12 | `g12-9b448c128-pqcount-t13.sh` | 9B | 448 px | H1 checkpoint 128 | GRPO, reward `format=1,pq=1,count=1` (temperature 1.3) | 2e-5 | 800 | 600 min | pair job 54197042 (failed at start; see below) |

`pair-g11-g12.sh` runs G11 and G12 side by side on one H200 (about 60 GB each); it calls the two scripts by their original FASRC paths (`slurm/*.sbatch`).

## What happened

- All four first-attempt jobs started on 2026-10-21 at 04:05 on the same node (`holygpu8a12204`).
- **G7b completed** (steps 430–800).
- **G9, G10, G11 and G12 failed within about 25 s at model load** with `CUDA error: CUDA-capable device(s) is/are busy or unavailable`, before any training step. Their output folders keep only `arguments.json`, `preflight.json` and `failure.json`.
- **The `*-try2.sh` scripts resubmitted them on 2026-10-22.** Each is identical except:
  - a new output folder,
  - `#SBATCH --exclude=holygpu8a12204`,
  - for G12 only, code at commit `4ab8da0`, which scores sampled answers at the sampling temperature. Runs at temperature 1.0 are unaffected by that change.

| Retry | Slurm job |
|---|---|
| G9 | `g9-27b448c128-pqcount-try2.sh`, sbatch 54690689 |
| G10 | `g10-27b448final-pqcount-try2.sh`, sbatch 54690691 |
| G11 + G12 | `pair-g11-g12-try2.sh` (calls the two `-try2` scripts), sbatch 54690692 |

Scripts are verbatim copies of the FASRC files (`slurm/<name>.sbatch`, renamed to `.sh`). Paths, run names and settings are described in `../README.md`. Outputs stay on FASRC under `runs/<output name>`.
