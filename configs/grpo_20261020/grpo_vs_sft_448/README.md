# GRPO vs continued SFT at 448 px, with and without a count term (C2x, G6x, G8x, g8r)

**Question.** At 448 px, starting from a short SFT warm-up (H1 checkpoint 128), does GRPO still beat continued SFT? Does adding a count term to the PQ reward (G8x) counter PQ's tendency to under-report nuclei? g8r repeats G8x with seed 43.

**Compared with.** C2x is the control for G6x, G8x and g8r (same start, 2 images per step). The 128-step start was chosen after seeing H1's validation curve. These runs were placed inside existing H200 allocations, so their speed is not comparable with single-tenant jobs, and they stopped at the time limit (360 min; 190 for g8r).

| Run | Script | Model | Input | Start | Training | LR | Max steps | Time limit | Launched as |
|---|---|---|---|---|---|---|---|---|---|
| C2x | `c2x-9b448c128-sftcont.sh` | 9B | 448 px | H1 checkpoint 128 | continued SFT | 1e-4 | 800 | 360 min | srun --overlap in 54050290 (G3) |
| G6x | `g6x-9b448c128-pq.sh` | 9B | 448 px | H1 checkpoint 128 | GRPO, reward `format=1,pq=1` | 2e-5 | 800 | 360 min | srun --overlap in 54050287 (G1) |
| G8x | `g8x-9b448c128-pqcount.sh` | 9B | 448 px | H1 checkpoint 128 | GRPO, reward `format=1,pq=1,count=1` | 2e-5 | 800 | 360 min | srun --overlap in 54050285 (G2) |
| g8r | `g8r-9b448c128-pqcount-seed43.sh` | 9B | 448 px | H1 checkpoint 128 | GRPO, reward `format=1,pq=1,count=1` (seed 43) | 2e-5 | 800 | 190 min | srun --overlap in 54050290 (G3), after C2x |

Scripts are verbatim copies of the FASRC files (`slurm/<name>.sbatch`, renamed to `.sh`). Paths, run names and settings are described in `../README.md`. Outputs stay on FASRC under `runs/<output name>`.
