# GRPO vs continued SFT at 224 px, and reward choice (C1, G1, G2, G3, G5)

**Question.** Starting from the same finished 9B SFT adapter, does GRPO beat simply continuing SFT for the same number of steps and images? Which reward works: foreground IoU (G1), per-crop PQ (G2), or the mentor's starting set of foreground IoU + count + separation (G3)? G5 repeats G2 with a 2.5× higher learning rate on an A100.

**Compared with.** C1 is the control for G1-G3: same start, images, D4 views, 2 images per step and 800 steps, with the SFT loss instead of GRPO. G5 ran on different hardware (A100) and is compared with its own starting evaluation.

| Run | Script | Model | Input | Start | Training | LR | Max steps | Time limit | Launched as |
|---|---|---|---|---|---|---|---|---|---|
| C1 | `c1-9b-sftcont.sh` | 9B | 224 px | 9B 224 px D4 SFT, final (512 steps) | continued SFT | 1e-4 | 800 | 570 min | sbatch 54050293 |
| G1 | `g1-9b-fgiou.sh` | 9B | 224 px | 9B 224 px D4 SFT, final (512 steps) | GRPO, reward `format=1,fg_iou=1` | 2e-5 | 800 | 570 min | sbatch 54050287 |
| G2 | `g2-9b-pq.sh` | 9B | 224 px | 9B 224 px D4 SFT, final (512 steps) | GRPO, reward `format=1,pq=1` | 2e-5 | 800 | 570 min | sbatch 54050285 |
| G3 | `g3-9b-mentor.sh` | 9B | 224 px | 9B 224 px D4 SFT, final (512 steps) | GRPO, reward `format=1,fg_iou=1,count=1,separation=1` | 2e-5 | 800 | 570 min | sbatch 54050290 |
| G5 | `g5-9b-pq-lr5e5-a100.sh` | 9B | 224 px | 9B 224 px D4 SFT, final (512 steps) | GRPO, reward `format=1,pq=1` | 5e-5 | 800 | 520 min | srun --overlap in A100 job 54046493 |

Scripts are verbatim copies of the FASRC files (`slurm/<name>.sbatch`, renamed to `.sh`). Paths, run names and settings are described in `../README.md`. Outputs stay on FASRC under `runs/<output name>`.
