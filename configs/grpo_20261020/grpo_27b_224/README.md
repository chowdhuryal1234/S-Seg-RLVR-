# GRPO on the 27B model at 224 px (G4)

**Question.** Does GRPO with the PQ reward also improve the strongest 224 px SFT model (Qwen3.6-27B)?

**Compared with.** Its own starting evaluation (the merged 27B SFT adapter). There is no matched continued-SFT control for 27B at 224 px.

| Run | Script | Model | Input | Start | Training | LR | Max steps | Time limit | Launched as |
|---|---|---|---|---|---|---|---|---|---|
| G4 | `g4-27b-pq.sh` | 27B | 224 px | 27B 224 px D4 SFT, final (512 steps) | GRPO, reward `format=1,pq=1` | 2e-5 | 800 | 540 min | sbatch 54050295 |

Scripts are verbatim copies of the FASRC files (`slurm/<name>.sbatch`, renamed to `.sh`). Paths, run names and settings are described in `../README.md`. Outputs stay on FASRC under `runs/<output name>`.
