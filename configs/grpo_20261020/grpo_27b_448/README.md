# GRPO on the strongest SFT start: 27B at 448 px (G7)

**Question.** Does GRPO with the PQ reward improve on the best SFT model, Qwen3.6-27B at 448 px (H2 checkpoint 128)?

**Compared with.** Its own start, H2's final (512-step) evaluation, and H2's best intermediate checkpoint (step 384, chosen after the fact). The run stopped at the time limit (540 min).

| Run | Script | Model | Input | Start | Training | LR | Max steps | Time limit | Launched as |
|---|---|---|---|---|---|---|---|---|---|
| G7 | `g7-27b448c128-pq.sh` | 27B | 448 px | H2 checkpoint 128 | GRPO, reward `format=1,pq=1` | 2e-5 | 800 | 540 min | sbatch 54062170 |

Scripts are verbatim copies of the FASRC files (`slurm/<name>.sbatch`, renamed to `.sh`). Paths, run names and settings are described in `../README.md`. Outputs stay on FASRC under `runs/<output name>`.
