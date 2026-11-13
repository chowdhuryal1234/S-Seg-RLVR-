# FASRC launch records, Nov 2026

These are the scripts used to launch the November 2026 FASRC runs, kept verbatim. They are not maintained tooling.

Each `*.sbatch.sh` file was submitted on FASRC as `<name>.sbatch`. The `.sh` suffix was added only so the repository content check accepts the file, so the references inside the scripts still use the `.sbatch` names.

| Folder | Experiment |
|---|---|
| `fasrc_trustmix_20261104/` | Trusted-mix SFT/GRPO lanes and validation suites. |
| `tiles256_20261105/` | SFT on 256 px all-nuclei tiles (`t256s1`). |
| `zoom_probe_20261105/` | 2x2 zoom inference probe. |
| `vision_lora_20261106/` | Vision-LoRA A/B study: SFT arms, greedy eval, and 8-sample support eval. `analyze_ab.py` holds the pre-registered decision rule and the paired tile bootstrap. |

The analysis helpers are in `scripts/analysis_202611/` (repository root):
- `never_found.py`, `rebin.py` and `gallery.py` for the never-found nucleus analysis.
- `run_stardist.py`, `run_instanseg.py` and `score_specialist.py` for the specialist-model baseline on validation tiles.

These scripts read validation tiles only. Results stay out of the repository.
