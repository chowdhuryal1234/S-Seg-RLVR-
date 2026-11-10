# NuNext reproduction (detection: SFT + GRPO)

An independent reimplementation of [NuNext: Reframing Nucleus Detection as Next-Point Detection](https://arxiv.org/abs/2603.07098)
(arXiv:2603.07098v1). The authors' code is not released, and v1 has no appendix. Hyperparameters the paper leaves
out are marked **assumed** below and exposed as command-line flags.

## Scope

| Component | Paper | Here |
|---|---|---|
| Next-point formulation, coordinate tokens, raster order | Sec. 3.1 | Implemented (`coords.py`) |
| Naive SFT (NSFT) | Table 5 | `sft.py --sigma 0` |
| Spatial-aware soft supervision (SASS) | Eq. 7–8 | `sft.py --sigma >0` |
| Chain-of-visual-thought (CoVT) | Eq. 9–10 | **Not implemented** (+0.003 F1 in Table 5; needs latent tokens decoded by SAM) |
| GRPO with distribution-matching F1 reward, format check | Eq. 11–17 | `grpo.py` |
| Low-variance group filtering (LVGF) | Sec. 3.4 | `grpo.py --delta` |
| Fine-grained advantage shaping (FGAS) | Eq. 18 | `grpo.py --fgas-decay` |
| Task-guided PQ reward, PromptNucSeg segmentation | Eq. 19 | Not implemented (next stage) |
| External benchmarks (CPM, Kumar, CoNSeP, …) | Tables 3–4 | Not implemented |

## Targets (PanNuke fold 3 detection F1, Table 5)

| Configuration | Paper F1 |
|---|---:|
| NSFT | 0.812 |
| NSFT + SASS | 0.819 |
| NSFT + SASS + CoVT | 0.822 |
| NSFT + GRPO | 0.830 |
| NSFT + SASS + CoVT + GRPO | 0.835 |
| … + LVGF | 0.837 |
| … + FGAS | 0.838 |
| … + LVGF + FGAS | 0.840 |
| … + LVGF + FGAS + task-guided reward | 0.842 |

The first row of Table 5 (no module ticked, F1 0.587) is not explained in v1. Most of the gain comes from SFT itself;
GRPO adds about 0.02. Because CoVT is skipped, the comparable chain here is NSFT (0.812) → NSFT + GRPO (0.830), then the
LVGF/FGAS increments on top.

## Settings

| Setting | Value | Source |
|---|---|---|
| Base model | Qwen2.5-VL-3B-Instruct | Paper |
| Data | PanNuke folds 1+2 train (5,179 images), fold 3 validation (2,722) | Paper |
| Instruction | "Point to nucleus. Output the coordinates in [x, y] format." | Paper |
| Output | x/y coordinate tokens per nucleus in raster order, then the end token | Paper (exact separators unstated; none used here) |
| Bins K | 256 (≈1 px on 256 px images) | **Assumed** |
| SASS σ | 1 bin | **Assumed** |
| Tuning | Full fine-tuning of the language model, embeddings and head; vision tower frozen; fp32 master weights, bf16 autocast | **Assumed** |
| New-token init | Mean of existing embeddings plus small noise | **Assumed** |
| Input size | 448 × 448 (16 × 16 visual tokens) | **Assumed** |
| SFT | 3 epochs, effective batch 16, lr 1e-5 cosine, 3% warm-up, D4 augmentation | **Assumed** |
| Match radius r | 12 px (6 µm at 40×, the usual PanNuke detection radius) | **Assumed** |
| F1 | Dataset-level (micro) over fold 3; mean per-image F1 also logged | **Assumed** |
| GRPO | Group size 8, 4 images per step, T = 1.0, lr 1e-6, k3 KL β = 0.04, 500 steps, one update per batch | **Assumed** (ε = 0.2 is the only value given; inactive with one update per batch) |
| LVGF δ | 0.02 (std of F1 within a group) | **Assumed**; tune from the logged `filtered_groups` |
| FGAS decay β | 0.5 | **Assumed** |
| Decoding | Greedy for evaluation; plain sampling (no top-k/top-p, no repetition penalty) for rollouts | **Assumed** |

## Stages

1. `scripts/prepare_pannuke.py`: PanNuke `images.npy`/`masks.npy` → PNGs and centroid manifests (`fold{k}.jsonl`).
2. `python -m nucleus_rl.nunext.sft` (`--sigma 0` for NSFT, `--sigma 1` for SASS). Goal: F1 ≈ 0.81.
3. `python -m nucleus_rl.nunext.grpo --init-model <sft>/model`, first plain, then `--delta 0.02 --fgas-decay 0.5`.
   Goal: about +0.02 F1 over the SFT start.
4. `python -m nucleus_rl.nunext.evaluate` re-scores any saved model on a manifest.

On Databricks, notebooks `scripts/dbx/10_nunext_data.py`, `11_nunext_sft.py` and `12_nunext_grpo.py` run these
stages (1×H100).

## Checks done before GPU runs

- Unit tests (`tests/test_nunext.py`): quantization, raster order, format check, Hungarian matching and the F1 = DQ
  identity, D4 geometry, soft labels (σ = 0 equals cross-entropy), LVGF on the paper's Fig. 3 example, FGAS signs.
- CPU smoke runs with a tiny random Qwen2.5-VL: SFT overfits one image to F1 = 1.0 through greedy decoding, and GRPO
  updates with non-zero gradients, KL and group filtering.

## Differences from the user-facing project pipeline

This package is separate from `nucleus_rl/sft.py` and `nucleus_rl/grpo.py` (JSON box+point prompts, LoRA, frozen SAM).
It adds tokens to the vocabulary, so models are saved in full rather than as adapters.
