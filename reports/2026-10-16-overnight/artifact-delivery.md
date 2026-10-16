# English artifact delivery

## Historical translation delivery

The original delivery translated 14 existing files and added 22 artifact files. Its audit verified that all 811 protected evidence files remained byte-identical. [translation-audit.json](translation-audit.json) is the immutable historical inventory and preservation record; it has not been rewritten to cover later work. Its Unicode-escaped historical filename preserves the existing local-note path.

The original translated PNGs were already English. The former preview_inputs page remains a historical pending-input preview. The original translation checks covered six renderer syntax checks, English prose, exact raw summaries, SVG/PNG inspection and the 811 protected-file hashes.

## Final morning roadmap update — October 16, 2026

**COMPLETE — all six runs finished their requested steps without a recorded failure.** Final snapshot: 2026-10-16 04:02 EDT. The six model runners total 4.205 hours; the supervisor window is 4.222 hours, excluding allocation setup. This completes the experiment plan; reliable nucleus detection remains unresolved.

Updated only `roadmap.md`, `index.html`, `roadmap.svg`, `roadmap.png` and this delivery note. The final roadmap replaces its earlier PENDING/RUNNING status with measured results from all six completed runs. It includes both 1,000-step seed-42 7B learning-rate arms, the 400-step 32B run, the 1,000-step selected-setting seed-43 repeat, and separate 20-step timing checks. The rationale diagram distinguishes October 9 training from October 15 frozen inference and the October 16 adaptation suite.

[Open roadmap](index.html) · [Markdown and evidence](roadmap.md) · [SVG](roadmap.svg) · [PNG](roadmap.png)

Validation64 contains 384 crop-level GT occurrences from 340 unique source nuclei, across four patients; these scores remain separate from benchmark8 / 51 and the older training8 / 41. The seed-43 repeat uses a setting chosen on reused validation. It is not independent test evidence. Training uses full-mask rewards and frozen SAM; no weak-label, SOTA or clinical result is claimed.

## Separate diagnostic and training reports

- [October 9 training](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/main/reports/2026-10-09/README.md)
- [Paired prompt test](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/prompt-ablation-20261016/reports/2026-10-15-prompt-ablation/README.md)
- [Model/segmenter comparisons](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/followup-cd-20261016/reports/2026-10-15-followup-cd/README.md)
- [Rectangle diagnostic](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/coordinate-calibration-20261016/reports/2026-10-15-coordinate-calibration/README.md)
- [One-nucleus diagnostic](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/single-nucleus-20261016/reports/2026-10-15-single-nucleus/README.md)
- [Three-nucleus diagnostic](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/multi-nucleus-20261016/reports/2026-10-15-multi-nucleus/README.md)
- [Candidate tasks](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/recognition-coordinate-20261016/reports/2026-10-15-recognition-coordinate/README.md)
- [Isolated patches](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/isolated-crop-20261016/reports/2026-10-15-isolated-crop/README.md)
- [7B timing check](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/benchmark-7b-20261016/reports/2026-10-16-benchmark-7b/README.md)
- [32B timing check](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/benchmark-32b-20261016/reports/2026-10-16-benchmark-32b/README.md)
- [7B · LR 1e-5 · seed 42](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/main-7b-lr1e-5-20261016/reports/2026-10-16-main-7b-lr1e-5/README.md)
- [7B · LR 5e-5 · seed 42](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/main-7b-lr5e-5-20261016/reports/2026-10-16-main-7b-lr5e-5/README.md)
- [32B · LR 1e-5 · seed 42](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/main-32b-lr1e-5-20261016/reports/2026-10-16-main-32b-lr1e-5/README.md)
- [7B · LR 5e-5 · seed 43](https://github.com/chowdhuryal1234/S-Seg-RLVR-/blob/codex/repeat-7b-seed43-20261016/reports/2026-10-16-repeat-7b-seed43/README.md)

These explicit cross-branch GitHub links allow this standalone roadmap report to resolve evidence before the other branches merge. Local roadmap figures and the historical audit remain relative links.

## Final-update checks

- All six source `result.json` files report completed requested steps, changed final adapters and unchanged SAM/vision hashes; the suite reports `finished` and zero child exit codes.
- Supporting tables are generated from saved before/after results and exact resource timings; benchmark8 and validation64 are separate.
- SVG and PNG share the same layout; PNG uses PIL and the repository font helper. Both are inspected for clipping and legibility.
- Local figure/document links, SVG syntax, complete English status wording and preservation of the prior cross-branch evidence URLs are checked.
- The historical translation audit SHA-256 remains `22ffc1504ef2d99d1e6befb968d56ddf02b8eee6fba8bb6f86c720c4465755c0`.

This final artifact update performs no GPU/model launch, Git operation, raw-data edit, frozen-code edit, reward change or protocol change. Publication and model-checkpoint backups are handled separately.

## Final preservation and publication verification

[Final verification record](final-verification.json) confirms 37,946 backed-up files (2,412,020,844 bytes) match the remote SHA-256 inventory exactly. All six sealed final checkpoints passed verification, and each final adapter matches its checkpoint. Persistent remote results and the complete local backup are retained. Checkpoint loading/resume on a GPU was not exercised by this audit.

All 18 experiment/infrastructure/report PRs are open drafts, attached to the project chat and unmerged; 103 expected report/code paths were verified at their live heads. The H200 allocation was released after training completed and backup verification passed, with 4h48m10s total allocation time. The existing A100 was retained. No additional allocation, extension or usage reset was used.
