"""Reproduction of NuNext (arXiv:2603.07098): nucleus detection as next-point prediction.

Scope: PanNuke detection with supervised fine-tuning (optionally with spatial-aware soft
supervision) followed by GRPO with the distribution-matching (F1) reward, low-variance
group filtering and fine-grained advantage shaping. See README.md for the assumptions
that fill gaps in the paper.
"""
