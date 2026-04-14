# KAT Baseline — Keypoint Action Tokens for In-Context Imitation Learning
#
# Project structure:
#   Core modules:
#     kat_eval.py              — Baseline evaluation (privileged GT state)
#     kat_eval_depth.py        — Depth-based vision evaluation
#     depth_object_detector.py — Object detection: background subtraction, robot masking,
#                                color filtering, cross-camera matching, offset calibration
#     camera_utils.py          — Camera projection, depth Z-buffer→meters (PyRep method)
#     run_sweep.py             — Full sweep driver (baseline)
#     run_sweep_depth.py       — Full sweep driver (depth pipeline)
#
#   scripts/                   — Utility scripts (smoke test, recording, plotting, viz)
#   diagnostics/               — Debugging and diagnostic scripts
#   archive/                   — Abandoned DINO-ViT pipeline (kept for reference)
#   cache/                     — LLM response cache (SHA256 → JSON)
#   results/                   — Evaluation result CSVs
