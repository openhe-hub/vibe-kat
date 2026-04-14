#!/usr/bin/env python3
"""KAT depth vision pipeline sweep.

Usage:
    # Smoke test (1 trial per config):
    python run_sweep_depth.py --smoke
    # Full sweep:
    python run_sweep_depth.py
    # Specific tasks:
    python run_sweep_depth.py --tasks reach_target push_button --n_demos 5
"""

import os
import sys
import argparse
import csv
import subprocess
import time

TASKS = ["reach_target", "push_button", "pick_up_cup", "take_lid_off_saucepan"]
N_DEMOS_LIST = [5, 10]
N_TRIALS_FULL = 25
EVAL_TIMEOUT = 300


def main():
    parser = argparse.ArgumentParser(description="KAT depth vision sweep")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 1-trial smoke pass")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--n_demos", nargs="+", type=int, default=None)
    parser.add_argument("--n_trials", type=int, default=None)
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else TASKS
    n_demos_list = args.n_demos if args.n_demos else N_DEMOS_LIST
    n_trials = args.n_trials if args.n_trials else (1 if args.smoke else N_TRIALS_FULL)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if args.output is None:
        suffix = "depth_sweep_smoke.csv" if args.smoke else "depth_sweep.csv"
        output_path = os.path.join(results_dir, suffix)
    else:
        output_path = args.output

    total_evals = len(tasks) * len(n_demos_list) * n_trials
    print(f"{'SMOKE PASS' if args.smoke else 'FULL SWEEP'} (depth pipeline)")
    print(f"Tasks: {tasks}")
    print(f"n_demos: {n_demos_list}")
    print(f"n_trials: {n_trials}")
    print(f"Total evaluations: {total_evals}")
    print("=" * 60)

    if os.path.exists(output_path):
        os.remove(output_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    completed = 0

    for task_name in tasks:
        for n_demos in n_demos_list:
            print(f"\n{'='*60}")
            print(f"[{completed}/{total_evals}] {task_name}, n_demos={n_demos}, n_trials={n_trials}")
            print(f"{'='*60}")

            timeout = max(EVAL_TIMEOUT, n_trials * 120)

            cmd = [
                sys.executable, os.path.join(script_dir, "kat_eval_depth.py"),
                "--task", task_name,
                "--n_demos", str(n_demos),
                "--n_trials", str(n_trials),
                "--output", output_path,
            ]

            t0 = time.time()
            try:
                result = subprocess.run(
                    cmd, timeout=timeout,
                    capture_output=False,
                )
                elapsed = time.time() - t0
                if result.returncode != 0:
                    print(f"  kat_eval_depth.py returned non-zero: {result.returncode}")
                else:
                    completed += n_trials
                print(f"  Elapsed: {elapsed:.1f}s")
            except subprocess.TimeoutExpired:
                elapsed = time.time() - t0
                print(f"  TIMEOUT after {elapsed:.1f}s — skipping {task_name} n_demos={n_demos}")
                os.makedirs(results_dir, exist_ok=True)
                fieldnames = ["task", "n_demos", "trial_id", "seed", "success",
                              "n_waypoints_executed", "parse_error", "execution_error"]
                with open(output_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if os.path.getsize(output_path) == 0:
                        writer.writeheader()
                    for t in range(n_trials):
                        writer.writerow({
                            "task": task_name, "n_demos": n_demos, "trial_id": t,
                            "seed": 1000 + t, "success": False, "n_waypoints_executed": 0,
                            "parse_error": "", "execution_error": f"TIMEOUT after {timeout}s",
                        })
                continue

    print(f"\n{'='*60}")
    print(f"DEPTH SWEEP COMPLETE. Results: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
