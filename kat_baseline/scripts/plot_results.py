#!/usr/bin/env python3
"""Plot KAT baseline results (Stage 4).

Reads sweep.csv, produces success rate vs n_demos plot with Wilson CIs.
"""

import os
import csv
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def wilson_ci(n_success, n_total, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n_total == 0:
        return 0, 0, 0
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return p, max(0, center - margin), min(1, center + margin)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "results", "sweep.csv")
    results_dir = os.path.join(script_dir, "results")

    # Parse CSV
    data = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row['task']
            n = int(row['n_demos'])
            data[task][n][0] += 1  # total
            if row['success'] == 'True':
                data[task][n][1] += 1  # success

    tasks = ['reach_target', 'push_button', 'pick_up_cup', 'take_lid_off_saucepan', 'stack_blocks']
    task_labels = ['Reach Target', 'Push Button', 'Pick Up Cup', 'Take Lid Off Saucepan', 'Stack Blocks']
    n_demos_list = [1, 2, 5, 10, 20]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    markers = ['o', 's', '^', 'D', 'v']

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (task, label) in enumerate(zip(tasks, task_labels)):
        rates = []
        ci_low = []
        ci_high = []
        valid_n = []
        for n in n_demos_list:
            if n in data[task] and data[task][n][0] > 0:
                total, succ = data[task][n]
                p, lo, hi = wilson_ci(succ, total)
                rates.append(p * 100)
                ci_low.append((p - lo) * 100)
                ci_high.append((hi - p) * 100)
                valid_n.append(n)

        if valid_n:
            ax.errorbar(valid_n, rates,
                       yerr=[ci_low, ci_high],
                       label=label, color=colors[i], marker=markers[i],
                       linewidth=2, markersize=8, capsize=4, capthick=1.5)

    ax.set_xscale('log')
    ax.set_xticks(n_demos_list)
    ax.set_xticklabels(n_demos_list)
    ax.set_xlabel('Number of In-Context Demos', fontsize=13)
    ax.set_ylabel('Success Rate (%)', fontsize=13)
    ax.set_title('KAT-style ICIL Baseline — GPT-4o on RLBench', fontsize=14)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()

    # Save
    png_path = os.path.join(results_dir, "results.png")
    pdf_path = os.path.join(results_dir, "results.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # Print table
    print(f"\n{'Task':25s}", end='')
    for n in n_demos_list:
        print(f"  n={n:2d}", end='')
    print()
    print('-' * 70)
    for task, label in zip(tasks, task_labels):
        print(f"{label:25s}", end='')
        for n in n_demos_list:
            if n in data[task] and data[task][n][0] > 0:
                total, succ = data[task][n]
                rate = 100 * succ / total
                print(f"  {rate:4.0f}%", end='')
            else:
                print(f"    ...", end='')
        print()


if __name__ == "__main__":
    main()
