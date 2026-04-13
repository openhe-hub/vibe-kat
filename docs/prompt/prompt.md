# KAT-style ICIL Baseline in RLBench — Implementation Spec

## Goal

Implement a Keypoint Action Tokens (KAT)-style in-context imitation learning baseline in RLBench. The "policy" is an LLM API call: given N demo (scene, action) pairs as context, predict an action waypoint sequence for a new scene. Measure success rate vs number of demos.

This is a baseline for a CoRL paper on multi-demo ICIL. It is NOT the main contribution. Keep the implementation minimal, faithful to the KAT paper's design choices, and reproducible.

## Non-goals (do NOT do these)

- No vision / DINO / keypoint detection. Use RLBench's privileged `task_low_dim_state` for perfect object poses.
- No training, no fine-tuning, no gradient steps.
- No closed-loop replanning. KAT is open-loop: one LLM call per episode, execute the predicted waypoint sequence, then evaluate.
- No abstractions, config systems, or dataclasses in the first version. Single-file script until it works end-to-end.
- No real-time control. Seconds of LLM latency per episode is fine.

## Staged delivery

**Stop after each stage and wait for review before proceeding.** Do not chain stages.

### Stage 1: Single-task, single-trial smoke test

Write ONE script `kat_smoke.py` that:

1. Launches RLBench with task `reach_target`
2. Uses action mode: `MoveArmThenGripper(arm=EndEffectorPoseViaPlanning(absolute_mode=True), gripper=Discrete())`
3. Uses observation config with `task_low_dim_state=True`
4. Collects 3 demos via `task.get_demos(3, live_demos=True)`
5. For each demo, extracts:
   - Scene: object poses from `obs.task_low_dim_state` (print its shape and contents first so we confirm structure)
   - Actions: subsample 10 keyframe waypoints from the demo's gripper pose trajectory. Use a simple uniform subsample for v1. Each waypoint = `[x, y, z, r1, r2, r3, r4, r5, r6, gripper]` where `r1..r6` is the 6D rotation representation (first two columns of the rotation matrix, flattened). **Do NOT use quaternions** — LLMs handle 6D rotation much better due to sign ambiguity issues with quats.
6. Resets the env to a new variation, builds a prompt with the 3 demos + the new scene, calls the LLM API once, parses the response, executes the waypoints, reads the success flag.
7. Prints: prompt token count, raw LLM response, parsed actions, per-waypoint execution status, final success.

**LLM call details:**
- Model: `claude-opus-4-6` (API key will be provided via env var `ANTHROPIC_API_KEY`). Use the Anthropic Python SDK.
- Temperature: 0
- Max tokens: 2000
- System prompt: one short paragraph explaining the format, coordinate frame (world frame, meters, 6D rotation, gripper 0=open/1=closed), and that the response must be a single JSON array of waypoint arrays with no prose.
- User message: the N demos formatted as `Scene: {...}\nActions: [[...], [...]]\n\n` blocks, followed by `Scene: {...}\nActions: ` for the query.

**Prompt format (use literally):**
```
Scene:
  object_0: [x, y, z, r1, r2, r3, r4, r5, r6]
  object_1: [x, y, z, r1, r2, r3, r4, r5, r6]
Actions:
  [[x, y, z, r1, r2, r3, r4, r5, r6, g], [...], ...]
```
Round all numbers to 3 decimal places.

**Parsing:**
- Extract the JSON array from the response. Accept responses with or without markdown code fences.
- Validate: must be a list of lists, each inner list length 10, all floats.
- Convert 6D rotation back to a proper rotation matrix (Gram-Schmidt on the two 3-vectors, then cross product for the third column), then to whatever format `EndEffectorPoseViaPlanning` expects (check RLBench source — likely quaternion xyzw).
- If parsing fails: log the failure, mark the trial as failed, do not retry the API call.

**Execution:**
- Execute waypoints sequentially via `task.step(action)`.
- If any step raises `InvalidActionError` or similar: log it, mark trial failed, break out of the loop.
- After all waypoints execute (or on failure), check `task._task.success()` or whatever RLBench exposes as the terminal success signal.

**Deliverable for Stage 1:** the script runs end-to-end on one trial and prints a clear success/failure with reasons. Do not proceed to Stage 2 until I confirm this works.

### Stage 2: Multi-trial evaluation on one task

Extend to N trials with proper seeding:

- Add a `--n_trials` flag (default 25)
- Add a `--n_demos` flag (default 5)
- Use a fixed list of test seeds so different `n_demos` configurations are evaluated on the same test scenes
- For each trial: resample the N context demos from a separate demo pool (not the test scene)
- **LLM response caching**: hash the prompt string (SHA256), cache responses to `cache/{hash}.json`. Always check cache before calling the API. This is mandatory — we will re-run experiments and cannot afford to repay for identical prompts.
- Output a CSV: `task, n_demos, trial_id, seed, success, n_waypoints_executed, parse_error, execution_error`

Deliverable: `python kat_eval.py --task reach_target --n_demos 5 --n_trials 25` runs and produces a CSV.

### Stage 3: Task and n_demos sweep

- Tasks: `reach_target`, `push_button`, `pick_up_cup`, `take_lid_off_saucepan`, `stack_blocks`
- n_demos: `[1, 2, 5, 10, 20]`
- Full sweep = 5 tasks × 5 n_demos × 25 trials = 625 evaluations
- Add a `run_sweep.py` that iterates and appends to a single results CSV
- **Before running the full sweep**, run a 1-trial smoke pass over the entire sweep grid to catch prompt-length issues, task-specific API quirks, and cache behavior. Report prompt token counts per (task, n_demos) combination so we can spot-check for context-window risk.

Deliverable: full results CSV.

### Stage 4: Plotting

- `plot_results.py` reads the CSV
- Produces: one figure, x-axis = n_demos (log scale), y-axis = success rate, one line per task, Wilson 95% confidence intervals as error bars
- Save as `results.png` and `results.pdf`

## Coordinate frame note

For v1, use world frame for everything. Do NOT do the "relative-to-keypoint" transformation that the original KAT paper uses. We'll add that as a separate ablation after the baseline numbers are in.

## Things to print loudly and ask me about if uncertain

- The exact structure of `obs.task_low_dim_state` for each task — print it on first demo collection and stop to confirm before building the prompt format.
- Whether `EndEffectorPoseViaPlanning` expects quaternion in xyzw or wxyz order — check RLBench source, don't guess.
- Whether gripper action is `[0, 1]` continuous or discrete — check and hardcode correctly.
- Any task where `task_low_dim_state` is empty or doesn't contain the relevant objects — flag it, don't silently use zeros.

## Budget

Total implementation: target 2–3 sessions. If Stage 1 takes more than one session because of RLBench API friction, stop and flag it — we'll debug manually before continuing.

## Repository layout

```
kat_baseline/
  kat_smoke.py        # Stage 1
  kat_eval.py         # Stage 2
  run_sweep.py        # Stage 3
  plot_results.py     # Stage 4
  cache/              # LLM response cache (gitignored)
  results/            # CSVs and plots
  README.md           # how to run each stage
```

## First action

Start Stage 1. Before writing any code, print a short plan: which RLBench imports you'll use, how you'll access `task_low_dim_state`, and the exact Anthropic SDK call you'll make. Wait for my confirmation, then implement.
