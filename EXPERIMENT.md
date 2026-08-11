# Autonomous experiment policy

This file is the source of truth for autonomous Codex experiments. It is
designed to be copied to other machine-learning repositories. Edit only the
project-specific configuration section when adapting it to another project.

## Project-specific configuration

- Primary metric: `Rank-1`
- Direction: higher is better
- Minimum accepted improvement: `0.001` absolute, equivalent to `0.1` percentage
  point when the metric is represented on a 0-to-1 scale
- Baseline model: `compact_cnn`
- Training command: `PYENV_VERSION=tf python train.py --cfg cfg/cfg.yaml`
- Result source: the final validation line and `validation_log.csv`
- Fixed training settings:

  ```yaml
  iterations: 30000
  checkpoint_interval: 30000
  ```

- Resource priority: improve the currently committed model without increasing
  inference operations or parameters. Increasing model computation is the last
  resort and requires evidence that lower-cost alternatives were exhausted.

For another repository, replace the metric with its primary metric such as
accuracy, mAP, F1, or loss; specify whether higher or lower is better; provide
one deterministic training command and one machine-readable result source.
Keep the minimum improvement equivalent to `0.1` percentage point unless the
project explicitly requires a different threshold.

## Required files

- `EXPERIMENT.md`: committed rules and accepted experiment history.
- `.experiment-history.md`: local-only history of attempted, rejected, blocked,
  and inconclusive experiments. It must be ignored by Git and never committed.
- `.codex-runs/`: local-only Codex execution logs. It must never be committed.

## Mandatory workflow

1. Before choosing a hypothesis, read this entire file, then read the entire
   local `.experiment-history.md` if it exists, inspect recent Git history, the
   current configuration, training code, evaluation code, and existing logs.
2. Propose exactly one falsifiable hypothesis. Run it only when it is relevant
   to the current committed baseline and has not already failed under an
   equivalent setup. Do not repeat a rejected experiment unless the new run
   explicitly addresses a documented flaw in the previous protocol.
3. Record the parent commit, hypothesis, single changed variable, exact command,
   configuration, and expected outcome in `.experiment-history.md` before
   modifying tracked files.
4. Modify the current repository directly. Do not create a worktree, clone,
   alternate checkout, or separate source directory for the experiment.
5. Experiments may change configuration, data sampling, augmentation, model
   structure, loss functions, optimizers, schedules, training logic, or
   evaluation logic. Keep each experiment focused on one hypothesis.
6. Run proportional syntax checks, unit/smoke tests, training, and evaluation.
   Never invent or estimate metrics. Use the configured primary metric from the
   designated result source.
   GPU validation and training commands require host GPU device access. Run
   `nvidia-smi -L`, TensorFlow GPU discovery, and the configured training command
   with escalated execution outside the workspace sandbox. Do not treat
   `PYENV_VERSION=tf` or `pyenv activate tf` as a substitute for GPU device
   access. If escalated GPU access cannot be granted, record the run as blocked
   and end that run without attempting another hypothesis.
7. Compare against the direct committed parent using the same dataset split,
   seed policy, training budget, evaluation protocol, and hardware-relevant
   settings. Use the fresh parent run to measure the experimental effect, but
   use the highest unrounded accepted Rank-1 in this file as the acceptance
   baseline. A candidate must exceed both baselines by the configured minimum
   improvement. Never lower or replace the historical-best baseline when a
   fresh parent run scores lower. Changing the evaluation protocol requires a
   separate experiment and must not be presented as a model-quality
   improvement.
8. Append the outcome and metrics to `.experiment-history.md` for every run.
9. If the primary metric improves by at least the configured threshold and all
   important guardrail metrics remain acceptable, append an entry to the
   accepted history at the bottom of this file and create one Git commit that
   includes the code/config changes and that accepted-history entry.
10. If the experiment fails, is inconclusive, or is blocked, do not commit its
    source/config changes. Manually reverse only the changes made by the current
    experiment and leave the tracked working tree exactly as it was before the
    run. Preserve the failed result only in `.experiment-history.md`.

## Git and filesystem prohibitions

The following commands and behaviors are forbidden during autonomous runs:

- `git reset` in every form
- `git rebase` in every form
- `git merge` in every form
- `git checkout` or `git switch`
- `git clean`
- `git restore`
- `rm`, `rmdir`, `unlink`, or other destructive shell deletion commands
- force push, ordinary push, branch deletion, history rewriting, commit amend,
  stash mutation, or editing another worktree
- deleting or overwriting datasets, checkpoints, user files, or unrelated
  changes

Use read-only Git commands to inspect history and diffs. Reverse an uncommitted
failed edit by applying a precise inverse patch or editing the affected lines
back to their pre-experiment content. Put disposable outputs only in ignored
checkpoint/log directories.

If a previously accepted commit must be undone, use `git revert <commit>` so
the history remains explicit. Do not rewrite or erase the earlier commit. A
revert is a maintenance action and must not be described as a successful model
experiment.

## Commit acceptance rules

- Never commit a candidate unless it improves by at least `0.001` absolute over
  both the fresh direct-parent result and the highest previously accepted
  unrounded Rank-1 in this file. For a lower-is-better metric, require the
  equivalent decrease against both baselines.
- Treat accepted metric history as monotonic: a newly accepted Rank-1 must be
  greater than the previous historical best by at least `0.001`. A lower fresh
  baseline caused by training variance does not lower this threshold.
- Compare unrounded metric values when available. A displayed rounding change
  alone is not sufficient evidence.
- One commit represents one successful hypothesis.
- Do not commit raw checkpoints, datasets, generated models, caches, raw Codex
  event logs, or `.experiment-history.md`.
- Include the previous historical-best metric and the new candidate metric in
  the commit message.
- Do not modify the policy sections of this file during an experiment. Only
  append a successful result to `Accepted experiment history`.
- Leave the repository clean after every successful commit or failed rollback,
  excluding ignored local experiment artifacts.

Recommended commit message:

```text
experiment: <short hypothesis> (<metric> <before> -> <after>)
```

## Hypothesis selection guidance

Prefer changes that improve the committed model at equal or lower inference
cost. A suggested order is:

1. Correctness bugs or train/evaluation mismatches.
2. Loss, sampling, optimizer, schedule, and regularization changes.
3. Data augmentation and input preprocessing changes.
4. Training-only heads or techniques that do not affect exported inference.
5. Rearranging existing computation or preserving useful spatial information.
6. Width, depth, resolution, parameters, or operation-count increases only as
   a final option.

For this repository, use `compact_cnn` unless an accepted experiment provides
evidence for changing the baseline. Do not spend experiments making MobileNet
or EfficientNet variants deeper merely because they are conventional names.

## Local failed-history format

Append each attempt to `.experiment-history.md` using this format:

```markdown
## <timestamp> — <hypothesis>

- Parent commit:
- Status: planned | accepted | rejected | inconclusive | blocked
- Changed variable:
- Command:
- Configuration:
- Baseline metric:
- Candidate metric:
- Delta:
- Guardrail metrics:
- Runtime:
- Conclusion:
- Reason not to repeat:
```

## Accepted experiment history

Successful experiments are appended here and committed with their code changes.

## 2026-08-11 — Reduce random-crop padding to preserve identity cues

- Parent commit: `a5630157b32484748350453a0c0e5f63a0679ace`
- Change: Reduced `random_crop_padding` from `10` to `4` for `compact_cnn`.
- Rank-1: `0.6440023752969121` → `0.6472684085510689` (`+0.0032660332541568`).
- Guardrails: Validation loss improved from `-0.42417898774147034` to `-0.4465788006782532`; AUC changed from `0.9864075751` to `0.9846521522`; EER from `0.05159000000000001` to `0.057984999999999995`; TAR@FAR1% from `0.795` to `0.77654`.
- Inference cost: Unchanged parameters and operations; augmentation is training-only.

## 2026-08-11 — Reduce random erasing to retain discriminative identity regions

- Parent commit: `96bf5ed658b67eab4bddd8c44f16c6c6f705cdc0`
- Change: Reduced `random_erasing_probability` from `0.5` to `0.25` for `compact_cnn`.
- Rank-1: `0.6442992874109263` → `0.6505344418052257` (`+0.0062351543942994`).
- Guardrails: Validation loss improved from `-0.4414414167404175` to `-0.459439218044281`; AUC changed from `0.9850128622` to `0.98461670875`; EER from `0.05689999999999998` to `0.057819999999999996`; TAR@FAR1% from `0.77777` to `0.77245`.
- Inference cost: Unchanged parameters and operations; augmentation is training-only.

## 2026-08-11 — Reduce color jitter to preserve clothing-color cues

- Parent commit: `ee5fb0e3fdd0a060aa880a908fdd524fd0f0631b`
- Change: Reduced `color_jitter` from `0.15` to `0.05` for `compact_cnn`.
- Rank-1: `0.6241092636579573` → `0.6582541567695962` (`+0.0341448931116389`).
- Guardrails: Validation loss improved from `-0.45757028460502625` to `-0.46080005168914795`; AUC improved from `0.9835561982500001` to `0.9847315404000001`; EER improved from `0.059975000000000014` to `0.055914999999999986`; TAR@FAR1% improved from `0.76726` to `0.77935`.
- Inference cost: Unchanged parameters and operations; augmentation is training-only.

## 2026-08-11 — Reduce L2 regularization to avoid underfitting the compact backbone

- Parent commit: `f370335fa63a71293678c2a5b952be0e444d8fbf`
- Change: Reduced `l2` from `0.0005` to `0.0002` for `compact_cnn`.
- Rank-1: `0.6514251781472684` → `0.6849762470308789` (`+0.0335510688836105`).
- Guardrails: Validation loss improved from `-0.475689172744751` to `-0.5171219706535339`; AUC changed from `0.9852943779499999` to `0.9850397041500001`; EER from `0.054805000000000006` to `0.05647999999999999`; TAR@FAR1% improved from `0.79017` to `0.79474`.
- Inference cost: Unchanged parameters and operations; L2 regularization is training-only.

## 2026-08-11 — Increase the negative-distance cap to sustain hard-negative separation

- Parent commit: `22ae668af23ce8dc84b6c724abc2ea98f9cf5e63`.
- Change: Increased `maximum_negative_distance` from `1.0` to `1.2` for `compact_cnn`.
- Rank-1: `0.6748812351543944` → `0.6763657957244655` (`+0.0014845605700711`).
- Guardrails: Validation loss improved from `-0.5147063136100769` to `-0.566238284111023`; AUC improved from `0.9849461981000001` to `0.98752020945`; EER improved from `0.05600000000000002` to `0.050175000000000025`; TAR@FAR1% improved from `0.78499` to `0.81142`.
- Inference cost: Unchanged parameters and operations; the negative-distance cap is training-only.

## 2026-08-11 — Lower the cosine final learning-rate ratio for finer late refinement

- Parent commit: `d158989cac20df0dc70662f5eca90e8d7aede17b`.
- Change: Lowered `lrf` from `0.05` to `0.01` for `compact_cnn`.
- Rank-1: `0.6760688836104513` → `0.6882422802850356` (`+0.0121733966745844`); the candidate also exceeded the previous historical best `0.6849762470308789` by `0.0032660332541568`.
- Guardrails: Validation loss improved from `-0.5632808208465576` to `-0.5689043998718262`; AUC changed from `0.98721888475` to `0.98689473365`; EER from `0.051300000000000005` to `0.051985000000000024`; TAR@FAR1% from `0.809` to `0.80639`.
- Inference cost: Unchanged parameters and operations; the learning-rate schedule is training-only.
