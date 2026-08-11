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
7. Compare against the direct committed parent using the same dataset split,
   seed policy, training budget, evaluation protocol, and hardware-relevant
   settings. Changing the evaluation protocol requires a separate experiment
   and must not be presented as a model-quality improvement.
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

- Never commit an improvement smaller than `0.001` absolute for a higher-is-
  better 0-to-1 metric. For a lower-is-better metric, require an equivalent
  decrease.
- Compare unrounded metric values when available. A displayed rounding change
  alone is not sufficient evidence.
- One commit represents one successful hypothesis.
- Do not commit raw checkpoints, datasets, generated models, caches, raw Codex
  event logs, or `.experiment-history.md`.
- Include the primary metric before and after in the commit message.
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
