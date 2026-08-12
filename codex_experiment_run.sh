#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
readonly GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --absolute-git-dir)"
readonly LOCAL_EXCLUDE="$GIT_DIR/info/exclude"
readonly POLICY_FILE="$REPO_ROOT/EXPERIMENT.md"
readonly LOCAL_HISTORY="$REPO_ROOT/.experiment-history.md"
readonly RUN_ROOT="$REPO_ROOT/.codex-runs"
readonly MAX_RUNS="${1:-${CODEX_EXPERIMENT_MAX_RUNS:-10}}"
readonly RUN_TIMEOUT="${CODEX_EXPERIMENT_TIMEOUT:-8h}"
readonly LOCK_FILE="${CODEX_EXPERIMENT_LOCK_FILE:-/tmp/codex-experiment-$(basename "$REPO_ROOT").lock}"

fail() {
    echo "error: $*" >&2
    exit 1
}

active_run_id=""
active_event_log=""
active_final_log=""
active_process_group=""

handle_interrupt() {
    local signal_name="$1"
    local exit_code="$2"
    local repository_status=""

    # Prevent a second signal from re-entering this handler while it reports
    # the state left by the interrupted Codex process.
    trap - INT TERM HUP

    echo >&2
    echo "experiment loop interrupted by $signal_name" >&2
    if [[ -n "$active_process_group" ]]; then
        kill -TERM -- "-$active_process_group" 2>/dev/null || true
        for _ in {1..20}; do
            if ! kill -0 "$active_process_group" 2>/dev/null; then
                break
            fi
            sleep 0.1
        done
        if kill -0 "$active_process_group" 2>/dev/null; then
            kill -KILL -- "-$active_process_group" 2>/dev/null || true
        fi
        wait "$active_process_group" 2>/dev/null || true
        active_process_group=""
    fi

    if [[ -n "$active_run_id" ]]; then
        echo "interrupted experiment: $active_run_id" >&2
        echo "partial event log: $active_event_log" >&2
        if [[ -f "$active_final_log" ]]; then
            echo "final message: $active_final_log" >&2
        else
            echo "final message was not produced" >&2
        fi
    fi

    repository_status="$(git -C "$REPO_ROOT" status --porcelain \
        --untracked-files=normal 2>/dev/null || true)"
    if [[ -n "$repository_status" ]]; then
        echo "the interrupted run left repository changes:" >&2
        printf '%s\n' "$repository_status" >&2
        echo "no automatic destructive cleanup was attempted; inspect the changes manually" >&2
    else
        echo "the tracked working tree is clean" >&2
    fi

    exit "$exit_code"
}

trap 'handle_interrupt SIGINT 130' INT
trap 'handle_interrupt SIGTERM 143' TERM
trap 'handle_interrupt SIGHUP 129' HUP

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

highest_accepted_rank1() {
    awk -F'`' '
        /^- Rank-1: / && ($4 + 0) > best { best = $4 + 0; value = $4 }
        END { if (value != "") print value }
    ' "$POLICY_FILE"
}

latest_accepted_rank1_line() {
    awk '/^- Rank-1: / { line = NR } END { if (line) print line }' \
        "$POLICY_FILE"
}

verify_recorded_baseline_is_current_head() {
    local metric_line="$1"
    local head_commit="$2"
    local metric_commit=""

    metric_commit="$(git -C "$REPO_ROOT" blame --porcelain \
        -L "$metric_line,$metric_line" -- EXPERIMENT.md | awk 'NR == 1 { print $1 }')"
    [[ "$metric_commit" == "$head_commit" ]] || fail \
        "latest accepted Rank-1 entry is not from current HEAD; commit and document the current baseline before skipping baseline training"
}

meets_minimum_improvement() {
    awk -v candidate="$1" -v baseline="$2" \
        'BEGIN { exit !((candidate - baseline) >= 0.001) }'
}

is_positive_integer "$MAX_RUNS" || fail "run count must be a positive integer"
[[ -f "$POLICY_FILE" ]] || fail "missing $POLICY_FILE"
command -v codex >/dev/null || fail "codex CLI is not installed"
command -v flock >/dev/null || fail "flock is not installed"
command -v timeout >/dev/null || fail "timeout is not installed"
command -v setsid >/dev/null || fail "setsid is not installed"

ensure_local_exclude() {
    local pattern="$1"
    if ! grep -Fxq "$pattern" "$LOCAL_EXCLUDE" 2>/dev/null; then
        printf '%s\n' "$pattern" >>"$LOCAL_EXCLUDE"
    fi
}

ensure_local_exclude '/.experiment-history.md'
ensure_local_exclude '/.codex-runs/'

mkdir -p "$RUN_ROOT"
if [[ ! -f "$LOCAL_HISTORY" ]]; then
    printf '# Local experiment history\n\n' >"$LOCAL_HISTORY"
    printf 'This file is local-only and must never be committed.\n' \
        >>"$LOCAL_HISTORY"
fi

exec 9>"$LOCK_FILE"
flock -n 9 || fail "another Codex experiment loop is already running"

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]]; then
    fail "the repository must be completely clean before automation starts"
fi

for ((run = 1; run <= MAX_RUNS; run++)); do
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]]; then
        fail "tracked or untracked changes remain from the previous run; inspect them manually"
    fi

    parent_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    timestamp="$(date +%Y%m%d-%H%M%S)"
    run_id="${timestamp}-$(printf '%03d' "$run")"
    event_log="$RUN_ROOT/$run_id.jsonl"
    final_log="$RUN_ROOT/$run_id.final.txt"
    prompt_file="$RUN_ROOT/$run_id.prompt.md"
    best_rank1_before="$(highest_accepted_rank1)"
    latest_rank1_line="$(latest_accepted_rank1_line)"
    [[ -n "$best_rank1_before" ]] \
        || fail "EXPERIMENT.md has no accepted Rank-1 baseline"
    [[ -n "$latest_rank1_line" ]] \
        || fail "EXPERIMENT.md has no latest accepted Rank-1 entry"
    verify_recorded_baseline_is_current_head "$latest_rank1_line" "$parent_commit"
    active_run_id="$run_id"
    active_event_log="$event_log"
    active_final_log="$final_log"

    echo "[$run/$MAX_RUNS] starting experiment $run_id at ${parent_commit:0:12}"

    codex_args=(
        exec
        --cd "$REPO_ROOT"
        --approve-for-me
        --json
        --output-last-message "$final_log"
    )
    if [[ -n "${CODEX_EXPERIMENT_MODEL:-}" ]]; then
        codex_args+=(--model "$CODEX_EXPERIMENT_MODEL")
    fi

    {
        cat "$POLICY_FILE"
        printf '\n## Current run\n\n'
        printf -- '- Run ID: `%s`\n' "$run_id"
        printf -- '- Parent commit: `%s`\n' "$parent_commit"
        printf -- '- Local history: `%s`\n' "$LOCAL_HISTORY"
        printf -- '- Maximum wall time: `%s`\n\n' "$RUN_TIMEOUT"
        printf -- '- Historical-best accepted Rank-1: `%s`\n' "$best_rank1_before"
        printf -- '- Minimum candidate Rank-1 for acceptance: historical best + `0.001`\n\n'
        printf 'The latest accepted metric entry in EXPERIMENT.md was verified '
        printf 'to come from the current HEAD, and the working tree was clean. '
        printf 'Treat the historical-best accepted Rank-1 as the current baseline. '
        printf 'For this run, this verified recorded baseline serves as both the '
        printf 'direct-parent and historical acceptance baseline and supersedes the '
        printf 'fresh-parent-training requirement in workflow rule 7. '
        printf 'Do not retrain the unchanged baseline. Inspect the current HEAD diff '
        printf 'and accepted entry for consistency, then modify exactly one candidate '
        printf 'variable and run only the candidate training/evaluation. If the HEAD '
        printf 'contains a relevant change not documented by the latest accepted entry, '
        printf 'record the run as blocked and do not train.\n\n'
        printf 'For every train.py training command in this run, add '
        printf '`--checkpoint-logs-only` so progress output is emitted only at '
        printf 'checkpoint intervals. Do not omit this option and do not use `tee`.\n\n'
        printf 'Perform exactly one experiment under these rules. '
        printf 'Work directly in this repository. Do not ask for input. '
        printf 'Request escalated execution outside the workspace sandbox for '
        printf 'GPU checks and training so host GPU devices are available.\n'
    } >"$prompt_file"

    set +e
    setsid timeout --signal=TERM "$RUN_TIMEOUT" \
        codex "${codex_args[@]}" - <"$prompt_file" >"$event_log" &
    active_process_group=$!
    wait "$active_process_group"
    codex_status=$?
    active_process_group=""
    set -e

    current_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    repository_status="$(git -C "$REPO_ROOT" status --porcelain \
        --untracked-files=normal)"

    if (( codex_status != 0 )); then
        echo "Codex exited with status $codex_status; see $event_log" >&2
        if [[ -n "$repository_status" ]]; then
            fail "the interrupted run left changes; no automatic destructive cleanup was attempted"
        fi
    elif [[ -n "$repository_status" ]]; then
        fail "Codex left uncommitted changes; inspect them manually (see $final_log)"
    elif [[ "$parent_commit" == "$current_commit" ]]; then
        echo "experiment produced no accepted commit; see $final_log"
    else
        commit_count="$(git -C "$REPO_ROOT" rev-list --count \
            "$parent_commit..$current_commit")"
        [[ "$commit_count" == "1" ]] \
            || fail "one experiment must create exactly one commit"

        if ! git -C "$REPO_ROOT" diff --name-only \
                "$parent_commit..$current_commit" \
                | grep -Fxq 'EXPERIMENT.md'; then
            fail "accepted commit does not include its EXPERIMENT.md history entry"
        fi
        if git -C "$REPO_ROOT" diff --name-only \
                "$parent_commit..$current_commit" \
                | grep -Eq '(^|/)(\.experiment-history\.md|\.codex-runs)(/|$)'; then
            fail "accepted commit contains a local-only experiment artifact"
        fi

        best_rank1_after="$(highest_accepted_rank1)"
        if ! meets_minimum_improvement "$best_rank1_after" "$best_rank1_before"; then
            fail "accepted commit did not improve historical-best Rank-1 by at least 0.001 ($best_rank1_before -> $best_rank1_after)"
        fi

        echo "accepted experiment commit:"
        git -C "$REPO_ROOT" log -1 --oneline
    fi

    active_run_id=""
    active_event_log=""
    active_final_log=""
done

echo "repository: $REPO_ROOT"
echo "local experiment history: $LOCAL_HISTORY"
echo "Codex run logs: $RUN_ROOT"
