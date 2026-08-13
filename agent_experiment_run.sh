#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
readonly GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --absolute-git-dir)"
readonly LOCAL_EXCLUDE="$GIT_DIR/info/exclude"
readonly POLICY_FILE="$REPO_ROOT/EXPERIMENT.md"
readonly LOCAL_HISTORY="$REPO_ROOT/.experiment-history.md"
readonly RUN_ROOT="$REPO_ROOT/.agent-runs"
readonly RUN_TIMEOUT="${AGENT_EXPERIMENT_TIMEOUT:-8h}"
readonly LOCK_FILE="${AGENT_EXPERIMENT_LOCK_FILE:-/tmp/agent-experiment-$(basename "$REPO_ROOT").lock}"

cd "$REPO_ROOT"

max_runs="${AGENT_EXPERIMENT_MAX_RUNS:-10}"
cli="${AGENT_EXPERIMENT_CLI:-codex}"
model="${AGENT_EXPERIMENT_MODEL:-}"
thinking_level="${AGENT_EXPERIMENT_THINKING_LEVEL:-}"
force_latest_best=false
run_count_set=false

fail() {
    echo "error: $*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [RUN_COUNT]

  --force-latest-best  Ignore whether the latest accepted metric belongs to
                       HEAD and force the highest Rank-1 in EXPERIMENT.md to
                       be used as the baseline without retraining it.
  --cli CLI            Agent CLI: codex or agy (default: codex).
  --model MODEL        Model (default: gpt-5.6-sol for codex,
                       gemini-3.6-flash for agy).
  --thinking-level LEVEL
                       Reasoning effort (default: low for codex, medium for agy).
  -h, --help           Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --force-latest-best)
            force_latest_best=true
            ;;
        --cli)
            (($# >= 2)) || fail "--cli requires a value"
            cli="$2"
            shift
            ;;
        --model)
            (($# >= 2)) || fail "--model requires a value"
            model="$2"
            shift
            ;;
        --thinking-level)
            (($# >= 2)) || fail "--thinking-level requires a value"
            thinking_level="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            (($# <= 1)) || fail "only one run count may be specified"
            if (($# == 1)); then
                max_runs="$1"
                run_count_set=true
            fi
            break
            ;;
        -* )
            fail "unknown option: $1"
            ;;
        *)
            [[ "$run_count_set" == false ]] || fail "only one run count may be specified"
            max_runs="$1"
            run_count_set=true
            ;;
    esac
    shift
done

case "$cli" in
    codex)
        model="${model:-gpt-5.6-sol}"
        thinking_level="${thinking_level:-low}"
        ;;
    agy)
        model="${model:-gemini-3.6-flash}"
        thinking_level="${thinking_level:-medium}"
        ;;
    *) fail "cli must be one of: codex, agy" ;;
esac

readonly MAX_RUNS="$max_runs"
readonly AGENT_CLI="$cli"
readonly AGENT_MODEL="$model"
readonly THINKING_LEVEL="$thinking_level"
readonly FORCE_LATEST_BEST="$force_latest_best"

active_run_id=""
active_event_log=""
active_final_log=""
active_process_group=""

handle_interrupt() {
    local signal_name="$1"
    local exit_code="$2"
    local repository_status=""

    # Prevent a second signal from re-entering this handler while it reports
    # the state left by the interrupted agent process.
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

fixed_training_iterations() {
    awk '
        /^[[:space:]]+iterations:[[:space:]]*[0-9]+[[:space:]]*$/ {
            print $2
            exit
        }
    ' "$POLICY_FILE"
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
[[ -n "$AGENT_MODEL" ]] || fail "model must not be empty"
if [[ "$AGENT_CLI" == agy ]]; then
    case "$THINKING_LEVEL" in
        low|medium|high) ;;
        *) fail "agy thinking level must be one of: low, medium, high" ;;
    esac
else
    case "$THINKING_LEVEL" in
        none|low|medium|high|xhigh|max) ;;
        *) fail "codex thinking level must be one of: none, low, medium, high, xhigh, max" ;;
    esac
fi
[[ -f "$POLICY_FILE" ]] || fail "missing $POLICY_FILE"
expected_iterations="$(fixed_training_iterations)"
is_positive_integer "$expected_iterations" \
    || fail "EXPERIMENT.md has no valid fixed training iterations value"
readonly EXPECTED_ITERATIONS="$expected_iterations"
command -v "$AGENT_CLI" >/dev/null || fail "$AGENT_CLI CLI is not installed"
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
ensure_local_exclude '/.agent-runs/'

mkdir -p "$RUN_ROOT"
if [[ ! -f "$LOCAL_HISTORY" ]]; then
    printf '# Local experiment history\n\n' >"$LOCAL_HISTORY"
    printf 'This file is local-only and must never be committed.\n' \
        >>"$LOCAL_HISTORY"
fi

exec 9>"$LOCK_FILE"
flock -n 9 || fail "another experiment loop is already running"

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]]; then
    fail "the repository must be completely clean before automation starts"
fi

echo "Experiment configuration:"
echo "  cli: $AGENT_CLI"
echo "  model: $AGENT_MODEL"
echo "  thinking level: $THINKING_LEVEL"
echo "  runs: $MAX_RUNS"
echo "  required training iterations: $EXPECTED_ITERATIONS"

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
    if [[ "$FORCE_LATEST_BEST" == false ]]; then
        verify_recorded_baseline_is_current_head "$latest_rank1_line" "$parent_commit"
    fi
    active_run_id="$run_id"
    active_event_log="$event_log"
    active_final_log="$final_log"

    echo "[$run/$MAX_RUNS] starting experiment $run_id at ${parent_commit:0:12} (cli=$AGENT_CLI, model=$AGENT_MODEL, thinking=$THINKING_LEVEL)"

    agent_args=(
        exec
        --cd "$REPO_ROOT"
        --approve-for-me
        --model "$AGENT_MODEL"
        --config "model_reasoning_effort=\"$THINKING_LEVEL\""
        --json
        --output-last-message "$final_log"
    )

    {
        cat "$POLICY_FILE"
        printf '\n## Current run\n\n'
        printf -- '- Run ID: `%s`\n' "$run_id"
        printf -- '- Parent commit: `%s`\n' "$parent_commit"
        printf -- '- Local history: `%s`\n' "$LOCAL_HISTORY"
        printf -- '- Maximum wall time: `%s`\n\n' "$RUN_TIMEOUT"
        printf -- '- Agent CLI: `%s`\n' "$AGENT_CLI"
        printf -- '- Agent model: `%s`\n' "$AGENT_MODEL"
        printf -- '- Thinking level: `%s`\n\n' "$THINKING_LEVEL"
        printf -- '- Required completed training iteration: `%s`\n\n' \
            "$EXPECTED_ITERATIONS"
        printf -- '- Historical-best accepted Rank-1: `%s`\n' "$best_rank1_before"
        printf -- '- Minimum candidate Rank-1 for acceptance: historical best + `0.001`\n\n'
        if [[ "$FORCE_LATEST_BEST" == true ]]; then
            printf 'The runner was started with `--force-latest-best`. Ignore '
            printf 'whether HEAD or any commits after the latest accepted entry are '
            printf 'documented in EXPERIMENT.md. Do not ask to document or measure '
            printf 'the current HEAD baseline. Force the historical-best accepted '
            printf 'Rank-1 above to be the current and acceptance baseline. '
        else
            printf 'The latest accepted metric entry in EXPERIMENT.md was verified '
            printf 'to come from the current HEAD, and the working tree was clean. '
            printf 'Treat the historical-best accepted Rank-1 as the current baseline. '
        fi
        printf 'For this run, this selected recorded baseline serves as both the '
        printf 'direct-parent and historical acceptance baseline and supersedes the '
        printf 'fresh-parent-training requirement in workflow rule 7. '
        printf 'Do not retrain the baseline. Modify exactly one candidate variable '
        printf 'and run only the candidate training/evaluation.\n\n'
        printf 'For every train.py training command in this run, add '
        printf '`--checkpoint-logs-only` so progress output is emitted only at '
        printf 'checkpoint intervals. Do not omit this option and do not use `tee`.\n\n'
        printf 'A training command that is still `in_progress` is running, not '
        printf 'finished, failed, or inconclusive. Run training in the foreground. '
        printf 'If command execution yields while it is still running or returns a '
        printf 'session ID, keep waiting or polling that exact command session until '
        printf 'it produces a completed event and exit code. While it is running, do '
        printf 'not launch another command, inspect checkpoints, evaluate results, '
        printf 'edit history, roll back the candidate, or produce a final response. '
        printf 'Do not abandon or terminate training merely because no output appears '
        printf 'between checkpoint intervals.\n\n'
        printf 'After the training command completes successfully, identify the '
        printf 'checkpoint directory created by this candidate run. Before any final '
        printf 'evaluation or acceptance/rejection decision, verify all of the '
        printf 'following: (1) the completed command exit code is zero; (2) that '
        printf 'directory contains `last_%s_iter.h5`; (3) it contains ' \
            "$EXPECTED_ITERATIONS"
        printf '`best_%s_iter_rank1_*.h5`; and (4) its `validation_log.csv` contains ' \
            "$EXPECTED_ITERATIONS"
        printf 'a row whose iteration is exactly `%s`. ' "$EXPECTED_ITERATIONS"
        printf 'Read Rank-1 only from that exact row. If any artifact is absent, first '
        printf 'check whether training is still running and continue waiting if so. '
        printf 'Only after the command has definitively completed with a nonzero exit '
        printf 'code or was externally interrupted may the run be recorded as blocked '
        printf 'or inconclusive; never perform final metric evaluation without all '
        printf 'four completion checks.\n\n'
        printf 'Perform exactly one experiment under these rules. '
        printf 'Work directly in this repository. Do not ask for input. '
        printf 'Request escalated execution outside the workspace sandbox for '
        printf 'GPU checks and training so host GPU devices are available.\n'
    } >"$prompt_file"

    set +e
    if [[ "$AGENT_CLI" == codex ]]; then
        setsid timeout --signal=TERM "$RUN_TIMEOUT" \
            codex "${agent_args[@]}" - <"$prompt_file" >"$event_log" &
    else
        # Unlike `codex exec -`, `agy --print` takes the prompt as the option's
        # value rather than consuming stdin. Keep the option and value in one
        # argument so the next flag cannot be mistaken for the prompt.
        agent_prompt="$(<"$prompt_file")"
        [[ -n "$agent_prompt" ]] || fail "generated agent prompt is empty"
        setsid timeout --signal=TERM "$RUN_TIMEOUT" \
            agy --dangerously-skip-permissions \
                --model "$AGENT_MODEL" --effort "$THINKING_LEVEL" \
                --print-timeout "$RUN_TIMEOUT" \
                --output-format text --log-file "$event_log" \
                --add-dir "$REPO_ROOT" \
                "--print=$agent_prompt" >"$final_log" &
    fi
    active_process_group=$!
    wait "$active_process_group"
    agent_status=$?
    active_process_group=""
    set -e

    current_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    repository_status="$(git -C "$REPO_ROOT" status --porcelain \
        --untracked-files=normal)"

    if (( agent_status != 0 )); then
        echo "$AGENT_CLI exited with status $agent_status; see $event_log" >&2
        if [[ -n "$repository_status" ]]; then
            fail "the interrupted run left changes; no automatic destructive cleanup was attempted"
        fi
    elif [[ -n "$repository_status" ]]; then
        fail "$AGENT_CLI left uncommitted changes; inspect them manually (see $final_log)"
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
                | grep -Eq '(^|/)(\.experiment-history\.md|\.agent-runs)(/|$)'; then
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
echo "Agent run logs: $RUN_ROOT"
