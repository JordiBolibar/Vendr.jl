#!/usr/bin/env bash
# Launch one 02_lambda_sweep.jl process per multiplier, batched MAX_PARALLEL at a time, each
# writing only its own mult<value>/ subfolder -- then merges them into the campaign-level
# aggregate (summary.csv, comparison.pdf, lcurve.pdf) automatically once every multiplier
# succeeds.
#
# Usage:
#   ./02_lambda_sweep_parallel.sh <law> <gsf> <epochs> <A_name> <mult1,mult2,...> [max_parallel]
#
# Example:
#   ./02_lambda_sweep_parallel.sh weertman 1 30,5 m15C 3000,10000,30000,100000,300000
#
# max_parallel (default 2) caps how many multipliers train at once. Each process's peak
# memory use scales with grid resolution (gridScalingFactor) and grows with training, so how
# many can safely run together depends on the machine and the resolution requested -- running
# too many at once can exhaust memory and abort a process mid-solve (Julia's GC crashing
# outright rather than erroring cleanly) rather than merely slow things down. Lower this for a
# memory-constrained machine or a low gridScalingFactor (finer grid); raise it if resources
# allow. Batches, not a true concurrency-limited pool: macOS ships bash 3.2, which has no
# `wait -n`, so each batch fully finishes before the next one starts.
set -euo pipefail

LAW="${1:?law required, e.g. weertman}"
GSF="${2:?gsf required, e.g. 1}"
EPOCHS="${3:?epochs required, e.g. 30,5}"
A_NAME="${4:?A_name required, e.g. m15C}"
MULTS="${5:?comma-separated multipliers required, e.g. 3000,10000,30000}"
MAX_PARALLEL="${6:-2}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="$(mktemp -d /tmp/lambda_sweep_parallel.XXXXXX)"
echo "Logs: $LOG_DIR  (max $MAX_PARALLEL at a time)"

cd "$REPO_ROOT"
IFS=',' read -ra mult_array <<< "$MULTS"
fail=0

n=${#mult_array[@]}
for ((start = 0; start < n; start += MAX_PARALLEL)); do
    batch=("${mult_array[@]:start:MAX_PARALLEL}")
    pids=()
    for mult in "${batch[@]}"; do
        log="$LOG_DIR/mult${mult}.log"
        # Piped through tee/sed so output shows live on the terminal, prefixed per multiplier
        # so interleaved streams within a batch stay attributable. Wrapped in a subshell that
        # exits with Julia's own code (PIPESTATUS[0]): otherwise $! below would be sed's PID,
        # and `wait` would report sed's (near-always successful) exit status instead of Julia's.
        (
            julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/02_lambda_sweep.jl \
                "$LAW" "$GSF" "$EPOCHS" "$mult" "$A_NAME" 2>&1 \
                | tee "$log" | sed -u "s/^/[mult${mult}] /"
            exit "${PIPESTATUS[0]}"
        ) &
        pids+=($!)
        # Staggered, not simultaneous: all processes independently read glathida's cached data
        # through HDF5 on startup, and HDF5 isn't safe for uncoordinated concurrent access from
        # separate processes -- launching together risks an `h5open` failure in one of them.
        sleep 10
        echo "mult=$mult  pid=$! log=$log"
    done

    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "FAILED: mult=${batch[$i]} -- see $LOG_DIR/mult${batch[$i]}.log"
            fail=1
        fi
    done
done

echo "All multipliers finished. Logs in $LOG_DIR"

TAG="${LAW}_${A_NAME}_gsf${GSF}_ep${EPOCHS//,/-}"
if [ "$fail" -eq 0 ]; then
    echo "Merging into the campaign aggregate ($TAG)..."
    julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/02_lambda_sweep_merge.jl "$TAG"
else
    # Not automatic on failure: merging silently over a partial/broken sweep would produce
    # aggregate plots that look complete but aren't. Run it by hand once you've checked the
    # failed multiplier's log, if you want to merge whatever did succeed anyway.
    echo "Skipping merge: at least one multiplier failed. Merge by hand once you've checked why:"
    echo "  julia +1.11 --project=. inversions/03_real/32_invert_C_gridded/02_lambda_sweep_merge.jl $TAG"
fi

exit $fail
