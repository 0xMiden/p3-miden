#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="${MODE:-bench}"              # bench | trace | both
VARIANT="${VARIANT:-both}"         # fri | lifted | both
FOLDING="${FOLDING:-8}"
LOG_TRACE_HEIGHTS="${LOG_TRACE_HEIGHTS:-17}"
THREADS="${THREADS:-16}"
BLOWUP="${BLOWUP:-8}"
QUERIES="${QUERIES:-32}"
GRINDING="${GRINDING:-16}"
REBUILD="${REBUILD:-1}"

if (( BLOWUP & (BLOWUP - 1) )); then
  echo "BLOWUP must be a power of two (got ${BLOWUP})" >&2
  exit 1
fi

LOG_BLOWUP="$(python3 - <<PY
import math
print(int(math.log2(${BLOWUP})))
PY)"

usage() {
  cat <<'EOF'
Usage: run_p3_benchmarks.sh [options]

Options:
  --mode bench|trace|both        What to run (default: bench)
  --variant fri|lifted|both      Which variant(s) to run (default: both)
  --folding N                    Folding factor (2,4,8,16)
  --log-trace H[,H...]           Log2 trace heights (e.g. 17 or 16,17,18)
  --threads N                    Rayon threads (default: 16)
  --blowup N                     Blowup factor (default: 8)
  --queries N                    Number of queries (default: 32)
  --grinding N                   Grinding bits (default: 16)
  --no-rebuild                   Skip clean rebuild (default: rebuild)
  -h, --help                     Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --folding) FOLDING="$2"; shift 2 ;;
    --log-trace) LOG_TRACE_HEIGHTS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --blowup) BLOWUP="$2"; shift 2 ;;
    --queries) QUERIES="$2"; shift 2 ;;
    --grinding) GRINDING="$2"; shift 2 ;;
    --no-rebuild) REBUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

CLEANED=0
maybe_clean() {
  if [[ "${REBUILD}" == "1" && "${CLEANED}" == "0" ]]; then
    (
      cd "${ROOT}"
      case "${VARIANT}" in
        fri) cargo clean -p p3-miden-fri ;;
        lifted) cargo clean -p p3-miden-lifted-fri ;;
        both) cargo clean -p p3-miden-fri -p p3-miden-lifted-fri ;;
      esac
    )
    CLEANED=1
  fi
}

run_bench_fri() {
  echo "== p3-miden-fri bench (fold=${FOLDING}, log-trace=${LOG_TRACE_HEIGHTS}) =="
  maybe_clean
  (
    cd "${ROOT}"
    P3_LOG_TRACE_HEIGHTS="${LOG_TRACE_HEIGHTS}" \
    P3_FOLDING="${FOLDING}" \
    P3_LOG_BLOWUP="${LOG_BLOWUP}" \
    P3_NUM_QUERIES="${QUERIES}" \
    P3_QUERY_POW_BITS="${GRINDING}" \
    RAYON_NUM_THREADS="${THREADS}" \
    cargo +nightly bench -p p3-miden-fri --bench pcs_criterion_winterfell --features parallel
  )
}

run_bench_lifted() {
  echo "== p3-miden-lifted-fri bench (fold=${FOLDING}, log-trace=${LOG_TRACE_HEIGHTS}) =="
  maybe_clean
  (
    cd "${ROOT}"
    P3_LOG_TRACE_HEIGHTS="${LOG_TRACE_HEIGHTS}" \
    P3_FOLDING="${FOLDING}" \
    P3_LOG_BLOWUP="${LOG_BLOWUP}" \
    P3_NUM_QUERIES="${QUERIES}" \
    P3_QUERY_POW_BITS="${GRINDING}" \
    RAYON_NUM_THREADS="${THREADS}" \
    cargo +nightly bench -p p3-miden-lifted-fri --bench pcs_criterion_winterfell --features parallel
  )
}

run_trace_fri() {
  echo "== p3-miden-fri forest log (fold=${FOLDING}, log-trace=${LOG_TRACE_HEIGHTS}) =="
  maybe_clean
  (
    cd "${ROOT}"
    RUST_LOG=info P3_LOG_FORMAT=forest \
    P3_LOG_TRACE_HEIGHTS="${LOG_TRACE_HEIGHTS}" \
    P3_FOLDING="${FOLDING}" \
    P3_LOG_BLOWUP="${LOG_BLOWUP}" \
    P3_NUM_QUERIES="${QUERIES}" \
    P3_QUERY_POW_BITS="${GRINDING}" \
    RAYON_NUM_THREADS="${THREADS}" \
    cargo +nightly bench -p p3-miden-fri --bench pcs_trace_winterfell --features parallel
  )
}

run_trace_lifted() {
  echo "== p3-miden-lifted-fri forest log (fold=${FOLDING}, log-trace=${LOG_TRACE_HEIGHTS}) =="
  maybe_clean
  (
    cd "${ROOT}"
    RUST_LOG=info P3_LOG_FORMAT=forest \
    P3_LOG_TRACE_HEIGHTS="${LOG_TRACE_HEIGHTS}" \
    P3_FOLDING="${FOLDING}" \
    P3_LOG_BLOWUP="${LOG_BLOWUP}" \
    P3_NUM_QUERIES="${QUERIES}" \
    P3_QUERY_POW_BITS="${GRINDING}" \
    RAYON_NUM_THREADS="${THREADS}" \
    cargo +nightly bench -p p3-miden-lifted-fri --bench pcs_trace_winterfell --features parallel
  )
}

run_bench() {
  case "${VARIANT}" in
    fri) run_bench_fri ;;
    lifted) run_bench_lifted ;;
    both) run_bench_fri; run_bench_lifted ;;
    *) echo "Invalid variant: ${VARIANT}" >&2; usage; exit 1 ;;
  esac
}

run_trace() {
  case "${VARIANT}" in
    fri) run_trace_fri ;;
    lifted) run_trace_lifted ;;
    both) run_trace_fri; run_trace_lifted ;;
    *) echo "Invalid variant: ${VARIANT}" >&2; usage; exit 1 ;;
  esac
}

case "${MODE}" in
  bench) run_bench ;;
  trace) run_trace ;;
  both)  run_bench; run_trace ;;
  *) echo "Invalid mode: ${MODE}" >&2; usage; exit 1 ;;
esac
