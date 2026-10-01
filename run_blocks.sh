#!/usr/bin/env bash
#
# run_blocks.sh  --  Processes fixed-size base blocks for a fixed k end to end
# (sieve -> PRP -> proof), records them in the ledger and stores the results.
# Resumable on TWO levels:
#   * block level: blocks whose proof is in coverage.tsv are skipped -- an
#     aborted run continues exactly where it left off.
#   * intra-block: the proof uses a persistent journal (.proofwork/) so a run
#     aborted mid-block only recomputes the one in-flight number.
#
# Usage:
#   ./run_blocks.sh --k K --from START --to END [options]
# Options:
#   --block SIZE   bases per block (default 1000000). Shrink it for large k,
#                  where even one block of 1e6 would take far too long.
#   --plimit P     sieve limit (default 1e7)        [must be < M(bmin)!]
#   --bases "..."  PRP bases (default "3 5 7")
#   --ecpp         prove with ECPP instead of APR-CL
#   -h | --help
#
# Processes [START, END) in steps of --block. START/END/SIZE should be chosen so
# the blocks tile consistently (multiples of SIZE).
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

BLOCK=1000000
K=""; FROM=""; TO=""; PLIMIT="1e7"; BASES="3 5 7"; METHOD="aprcl"; PROVE_FLAG=""

usage() { awk 'NR==1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --k)      K="$2"; shift 2 ;;
        --from)   FROM="$2"; shift 2 ;;
        --to)     TO="$2"; shift 2 ;;
        --block)  BLOCK="$2"; shift 2 ;;
        --plimit) PLIMIT="$2"; shift 2 ;;
        --bases)  BASES="$2"; shift 2 ;;
        --ecpp)   METHOD="ecpp"; PROVE_FLAG="--ecpp"; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 2 ;;
    esac
done
[ -n "$K" ] && [ -n "$FROM" ] && [ -n "$TO" ] || { echo "Error: --k, --from, --to required." >&2; usage 2; }

N=$(( 1 << K ))
BASES_CSV=$(printf '%s' "$BASES" | tr ' ' ',')
mkdir -p .proofwork "results/primes/k$K"
[ -f results/primes.tsv ] || \
    printf '# date\tk\tblock_start\tblock_end\tn_primes\tmax_digits\tmethod\tcommit\thost\n' > results/primes.tsv

for (( start=FROM; start<TO; start+=BLOCK )); do
    end=$(( start + BLOCK ))
    bmin=$(( start + 1 )); [ "$start" -eq 0 ] && bmin=3

    # --- block level: already proved? -> skip ---
    if awk -F'\t' -v k="$K" -v s="$start" \
        '$2==k && $3==s && $5=="proof" { f=1 } END { exit f?0:1 }' coverage.tsv 2>/dev/null; then
        echo "k=$K block [$start,$end): already proved -> skipped"
        continue
    fi

    # --- safety guard: the sieve is correct only if M(bmin) > plimit ---
    if ! awk -v n="$N" -v b="$bmin" -v p="$PLIMIT" \
        'BEGIN { exit (n*log(b) - log(2) > log(p)) ? 0 : 1 }'; then
        echo "ERROR: plimit=$PLIMIT too large for k=$K, bmin=$bmin (M(bmin) <= plimit)." >&2
        echo "The sieve could strike out real primes. Lower plimit." >&2
        exit 1
    fi

    S=$(mktemp); P=$(mktemp); PR=$(mktemp)
    journal=".proofwork/k${K}_${start}-${end}.done"

    ./build/hgfn_sieve --k "$K" --bmin "$bmin" --bmax "$end" --plimit "$PLIMIT" --out "$S" >/dev/null 2>&1
    nsieve=$(grep -c '^[0-9]' "$S" || true); nsieve=${nsieve:-0}

    ./build/prp_test --bases "$BASES" "$S" --out "$P" >/dev/null 2>&1
    nprp=$(grep -c '^[0-9]' "$P" || true); nprp=${nprp:-0}

    # Proof with a persistent journal (intra-block checkpoint)
    ./prove.sh $PROVE_FLAG --journal "$journal" "$P" --out "$PR" >/dev/null 2>&1
    nprimes=$(grep -c '^[0-9]' "$PR" || true); nprimes=${nprimes:-0}

    { echo "# proven primes: b with (b^$N+1)/2 prime ($METHOD), block [$start,$end)"
      grep '^[0-9]' "$PR" || true; } > "results/primes/k$K/${start}-${end}.txt"

    ./coverage.sh record "$K" "$start" "$end" sieve "plimit=$PLIMIT" "$nsieve" >/dev/null
    ./coverage.sh record "$K" "$start" "$end" prp   "bases=$BASES_CSV"  "$nprp"   >/dev/null
    ./coverage.sh record "$K" "$start" "$end" proof "method=$METHOD"    "$nprimes" >/dev/null

    if [ "$nprimes" -gt 0 ]; then
        maxb=$(grep '^[0-9]' "$PR" | sort -n | tail -1)
        dig=$(python3 -c "import sys;sys.set_int_max_str_digits(1000000);b=$maxb;print(len(str((b**$N+1)//2)))")
    else
        dig=0
    fi
    date=$(date -u +%Y-%m-%d); commit=$(git rev-parse --short HEAD 2>/dev/null || echo -); host=$(hostname -s 2>/dev/null || echo -)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$date" "$K" "$start" "$end" "$nprimes" "$dig" "$METHOD" "$commit" "$host" >> results/primes.tsv

    rm -f "$S" "$P" "$PR" "$journal"   # block done -> drop the intra-block journal
    echo "k=$K block [$start,$end): sieve=$nsieve prp=$nprp primes=$nprimes (max $dig digits)"
done
