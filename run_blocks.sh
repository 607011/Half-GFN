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
# Reuse of banked artifacts (avoids recomputing finished stages):
#   * sieve: if results/candidates/k<K>/<s>-<e>.txt exists AND its sieve row is in
#     coverage.tsv, it is reused instead of re-sieving (e.g. a block pre-sieved by
#     sieve_k10.sh). Sieve output is otherwise banked there (local, gitignored).
#   * PRP: results/prp/k<K>/<s>-<e>.txt is the proof-queue staging file; reused if
#     present (with its coverage row), and deleted once the block is proved.
#   The coverage row is the completion marker, so a truncated file is never reused.
#   -y forces a fresh recomputation (overwrites banked files).
#
# Usage:
#   ./run_blocks.sh --k K --bmin B --bmax C [options]
#   Tests the odd bases b with B <= b <= C (INCLUSIVE; b >= 3) end to end.
# Options:
#   --block SIZE   tiling granularity: SIZE integers per block (default 1000000).
#                  It does NOT set the range -- that is --bmin/--bmax. SIZE only
#                  chops the range into per-block units (the unit of coverage rows,
#                  of block-level resume, and of a distributed 'claim'). The
#                  interval size (bmax-bmin+1) MUST be a whole multiple of SIZE.
#                  Shrink SIZE for large k, where even one 1e6 block would take far
#                  too long; for a small range, set SIZE to the range size (= one
#                  block), e.g. --bmin 0 --bmax 999 --block 1000.
#   --plimit P     sieve limit (default 1e7)        [must be < M(bmin)!]
#   --bases "..."  PRP bases (default: first 13 primes, "2 3 5 ... 41")
#   (proof method is AUTOMATIC by size unless forced: small -> APR-CL, large
#    (>= ~3000 digits) -> resumable ECPP; memory is auto-capped to ~3/4 RAM)
#   --aprcl        force APR-CL for the proof
#   --ecpp         force plain ECPP for the proof
#   --primecert    force resumable ECPP (checkpointed descent; survives a crash)
#   --maxmem BYTES cap the prover's memory (prove.sh --maxmem); default ~3/4 of RAM,
#                  so a large-k proof cannot swap the machine
#   -v | --verbose pass -v to hgfn_sieve/prp_test and show their output
#   -y | --yes     recompute/overwrite already-computed blocks without asking
#   --gpu          run the PRP stage on the GPU NTT engine (ntt_metal, Fermat PRP);
#                  sound (drops no primes) and worthwhile only for large k
#   -h | --help
#
# If the requested range contains blocks that are already computed (present in
# coverage.tsv), the script reports them and asks whether to recompute and
# overwrite. Answering no (or a non-interactive run without -y) keeps the existing
# results and computes only the missing blocks. -y recomputes and overwrites all.
#
# The interval [bmin,bmax] is inclusive and must be a whole number of blocks. Each
# block covers the inclusive sub-range [block_bmin, block_bmax]. For resumable
# multi-block sweeps across runs, keep bmin aligned to SIZE so blocks tile the same.
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

BLOCK=1000000
K=""; BMIN=""; BMAX=""; PLIMIT="1e7"; METHOD="aprcl"; PROVE_FLAG=""
EXTRA_FLAG=""; METHOD_SET=0
BASES="2 3 5 7 11 13 17 19 23 29 31 37 41"
VERBOSE=0; VFLAG=""; ASSUME_YES=0; GPU=0

usage() { awk 'NR==1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --k)      K="$2"; shift 2 ;;
        --bmin)   BMIN="$2"; shift 2 ;;
        --bmax)   BMAX="$2"; shift 2 ;;
        --block)  BLOCK="$2"; shift 2 ;;
        --plimit) PLIMIT="$2"; shift 2 ;;
        --bases)  BASES="$2"; shift 2 ;;
        --aprcl)  METHOD="aprcl"; PROVE_FLAG="--aprcl"; METHOD_SET=1; shift ;;
        --ecpp)   METHOD="ecpp"; PROVE_FLAG="--ecpp"; METHOD_SET=1; shift ;;
        --primecert) METHOD="ecpp-primecert"; PROVE_FLAG="--primecert"; METHOD_SET=1; shift ;;
        --maxmem) EXTRA_FLAG="$EXTRA_FLAG --maxmem $2"; shift 2 ;;
        -v|--verbose) VERBOSE=1; VFLAG="-v"; shift ;;
        -y|--yes) ASSUME_YES=1; shift ;;
        --gpu)    GPU=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 2 ;;
    esac
done
[ -n "$K" ] && [ -n "$BMIN" ] && [ -n "$BMAX" ] || { echo "Error: --k, --bmin, --bmax required." >&2; usage 2; }
[ "$BMAX" -ge "$BMIN" ] || { echo "Error: --bmax ($BMAX) must be >= --bmin ($BMIN)." >&2; exit 2; }
# The interval [bmin,bmax] (inclusive) must be a whole number of blocks.
len=$(( BMAX - BMIN + 1 ))
if [ $(( len % BLOCK )) -ne 0 ]; then
    nblk=$(( (len + BLOCK - 1) / BLOCK ))
    suggest=$(( BMIN + nblk * BLOCK - 1 ))
    echo "Error: the interval [$BMIN,$BMAX] spans $len bases, not a multiple of --block ($BLOCK)." >&2
    echo "Use a multiple, e.g. --bmax $suggest, or a --block that divides $len." >&2
    exit 2
fi

# Diagnostic output of the called tools is tagged ([SIEVE]/[PRP]/[PROVE]) and sent
# to fd 3: the terminal when -v is set, otherwise /dev/null. The result files
# (--out) are unaffected either way.
if [ "$VERBOSE" -eq 1 ]; then exec 3>&1; else exec 3>/dev/null; fi

N=$(( 1 << K ))
BASES_CSV=$(printf '%s' "$BASES" | tr ' ' ',')
mkdir -p .proofwork "results/primes/k$K" "results/candidates/k$K" "results/prp/k$K"
[ -f results/primes.tsv ] || \
    printf '# date\tk\tblock_start\tblock_end\tn_primes\tmax_digits\tmethod\tcommit\thost\n' > results/primes.tsv

# --- Pre-scan: which blocks in [BMIN,BMAX] already have results in coverage.tsv? ---
# Matched on the exact (k, block_bmin, block_bmax), so a different block size with
# the same start does NOT count as this block. A block is "done" once its proof is
# recorded; "partial" means sieve and/or PRP are recorded but the proof is not.
# OVERWRITE=1 means recompute done blocks (overwrite); 0 means skip them (resume).
OVERWRITE=0
done_blocks=()
partial_blocks=()
for (( s=BMIN; s<=BMAX; s+=BLOCK )); do
    e=$(( s + BLOCK - 1 < BMAX ? s + BLOCK - 1 : BMAX ))   # inclusive block upper
    stages=$(awk -F'\t' -v k="$K" -v st="$s" -v en="$e" \
        '$2==k && $3==st && $4==en { seen[$5]=1 } END { for (x in seen) printf "%s ", x }' \
        coverage.tsv 2>/dev/null)
    case " $stages " in
        *" proof "*)            done_blocks+=( "$s" ) ;;
        *" sieve "*|*" prp "*)  partial_blocks+=( "$s" ) ;;
    esac
done

# (a) Report blocks with partial results (sieve/PRP present, proof pending). They
# are recomputed either way (no proof yet) -- this is informational only.
if [ "${#partial_blocks[@]}" -gt 0 ]; then
    echo "note: ${#partial_blocks[@]} block(s) in [$BMIN,$BMAX] have partial results" \
         "(sieve/PRP recorded, proof pending) and will be (re)computed:"
    show=6; [ "${#partial_blocks[@]}" -lt "$show" ] && show="${#partial_blocks[@]}"
    for (( i=0; i<show; i++ )); do
        ps="${partial_blocks[$i]}"; pe=$(( ps + BLOCK - 1 < BMAX ? ps + BLOCK - 1 : BMAX ))
        echo "  k=$K block [$ps..$pe]"
    done
    [ "${#partial_blocks[@]}" -gt "$show" ] && echo "  ... and $(( ${#partial_blocks[@]} - show )) more"
fi

if [ "${#done_blocks[@]}" -gt 0 ]; then
    first="${done_blocks[0]}"; last="${done_blocks[${#done_blocks[@]}-1]}"
    if [ "$ASSUME_YES" -eq 1 ]; then
        OVERWRITE=1
        echo "note: ${#done_blocks[@]} block(s) in [$BMIN,$BMAX] already computed -> recomputing and overwriting (-y)."
    elif [ -t 0 ]; then
        echo "k=$K: ${#done_blocks[@]} of the requested blocks are already computed" \
             "(first start=$first, last start=$last)."
        printf "Recompute and overwrite them? [y/N] "
        read -r ans || ans=""
        case "$ans" in
            y|Y|yes|YES|Yes) OVERWRITE=1 ;;
            *) OVERWRITE=0; echo "Keeping existing results; computing only the missing blocks." ;;
        esac
    else
        echo "note: ${#done_blocks[@]} block(s) in [$BMIN,$BMAX] already computed -> skipping them." \
             "Use -y to recompute and overwrite." >&2
    fi
fi

# A banked artifact is reused only if BOTH its file and its coverage row exist.
# The coverage row is written only after a stage finishes, so this never reuses a
# truncated/aborted file (which, for the sieve, could be missing real primes).
has_stage() {  # $1 = sieve|prp ; uses $K,$start,$end
    awk -F'\t' -v k="$K" -v s="$start" -v e="$end" -v st="$1" \
        '$2==k && $3==s && $4==e && $5==st { f=1 } END { exit f?0:1 }' coverage.tsv 2>/dev/null
}

for (( start=BMIN; start<=BMAX; start+=BLOCK )); do
    end=$(( start + BLOCK - 1 < BMAX ? start + BLOCK - 1 : BMAX ))   # inclusive block upper
    bmin=$start; [ "$bmin" -lt 3 ] && bmin=3   # the smallest valid base is 3

    # --- block level: already proved? -> skip, unless the user chose to overwrite ---
    if [ "$OVERWRITE" -eq 0 ] && awk -F'\t' -v k="$K" -v s="$start" -v e="$end" \
        '$2==k && $3==s && $4==e && $5=="proof" { f=1 } END { exit f?0:1 }' coverage.tsv 2>/dev/null; then
        echo "k=$K block [$start..$end]: already proved -> skipped"
        continue
    fi

    cand_file="results/candidates/k$K/${start}-${end}.txt"
    prp_file="results/prp/k$K/${start}-${end}.txt"
    journal=".proofwork/k${K}_${start}-${end}.done"
    # When overwriting an existing block, drop its proof journal so it is a full
    # fresh recomputation rather than a resume.
    [ "$OVERWRITE" -eq 1 ] && rm -f "$journal"

    # --- Stage 1: sieve (reuse banked candidates if present) ---
    if [ "$OVERWRITE" -eq 0 ] && [ -f "$cand_file" ] && has_stage sieve; then
        nsieve=$(grep -c '^[0-9]' "$cand_file" || true); nsieve=${nsieve:-0}
        echo "k=$K block [$start..$end]: reusing banked candidates ($nsieve)"
    else
        # safety guard: the sieve is correct only if M(bmin) > plimit
        if ! awk -v n="$N" -v b="$bmin" -v p="$PLIMIT" \
            'BEGIN { exit (n*log(b) - log(2) > log(p)) ? 0 : 1 }'; then
            echo "ERROR: plimit=$PLIMIT too large for k=$K, bmin=$bmin (M(bmin) <= plimit)." >&2
            echo "The sieve could strike out real primes. Lower plimit." >&2
            exit 1
        fi
        ./build/hgfn_sieve $VFLAG --k "$K" --bmin "$bmin" --bmax "$end" --plimit "$PLIMIT" \
            --out "$cand_file.tmp" 2>&1 | awk '{ print "[SIEVE]", $0; fflush() }' >&3
        mv -f "$cand_file.tmp" "$cand_file"      # atomic -> a banked file is always complete
        nsieve=$(grep -c '^[0-9]' "$cand_file" || true); nsieve=${nsieve:-0}
        ./coverage.sh record "$K" "$start" "$end" sieve "plimit=$PLIMIT" "$nsieve" >/dev/null
    fi

    # --- Stage 2: PRP (reuse banked survivors if present) ---
    if [ "$OVERWRITE" -eq 0 ] && [ -f "$prp_file" ] && has_stage prp; then
        nprp=$(grep -c '^[0-9]' "$prp_file" || true); nprp=${nprp:-0}
        echo "k=$K block [$start..$end]: reusing banked PRP survivors ($nprp)"
    elif [ "$GPU" -eq 1 ]; then
        # GPU path: NTT-accelerated Fermat PRP (ntt_metal). Sound as a pre-proof
        # filter -- it drops no primes; a few extra Fermat pseudoprimes just reach
        # the (authoritative) proof. Worthwhile only for large k.
        [ -x ./build/ntt_metal ] || { echo "ERROR: --gpu needs ./build/ntt_metal (build it first)." >&2; exit 1; }
        ./build/ntt_metal --prp "$cand_file" --out "$prp_file.tmp" --bases "$BASES" \
            --journal "$prp_file.journal" $VFLAG 2>&1 | awk '{ print "[PRP]", $0; fflush() }' >&3
        mv -f "$prp_file.tmp" "$prp_file"
        nprp=$(grep -c '^[0-9]' "$prp_file" || true); nprp=${nprp:-0}
        ./coverage.sh record "$K" "$start" "$end" prp "bases=$BASES_CSV,gpu-fermat" "$nprp" >/dev/null
    else
        ./build/prp_test $VFLAG --bases "$BASES" "$cand_file" --out "$prp_file.tmp" 2>&1 \
            | awk '{ print "[PRP]", $0; fflush() }' >&3
        mv -f "$prp_file.tmp" "$prp_file"
        nprp=$(grep -c '^[0-9]' "$prp_file" || true); nprp=${nprp:-0}
        ./coverage.sh record "$K" "$start" "$end" prp "bases=$BASES_CSV" "$nprp" >/dev/null
    fi

    # --- Stage 3: proof (always; persistent journal for intra-block resume) ---
    # Auto method unless the user forced one: large numbers (>= ~3000 digits) -> the
    # resumable ECPP (primecert), small -> APR-CL. Matches prove.sh's own auto, but
    # resolved here too so the coverage/results method label is accurate.
    pflag="$PROVE_FLAG"; pmethod="$METHOD"
    if [ "$METHOD_SET" -eq 0 ]; then
        pdigits=$(awk -v n="$N" -v b="$end" 'BEGIN { printf "%d", n * log(b) / log(10) }')
        if [ "${pdigits:-0}" -ge 3000 ]; then pflag="--primecert"; pmethod="ecpp-primecert";
        else pflag="--aprcl"; pmethod="aprcl"; fi
    fi
    PR=$(mktemp)
    ./prove.sh $pflag $EXTRA_FLAG --journal "$journal" "$prp_file" --out "$PR" 2>&1 \
        | awk '{ print "[PROVE]", $0; fflush() }' >&3
    nprimes=$(grep -c '^[0-9]' "$PR" || true); nprimes=${nprimes:-0}

    { echo "# proven primes: b with (b^$N+1)/2 prime ($pmethod), block [$start..$end]"
      grep '^[0-9]' "$PR" || true; } > "results/primes/k$K/${start}-${end}.txt"

    ./coverage.sh record "$K" "$start" "$end" proof "method=$pmethod" "$nprimes" >/dev/null

    if [ "$nprimes" -gt 0 ]; then
        maxb=$(grep '^[0-9]' "$PR" | sort -n | tail -1)
        dig=$(python3 -c "import sys;sys.set_int_max_str_digits(1000000);b=$maxb;print(len(str((b**$N+1)//2)))")
    else
        dig=0
    fi
    date=$(date -u +%Y-%m-%d); commit=$(git rev-parse --short HEAD 2>/dev/null || echo -); host=$(hostname -s 2>/dev/null || echo -)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$date" "$K" "$start" "$end" "$nprimes" "$dig" "$pmethod" "$commit" "$host" >> results/primes.tsv

    # Block proved: drop the proof scratch, the PRP staging file (proof-queue
    # convention: the verified result now lives in results/primes/) and the
    # intra-block journal. Keep the banked candidates (reproducible, gitignored,
    # reusable for a re-PRP with other bases).
    rm -f "$PR" "$prp_file" "$journal"
    echo "k=$K block [$start..$end]: sieve=$nsieve prp=$nprp primes=$nprimes (max $dig digits)"
done
