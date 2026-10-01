#!/usr/bin/env bash
#
# run_pfgw.sh  --  PRP/primality test of the sieve survivors with pfgw (OpenPFGW)
#
# Second step of the solo workflow: the C++ sieve (hgfn_sieve) produces a list
# of candidate bases b whose M(b) = (b^N + 1)/2 has no small factor. This script
# tests each of them with pfgw for (probable) primality.
#
# The exponent N = 2^k is read automatically from the candidate file's header
# line (e.g. "# (b^32768+1)/2, sieved up to p = ...").
#
# Usage:
#   ./run_pfgw.sh [options] [candidate-file]        (default: kand.txt)
#
# Options:
#   --pfgw PATH     path/name of the pfgw binary (otherwise auto-detected)
#   --exp E         override the exponent N (otherwise from the header)
#   --prp-base B    Fermat PRP base (pfgw -b<B>); without it, the pfgw default
#   --limit N       test only the first N bases (0 = all, default 0)
#   --out FILE      write the found (P)PRP bases here (default prp.txt)
#   --extra "..."   extra pfgw flags (e.g. "-tc" for an N-1/N+1 proof)
#   -h | --help     this help
#
# Example:
#   ./hgfn_sieve --k 15 --bmax 1000001 --plimit 1e9 --out kand.txt
#   ./run_pfgw.sh --limit 50 kand.txt
#
set -euo pipefail

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
CANDFILE=""
PFGW=""
EXP=""
PRPBASE=""
LIMIT=0
OUT="prp.txt"
EXTRA=""

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --pfgw)     PFGW="$2"; shift 2 ;;
        --exp)      EXP="$2"; shift 2 ;;
        --prp-base) PRPBASE="$2"; shift 2 ;;
        --limit)    LIMIT="$2"; shift 2 ;;
        --out)      OUT="$2"; shift 2 ;;
        --extra)    EXTRA="$2"; shift 2 ;;
        -h|--help)  usage 0 ;;
        -*)         echo "Unknown option: $1" >&2; usage 2 ;;
        *)          CANDFILE="$1"; shift ;;
    esac
done
[ -z "$CANDFILE" ] && CANDFILE="kand.txt"

if [ ! -f "$CANDFILE" ]; then
    echo "Candidate file not found: $CANDFILE" >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Find pfgw
# --------------------------------------------------------------------------
if [ -z "$PFGW" ]; then
    for cand in pfgw64 pfgw pfgw-x86_64 ./pfgw64 ./pfgw; do
        if command -v "$cand" >/dev/null 2>&1; then PFGW="$cand"; break; fi
    done
fi
if [ -z "$PFGW" ] || ! command -v "$PFGW" >/dev/null 2>&1; then
    cat >&2 <<'EOF'
pfgw was not found.

  OpenPFGW is available as a binary at:  https://sourceforge.net/projects/openpfgw/
  Then put it on your PATH or pass it with --pfgw /path/to/pfgw64.
  (On Apple Silicon the x86_64 build runs under Rosetta 2.)
EOF
    exit 127
fi

# --------------------------------------------------------------------------
# Determine the exponent (from the header:  "# (b^EXP+1)/2, ...")
# --------------------------------------------------------------------------
if [ -z "$EXP" ]; then
    EXP=$(sed -nE '1 s/.*b\^([0-9]+).*/\1/p' "$CANDFILE" || true)
fi
if ! [[ "$EXP" =~ ^[0-9]+$ ]]; then
    echo "Exponent not readable from header -- please pass --exp N." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Build the expression file:  one (b^EXP+1)/2 per line
# --------------------------------------------------------------------------
WORK="pfgw_work"
mkdir -p "$WORK"
EXPRFILE="$WORK/expr.txt"
LOGFILE="$WORK/pfgw.log"

# Number lines only (skip comments like '# ...'), optionally capped.
awk -v ex="$EXP" -v lim="$LIMIT" '
    /^[0-9]+/ { c++; if (lim>0 && c>lim) exit; printf "(%s^%s+1)/2\n", $1, ex }
' "$CANDFILE" > "$EXPRFILE"

NCAND=$(wc -l < "$EXPRFILE" | tr -d ' ')
if [ "$NCAND" -eq 0 ]; then
    echo "No candidates found in $CANDFILE." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Assemble the pfgw flags
# --------------------------------------------------------------------------
PFGW_FLAGS=()
[ -n "$PRPBASE" ] && PFGW_FLAGS+=("-b${PRPBASE}")
# shellcheck disable=SC2206
[ -n "$EXTRA" ]  && PFGW_FLAGS+=($EXTRA)
PFGW_FLAGS+=("$EXPRFILE")

echo "pfgw:        $PFGW"
echo "exponent N:  $EXP   (M(b) = (b^N+1)/2)"
echo "candidates:  $NCAND from $CANDFILE${LIMIT:+ }$( [ "$LIMIT" -gt 0 ] && echo "(capped at $LIMIT)")"
echo "result ->    $OUT   (raw log: $LOGFILE)"
echo "flags:       ${PFGW_FLAGS[*]}"
echo "--------------------------------------------------------------------------"

: > "$OUT"
found=0

# Run pfgw; evaluate result lines live and collect PRP/prime bases.
# Check pfgw's return code via PIPESTATUS.
set +e
"$PFGW" "${PFGW_FLAGS[@]}" 2>&1 | tee "$LOGFILE" | while IFS= read -r line; do
    printf '%s\n' "$line"
    case "$line" in
        *"PRP!"*|*"is prime"*|*"is 3-PRP"*)
            b=$(printf '%s\n' "$line" | sed -nE 's/.*\(([0-9]+)\^.*/\1/p')
            if [ -n "$b" ]; then
                echo "$b" >> "$OUT"
                found=$((found+1))
            fi
            ;;
    esac
done
rc=${PIPESTATUS[0]}
set -e

# Sort/deduplicate the bases numerically
if [ -s "$OUT" ]; then
    sort -n -u "$OUT" -o "$OUT"
fi
NFOUND=$(grep -c '^[0-9]' "$OUT" 2>/dev/null || echo 0)

echo "--------------------------------------------------------------------------"
echo "Done. $NFOUND (probable) prime(s) found -> $OUT"
if [ "$rc" -ne 0 ]; then
    echo "Note: pfgw exited with code $rc (see $LOGFILE)." >&2
fi
exit "$rc"
