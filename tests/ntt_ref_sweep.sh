#!/usr/bin/env bash
# Stage-1 validation sweep for the large-k GPU PRP reference (ntt_ref).
#
# Checks two things against GMP across k = 4..KMAX and several bases b:
#   1. --selftest : random negacyclic multiply/square mod (b^N+1) == GMP
#   2. full run   : a^(M-1) mod M, M=(b^N+1)/2, bit-exact == mpz_powm
#
# ntt_ref exits 0 on match / ALL OK, non-zero otherwise, so we just check $?.
#
# Usage: tests/ntt_ref_sweep.sh [path-to-ntt_ref] [KMAX]
set -u

BIN="${1:-./build-release/ntt_ref.exe}"
KMAX="${2:-10}"
[ -x "$BIN" ] || BIN="${BIN%.exe}"          # allow non-Windows name
if [ ! -x "$BIN" ]; then echo "ntt_ref not found at $BIN"; exit 2; fi

# A mix of bases: small, medium, large (3-prime regime), and known-composite ones.
BASES="3 5 7 9 101 9999 99999"

fail=0
run() {   # description, args...
    local desc="$1"; shift
    if "$BIN" "$@" >/dev/null 2>&1; then
        echo "  ok    $desc"
    else
        echo "  FAIL  $desc"; fail=$((fail+1))
    fi
}

echo "== selftest (random negamul/square vs GMP) =="
for k in $(seq 4 "$KMAX"); do
    for b in $BASES; do
        run "selftest k=$k b=$b" --selftest --k "$k" --b "$b" --n 20
    done
done

echo "== full PRP a^(M-1) mod M, bit-exact vs GMP =="
for k in $(seq 4 "$KMAX"); do
    for b in $BASES; do
        run "prp k=$k b=$b" --k "$k" --b "$b" --base 3
    done
done

echo
if [ "$fail" -eq 0 ]; then
    echo "ALL PASS"
else
    echo "$fail FAILURE(S)"
fi
exit "$fail"
