#!/usr/bin/env bash
#
# hunt_partitional.sh -- patient, resumable PRP hunt for Partitional numbers.
#
#   Pi(n) = p(0) * p(1) * ... * p(n)      (product of partition numbers)
#   candidates:  Pi(n) + 1   and   Pi(n) - 1
#
# This is the SEARCH stage only: it PRP-tests candidates with PARI's
# ispseudoprime (BPSW -- no known counterexample) and logs the hits. Each hit
# is a *probable* prime; turn it into a deterministic proof afterwards with the
# fast N-1 / N+1 (Pocklington/BLS) provers:
#
#     gp -q partitional.gp        # then:  prove(<n>)
#
# Properties for an unattended long run on a spare machine:
#   * incremental: Pi(n) is carried as a running product (one multiply per n),
#     not recomputed -- and it does NOT factor anything (search needs no factors)
#   * nice -n 19: stays out of the way of everything else on the box
#   * append-only hit log (crash-safe) + truncate-written state file
#   * resumable: restarts just past the furthest n already covered (max of the
#     state file and the last logged hit), so a reboot/Ctrl-C loses no ground
#     and never double-logs a hit
#
# Usage:
#   ./hunt_partitional.sh                 # resume (or start at n=2) and grind up
#   ./hunt_partitional.sh --from 400      # force a start point
#   ./hunt_partitional.sh --to 5000       # stop after n=5000 (default: 1000000)
#   nohup ./hunt_partitional.sh >hunt.out 2>&1 &    # detach on the "Topf"
#
# Heads-up: Partitional primes are sparse and the heuristic expected count
# converges (ln Pi(n) ~ n^1.5), so hits thin out fast beyond small n. This is a
# patience machine, not a quick win.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

LOG="results/partitional_prp.log"
STATE="results/partitional_prp.state"
FROM=""
TO=1000000
NICE="nice -n 19"

usage() { awk 'NR==1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --from)  FROM="$2"; shift 2 ;;
        --to)    TO="$2"; shift 2 ;;
        --log)   LOG="$2"; shift 2 ;;
        --state) STATE="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 2 ;;
    esac
done

command -v gp >/dev/null 2>&1 || { echo "ERROR: PARI/GP (gp) not found." >&2; exit 1; }
mkdir -p "$(dirname "$LOG")"

# Resume point: 1 + max(state value, largest n already in the log). Taking the
# max of both means a crash between logging a hit and writing the state cannot
# lose the hit's n nor re-log it.
last=0
if [ -f "$STATE" ]; then
    s=$(cat "$STATE" 2>/dev/null || echo 0); [ -n "$s" ] && last="$s"
fi
if [ -s "$LOG" ]; then
    lm=$(awk -F'\t' '/^[0-9]/ { if ($1+0 > m) m = $1+0 } END { print m+0 }' "$LOG")
    [ "$lm" -gt "$last" ] && last="$lm"
fi
if [ -n "$FROM" ]; then START="$FROM"; else START=$(( last + 1 )); fi
[ "$START" -lt 2 ] && START=2

nhits=0
[ -s "$LOG" ] && nhits=$(grep -c '^[0-9]' "$LOG" 2>/dev/null || echo 0)
echo "[HUNT] Partitional PRP search: n=$START..$TO"
echo "[HUNT] log=$LOG  state=$STATE  (hits so far: $nhits)"
echo "[HUNT] nice=$NICE ; prove a hit later with:  gp -q partitional.gp  ->  prove(n)"

# One gp process carries the running product and does all PRP tests. It writes
# hits to the log (append) and the current n to the state file (truncate) itself,
# so persistence does not depend on this shell seeing gp's stdout.
$NICE gp -q 2>&1 <<GP
prt0 = prod(k = 0, $START - 1, numbpart(k));   /* Pi(START-1); =1 for START=2 */
{
   my(P = prt0, d, f);
   for (n = $START, $TO,
      P *= numbpart(n);
      d = #Str(P);
      if (ispseudoprime(P + 1),
         write("$LOG", Str(n, "\t+1\t", d, "\tPRP"));
         print("  HIT  n=", n, "  Pi(n)+1  ", d, " digits  (PRP)"));
      if (ispseudoprime(P - 1),
         write("$LOG", Str(n, "\t-1\t", d, "\tPRP"));
         print("  HIT  n=", n, "  Pi(n)-1  ", d, " digits  (PRP)"));
      f = fileopen("$STATE", "w"); filewrite(f, Str(n)); fileclose(f);
      if (n % 25 == 0, print("  ... n=", n, "  (", d, " digits)"))
   );
}
GP

echo "[HUNT] done (reached n=$TO or stopped)."
