#!/usr/bin/env bash
#
# coverage.sh  --  Progress / provenance ledger for the prime search.
#
# The project runs over time and across several machines. This append-only ledger
# (coverage.tsv) records which base BLOCKS are done for which STAGE with which
# PARAMETERS -- unambiguous, auditable, and git-mergeable (each completion is
# exactly one new line at the end).
#
# Block size: 1,000,000 bases. Block b covers the odd bases in
# [b, b+1,000,000). block_start should be a multiple of 1,000,000.
#
# Stages: sieve  (params: plimit=...)
#         prp    (params: bases=3,5,7)
#         proof  (params: method=aprcl|ecpp)
#
# Usage:
#   ./coverage.sh record K BLOCK_START STAGE PARAMS COUNT
#        e.g.  ./coverage.sh record 15 0 sieve plimit=1e9 80375
#   ./coverage.sh status      # generate STATUS.md from coverage.tsv
#   ./coverage.sh todo        # blocks with PRP done but proof pending
#   ./coverage.sh claim   K BLOCK_START STAGE [TTL_HOURS]   # reserve a block
#   ./coverage.sh renew   K BLOCK_START STAGE [TTL_HOURS]   # extend the lease
#   ./coverage.sh release K BLOCK_START STAGE               # release the lease
#   ./coverage.sh claims                                    # active reservations
#
# DISTRIBUTION WITHOUT DOUBLE WORK (several machines, git as the lock):
#   1. git pull
#   2. pick a free block (status/todo; not 'done', not actively 'claim'ed)
#   3. ./coverage.sh claim ... && git add claims.tsv && git commit && git push
#   4. push REJECTED? -> git pull, back to 2 (nothing has been computed yet)
#   5. push OK? -> ONLY NOW compute. Then ./coverage.sh record ... + push.
#   The iron rule: computation starts ONLY after a successful claim push. The
#   loser of a race loses milliseconds (re-pick), never CPU time.
#
set -euo pipefail

LEDGER="coverage.tsv"
CLAIMS="claims.tsv"
BLOCK=1000000
TAB=$(printf '\t')
# Lease duration in hours (configurable). Default 7 days -- long tasks
# (deep sieving, big proofs) must not expire mid-run.
# Even longer runs: extend with 'renew'.
TTL_HOURS="${COVERAGE_TTL_HOURS:-168}"

ensure_ledger() {
    if [ ! -f "$LEDGER" ]; then
        printf '# date\tk\tblock_start\tblock_end\tstage\tparams\tcount\tcommit\thost\n' > "$LEDGER"
    fi
}

ensure_claims() {
    if [ ! -f "$CLAIMS" ]; then
        printf '# claimed_epoch\tk\tblock_start\tstage\thost\texpires_epoch\n' > "$CLAIMS"
    fi
}

# Append a claim line (claim/renew: expires = now + ttl; release: now).
append_claim() {
    local K="$1" BS="$2" STAGE="$3" EXPIRES="$4"
    local now host
    now=$(date +%s)
    host=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo "-")
    ensure_claims
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "$K" "$BS" "$STAGE" "$host" "$EXPIRES" >> "$CLAIMS"
}

cmd_record() {
    if [ $# -lt 5 ]; then
        echo "Usage: $0 record K BLOCK_START STAGE PARAMS COUNT" >&2
        exit 2
    fi
    local K="$1" BS="$2" STAGE="$3" PARAMS="$4" COUNT="$5"
    case "$STAGE" in
        sieve|prp|proof) ;;
        *) echo "Unknown stage: $STAGE (sieve|prp|proof)" >&2; exit 2 ;;
    esac
    if [ $(( BS % BLOCK )) -ne 0 ]; then
        echo "Warning: block_start $BS is not a multiple of $BLOCK." >&2
    fi
    local BE=$(( BS + BLOCK ))
    local date commit host
    date=$(date -u +%Y-%m-%d)
    commit=$(git rev-parse --short HEAD 2>/dev/null || echo "-")
    host=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo "-")
    ensure_ledger
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$date" "$K" "$BS" "$BE" "$STAGE" "$PARAMS" "$COUNT" "$commit" "$host" >> "$LEDGER"
    echo "recorded: k=$K block=$BS-$BE $STAGE $PARAMS count=$COUNT"
}

# Prints "host (Nh left)" if (k,block,stage) is currently actively claimed.
active_claim() {
    local K="$1" BS="$2" STAGE="$3"
    [ -f "$CLAIMS" ] || return 0
    awk -F'\t' -v k="$K" -v bs="$BS" -v st="$STAGE" -v now="$(date +%s)" '
        /^#/ { next }
        $2==k && $3==bs && $4==st && ($1+0) >= ts { ts=$1+0; host=$5; xp=$6+0 }
        END { if (xp > now) { printf "%s (%dh left)", host, int((xp-now)/3600) } }
    ' "$CLAIMS"
}

cmd_claim() {
    [ $# -ge 3 ] || { echo "Usage: $0 claim K BLOCK_START STAGE [TTL_HOURS]" >&2; exit 2; }
    local K="$1" BS="$2" STAGE="$3" ttl="${4:-$TTL_HOURS}"
    case "$STAGE" in sieve|prp|proof) ;; *) echo "Unknown stage: $STAGE" >&2; exit 2 ;; esac
    local info; info=$(active_claim "$K" "$BS" "$STAGE")
    if [ -n "$info" ]; then
        echo "Already claimed: k=$K block=$BS $STAGE by $info" >&2
        echo "(use 'renew' on this host, or wait for the lease to lapse)" >&2
        exit 1
    fi
    local expires=$(( $(date +%s) + ttl * 3600 ))
    append_claim "$K" "$BS" "$STAGE" "$expires"
    echo "claimed: k=$K block=$BS $STAGE (lease ${ttl}h) -- now 'git push'"
}

cmd_renew() {
    [ $# -ge 3 ] || { echo "Usage: $0 renew K BLOCK_START STAGE [TTL_HOURS]" >&2; exit 2; }
    local K="$1" BS="$2" STAGE="$3" ttl="${4:-$TTL_HOURS}"
    local expires=$(( $(date +%s) + ttl * 3600 ))
    append_claim "$K" "$BS" "$STAGE" "$expires"
    echo "renewed: k=$K block=$BS $STAGE (lease +${ttl}h)"
}

cmd_release() {
    [ $# -ge 3 ] || { echo "Usage: $0 release K BLOCK_START STAGE" >&2; exit 2; }
    append_claim "$1" "$2" "$3" "$(date +%s)"   # expires = now -> free immediately
    echo "released: k=$1 block=$2 $3"
}

cmd_claims() {
    [ -f "$CLAIMS" ] || { echo "No claims."; return 0; }
    echo "Active claims:"
    awk -F'\t' -v now="$(date +%s)" '
        /^#/ { next }
        { key=$2 SUBSEP $3 SUBSEP $4; if (($1+0) >= ts[key]) { ts[key]=$1+0; host[key]=$5; xp[key]=$6+0 } }
        END {
            n=0;
            for (key in xp) {
                if (xp[key] > now) {
                    split(key, a, SUBSEP);
                    printf "  k=%s  block=%s  %s  ->  %s  (%dh left)\n",
                           a[1], a[2], a[3], host[key], int((xp[key]-now)/3600);
                    n++;
                }
            }
            if (n==0) { print "  (none)" }
        }
    ' "$CLAIMS" | sort
}

cmd_status() {
    [ -f "$LEDGER" ] || { echo "No $LEDGER present." >&2; exit 1; }
    {
        echo "# Coverage status"
        echo
        echo "_Generated automatically from \`coverage.tsv\` (block size ${BLOCK})._"
        echo "_Do not edit by hand -- run \`./coverage.sh status\`._"
        # For each (k, block) collect the most recently reported params per stage,
        # then emit as Markdown tables sorted numerically by k and block.
        awk -F'\t' '
            /^#/ { next }
            NF >= 6 {
                k=$2; bs=$3; be=$4; st=$5; pa=$6; co=$7;
                key=k SUBSEP bs; seen[key]=1; endv[key]=be;
                if (st=="sieve")      { sieve[key]=pa (co!=""?" ("co")":"") }
                else if (st=="prp")   { prp[key]=pa (co!=""?" ("co")":"") }
                else if (st=="proof") { proof[key]=pa (co!=""?" ("co")":"") }
            }
            END {
                for (key in seen) {
                    split(key, a, SUBSEP);
                    s = (key in sieve) ? sieve[key] : "—";
                    p = (key in prp)   ? prp[key]   : "—";
                    r = (key in proof) ? proof[key] : "—";
                    printf "%s\t%s\t%s\t%s\t%s\t%s\n", a[1], a[2], endv[key], s, p, r;
                }
            }
        ' "$LEDGER" | sort -t"$TAB" -k1,1n -k2,2n | awk -F'\t' '
            BEGIN { curk="" }
            {
                if ($1 != curk) {
                    curk=$1;
                    printf "\n## k = %s\n\n", curk;
                    print "| block (bases) | sieve | PRP | proof |";
                    print "|---|---|---|---|";
                }
                printf "| %s–%s | %s | %s | %s |\n", $2, $3, $4, $5, $6;
            }
        '
        echo
    } > STATUS.md
    echo "STATUS.md updated."
}

cmd_todo() {
    [ -f "$LEDGER" ] || { echo "No $LEDGER present." >&2; exit 1; }
    ensure_claims
    echo "Proof queue (PRP done, proof pending; [claimed] = running elsewhere):"
    # First file: claims (collect active proof claims). Second: coverage.
    awk -F'\t' -v now="$(date +%s)" '
        FNR==NR {
            if ($0 ~ /^#/) { next }
            ck=$2 SUBSEP $3 SUBSEP $4;
            if (($1+0) >= cts[ck]) { cts[ck]=$1+0; cexp[ck]=$6+0 }
            next
        }
        /^#/ { next }
        NF >= 6 {
            key=$2 SUBSEP $3; endv[key]=$4;
            if ($5=="prp")   { hasprp[key]=1;   prpparams[key]=$6 }
            if ($5=="proof") { hasproof[key]=1 }
        }
        END {
            n=0;
            for (key in hasprp) {
                if (!(key in hasproof)) {
                    split(key, a, SUBSEP);
                    ck=a[1] SUBSEP a[2] SUBSEP "proof";
                    claimed = (ck in cexp && cexp[ck] > now) ? "  [claimed]" : "";
                    printf "  k=%s  block=%s-%s  (%s)%s\n", a[1], a[2], endv[key], prpparams[key], claimed;
                    n++;
                }
            }
            if (n==0) { print "  (empty)" }
        }
    ' "$CLAIMS" "$LEDGER" | sort
}

main() {
    local cmd="${1:-}"
    shift || true
    case "$cmd" in
        record)  cmd_record "$@" ;;
        status)  cmd_status "$@" ;;
        todo)    cmd_todo "$@" ;;
        claim)   cmd_claim "$@" ;;
        renew)   cmd_renew "$@" ;;
        release) cmd_release "$@" ;;
        claims)  cmd_claims "$@" ;;
        -h|--help|"") awk 'NR==1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0" ;;
        *) echo "Unknown command: $cmd" >&2; exit 2 ;;
    esac
}

main "$@"
