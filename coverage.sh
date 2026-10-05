#!/usr/bin/env bash
#
# coverage.sh  --  Progress / provenance ledger for the prime search.
#
# The project runs over time and across several machines. This append-only ledger
# (coverage.tsv) records which base BLOCKS are done for which STAGE with which
# PARAMETERS -- unambiguous, auditable, and git-mergeable (each completion is
# exactly one new line at the end).
#
# Block sizes may VARY (e.g. orders of magnitude smaller for large k, where even
# one base is expensive). A block covers the odd bases in [block_start, block_end]
# (inclusive); both bounds are stored explicitly, so no fixed size is baked in.
# Within one k, tile consistently at whatever size you choose for that k.
#
# Stages: sieve  (params: plimit=...)
#         prp    (params: bases=3,5,7)
#         proof  (params: method=aprcl|ecpp)
#
# Usage:
#   ./coverage.sh record K BLOCK_START BLOCK_END STAGE PARAMS COUNT
#        e.g.  ./coverage.sh record 15 0 999999 sieve plimit=1e9 80375   (block [0..999999])
#   ./coverage.sh status      # generate STATUS.md from coverage.tsv
#   ./coverage.sh todo        # blocks with PRP done but proof pending
#   ./coverage.sh claim   K BLOCK_START BLOCK_END STAGE [TTL_HOURS]  # reserve a block
#   ./coverage.sh renew   K BLOCK_START BLOCK_END STAGE [TTL_HOURS]  # extend the lease
#   ./coverage.sh release K BLOCK_START BLOCK_END STAGE              # release the lease
#   ./coverage.sh claims                                            # active reservations
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
        printf '# claimed_epoch\tk\tblock_start\tblock_end\tstage\thost\texpires_epoch\n' > "$CLAIMS"
    fi
}

# Append a claim line (claim/renew: expires = now + ttl; release: now).
append_claim() {
    local K="$1" BS="$2" BE="$3" STAGE="$4" EXPIRES="$5"
    local now host
    now=$(date +%s)
    host=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo "-")
    ensure_claims
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "$K" "$BS" "$BE" "$STAGE" "$host" "$EXPIRES" >> "$CLAIMS"
}

cmd_record() {
    if [ $# -lt 6 ]; then
        echo "Usage: $0 record K BLOCK_START BLOCK_END STAGE PARAMS COUNT" >&2
        exit 2
    fi
    local K="$1" BS="$2" BE="$3" STAGE="$4" PARAMS="$5" COUNT="$6"
    case "$STAGE" in
        sieve|prp|proof) ;;
        *) echo "Unknown stage: $STAGE (sieve|prp|proof)" >&2; exit 2 ;;
    esac
    if [ "$BE" -lt "$BS" ]; then
        echo "Warning: block_end $BE < block_start $BS." >&2
    fi
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
    local K="$1" BS="$2" BE="$3" STAGE="$4"
    [ -f "$CLAIMS" ] || return 0
    awk -F'\t' -v k="$K" -v bs="$BS" -v be="$BE" -v st="$STAGE" -v now="$(date +%s)" '
        /^#/ { next }
        $2==k && $3==bs && $4==be && $5==st && ($1+0) >= ts { ts=$1+0; host=$6; xp=$7+0 }
        END { if (xp > now) { printf "%s (%dh left)", host, int((xp-now)/3600) } }
    ' "$CLAIMS"
}

cmd_claim() {
    [ $# -ge 4 ] || { echo "Usage: $0 claim K BLOCK_START BLOCK_END STAGE [TTL_HOURS]" >&2; exit 2; }
    local K="$1" BS="$2" BE="$3" STAGE="$4" ttl="${5:-$TTL_HOURS}"
    case "$STAGE" in sieve|prp|proof) ;; *) echo "Unknown stage: $STAGE" >&2; exit 2 ;; esac
    local info; info=$(active_claim "$K" "$BS" "$BE" "$STAGE")
    if [ -n "$info" ]; then
        echo "Already claimed: k=$K block=$BS-$BE $STAGE by $info" >&2
        echo "(use 'renew' on this host, or wait for the lease to lapse)" >&2
        exit 1
    fi
    local expires=$(( $(date +%s) + ttl * 3600 ))
    append_claim "$K" "$BS" "$BE" "$STAGE" "$expires"
    echo "claimed: k=$K block=$BS-$BE $STAGE (lease ${ttl}h) -- now 'git push'"
}

cmd_renew() {
    [ $# -ge 4 ] || { echo "Usage: $0 renew K BLOCK_START BLOCK_END STAGE [TTL_HOURS]" >&2; exit 2; }
    local K="$1" BS="$2" BE="$3" STAGE="$4" ttl="${5:-$TTL_HOURS}"
    local expires=$(( $(date +%s) + ttl * 3600 ))
    append_claim "$K" "$BS" "$BE" "$STAGE" "$expires"
    echo "renewed: k=$K block=$BS-$BE $STAGE (lease +${ttl}h)"
}

cmd_release() {
    [ $# -ge 4 ] || { echo "Usage: $0 release K BLOCK_START BLOCK_END STAGE" >&2; exit 2; }
    append_claim "$1" "$2" "$3" "$4" "$(date +%s)"   # expires = now -> free immediately
    echo "released: k=$1 block=$2-$3 $4"
}

cmd_claims() {
    [ -f "$CLAIMS" ] || { echo "No claims."; return 0; }
    echo "Active claims:"
    awk -F'\t' -v now="$(date +%s)" '
        /^#/ { next }
        { key=$2 SUBSEP $3 SUBSEP $4 SUBSEP $5; if (($1+0) >= ts[key]) { ts[key]=$1+0; host[key]=$6; xp[key]=$7+0 } }
        END {
            n=0;
            for (key in xp) {
                if (xp[key] > now) {
                    split(key, a, SUBSEP);
                    printf "  k=%s  block=%s-%s  %s  ->  %s  (%dh left)\n",
                           a[1], a[2], a[3], a[4], host[key], int((xp[key]-now)/3600);
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
        echo "_Generated automatically from \`coverage.tsv\` (block sizes may vary by k)._"
        echo "_Do not edit by hand -- run \`./coverage.sh status\`._"
        # For each (k, block) collect the most recently reported params per stage,
        # then emit as Markdown tables sorted numerically by k and block.
        awk -F'\t' '
            /^#/ { next }
            NF >= 6 {
                k=$2; bs=$3; be=$4; st=$5; pa=$6; co=$7;
                key=k SUBSEP bs SUBSEP be; seen[key]=1;
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
                    printf "%s\t%s\t%s\t%s\t%s\t%s\n", a[1], a[2], a[3], s, p, r;
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
            ck=$2 SUBSEP $3 SUBSEP $4 SUBSEP $5;
            if (($1+0) >= cts[ck]) { cts[ck]=$1+0; cexp[ck]=$7+0 }
            next
        }
        /^#/ { next }
        NF >= 6 {
            key=$2 SUBSEP $3 SUBSEP $4;
            if ($5=="prp")   { hasprp[key]=1;   prpparams[key]=$6 }
            if ($5=="proof") { hasproof[key]=1 }
        }
        END {
            n=0;
            for (key in hasprp) {
                if (!(key in hasproof)) {
                    split(key, a, SUBSEP);
                    ck=a[1] SUBSEP a[2] SUBSEP a[3] SUBSEP "proof";
                    claimed = (ck in cexp && cexp[ck] > now) ? "  [claimed]" : "";
                    printf "  k=%s  block=%s-%s  (%s)%s\n", a[1], a[2], a[3], prpparams[key], claimed;
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
