#!/usr/bin/env bash
#
# coverage.sh  --  Fortschritts-/Provenance-Register fuer die Primzahlsuche.
#
# Das Projekt laeuft ueber Zeit und mehrere Maschinen. Dieses append-only Register
# (coverage.tsv) haelt fest, welche Basis-BLOECKE fuer welche STUFE mit welchen
# PARAMETERN abgeschlossen sind -- eindeutig, nachvollziehbar und git-mergebar
# (jede Fertigmeldung ist genau eine neue Zeile am Ende).
#
# Blockgroesse: 1.000.000 Basen. Block b deckt die ungeraden Basen in
# [b, b+1.000.000) ab. block_start sollte ein Vielfaches von 1.000.000 sein.
#
# Stufen:  sieve  (params: plimit=...)
#          prp    (params: bases=3,5,7)
#          proof  (params: method=aprcl|ecpp)
#
# Aufruf:
#   ./coverage.sh record K BLOCK_START STAGE PARAMS COUNT
#        z. B.  ./coverage.sh record 15 0 sieve plimit=1e9 80375
#   ./coverage.sh status      # erzeugt STATUS.md aus coverage.tsv
#   ./coverage.sh todo        # Bloecke mit PRP fertig, aber Beweis offen
#   ./coverage.sh claim   K BLOCK_START STAGE [TTL_STUNDEN]   # Block reservieren
#   ./coverage.sh renew   K BLOCK_START STAGE [TTL_STUNDEN]   # Lease verlaengern
#   ./coverage.sh release K BLOCK_START STAGE                 # Lease freigeben
#   ./coverage.sh claims                                      # aktive Reservierungen
#
# VERTEILUNG OHNE DOPPELARBEIT (mehrere Maschinen, Git als Sperre):
#   1. git pull
#   2. freien Block waehlen (status/todo; nicht 'done', nicht aktiv 'claim')
#   3. ./coverage.sh claim ... && git add claims.tsv && git commit && git push
#   4. Push ABGELEHNT? -> git pull, zurueck zu 2 (es wurde NOCH NICHT gerechnet)
#   5. Push OK? -> ERST JETZT rechnen. Danach ./coverage.sh record ... + push.
#   Die eiserne Regel: Rechnen beginnt NUR nach erfolgreichem Claim-Push. Der
#   Verlierer eines Rennens verliert Millisekunden (Neuwahl), nie CPU-Zeit.
#
set -euo pipefail

LEDGER="coverage.tsv"
CLAIMS="claims.tsv"
BLOCK=1000000
TAB=$(printf '\t')
# Lease-Dauer in Stunden (konfigurierbar). Default 7 Tage -- lange Aufgaben
# (tiefes Sieben, grosse Beweise) sollen nicht mitten im Lauf verfallen.
# Noch laengere Laeufe: per 'renew' verlaengern.
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

# Haengt eine Claim-Zeile an (claim/renew: expires = jetzt + ttl; release: jetzt).
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
        echo "Aufruf: $0 record K BLOCK_START STAGE PARAMS COUNT" >&2
        exit 2
    fi
    local K="$1" BS="$2" STAGE="$3" PARAMS="$4" COUNT="$5"
    case "$STAGE" in
        sieve|prp|proof) ;;
        *) echo "Unbekannte Stufe: $STAGE (sieve|prp|proof)" >&2; exit 2 ;;
    esac
    if [ $(( BS % BLOCK )) -ne 0 ]; then
        echo "Warnung: block_start $BS ist kein Vielfaches von $BLOCK." >&2
    fi
    local BE=$(( BS + BLOCK ))
    local date commit host
    date=$(date -u +%Y-%m-%d)
    commit=$(git rev-parse --short HEAD 2>/dev/null || echo "-")
    host=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo "-")
    ensure_ledger
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$date" "$K" "$BS" "$BE" "$STAGE" "$PARAMS" "$COUNT" "$commit" "$host" >> "$LEDGER"
    echo "notiert: k=$K block=$BS-$BE $STAGE $PARAMS count=$COUNT"
}

# Liefert "host (Rest h)" wenn (k,block,stage) aktuell aktiv beansprucht ist.
active_claim() {
    local K="$1" BS="$2" STAGE="$3"
    [ -f "$CLAIMS" ] || return 0
    awk -F'\t' -v k="$K" -v bs="$BS" -v st="$STAGE" -v now="$(date +%s)" '
        /^#/ { next }
        $2==k && $3==bs && $4==st && ($1+0) >= ts { ts=$1+0; host=$5; xp=$6+0 }
        END { if (xp > now) { printf "%s (noch %dh)", host, int((xp-now)/3600) } }
    ' "$CLAIMS"
}

cmd_claim() {
    [ $# -ge 3 ] || { echo "Aufruf: $0 claim K BLOCK_START STAGE [TTL_STUNDEN]" >&2; exit 2; }
    local K="$1" BS="$2" STAGE="$3" ttl="${4:-$TTL_HOURS}"
    case "$STAGE" in sieve|prp|proof) ;; *) echo "Unbekannte Stufe: $STAGE" >&2; exit 2 ;; esac
    local info; info=$(active_claim "$K" "$BS" "$STAGE")
    if [ -n "$info" ]; then
        echo "Schon beansprucht: k=$K block=$BS $STAGE von $info" >&2
        echo "(auf diesem Host 'renew' nutzen, oder Lease abwarten)" >&2
        exit 1
    fi
    local expires=$(( $(date +%s) + ttl * 3600 ))
    append_claim "$K" "$BS" "$STAGE" "$expires"
    echo "beansprucht: k=$K block=$BS $STAGE (Lease ${ttl}h) -- jetzt 'git push'"
}

cmd_renew() {
    [ $# -ge 3 ] || { echo "Aufruf: $0 renew K BLOCK_START STAGE [TTL_STUNDEN]" >&2; exit 2; }
    local K="$1" BS="$2" STAGE="$3" ttl="${4:-$TTL_HOURS}"
    local expires=$(( $(date +%s) + ttl * 3600 ))
    append_claim "$K" "$BS" "$STAGE" "$expires"
    echo "erneuert: k=$K block=$BS $STAGE (Lease +${ttl}h)"
}

cmd_release() {
    [ $# -ge 3 ] || { echo "Aufruf: $0 release K BLOCK_START STAGE" >&2; exit 2; }
    append_claim "$1" "$2" "$3" "$(date +%s)"   # expires = jetzt -> sofort frei
    echo "freigegeben: k=$1 block=$2 $3"
}

cmd_claims() {
    [ -f "$CLAIMS" ] || { echo "Keine Claims."; return 0; }
    echo "Aktive Claims:"
    awk -F'\t' -v now="$(date +%s)" '
        /^#/ { next }
        { key=$2 SUBSEP $3 SUBSEP $4; if (($1+0) >= ts[key]) { ts[key]=$1+0; host[key]=$5; xp[key]=$6+0 } }
        END {
            n=0;
            for (key in xp) {
                if (xp[key] > now) {
                    split(key, a, SUBSEP);
                    printf "  k=%s  block=%s  %s  ->  %s  (noch %dh)\n",
                           a[1], a[2], a[3], host[key], int((xp[key]-now)/3600);
                    n++;
                }
            }
            if (n==0) { print "  (keine)" }
        }
    ' "$CLAIMS" | sort
}

cmd_status() {
    [ -f "$LEDGER" ] || { echo "Kein $LEDGER vorhanden." >&2; exit 1; }
    {
        echo "# Coverage status"
        echo
        echo "_Automatisch aus \`coverage.tsv\` erzeugt (Blockgroesse ${BLOCK})._"
        echo "_Nicht von Hand bearbeiten -- \`./coverage.sh status\` ausfuehren._"
        # Pro (k, block) die zuletzt gemeldeten Parameter je Stufe sammeln,
        # dann numerisch nach k und block sortiert als Markdown-Tabellen ausgeben.
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
    echo "STATUS.md aktualisiert."
}

cmd_todo() {
    [ -f "$LEDGER" ] || { echo "Kein $LEDGER vorhanden." >&2; exit 1; }
    ensure_claims
    echo "Beweis-Warteschlange (PRP fertig, Beweis offen; [beansprucht] = laeuft woanders):"
    # Erste Datei: claims (aktive proof-Claims sammeln). Zweite: coverage.
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
                    claimed = (ck in cexp && cexp[ck] > now) ? "  [beansprucht]" : "";
                    printf "  k=%s  block=%s-%s  (%s)%s\n", a[1], a[2], endv[key], prpparams[key], claimed;
                    n++;
                }
            }
            if (n==0) { print "  (leer)" }
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
        *) echo "Unbekannter Befehl: $cmd" >&2; exit 2 ;;
    esac
}

main "$@"
