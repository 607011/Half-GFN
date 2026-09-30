#!/usr/bin/env bash
#
# prove.sh  --  Deterministischer Primzahl-BEWEIS der PRP-Ueberlebenden,
#               ARM-nativ ueber PARI/GP (isprime = APR-CL / ECPP, GMP-Kernel).
#
# Dritte Stufe des Solo-Workflows:
#   hgfn_sieve  ->  kand.txt   (kein kleiner Faktor)
#   prp_test    ->  prp.txt    (probable prime, schneller Filter)
#   prove.sh    ->  primes.txt (BEWIESEN prim)
#
# WARUM kein Pepin-Test:
#   Fuer M(b) = (b^N+1)/2 (N = 2^k, b ungerade) gibt es keinen klassischen
#   Pepin-/Pocklington-Beweis. Grund: modulo M gilt b^N = -1, d.h. b hat immer
#   Ordnung 2N -- voellig unabhaengig davon, ob M prim ist; b taugt also nicht
#   als Pepin-Basis. Und M-1 = (b^N-1)/2 zerfaellt algebraisch in
#   (b-1)(b+1)(b^2+1)...(b^(N/2)+1)/2; der groesste Faktor b^(N/2)+1 ist ~sqrt(M)
#   und selbst eine verallgemeinerte Fermat-Zahl -- ihn zu faktorisieren ist so
#   schwer wie das Ausgangsproblem. Damit reicht der faktorisierte Anteil von
#   M-1 (bzw. M+1) fuer keinen N-1/N+1-Beweis. Der korrekte Beweis fuer diese
#   Form ist ein allgemeines Verfahren: APR-CL (Default) oder ECPP.
#
# Aufruf:
#   ./prove.sh [Optionen] [datei]         (Default: prp.txt)
# Optionen:
#   --exp N        Exponent ueberschreiben (sonst aus Header)
#   --limit N      nur die ersten N Basen beweisen (0 = alle)
#   --out FILE     bewiesene Primzahl-Basen hierhin (Default primes.txt)
#   --ecpp         ECPP statt APR-CL (isprime(.,2); liefert Zertifikat, oft
#                  schneller bei sehr grossen Zahlen)
#   --stack BYTES  PARI-Stackgroesse (Default 2000000000 = ~2 GB)
#   --journal FILE jede bewiesene/getestete Basis wird sofort protokolliert; ein
#                  erneuter Aufruf mit gleichem Journal ueberspringt sie (Resume).
#                  Strg-C beendet sauber, das Journal bleibt erhalten.
#   -h | --help
#
set -euo pipefail

INFILE=""
EXP=""
LIMIT=0
OUT="primes.txt"
FLAG=0                    # 0 = isprime-Default (APR-CL), 2 = ECPP
STACK=2000000000
JOURNAL=""
INTERRUPTED=0

usage() { sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --exp)    EXP="$2"; shift 2 ;;
        --limit)  LIMIT="$2"; shift 2 ;;
        --out)    OUT="$2"; shift 2 ;;
        --ecpp)   FLAG=2; shift ;;
        --stack)  STACK="$2"; shift 2 ;;
        --journal) JOURNAL="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        -*)       echo "Unbekannte Option: $1" >&2; usage 2 ;;
        *)        INFILE="$1"; shift ;;
    esac
done
[ -z "$INFILE" ] && INFILE="prp.txt"

command -v gp >/dev/null 2>&1 || {
    echo "PARI/GP (gp) nicht gefunden. Installation:  brew install pari" >&2; exit 127; }
[ -f "$INFILE" ] || { echo "Datei nicht gefunden: $INFILE" >&2; exit 1; }

# Exponent aus Header "# (b^EXP+1)/2, ..." lesen, falls nicht gesetzt
if [ -z "$EXP" ]; then
    EXP=$(sed -nE '1 s/.*b\^([0-9]+).*/\1/p' "$INFILE" || true)
fi
[[ "$EXP" =~ ^[0-9]+$ ]] || { echo "Exponent unlesbar -- bitte --exp N angeben." >&2; exit 1; }

WORK="prove_work"; mkdir -p "$WORK"
BASES="$WORK/bases.txt"; SCRIPT="$WORK/prove.gp"; LOG="$WORK/gp.log"

# Vollstaendige Kandidatenliste (unter Beachtung von --limit).
ALL="$WORK/all.txt"
awk -v lim="$LIMIT" '
    /^[0-9]+/ { c++; if (lim>0 && c>lim) exit; print $1 }
' "$INFILE" > "$ALL"
NALL=$(wc -l < "$ALL" | tr -d ' ')
[ "$NALL" -gt 0 ] || { echo "Keine Basen in $INFILE." >&2; exit 1; }

# Journal (Resume): bereits getestete Basen (Spalte 1) herausfiltern.
NDONE=0
if [ -n "$JOURNAL" ] && [ -f "$JOURNAL" ]; then
    awk 'NR==FNR { d[$1]=1; next } !($1 in d)' "$JOURNAL" "$ALL" > "$BASES"
    NDONE=$(awk 'NR==FNR{a[$1]=1;next} ($1 in a){c++} END{print c+0}' "$ALL" "$JOURNAL")
else
    cp "$ALL" "$BASES"
fi
NB=$(wc -l < "$BASES" | tr -d ' ')

METHOD=$([ "$FLAG" -eq 2 ] && echo "ECPP" || echo "APR-CL")

# Hinweis: GP-Skript ohne "\\"-Kommentare erzeugen -- im unquoted Heredoc wuerde
# bash "\\" zu "\" verkuerzen und GP das als Befehl fehldeuten.
cat > "$SCRIPT" <<GP
default(parisizemax, $((STACK * 8)));
N = $EXP;
v = readvec("$BASES");
{
for(i = 1, #v,
    b = v[i];
    M = (b^N + 1)/2;
    gettime();
    r = isprime(M, $FLAG);
    dt = gettime();
    print("RESULT ", b, " ", r, " ", dt);
);
}
GP

echo "Beweis:      $METHOD (PARI/GP, ARM-nativ)"
if [ -n "$JOURNAL" ]; then
    echo "Kandidaten:  $NALL gesamt, $NDONE laut Journal erledigt, $NB zu beweisen  (aus $INFILE)"
    echo "Journal:     $JOURNAL"
else
    echo "Kandidaten:  $NB aus $INFILE"
fi
echo "Zahl:        M(b) = (b^$EXP+1)/2"
echo "Ergebnis ->  $OUT   (Log: $LOG)"
echo "--------------------------------------------------------------------------"

# OUT mit den bereits im Journal bewiesenen Primzahlen vorbefuellen.
: > "$OUT"
if [ -n "$JOURNAL" ] && [ -f "$JOURNAL" ]; then
    awk '$2==1 { print $1 }' "$JOURNAL" >> "$OUT"
fi

# Strg-C / kill vormerken -- das Journal (Anhaengen pro Ergebnis) bleibt konsistent.
trap 'INTERRUPTED=1' INT TERM

rc=0
if [ "$NB" -eq 0 ]; then
    echo "  (nichts zu tun -- alle Kandidaten bereits im Journal)"
else
    set +e
    gp -q -s "$STACK" "$SCRIPT" 2>&1 | tee "$LOG" | while IFS= read -r line; do
        case "$line" in
            RESULT\ *)
                set -- $line          # RESULT b r dt
                b=$2; r=$3; dt=$4
                # Ergebnis sofort dauerhaft ins Journal (getestet = uebersprungen bei Resume).
                [ -n "$JOURNAL" ] && printf '%s %s\n' "$b" "$r" >> "$JOURNAL"
                if [ "$r" = "1" ]; then
                    echo "$b" >> "$OUT"
                    printf "  (%s^%s+1)/2  BEWIESEN PRIM   (%s ms)\n" "$b" "$EXP" "$dt"
                else
                    printf "  (%s^%s+1)/2  zusammengesetzt  (%s ms)\n" "$b" "$EXP" "$dt"
                fi
                ;;
            *) printf '%s\n' "$line" ;;   # gp-Meldungen/Fehler durchreichen
        esac
    done
    rc=${PIPESTATUS[0]}
    set -e
fi

if [ -s "$OUT" ]; then sort -n -u "$OUT" -o "$OUT"; fi
nprime=$(grep -c '^[0-9]' "$OUT" 2>/dev/null || echo 0)

echo "--------------------------------------------------------------------------"
if [ "$INTERRUPTED" -eq 1 ] || [ "$rc" -ge 128 ]; then
    echo "Abgebrochen. $nprime Primzahl(en) bisher -> $OUT"
    if [ -n "$JOURNAL" ]; then
        echo "Fortschritt im Journal $JOURNAL -- gleicher Aufruf setzt fort."
    else
        echo "Ohne --journal geht der Fortschritt der laufenden Beweise verloren."
    fi
    exit 130
fi
echo "Fertig. $nprime bewiesene Primzahl(en) -> $OUT"
[ "$rc" -ne 0 ] && { echo "Hinweis: gp endete mit Code $rc (siehe $LOG)." >&2; exit "$rc"; }
exit 0
