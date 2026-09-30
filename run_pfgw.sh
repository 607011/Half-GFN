#!/usr/bin/env bash
#
# run_pfgw.sh  --  PRP-/Primzahltest der Sieb-Ueberlebenden mit pfgw (OpenPFGW)
#
# Zweiter Schritt des Solo-Workflows: Das C++-Sieb (hgfn_sieve) liefert eine
# Kandidatenliste von Basen b, deren M(b) = (b^N + 1)/2 keinen kleinen Faktor
# hat. Dieses Skript testet jede davon mit pfgw auf (probable) Primalitaet.
#
# Der Exponent N = 2^k wird automatisch aus der Header-Zeile der
# Kandidatendatei gelesen (z. B. "# (b^32768+1)/2, gesiebt bis p = ...").
#
# Aufruf:
#   ./run_pfgw.sh [Optionen] [kandidatendatei]      (Default: kand.txt)
#
# Optionen:
#   --pfgw PATH     Pfad/Name des pfgw-Binaries (sonst Autoerkennung)
#   --exp E         Exponent N ueberschreiben (sonst aus Header)
#   --prp-base B    Fermat-PRP-Basis (pfgw -b<B>); ohne Angabe pfgw-Default
#   --limit N       nur die ersten N Basen testen (0 = alle, Default 0)
#   --out FILE      gefundene (P)PRP-Basen hierhin (Default prp.txt)
#   --extra "..."   zusaetzliche pfgw-Flags (z. B. "-tc" fuer N-1/N+1-Beweis)
#   -h | --help     diese Hilfe
#
# Beispiel:
#   ./hgfn_sieve --k 15 --bmax 1000001 --plimit 1e9 --out kand.txt
#   ./run_pfgw.sh --limit 50 kand.txt
#
set -euo pipefail

# --------------------------------------------------------------------------
# Argumente
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
        -*)         echo "Unbekannte Option: $1" >&2; usage 2 ;;
        *)          CANDFILE="$1"; shift ;;
    esac
done
[ -z "$CANDFILE" ] && CANDFILE="kand.txt"

if [ ! -f "$CANDFILE" ]; then
    echo "Kandidatendatei nicht gefunden: $CANDFILE" >&2
    exit 1
fi

# --------------------------------------------------------------------------
# pfgw finden
# --------------------------------------------------------------------------
if [ -z "$PFGW" ]; then
    for cand in pfgw64 pfgw pfgw-x86_64 ./pfgw64 ./pfgw; do
        if command -v "$cand" >/dev/null 2>&1; then PFGW="$cand"; break; fi
    done
fi
if [ -z "$PFGW" ] || ! command -v "$PFGW" >/dev/null 2>&1; then
    cat >&2 <<'EOF'
pfgw wurde nicht gefunden.

  OpenPFGW gibt es als Binary bei:  https://sourceforge.net/projects/openpfgw/
  Danach entweder ins PATH legen oder mit --pfgw /pfad/zu/pfgw64 angeben.
  (Auf Apple Silicon laeuft der x86_64-Build unter Rosetta 2.)
EOF
    exit 127
fi

# --------------------------------------------------------------------------
# Exponent bestimmen (aus Header:  "# (b^EXP+1)/2, ...")
# --------------------------------------------------------------------------
if [ -z "$EXP" ]; then
    EXP=$(sed -nE '1 s/.*b\^([0-9]+).*/\1/p' "$CANDFILE" || true)
fi
if ! [[ "$EXP" =~ ^[0-9]+$ ]]; then
    echo "Exponent nicht aus Header lesbar -- bitte mit --exp N angeben." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Ausdrucksdatei bauen:  je Zeile  (b^EXP+1)/2
# --------------------------------------------------------------------------
WORK="pfgw_work"
mkdir -p "$WORK"
EXPRFILE="$WORK/expr.txt"
LOGFILE="$WORK/pfgw.log"

# Nur Zahlen-Zeilen (Kommentare wie '# ...' ueberspringen), optional gekappt.
awk -v ex="$EXP" -v lim="$LIMIT" '
    /^[0-9]+/ { c++; if (lim>0 && c>lim) exit; printf "(%s^%s+1)/2\n", $1, ex }
' "$CANDFILE" > "$EXPRFILE"

NCAND=$(wc -l < "$EXPRFILE" | tr -d ' ')
if [ "$NCAND" -eq 0 ]; then
    echo "Keine Kandidaten in $CANDFILE gefunden." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# pfgw-Flags zusammenstellen
# --------------------------------------------------------------------------
PFGW_FLAGS=()
[ -n "$PRPBASE" ] && PFGW_FLAGS+=("-b${PRPBASE}")
# shellcheck disable=SC2206
[ -n "$EXTRA" ]  && PFGW_FLAGS+=($EXTRA)
PFGW_FLAGS+=("$EXPRFILE")

echo "pfgw:        $PFGW"
echo "Exponent N:  $EXP   (M(b) = (b^N+1)/2)"
echo "Kandidaten:  $NCAND aus $CANDFILE${LIMIT:+ }$( [ "$LIMIT" -gt 0 ] && echo "(auf $LIMIT begrenzt)")"
echo "Ergebnis ->  $OUT   (Roh-Log: $LOGFILE)"
echo "Flags:       ${PFGW_FLAGS[*]}"
echo "--------------------------------------------------------------------------"

: > "$OUT"
found=0

# pfgw ausfuehren; Ergebnis-Zeilen live auswerten und PRP/Prime-Basen sammeln.
# Rueckgabecode von pfgw ueber PIPESTATUS pruefen.
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

# Basen numerisch sortieren/deduplizieren
if [ -s "$OUT" ]; then
    sort -n -u "$OUT" -o "$OUT"
fi
NFOUND=$(grep -c '^[0-9]' "$OUT" 2>/dev/null || echo 0)

echo "--------------------------------------------------------------------------"
echo "Fertig. $NFOUND (probable) Primzahl(en) gefunden -> $OUT"
if [ "$rc" -ne 0 ]; then
    echo "Hinweis: pfgw endete mit Code $rc (siehe $LOGFILE)." >&2
fi
exit "$rc"
