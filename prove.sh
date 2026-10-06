#!/usr/bin/env bash
#
# prove.sh  --  Deterministic primality PROOF of the PRP survivors,
#               ARM-native via PARI/GP (isprime = APR-CL / ECPP, GMP kernel).
#
# Third stage of the solo workflow:
#   hgfn_sieve  ->  kand.txt   (no small factor)
#   prp_test    ->  prp.txt    (probable prime, fast filter)
#   prove.sh    ->  primes.txt (PROVEN prime)
#
# WHY no Pepin test:
#   For M(b) = (b^N+1)/2 (N = 2^k, b odd) there is no classical Pepin/Pocklington
#   proof. Reason: modulo M we have b^N = -1, i.e. b always has order 2N --
#   independent of whether M is prime, so b is useless as a Pepin base. And
#   M-1 = (b^N-1)/2 factors algebraically into (b-1)(b+1)(b^2+1)...(b^(N/2)+1)/2;
#   the largest factor b^(N/2)+1 is ~sqrt(M) and itself a generalized Fermat
#   number -- factoring it is as hard as the original problem. So the factored
#   part of M-1 (or M+1) is not enough for an N-1/N+1 proof. The correct proof
#   for this form is a general method: APR-CL (default) or ECPP.
#
# Usage:
#   ./prove.sh [options] [file]           (default: prp.txt)
# Options:
#   --exp N        override the exponent (otherwise from the header)
#   --limit N      prove only the first N bases (0 = all)
#   --out FILE     write the proven prime bases here (default primes.txt)
# The proof method is chosen AUTOMATICALLY by number size unless one of the next
# three is given: small -> isprime/APR-CL, large (>= ~3000 digits) -> resumable ECPP.
#   --aprcl        force APR-CL (isprime default)
#   --ecpp         force ECPP (isprime(.,2); yields a certificate, often faster for
#                  very large numbers)
#   --primecert    force resumable ECPP via primecert: build the Atkin-Morain descent
#                  in chunks, checkpointing the partial certificate after each, so a
#                  proof that runs for hours/days survives a crash and resumes
#                  (checkpoints in prove_work/cert_<N>_<b>.gp). Each proof is
#                  verified with primecertisvalid.
#   --stack BYTES  initial PARI stack (default ~1 GB); grows on demand up to --maxmem
#   --maxmem BYTES parisizemax cap (default ~3/4 of physical RAM). Bounds gp's memory
#                  so a large-k proof (k>=11 can need many GB) cannot swallow all RAM
#                  and swap the machine; if a proof needs more it fails cleanly.
#   --journal FILE journaling is ON by default (file "<infile>.journal"); every
#                  proved/tested base is logged immediately and skipped on a rerun
#                  (resume). The journal is removed once all candidates are proved;
#                  Ctrl-C exits cleanly and keeps it.
#   --no-journal   do not read or write a resume journal
#   -h | --help
#
set -euo pipefail

INFILE=""
EXP=""
LIMIT=0
OUT="primes.txt"
FLAG=0                    # 0 = isprime default (APR-CL), 2 = ECPP
PRIMECERT=0               # 1 = resumable ECPP via primecert (checkpointed descent)
METHOD_SET=0              # 1 once a method flag is given; else choose automatically
AUTO_CERT_DIGITS=3000     # auto: at >= this many decimal digits, use resumable ECPP
STACK=1000000000         # initial PARI stack (-s); grows on demand up to MAXMEM
MAXMEM=""                # parisizemax cap in bytes; empty = RAM-aware default
JOURNAL=""
NO_JOURNAL=0
INTERRUPTED=0

usage() { awk 'NR==1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --exp)    EXP="$2"; shift 2 ;;
        --limit)  LIMIT="$2"; shift 2 ;;
        --out)    OUT="$2"; shift 2 ;;
        --aprcl)  FLAG=0; PRIMECERT=0; METHOD_SET=1; shift ;;
        --ecpp)   FLAG=2; METHOD_SET=1; shift ;;
        --primecert) PRIMECERT=1; METHOD_SET=1; shift ;;
        --stack)  STACK="$2"; shift 2 ;;
        --maxmem) MAXMEM="$2"; shift 2 ;;
        --journal) JOURNAL="$2"; shift 2 ;;
        --no-journal) NO_JOURNAL=1; shift ;;
        -h|--help) usage 0 ;;
        -*)       echo "Unknown option: $1" >&2; usage 2 ;;
        *)        INFILE="$1"; shift ;;
    esac
done
[ -z "$INFILE" ] && INFILE="prp.txt"

# Memory cap (parisizemax). PARI grows its stack on demand up to this bound, so a
# large-k proof (k>=11 can need many GB) must not be allowed to swallow all RAM and
# swap the machine to a halt. Default: ~3/4 of physical RAM, so it leaves headroom
# and -- if the proof needs more than that -- fails cleanly ("the PARI stack
# overflows") instead of thrashing. Override with --maxmem BYTES.
if [ -z "$MAXMEM" ]; then
    RAM=0
    if [ "$(uname)" = "Darwin" ]; then
        RAM=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    elif [ -r /proc/meminfo ]; then
        RAM=$(awk '/^MemTotal:/ { print $2 * 1024; exit }' /proc/meminfo)
    fi
    if [ "${RAM:-0}" -gt 0 ]; then MAXMEM=$(( RAM * 3 / 4 )); else MAXMEM=$(( STACK * 8 )); fi
fi
[ "$MAXMEM" -lt "$STACK" ] && MAXMEM="$STACK"   # parisizemax must be >= initial stack

# Journaling is on by default: without --journal, use "<infile>.journal" so a
# resume is tied to this input. --no-journal opts out.
if [ "$NO_JOURNAL" -eq 1 ]; then
    JOURNAL=""
elif [ -z "$JOURNAL" ]; then
    JOURNAL="${INFILE}.journal"
fi

command -v gp >/dev/null 2>&1 || {
    echo "PARI/GP (gp) not found. Install:  brew install pari" >&2; exit 127; }
[ -f "$INFILE" ] || { echo "File not found: $INFILE" >&2; exit 1; }

# Read the exponent from the header "# (b^EXP+1)/2, ..." if not set
if [ -z "$EXP" ]; then
    EXP=$(sed -nE '1 s/.*b\^([0-9]+).*/\1/p' "$INFILE" || true)
fi
[[ "$EXP" =~ ^[0-9]+$ ]] || { echo "Exponent not readable -- please pass --exp N." >&2; exit 1; }

WORK="prove_work"; mkdir -p "$WORK"
BASES="$WORK/bases.txt"; SCRIPT="$WORK/prove.gp"; LOG="$WORK/gp.log"

# Full candidate list (respecting --limit).
ALL="$WORK/all.txt"
awk -v lim="$LIMIT" '
    /^[0-9]+/ { c++; if (lim>0 && c>lim) exit; print $1 }
' "$INFILE" > "$ALL"
NALL=$(wc -l < "$ALL" | tr -d ' ')
# Empty input is not an error: nothing to prove -> 0 primes, clean exit. (A block
# with 0 PRP survivors is a legitimate result, common for small blocks at large k.)
if [ "$NALL" -eq 0 ]; then
    : > "$OUT"
    [ -n "$JOURNAL" ] && rm -f "$JOURNAL"
    echo "No bases in $INFILE -- nothing to prove (0 proven)."
    exit 0
fi

# Auto method: without an explicit --aprcl/--ecpp/--primecert, pick by size. Small
# numbers -> isprime (APR-CL), fast and simple. Large numbers -> resumable ECPP
# (primecert), because the proof then runs long enough that crash-resume matters.
AUTO_NOTE=""
if [ "$METHOD_SET" -eq 0 ]; then
    maxb=$(sort -n "$ALL" | tail -1)
    DIGITS=$(awk -v n="$EXP" -v b="$maxb" 'BEGIN { printf "%d", n * log(b) / log(10) }')
    if [ "${DIGITS:-0}" -ge "$AUTO_CERT_DIGITS" ]; then
        PRIMECERT=1; AUTO_NOTE=" [auto: ~$DIGITS digits >= $AUTO_CERT_DIGITS]"
    else
        AUTO_NOTE=" [auto: ~$DIGITS digits]"
    fi
fi

# Journal (resume): drop bases already tested (column 1).
NDONE=0
if [ -n "$JOURNAL" ] && [ -f "$JOURNAL" ]; then
    awk 'NR==FNR { d[$1]=1; next } !($1 in d)' "$JOURNAL" "$ALL" > "$BASES"
    NDONE=$(awk 'NR==FNR{a[$1]=1;next} ($1 in a){c++} END{print c+0}' "$ALL" "$JOURNAL")
else
    cp "$ALL" "$BASES"
fi
NB=$(wc -l < "$BASES" | tr -d ' ')

if [ "$PRIMECERT" -eq 1 ]; then METHOD="ECPP/primecert (resumable)";
else METHOD=$([ "$FLAG" -eq 2 ] && echo "ECPP" || echo "APR-CL"); fi

# Note: generate the GP script without "\\" comments -- in an unquoted heredoc
# bash would shorten "\\" to "\" and GP would misread it as a command.
if [ "$PRIMECERT" -eq 1 ]; then
    # Resumable ECPP: build the Atkin-Morain descent in chunks (primecert with a
    # decreasing partial threshold), persisting the partial certificate after each
    # chunk. A killed run resumes from the last chunk. Each proof is validated with
    # primecertisvalid. Cert files are named by (exponent, b) so a stale cert for a
    # different M is never reused (also guarded by CC[1][1] == M).
    cat > "$SCRIPT" <<GP
default(parisizemax, $MAXMEM);
N = $EXP;
cdir = "$WORK";
v = readvec("$BASES");
frontier(X) = if(type(X)=="t_INT", #binary(X), my(L=X[#X]); #binary((L[1]+1-L[2])/L[3]));
chkpt(f) = { my(t=Str(f,".tmp")); system(Str("rm -f ",t)); write(t,"CC=",CC,";"); system(Str("mv -f ",t," ",f)); };
proveM(M, f) = {
   CC = M;
   iferr(read(f), e, 0);
   if(type(CC) != "t_INT" && CC[1][1] != M, CC = M);
   while(frontier(CC) > 64,
      my(p = max(frontier(CC)\2, 64));
      CC = primecert(CC, 0, p);
      if(CC == 0, return(0));
      chkpt(f);
   );
   CC = primecert(CC);
   if(CC == 0, return(0));
   chkpt(f);
   if(type(CC) == "t_INT", 1, primecertisvalid(CC));
};
{
for(i = 1, #v,
    b = v[i];
    M = (b^N + 1)/2;
    f = Str(cdir, "/cert_", N, "_", b, ".gp");
    gettime();
    r = proveM(M, f);
    dt = gettime();
    if(r, system(Str("rm -f ", f)));
    print("RESULT ", b, " ", r, " ", dt);
);
}
GP
else
    cat > "$SCRIPT" <<GP
default(parisizemax, $MAXMEM);
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
fi
echo "quit;" >> "$SCRIPT"   # ensure gp exits after the script (no interactive hang)

echo "Proof:       $METHOD (PARI/GP, ARM-native)$AUTO_NOTE"
echo "Memory:      parisizemax $(( MAXMEM / 1024 / 1024 )) MiB (initial stack $(( STACK / 1024 / 1024 )) MiB)"
if [ -n "$JOURNAL" ]; then
    echo "Candidates:  $NALL total, $NDONE done per journal, $NB to prove  (from $INFILE)"
    echo "Journal:     $JOURNAL"
else
    echo "Candidates:  $NB from $INFILE"
fi
echo "Number:      M(b) = (b^$EXP+1)/2"
echo "Result ->    $OUT   (log: $LOG)"
echo "--------------------------------------------------------------------------"

# Prefill OUT with the primes already proven in the journal.
: > "$OUT"
if [ -n "$JOURNAL" ] && [ -f "$JOURNAL" ]; then
    awk '$2==1 { print $1 }' "$JOURNAL" >> "$OUT"
fi

# Note a Ctrl-C / kill -- the journal (appended per result) stays consistent.
trap 'INTERRUPTED=1' INT TERM

rc=0
if [ "$NB" -eq 0 ]; then
    echo "  (nothing to do -- all candidates already in the journal)"
else
    set +e
    gp -q -s "$STACK" "$SCRIPT" < /dev/null 2>&1 | tee "$LOG" | while IFS= read -r line; do
        case "$line" in
            RESULT\ *)
                set -- $line          # RESULT b r dt
                b=$2; r=$3; dt=$4
                # Persist the result to the journal at once (tested = skipped on resume).
                [ -n "$JOURNAL" ] && printf '%s %s\n' "$b" "$r" >> "$JOURNAL"
                if [ "$r" = "1" ]; then
                    echo "$b" >> "$OUT"
                    printf "  (%s^%s+1)/2  PROVEN PRIME   (%s ms)\n" "$b" "$EXP" "$dt"
                else
                    printf "  (%s^%s+1)/2  composite      (%s ms)\n" "$b" "$EXP" "$dt"
                fi
                ;;
            *) printf '%s\n' "$line" ;;   # pass through gp messages/errors
        esac
    done
    rc=${PIPESTATUS[0]}
    set -e
fi

if [ -s "$OUT" ]; then sort -n -u "$OUT" -o "$OUT"; fi
nprime=$(grep -c '^[0-9]' "$OUT" 2>/dev/null || echo 0)

echo "--------------------------------------------------------------------------"
if [ "$INTERRUPTED" -eq 1 ] || [ "$rc" -ge 128 ]; then
    echo "Aborted. $nprime prime(s) so far -> $OUT"
    if [ -n "$JOURNAL" ]; then
        echo "Progress is in the journal $JOURNAL -- the same command resumes."
    else
        echo "Without --journal the progress of running proofs is lost."
    fi
    exit 130
fi
echo "Done. $nprime proven prime(s) -> $OUT"
[ "$rc" -ne 0 ] && { echo "Note: gp exited with code $rc (see $LOG)." >&2; exit "$rc"; }
# Completed cleanly -> the journal is obsolete; remove it.
[ -n "$JOURNAL" ] && rm -f "$JOURNAL"
exit 0
