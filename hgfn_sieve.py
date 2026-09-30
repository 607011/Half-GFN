#!/usr/bin/env python3
"""
Sieb fuer halbierte verallgemeinerte Fermat-Zahlen  M(b) = (b^N + 1) / 2
mit N = 2^k und ungerader Basis b in [bmin, bmax].

Mathematischer Kern:
  Ist p ein ungerader Primteiler von b^N + 1, dann gilt b^N = -1 (mod p),
  also hat b modulo p die Ordnung 2N. Daraus folgt p = 1 (mod 2N).
  -> Es genuegt, nur Primzahlen p = 2N*j + 1 zu betrachten.
  Fuer so ein p hat x^N = -1 (mod p) genau N Loesungen: die ungeraden
  Potenzen r^1, r^3, ..., r^(2N-1) einer primitiven 2N-ten Einheitswurzel r.
  Jede Basis b, die modulo p zu einer dieser Loesungen kongruent ist,
  hat den Faktor p und wird gestrichen.

Aufruf (Beispiel):
  python3 hgfn_sieve.py --k 15 --bmin 3 --bmax 1000001 --plimit 1e9 --out kandidaten.txt
"""

import argparse
import math
import time


# ---------------------------------------------------------------------------
# Deterministischer Miller-Rabin-Test (korrekt fuer alle n < 3.3 * 10^24)
# ---------------------------------------------------------------------------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for q in _MR_BASES:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def primitive_2N_root(p: int, N: int) -> int:
    """Liefert r mit r^N = -1 (mod p), also eine primitive 2N-te Einheitswurzel."""
    e = (p - 1) // (2 * N)
    a = 2
    while True:
        r = pow(a, e, p)
        if pow(r, N, p) == p - 1:
            return r
        a += 1


def strike_odd_powers(alive: bytearray, b0: int, bmax: int) -> None:
    """Streicht alle b = c^e (e >= 3 ungerade) im Bereich. Dann ist b^N + 1
    algebraisch durch c^N + 1 teilbar und damit sicher zusammengesetzt.
    Da b ungerade ist, muss auch c ungerade sein. Reine Ganzzahl-Arithmetik,
    Aufwand etwa O(bmax^(1/3))."""
    e = 3
    while 3 ** e <= bmax:
        c = 3
        while True:
            b = c ** e
            if b > bmax:
                break
            if b >= b0:
                alive[(b - b0) // 2] = 0
            c += 2
        e += 2


# ---------------------------------------------------------------------------
# Das eigentliche Sieb
# ---------------------------------------------------------------------------
def sieve(k: int, bmin: int, bmax: int, plimit: int, report_every: float = 10.0):
    N = 1 << k
    step = 2 * N

    # Nur ungerade Basen: Index i  <->  b = b0 + 2*i
    b0 = bmin if bmin % 2 == 1 else bmin + 1
    count = (bmax - b0) // 2 + 1
    alive = bytearray([1]) * count

    # Algebraisch zusammengesetzte Basen vorab streichen
    strike_odd_powers(alive, b0, bmax)

    t_start = t_last = time.time()
    primes_used = 0
    left = sum(alive)          # einmalig zaehlen, danach mitfuehren

    # Schleife ueber alle Siebprimzahlen p = 2N*j + 1 <= plimit
    p = step + 1
    while p <= plimit:
        if is_prime(p):
            primes_used += 1
            r = primitive_2N_root(p, N)
            r2 = r * r % p
            x = r
            # Schleife ueber die N Wurzeln x = r^(2i+1) von x^N = -1 (mod p)
            for _ in range(N):
                # Wir brauchen b = x (mod p) UND b ungerade -> b = y (mod 2p)
                y = x if x % 2 == 1 else x + p
                # kleinstes b >= b0 mit b = y (mod 2p)
                first = b0 + ((y - b0) % (2 * p))
                # Schleife ueber die Basen: im Index-Raum ist der Abstand p
                idx = (first - b0) // 2
                # (M(b) ist stets viel groesser als p, daher kein Sonderfall)
                while idx < count:
                    if alive[idx]:
                        alive[idx] = 0
                        left -= 1
                    idx += p
                x = x * r2 % p

            now = time.time()
            if now - t_last >= report_every:
                t_last = now
                print(f"p = {p:>16,}  Primzahlen: {primes_used:>8,}  "
                      f"verbleibend: {left:>10,}  Zeit: {now - t_start:,.0f} s",
                      flush=True)
        p += step

    survivors = [b0 + 2 * i for i in range(count) if alive[i]]
    return survivors, primes_used, time.time() - t_start


# ---------------------------------------------------------------------------
# Kommandozeile
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Sieb fuer (b^(2^k)+1)/2, b ungerade")
    ap.add_argument("--k", type=int, required=True, help="Exponent: N = 2^k")
    ap.add_argument("--bmin", type=int, default=3)
    ap.add_argument("--bmax", type=int, required=True)
    ap.add_argument("--plimit", type=float, default=1e8, help="Siebgrenze")
    ap.add_argument("--out", default="kandidaten.txt")
    args = ap.parse_args()

    survivors, nprimes, secs = sieve(args.k, args.bmin, args.bmax, int(args.plimit))

    total = (args.bmax - args.bmin) // 2 + 1
    print(f"\nN = 2^{args.k} = {1 << args.k}, Basen {args.bmin}..{args.bmax}")
    print(f"{nprimes:,} Siebprimzahlen bis {int(args.plimit):,} in {secs:,.1f} s")
    print(f"{len(survivors):,} von ca. {total:,} ungeraden Basen ueberleben "
          f"({100 * len(survivors) / max(total, 1):.2f} %)")

    with open(args.out, "w") as f:
        f.write(f"# (b^{1 << args.k}+1)/2, gesiebt bis p = {int(args.plimit)}\n")
        for b in survivors:
            f.write(f"{b}\n")
    print(f"Kandidaten geschrieben nach {args.out}")


if __name__ == "__main__":
    main()
