#!/usr/bin/env python3
"""
Sieve for halved generalized Fermat numbers  M(b) = (b^N + 1) / 2
with N = 2^k and odd base b in [bmin, bmax].

Mathematical core:
  If p is an odd prime divisor of b^N + 1, then b^N = -1 (mod p), so b has
  order 2N modulo p. It follows that p = 1 (mod 2N).
  -> It is enough to consider only primes p = 2N*j + 1.
  For such a p, x^N = -1 (mod p) has exactly N solutions: the odd powers
  r^1, r^3, ..., r^(2N-1) of a primitive 2N-th root of unity r.
  Every base b congruent modulo p to one of these solutions has the factor p
  and is struck out.

Example:
  python3 hgfn_sieve.py --k 15 --bmin 3 --bmax 1000001 --plimit 1e9 --out candidates.txt
"""

import argparse
import math
import time


# ---------------------------------------------------------------------------
# Deterministic Miller-Rabin test (correct for all n < 3.3 * 10^24)
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
# Helper functions
# ---------------------------------------------------------------------------
def primitive_2N_root(p: int, N: int) -> int:
    """Return r with r^N = -1 (mod p), i.e. a primitive 2N-th root of unity."""
    e = (p - 1) // (2 * N)
    a = 2
    while True:
        r = pow(a, e, p)
        if pow(r, N, p) == p - 1:
            return r
        a += 1


def strike_odd_powers(alive: bytearray, b0: int, bmax: int) -> None:
    """Strike all b = c^e (e >= 3 odd) in the range. Then b^N + 1 is
    algebraically divisible by c^N + 1 and hence certainly composite.
    Since b is odd, c must be odd too. Pure integer arithmetic,
    cost about O(bmax^(1/3))."""
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
# The sieve itself
# ---------------------------------------------------------------------------
def sieve(k: int, bmin: int, bmax: int, plimit: int, report_every: float = 10.0):
    N = 1 << k
    step = 2 * N

    # Odd bases only: index i  <->  b = b0 + 2*i
    b0 = bmin if bmin % 2 == 1 else bmin + 1
    count = (bmax - b0) // 2 + 1
    alive = bytearray([1]) * count

    # Strike algebraically composite bases up front
    strike_odd_powers(alive, b0, bmax)

    t_start = t_last = time.time()
    primes_used = 0
    left = sum(alive)          # count once, then keep it updated

    # Loop over all sieve primes p = 2N*j + 1 <= plimit
    p = step + 1
    while p <= plimit:
        if is_prime(p):
            primes_used += 1
            r = primitive_2N_root(p, N)
            r2 = r * r % p
            x = r
            # Loop over the N roots x = r^(2i+1) of x^N = -1 (mod p)
            for _ in range(N):
                # We need b = x (mod p) AND b odd -> b = y (mod 2p)
                y = x if x % 2 == 1 else x + p
                # smallest b >= b0 with b = y (mod 2p)
                first = b0 + ((y - b0) % (2 * p))
                # Loop over the bases: in index space the stride is p
                idx = (first - b0) // 2
                # (M(b) is always much larger than p, so no special case)
                while idx < count:
                    if alive[idx]:
                        alive[idx] = 0
                        left -= 1
                    idx += p
                x = x * r2 % p

            now = time.time()
            if now - t_last >= report_every:
                t_last = now
                print(f"p = {p:>16,}  primes: {primes_used:>8,}  "
                      f"remaining: {left:>10,}  time: {now - t_start:,.0f} s",
                      flush=True)
        p += step

    survivors = [b0 + 2 * i for i in range(count) if alive[i]]
    return survivors, primes_used, time.time() - t_start


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Sieve for (b^(2^k)+1)/2, b odd")
    ap.add_argument("--k", type=int, required=True, help="exponent: N = 2^k")
    ap.add_argument("--bmin", type=int, default=3)
    ap.add_argument("--bmax", type=int, required=True)
    ap.add_argument("--plimit", type=float, default=1e8, help="sieve limit")
    ap.add_argument("--out", default="candidates.txt")
    args = ap.parse_args()

    survivors, nprimes, secs = sieve(args.k, args.bmin, args.bmax, int(args.plimit))

    total = (args.bmax - args.bmin) // 2 + 1
    print(f"\nN = 2^{args.k} = {1 << args.k}, bases {args.bmin}..{args.bmax}")
    print(f"{nprimes:,} sieve primes up to {int(args.plimit):,} in {secs:,.1f} s")
    print(f"{len(survivors):,} of ~{total:,} odd bases survive "
          f"({100 * len(survivors) / max(total, 1):.2f} %)")

    with open(args.out, "w") as f:
        f.write(f"# (b^{1 << args.k}+1)/2, sieved up to p = {int(args.plimit)}\n")
        for b in survivors:
            f.write(f"{b}\n")
    print(f"Candidates written to {args.out}")


if __name__ == "__main__":
    main()
