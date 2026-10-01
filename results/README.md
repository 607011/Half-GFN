# Results

Small, valuable, kept-in-git outputs of the search. Bulk reproducible intermediates
(full sieve candidate lists, `*.ckpt`, `*.done`) stay out of git — see the top-level
README. Progress/provenance lives in [`../coverage.tsv`](../coverage.tsv); this
folder holds the actual numbers those records point to.

## Layout

```
results/
  primes.tsv                      # one SUMMARY row per proven block
  primes/k<K>/<start>-<end>.txt    # proven prime bases of that block
  prp/k<K>/<start>-<end>.txt       # PRP survivors of that block
```

Per-block lists scale: small `k` yields many primes (thousands per block), large `k`
very few. One row per prime would bloat a single file, so the actual bases live in
per-block files and `primes.tsv` only summarizes.

## `primes.tsv` (summary)

Tab-separated, append-only. One row per proven block:

```
# date   k   block_start   block_end   n_primes   max_digits   method   commit   host
```

- `n_primes`: how many b in the block give a proven prime M(b) = (b^N+1)/2, N = 2^k.
- `max_digits`: decimal digits of the largest proven M(b) in the block.
- `method`: `aprcl` or `ecpp`; `commit`: git short hash of the tools used.

## `primes/k<K>/<start>-<end>.txt` and `prp/k<K>/<start>-<end>.txt`

Proven prime bases (resp. PRP survivors) of one block: a `#` header
then one base `b` per line.

`prp/` is a **staging area — the proof queue**: it holds the PRP survivors only for
blocks whose proof is still pending (so the expensive sieve+PRP work is not lost and
`./coverage.sh todo` is directly actionable). **Once a block is proved, delete its
`prp/` file** — the verified result lives in `primes/`, and `coverage.tsv` keeps the
PRP count for provenance. (If a block ever has pseudoprimes, i.e. `n_prp > n_primes`,
those `prp \ primes` bases are the only thing lost on deletion; record them separately
first if you care — so far every block has had zero.)
