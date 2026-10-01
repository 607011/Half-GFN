# Results

Small, valuable, kept-in-git outputs of the search. Bulk reproducible intermediates
(full sieve candidate lists, `*.ckpt`, `*.done`) stay out of git — see the top-level
README. Progress/provenance lives in [`../coverage.tsv`](../coverage.tsv); this
folder holds the actual numbers those records point to.

## Layout

```
results/
  primes.tsv              # proven primes (the crown jewels)
  prp/k<K>/<start>-<end>.txt   # PRP survivors per block (inputs to the proof stage)
```

## `primes.tsv`

Tab-separated, append-only. One row per proven prime M(b) = (b^N+1)/2, N = 2^k:

```
# date   k   base   digits   method   commit   host
```

- `method`: `aprcl` or `ecpp` (from `prove.sh`).
- `commit`: git short hash of the tools used, for reproducibility.

## `prp/k<K>/<start>-<end>.txt`

The PRP survivors of one 1,000,000-base block (same format as a candidate file:
a `# (b^N+1)/2, ...` header then one base per line). These are the exact inputs the
proof stage consumes; keeping them means the expensive sieve+PRP work is not lost and
the proof queue (`./coverage.sh todo`) is directly actionable.
