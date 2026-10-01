# TODO / Roadmap

Deferred ideas, newest concerns first. Not commitments — a parking lot so nothing
gets lost.

## Real distributed computing with cross-verification

Open the search to **external contributors** (beyond the owner's own machines).
Today's coordination (git-mediated claims in `coverage.sh`) assumes all workers are
trusted, so a self-reported "done" is taken at face value. With strangers that no
longer holds — a wrong or fabricated result could poison the ledger and cause a real
prime to be missed.

What this needs:
- **Result verification.** Either every block is independently recomputed by a
  second worker and the results must agree, or we lean on **self-checking proofs**
  (ECPP produces a certificate anyone can verify cheaply; PRP residues can be
  double-checked). Primality proofs are the easy part (certificates); sieve and PRP
  coverage are the hard part to trust.
- **A coordinator or signed submissions.** A small server (BOINC-style) that hands
  out ranges and validates returns, or signed result submissions + a verification
  pass before a block counts as done.
- **Redundancy policy.** Decide replication factor (e.g. every block done by ≥2
  independent workers) vs. trust-but-audit (spot-check a random sample).

Until then: owner's own machines only, trusted, no verification needed.

## Other deferred items

- **CUDA port of `prp_metal`** for the Windows RTX 4060 (same Montgomery math as the
  Metal kernel; CUDA also gives real per-thread branching, avoiding the SIMD-
  divergence penalty seen with multiple bases on Metal).
- **FFT/NTT GPU PRP for large k** (tens of thousands of digits) — one big squaring
  spread across the GPU, à la `genefer`/`gpuOwl`; the current one-thread-per-candidate
  kernel is only for medium sizes.
- **Auto-renew leases** from the long-running tools: have `hgfn_sieve` (at each
  checkpoint) and `prove.sh` call `coverage.sh renew`, so the lease TTL can be short
  while a live worker keeps its claim fresh.
- **Nicer CLI parsing** (getopt-cpp or CLI11) for the C++ tools — evaluated, not yet
  adopted.
- **Auto-record coverage** from the tools on successful completion, instead of a
  manual `coverage.sh record` step.
