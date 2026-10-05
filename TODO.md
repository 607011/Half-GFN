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

**Central enforcement (not a client hook).** The local `githooks/pre-commit` only
gives fast feedback on one machine and is bypassable / absent on forks. In the
current trusted direct-push model that is enough — the cleanup commit propagates
with the push, so nothing central is needed. With PRs from untrusted contributors it
is not: GitHub.com has no server-side hooks (`pre-receive` is Enterprise-only), so the
authoritative layer is **GitHub Actions**:
- `on: pull_request` — a required status check that validates invariants read-only
  (no `prp/` for a proved block, `n_prp == n_primes`, well-formed `coverage.tsv`,
  claims respected) so a violating PR cannot merge.
- `on: push`/scheduled — a bot job that authoritatively reconciles and commits the
  cleanup back to `main` (the "central aufräumen" after an accepted PR).
Same CI gate is where result **verification** lives (re-check certificates, recompute
samples) before a block counts as done.

Until then: owner's own machines only, trusted, no verification needed.

## Other deferred items

- **CUDA port of `prp_metal`** for the Windows RTX 4060 (same Montgomery math as the
  Metal kernel; CUDA also gives real per-thread branching, avoiding the SIMD-
  divergence penalty seen with multiple bases on Metal).
- **FFT/NTT GPU PRP for large k — DONE (Apple Silicon / Metal).** One big squaring
  spread across the GPU via an integer NTT; `ntt_metal` (see
  [`docs/metal-largek-prp.md`](docs/metal-largek-prp.md)) crosses over the CPU at
  k=16 and runs the production pipeline via `run_blocks.sh --gpu`, with
  checkpoint/resume. Still open on this engine: the four-step tiling (stage 5b,
  speed only) and a sound error check (Gerbicz–Li). The CUDA sibling for the RTX
  4060 is the separate item above (design in
  [`docs/cuda-largek-prp.md`](docs/cuda-largek-prp.md)).
- **Checkpointable large-k proofs via `primecert` — DONE.** `prove.sh --primecert`
  (and `run_blocks.sh --primecert`) builds the Atkin-Morain ECPP descent in chunks
  with `primecert(C, 0, partial)` at a decreasing threshold, persisting the partial
  certificate after each chunk (`prove_work/cert_<N>_<b>.gp`), so a proof that runs
  for hours/days survives a crash and resumes from the last chunk. Each proof is
  verified with `primecertisvalid`; cert files are keyed by (exponent, b) and
  guarded by `CC[1][1] == M` so a stale certificate is never reused. Kept plain
  `isprime` (APR-CL) as the default for small k. (The certificate is also the
  verification primitive for the distributed/cross-verification item above.)
- **Auto-renew leases** from the long-running tools: have `hgfn_sieve` (at each
  checkpoint) and `prove.sh` call `coverage.sh renew`, so the lease TTL can be short
  while a live worker keeps its claim fresh.
- **Nicer CLI parsing** (getopt-cpp or CLI11) for the C++ tools — evaluated, not yet
  adopted.
- **Auto-record coverage** from the tools on successful completion, instead of a
  manual `coverage.sh record` step. (Largely moot for the main path: `run_blocks.sh`
  already records every stage. This only matters for running the tools standalone,
  where they would need to be told the block's (k, bmin, bmax).)
