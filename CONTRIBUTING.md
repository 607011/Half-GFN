# Contributing

Two ways to help: **contribute compute** (run part of the search) or
**contribute code**. The README explains what each tool does and how to run it;
this file explains how to take part without stepping on anyone else.

> Current model: the search runs on the maintainer's own, trusted machines. Opening
> it to outside contributors needs result verification (re-runs or ECPP
> certificates) — see [TODO.md](TODO.md). The workflow below assumes trusted
> workers.

## Build

Dependencies: a C++17 compiler, [GMP](https://gmplib.org/) (for `prp_test`) and
[PARI/GP](https://pari.math.u-bordeaux.fr/) — the `gp` binary — for `prove.sh`.

```bash
# macOS:   brew install gmp pari
# Debian:  sudo apt install libgmp-dev pari-gp ninja-build cmake
cmake -G Ninja -B build && ninja -C build
git config core.hooksPath githooks   # once per clone: enables the cleanup hook
```

Quick sanity check (should report 21 primes):

```bash
./build/hgfn_sieve --k 4 --bmax 399 --plimit 1e5 --out /tmp/c.txt
./build/prp_test --bases "3 5 7" /tmp/c.txt --out /tmp/p.txt   # -> 21 PRP
```

## Contributing compute

Work is tracked in **blocks** of bases per exponent `k` (size configurable via
`run_blocks.sh --block`, default 1,000,000 — shrink it for large `k`), recorded in
`coverage.tsv` (progress) and `claims.tsv` (who is working on what). The git repo
itself is the coordination lock — no server.

**The iron rule: start computing only after your claim push succeeds.** A claim push
takes seconds; the work takes minutes to days. If two machines race, exactly one
push wins and the other is rejected — the loser has computed *nothing* and simply
picks another block. This is what makes double work impossible.

1. `git pull`
2. Find a free block:
   ```bash
   ./coverage.sh status     # what is done
   ./coverage.sh todo       # proof queue (and what is already claimed)
   ```
3. Claim it, then publish the claim **before** computing:
   ```bash
   ./coverage.sh claim 4 16000000 proof        # k=4, block [16e6, 17e6)
   git add claims.tsv && git commit -m "claim k4 b16 proof" && git push
   #   push rejected? -> git pull, pick another block, retry (nothing computed yet)
   ```
4. Only after the push lands, compute. The easy path runs the whole pipeline and
   records every stage for you:
   ```bash
   ./run_blocks.sh --k 4 --from 16000000 --to 17000000
   ```
   (Or run the stages by hand: `hgfn_sieve` → `prp_test` → `prove.sh`.)
5. Publish the results:
   ```bash
   git add -A && git commit -m "k4 block 16 (proved)" && git push
   ```

Leases exist only for crash recovery: a claim expires after `COVERAGE_TTL_HOURS`
(default 168 h). For a run longer than the TTL, extend it with
`./coverage.sh renew ...`; `./coverage.sh release ...` frees a block early, and a
recorded `proof` releases it implicitly.

**Picking work.** Go deeper in one `k` (more blocks) or wider across `k`. Keep the
sieve valid: `plimit` must stay below `M(bmin)` for the block, i.e. small `k` needs a
small `plimit` (`run_blocks.sh` enforces this and refuses otherwise).

## What goes in git

- **Commit:** source, `CMakeLists.txt`, the ledgers (`coverage.tsv`, `claims.tsv`),
  `results/` (proven primes + summary), `STATUS.md`, docs.
- **Do not commit** (already git-ignored): `build/`, candidate files, `*.ckpt`,
  `*.done`, `.proofwork/`.
- A **proved** block keeps only `results/primes/`; its `results/prp/` staging file is
  removed (the `pre-commit` hook does this, or delete it by hand).
- `STATUS.md` is generated — never edit it by hand; run `./coverage.sh status`.

## Contributing code

- **English only** — code, comments and program output.
- **C/C++: brace every block**, including single-statement `if`/`else`/`for`/`while`.
- **Build system is CMake + Ninja** (there is no Makefile); keep dependencies minimal
  (GMP and PARI/GP only).
- Keep the three correctness checks intact: `prp_test` / `prp_metal` cross-check
  against each other and the CPU, and the proof stage is the final arbiter. If you
  touch the sieve or a test, re-run the k=4 sanity check above.
- Retired-but-kept code lives in `___attic/`.

See the [README](README.md) for the mathematics and per-tool details, and
[TODO.md](TODO.md) for the roadmap.
