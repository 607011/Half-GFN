# HGFN — Half Generalized Fermat Number prime hunting

A small, self-contained toolchain for hunting primes of the form

```
M(b) = (b^N + 1) / 2 ,   N = 2^k ,   b odd
```

("Half Generalized Fermat Numbers"). It runs entirely offline, independent of any
collaborative prime search, and takes you from a raw base range all the way to a
rigorous primality **proof** in three stages:

| Stage | Tool | Purpose | Resumable |
|-------|------|---------|-----------|
| 1. Sieve | `hgfn_sieve` (C++) | discard bases whose `M(b)` has a small factor | ✅ checkpoint |
| 2. PRP filter | `prp_test` (C++ / GMP) | cheap strong probable-prime test | ✅ journal |
| 3. Proof | `prove.sh` (PARI/GP) | deterministic primality proof (APR-CL / ECPP) | ✅ journal |

Everything is native `arm64` on Apple Silicon — no x86, no Rosetta, no FFT
assembler libraries. It also builds and runs on any platform with a C++17
compiler, GMP and PARI/GP.

---

## The math

Write `N = 2^k` and let `b` be odd. Then `M(b) = (b^N + 1)/2` is an odd integer
(the odd part of the always-even generalized Fermat number `b^N + 1`).

**Sieve congruence.** If an odd prime `p` divides `b^N + 1`, then `b^N ≡ -1 (mod p)`,
so `b` has order `2N` modulo `p`, hence

```
p ≡ 1 (mod 2N).
```

It therefore suffices to consider sieve primes of the form `p = 2N·j + 1`. For such
a `p`, the equation `x^N ≡ -1 (mod p)` has exactly `N` solutions — the odd powers
`r, r^3, …, r^(2N-1)` of a primitive `2N`-th root of unity `r`. Every base `b`
congruent to one of them is divisible by `p` and gets struck out. Bases that are
odd perfect powers `b = c^e` (odd `e ≥ 3`) are struck algebraically up front, since
then `c^N + 1 | b^N + 1`.

**Why there is no Pépin test for `M(b)`.** Two obstructions:

1. Modulo `M` we always have `b^N ≡ -1`, so `b` has order `2N` regardless of
   whether `M` is prime — `b` is useless as a Pépin base.
2. `M - 1 = (b^N - 1)/2 = (b-1)(b+1)(b²+1)(b⁴+1)···(b^(N/2)+1) / 2`. The largest
   factor `b^(N/2)+1 ≈ √M` is itself a generalized Fermat number; factoring it is
   as hard as the original problem. So neither an N−1 (Pocklington) nor an N+1
   proof is feasible, and Proth/Riesel do not apply either (the power of two in
   `M-1` is only `2^(k+1)`, far too small).

The correct rigorous proof for this shape is therefore a **general-purpose** method:
APR-CL (the default) or ECPP. That is what stage 3 uses.

---

## Requirements

- A C++17 compiler (`clang++` or `g++`).
- [GMP](https://gmplib.org/) — for `prp_test`.
- [PARI/GP](https://pari.math.u-bordeaux.fr/) — the `gp` binary, for `prove.sh`.

On macOS with Homebrew:

```bash
brew install gmp pari
```

CMake locates GMP automatically via `pkg-config`.

---

## Build

The project uses CMake with the Ninja generator:

```bash
cmake -G Ninja -B build
ninja -C build
```

The binaries land in `build/` (`build/hgfn_sieve`, `build/prp_test`). `prove.sh` is
a shell script and needs no compilation. `cmake --install build` installs the tools
(prefix via `-DCMAKE_INSTALL_PREFIX=...`).

CMake finds GMP through `pkg-config` (with a manual search as fallback; set the
`GMP_DIR` environment variable to point at a custom prefix). PARI/GP is only needed
at runtime by `prove.sh`.

> Note: `-march=native` is enabled by default for maximum speed, which ties the
> binaries to this machine's CPU. Configure with `-DHGFN_NATIVE=OFF` for portable
> binaries.

---

## Quick start (full workflow)

```bash
# 1. Sieve base range for k = 15 (N = 32768) up to sieve prime 1e9
./hgfn_sieve --k 15 --bmax 1000001 --plimit 1e9 --out kand.txt

# 2. Fast strong-PRP filter with several bases
./prp_test --bases "3 5 7" kand.txt --out prp.txt

# 3. Prove the survivors (deterministic)
./prove.sh prp.txt --out primes.txt
```

Each stage reads the exponent `N` automatically from the header line of its input
file, so you only specify `--k` once.

**Want to take part?** See [CONTRIBUTING.md](CONTRIBUTING.md) for how to claim a block
without duplicating work, run the pipeline, and record results — plus the code
conventions.

---

## Stage 1 — `hgfn_sieve`

Multithreaded sieve over sieve primes `p = 2N·j + 1`. Writes surviving bases (one
per line) to the output file, with a header recording `N` and the sieve limit.

```
./hgfn_sieve --k K --bmax BMAX [options]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--k K` | *(required)* | exponent, `N = 2^k` |
| `--bmax BMAX` | *(required)* | upper base bound |
| `--bmin B` | `3` | lower base bound |
| `--plimit P` | `1e8` | sieve up to this prime (accepts `1e9` style) |
| `--out FILE` | `candidates.txt` | output file for surviving bases |
| `--threads N` | all cores | worker threads |
| `--checkpoint FILE` | *(off)* | enable checkpointing to `FILE` |
| `--checkpoint-interval S` | `60` | seconds between checkpoints |
| `--pause-file FILE` | *(off)* | pause at block boundaries while `FILE` exists |

### Checkpointing

The sieve holds one large cumulative state (the `alive` bitmap built up over
billions of primes), so it uses a **snapshot checkpoint**. Candidates are processed
in blocks; all threads synchronise at each block boundary, guaranteeing a clean
"all `j < next_j` done" resume point. The checkpoint (parameters + `next_j` + the
full bitmap) is written atomically (`.tmp` + `rename`).

```bash
./hgfn_sieve --k 6 --bmax 5000001 --plimit 1e10 --checkpoint sieve.ckpt
```

- If the checkpoint file exists, the run **resumes automatically** — just repeat
  the same command.
- **Ctrl-C / SIGTERM** writes a checkpoint at the next block boundary and exits
  cleanly (exit code 130), without writing a partial candidate file.
- `--plimit` may be **increased** on resume to deepen an existing sieve.
- A checkpoint is refused if `k`, `bmin` or `bmax` differ from the current call.

> The checkpoint format is raw little-endian bytes and is machine-local (not
> portable across architectures).

### Pausing

With `--pause-file FILE`, the sieve pauses at the next block boundary whenever
`FILE` exists and resumes when it is removed — a cross-platform trigger that also
works for background and piped runs (no TTY required):

```bash
touch sieve.pause   # pause after the current block (a checkpoint is written first)
rm sieve.pause      # resume
```

Combine it with `--checkpoint` so a paused run can also be killed safely.

---

## Stage 2 — `prp_test`

Strong (Miller-Rabin) probable-prime test of each surviving `M(b)`, using GMP.
This is the cheap filter — a passed PRP test is **not** a proof.

```
./prp_test [options] [candidate-file]        # default: kand.txt
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--bases "a b c"` | `3` | space-separated PRP bases |
| `--exp N` | *(from header)* | override exponent |
| `--limit N` | `0` (all) | test only the first `N` candidates |
| `--out FILE` | `prp.txt` | output file for PRP bases |
| `--threads N` | all cores | worker threads |
| `--journal FILE` | *(off)* | enable journaling / resume |
| `--verbose` | off | also report composite candidates |

Use several bases: a single base has pseudoprimes. For example `(81^1024+1)/2`
passes base 3 but is composite (base 5 catches it, and the proof stage rejects it
outright).

### Journaling

Because each candidate is an independent, idempotent test, `prp_test` resumes via a
lightweight **journal** rather than a snapshot. With `--journal FILE`, every tested
base is appended immediately as `<base> <0|1>` (`1` = PRP) and flushed. On restart
with the same journal, already-tested bases are skipped — including the composites,
which are where the time goes.

```bash
./prp_test --bases "3 5 7" --journal prp.done kand.txt --out prp.txt
```

Ctrl-C stops cleanly: in-flight tests finish and are journaled, and `prp.txt` is
written only on a complete run (until then, the journal is the source of truth).

---

## Stage 3 — `prove.sh`

Deterministic primality **proof** of the PRP survivors, via PARI/GP's `isprime`
(APR-CL by default, ECPP optional). Native `arm64`, GMP-backed.

```
./prove.sh [options] [input-file]            # default: prp.txt
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--exp N` | *(from header)* | override exponent |
| `--limit N` | `0` (all) | prove only the first `N` bases |
| `--out FILE` | `primes.txt` | output file for proven-prime bases |
| `--ecpp` | off | use ECPP (`isprime(.,2)`, yields a certificate; often faster for very large numbers) |
| `--stack BYTES` | `2000000000` | PARI stack size (`parisizemax` grows up to 8× this) |
| `--journal FILE` | *(off)* | enable journaling / resume |

```bash
./prove.sh --journal proof.done prp.txt --out primes.txt
```

Journaling works exactly as in `prp_test` (one `<base> <0|1>` line per result).
This matters here because a single proof can take a long time (see below). Ctrl-C
leaves the journal consistent; re-run the same command to continue.

---

## Performance notes (honest)

- The **sieve** is fast and scales across cores. Small `k` with a large `--plimit`
  is the long-running case (that's what checkpointing is for).
- The **PRP test** is cheap: seconds per ~1000-digit number, and it parallelises
  over candidates.
- The **proof** is inherently expensive. APR-CL/ECPP scale quasi-polynomially in
  the digit count:
  - ~250 digits: milliseconds
  - ~3000 digits (e.g. `(827^1024+1)/2`, proven prime): **~106 minutes**
  - k=15 (~31000 digits): **hours to days** — use `--ecpp` here.

This is exactly why the pipeline order matters: sieve cheaply, PRP-filter cheaply,
and only then pay for a proof on the few survivors.

---

## GPU acceleration (experimental, macOS / Metal)

`prp_metal` is a GPU prototype of the PRP filter for **Regime A** — many
medium-sized candidates, one GPU thread per candidate. The CPU (GMP) prepares each
`M = (b^N+1)/2` and its Montgomery constants; the GPU does the heavy part — a
**strong Miller-Rabin** test to base `a` (decompose `M-1 = d·2^s`, then `a^d` and
the squarings) — using CIOS Montgomery multiplication in 32-bit limbs, staying in
the Montgomery domain throughout. It supports **several bases in one run** (a
candidate must pass the strong test for all of them) and checks that the GPU result
matches the same test on the CPU exactly (and it agrees with `prp_test`).

```bash
ninja -C build prp_metal         # macOS only; needs Metal + GMP
./build/prp_metal --bases "3 5 7" kand.txt
```

### Benchmark (Apple M2 Pro, 10 CPU threads vs GPU, machine otherwise idle)

GPU kernel vs the same strong Miller-Rabin test on all CPU cores:

| k | number size | candidates | 1 base | 3 bases (`3 5 7`) |
|---|-------------|-----------:|-------:|------------------:|
| 3 | ~175 bit (NL=6)  | 340,191 | **3.8×** | 2.5× |
| 4 | ~344 bit (NL=11) | 447,528 | **3.5×** | 1.4× |
| 6 | ~1211 bit (NL=38) | 80,375 | 1.1× | 0.3× |

(kernel speedup; GPU and CPU PRP sets matched exactly in every run.) Numbers are
lower than on a loaded machine — a busy CPU (e.g. BOINC running) slows the CPU
baseline and inflates the apparent GPU advantage; measure on an idle machine.

**Why extra bases help the GPU less than the CPU — SIMD divergence.** On the CPU,
most composites fail the first base and are dropped immediately, so three bases cost
barely more than one. On the GPU, threads run in lockstep within a SIMD group: if
*any* thread in the group is a survivor that needs all three bases, the whole group
pays for all three. So the GPU does close to 3× the work while the CPU does barely
more — which is why the 3-base speedups are markedly lower, and dip below 1× once
the per-thread work is already heavy (`NL=38`). Multiple bases are still useful as a
stronger filter; just expect the GPU edge to shrink.

> Work is dispatched in chunks (one command buffer each) so no single buffer runs
> long enough to hit the GPU watchdog — a multi-second dispatch had threads aborted
> mid-flight, silently producing too few PRPs (always cross-check GPU against CPU, as
> this tool does). The chunks are all enqueued first and awaited only at the end, so
> the GPU runs them back-to-back without per-chunk CPU stalls (waiting after every
> small chunk was itself a 5× slowdown).

**The lesson:** this one-thread-per-candidate approach wins when the numbers are
small enough to stay in registers (high GPU occupancy) and there are many of them.
As the limb count grows the per-thread big-integer arrays spill and the schoolbook
`O(limbs²)` cost rises, so the advantage fades — around `NL≈38` it merely ties
highly-tuned GMP on the CPU. For **large k** (tens of thousands of digits) the right
approach is a different one entirely: a single FFT/NTT-based squaring spread across
the whole GPU (as in `genefer`/`gpuOwl`), not one thread per candidate.

> Prototype limit: `NL ≤ 128` limbs (~4096 bit). CUDA (for NVIDIA) is a planned
> port of the same kernel; the Montgomery math is identical.

## Retired (`___attic/`)

- `___attic/run_pfgw.sh` — an earlier wrapper driving
  [OpenPFGW](https://sourceforge.net/projects/openpfgw/) for the PRP stage.
  Superseded by `prp_test`: PFGW ships only x86 binaries (Rosetta, going away in a
  future macOS), whereas `prp_test` is ARM-native.
- `___attic/hgfn_sieve.py` — the original Python reference implementation of the
  sieve, superseded by `hgfn_sieve.cpp` but kept as a readable spec.

---

## Tracking progress across runs and machines

A long search spread over time and several machines needs a record of *what has
been done with which parameters*. `coverage.sh` maintains an append-only ledger,
`coverage.tsv`, separating three kinds of artifact by how reproducible they are:

| artifact | example | in git? |
|---|---|---|
| the ledger (which blocks/stages/params are done) | `coverage.tsv` | **yes** — small, the index of the work |
| small valuable results | PRP survivors, proven primes | **yes** |
| bulk reproducible intermediates | full candidate lists, `*.ckpt`, `*.done` | **no** (gitignored) |

Work is tracked in fixed, non-overlapping **blocks of 1,000,000 bases** (block `b`
covers odd bases in `[b, b+1000000)`). "Done" always carries its parameters — sieved
to `plimit=1e9` is not the same as `1e12`, and PRP with bases `{3,5,7}` is not `{3}` —
so a completed block is unambiguous and reproducible.

```bash
# after finishing a stage on a block, record it:
./coverage.sh record 15 0 sieve plimit=1e9 157063
./coverage.sh record 8  0 prp   bases=3,5,7 12345

./coverage.sh status   # regenerate STATUS.md (human-readable table)
./coverage.sh todo     # blocks where PRP is done but the proof is still pending
```

`STATUS.md` is generated from the ledger — never edit it by hand. Each completion is
a single appended line, so two machines rarely produce a merge conflict; regenerate
`STATUS.md` after merging. `todo` is the proof queue: it lists exactly the blocks
whose sieve+PRP are complete, so a deterministic proof can run over them.

### Distributing work across machines (no wasted CPU)

Several machines coordinate through the git repo itself — no server. A short-lived
**claim** (lease) in `claims.tsv` reserves a block before work starts:

```bash
git pull
./coverage.sh todo                     # or status — find a free block
./coverage.sh claim 8 0 proof          # reserve it (lease, default 7 days)
git add claims.tsv && git commit -m "claim k8 b0 proof" && git push
#   push rejected?  ->  git pull, pick another block, retry (nothing computed yet)
#   push accepted?  ->  ONLY NOW start computing
./prove.sh ...                         # the actual (hours/days) work
./coverage.sh record 8 0 proof method=aprcl 123   # done = releases the block
git add -A && git commit && git push
```

**Why this wastes zero CPU.** The lock is the claim *push*, which takes seconds; the
work takes hours or days. `git push` to the shared branch is atomic — if two machines
race, exactly one push succeeds and the other is rejected. The loser hasn't computed
anything yet; it just pulls and picks another block (milliseconds lost, never CPU
time). The iron rule: **computation begins only after the claim push succeeds.**

Leases exist only for crash recovery: a claim expires after the TTL
(`COVERAGE_TTL_HOURS`, default 168 h) so a dead machine's block frees up. For runs
longer than the TTL, extend with `./coverage.sh renew`; `release` frees a block early,
and a `record` (done) releases it implicitly. This assumes trusted workers (your own
machines) — opening the search to outside contributors needs result verification, see
[TODO.md](TODO.md).

### Automatic staging cleanup (git hook)

Once a block is proved, its `results/prp/` staging file is redundant (see below). A
tracked `pre-commit` hook removes it automatically when you commit the `proof` record
— but only when nothing is lost (`n_prp == n_primes`); otherwise it warns and keeps
the file. Enable it once per clone:

```bash
git config core.hooksPath githooks
```

The hook is a convenience (bypassable with `git commit --no-verify`); the invariant
it enforces — a proved block keeps only `results/primes/`, not `results/prp/` — also
holds if you delete the file by hand.

## File formats

- **Candidate / PRP file:** first line is a header
  `# (b^N+1)/2, ...`; remaining lines are bases (one integer per line). All three
  stages parse `N` from this header.
- **Journal file (`prp_test`, `prove.sh`):** one line per tested base,
  `<base> <0|1>` (`1` = PRP / proven prime). Append-only; safe to delete to start
  over.
- **Checkpoint file (`hgfn_sieve`):** binary snapshot, machine-local.

Suggested `.gitignore` for runtime artifacts:

```
*.ckpt
*.done
prove_work/
kand*.txt
prp.txt
primes.txt
```

---

## Files

| File | Description |
|------|-------------|
| `hgfn_sieve.cpp` | the sieve (C++17, std::thread, checkpointing) |
| `prp_test.cpp` | strong PRP test (C++17, GMP, journaling) |
| `prp_metal.mm` | GPU PRP prototype (Apple Metal, macOS only) |
| `prove.sh` | primality proof driver (PARI/GP, journaling) |
| `___attic/` | retired files kept for reference (original Python sieve, legacy PFGW wrapper) |
| `CMakeLists.txt` | CMake build (Ninja generator) for `hgfn_sieve` and `prp_test` |
| `coverage.sh` | progress ledger: record completed blocks, generate `STATUS.md`, list the proof queue |
| `coverage.tsv` | append-only coverage ledger (which blocks/stages/params are done) |
| `claims.tsv` | append-only lease ledger (which blocks are currently being worked on) |
| `STATUS.md` | human-readable coverage table (generated) |
| `results/` | small kept results: proven primes (`primes.tsv`) and PRP survivors per block |
| `TODO.md` | deferred ideas / roadmap |
| `CONTRIBUTING.md` | how to contribute compute (claim/run/record) and code |
| `githooks/pre-commit` | removes a proved block's PRP staging file on commit (enable: `git config core.hooksPath githooks`) |
