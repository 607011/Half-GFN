# Benchmarks

Measured, reproducible numbers for the PRP stage. Append new runs; don't rewrite
history. Each entry records hardware, build type, exact command, and the result.

---

## 2026-10-02 — `prp_cuda` vs. CPU (GMP), medium k

**Question.** Does the one-thread-per-candidate CUDA kernel (`prp_cuda.cu`) beat
the multi-threaded GMP CPU path for the sizes it supports?

**Hardware.**
- GPU: NVIDIA GeForce RTX 4060 (Ada, `sm_89`, 8 GB, driver 591.86), CUDA 13.4.
- CPU: Intel Core i5-14500, 20 logical cores (used as 20 GMP threads).

**Build.** Release (`-DCMAKE_BUILD_TYPE=Release`), vcpkg release GMP, MSVC 19.42.
Single base (`3`), one strong Miller-Rabin test. BOINC paused during the run.

**Method.** Candidates generated with `hgfn_sieve --bmin 3 --bmax 50000
--plimit 1e6`; `prp_cuda <file>` runs GPU and CPU on the same list and checks
them against each other. The kernel caps at `MAXNL = 64` limbs (2048 bits).

| k | N = 2^k | limbs (n) | candidates | GPU total | CPU (20T) | GPU / CPU | correctness |
|---|--------:|----------:|-----------:|----------:|----------:|:---------:|:-----------:|
| 5 | 32      | 16        | 7 369      | 0.038 s   | 0.023 s   | **0.6×**  | 580 = 580 ✓ |
| 6 | 64      | 32        | 8 036      | 0.269 s   | 0.140 s   | **0.5×**  | 330 = 330 ✓ |
| 7 | 128     | 63        | 6 321      | 1.686 s   | 0.782 s   | **0.5×**  | 146 = 146 ✓ |
| 5 | 32      | 20        | 88 121     | 0.700 s   | 0.415 s   | **0.6×**  | 5497 = 5497 ✓ |

**Result.** The CPU is ~1.7–2× faster than the GPU across every size tested.
The GPU kernel never wins. Correctness is exact everywhere (0 mismatches,
GPU PRP count == CPU PRP count).

**Two hypotheses ruled out.**
- *Not launch-overhead bound.* Growing the candidate count 12× (7 369 → 88 121)
  leaves the ratio at 0.6×. The GPU is compute bound per element, not startup bound.
- *Not fixed by `MAXNL`.* Rebuilding with `-DMAXNL=20` (tight fit for the k=5 set,
  n=20) instead of the default 64 gave identical time (0.711 s vs 0.694 s). The
  per-thread limb arrays are not the bottleneck.

**Root cause.** One thread per candidate runs a whole modular-exponentiation chain
serially with schoolbook O(n²) CIOS Montgomery. A single Ada core doing O(n²)
bignum work is far slower than a Raptor Cove core, and GMP uses sub-quadratic
multiplication (Karatsuba/Toom) across 20 cores. The gap widens slightly with n
(k=7, n=63 is the worst ratio). This matches the kernel's own header note: it is a
medium-size prototype, and large k needs an FFT approach, not one thread per candidate.

**Reproduce.**
```bash
cmake -G Ninja -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build-release
for k in 5 6 7; do
  ./build-release/hgfn_sieve --k $k --bmin 3 --bmax 50000 --plimit 1e6 --out k$k.txt
  ./build-release/prp_cuda k$k.txt
done
```

**Takeaway for the roadmap.** The one-thread-per-candidate kernel is a dead end for
beating the CPU; keep it only as a correctness oracle / medium-k convenience. The
large-k target (k ≥ 16, numbers of ~10^5–10^6 digits) requires a different
paradigm: one big squaring spread across the whole GPU via FFT/NTT. See
[`docs/cuda-largek-prp.md`](docs/cuda-largek-prp.md).

---

## 2026-10-02 — `ntt_metal` large-k NTT engine vs. CPU (GMP), Apple Silicon

**Question.** Does the integer-NTT large-k engine (`ntt_metal.mm`, one squaring
spread across the whole GPU) beat the CPU per modular squaring, and where is the
crossover k?

**Hardware.** Apple M2 Pro (10-core GPU, unified memory). CPU baseline is one
GMP core (a single strong-PRP squaring does not parallelise across cores).

**Build.** `clang++ -O3 -ObjC++ -fobjc-arc`, GMP via Homebrew, integer NTT with
2 primes (b=101), 32-bit Montgomery modmul, CPU-side CRT + carry over unified
memory. Metric: milliseconds per modular squaring mod (b^N+1) (`--selftest bench`).

| k | N = 2^k | GPU ms/sq | CPU ms/sq | speedup | GPU dispatch | CPU CRT+carry |
|---|--------:|----------:|----------:|:-------:|-------------:|--------------:|
| 13 |   8 192 | 1.24 | 0.20  | 0.16× | — | — |
| 14 |  16 384 | 1.22 | 0.51  | 0.42× | — | — |
| 15 |  32 768 | 1.51 | 1.13  | 0.75× | — | — |
| 16 |  65 536 | 2.52 | 2.71  | **1.08×** | 1.05 ms | 1.25 ms |
| 17 | 131 072 | 4.05 | 5.70  | **1.41×** | 1.31 ms | 2.47 ms |
| 18 | 262 144 | 7.34 | 12.08 | **1.65×** | — | — |

**Result.** Crossover at **k = 16** on the M2 Pro (one k later than the RTX 4060's
k = 15, as expected for an integrated vs. a discrete GPU), scaling to 1.65× at
k = 18. The GPU dispatch time stays nearly flat with k (0.7 → 1.3 ms), exactly the
NTT signature; the CPU/GMP curve rises steeply (0.2 → 12 ms). Correctness is exact
(residue + verdict match GMP; see `--selftest prp`/`negamul`).

**What moved the needle.** The first working engine was 65–470× *slower*, and
profiling (not guessing) showed 74–94% of the time was CPU post-processing, not
the GPU: the CRT reconstruction used mpz Garner **per coefficient**, with a
`modinv` per coefficient. Three fixes, each measured:
- mpz Garner -> pure integer (`__int128`) CRT: ~3× on the CPU part;
- precompute the (prime-set-constant) Garner inverses once, not per coefficient: ~2× more;
- 2-prime fast path in all-`u64` (p0·p1 < 2^62, no 128-bit division): crossover.

The GPU transform itself was competitive from the start; the lesson was that on
unified memory the CPU post-processing is the thing to watch.

**Still open.** Four-step/threadgroup tiling (fewer global passes) and batching the
primes into one command buffer would cut the GPU part further and likely lower the
crossover k; Gerbicz–Li + checkpoint are required before trusting long large-k runs.

**Reproduce.**
```bash
clang++ -std=c++17 -O3 -ObjC++ -fobjc-arc ntt_metal.mm \
  -I$(brew --prefix gmp)/include -L$(brew --prefix gmp)/lib -lgmp \
  -framework Metal -framework Foundation -o ntt_metal
for k in 13 14 15 16 17 18; do ./ntt_metal --selftest bench --k $k --b 101 --reps 40; done
```

---

## 2026-10-05 — `ntt_metal` vs. CPU, fresh sweep (BOINC stopped)

GPU NTT engine vs. one GMP core, ms per modular squaring mod (b^N+1), b=101, 60
iters/point, Apple M2 Pro, machine otherwise idle (BOINC paused). `--selftest bench`.

| k | N = 2^k | GPU ms/sq | CPU ms/sq | speedup | GPU dispatch | CPU CRT+carry |
|---|--------:|----------:|----------:|:-------:|-------------:|--------------:|
| 13 |   8 192 | 0.653 | 0.190 | 0.29× | — | — |
| 14 |  16 384 | 0.842 | 0.499 | 0.59× | — | — |
| 15 |  32 768 | 1.236 | 1.158 | 0.94× | — | — |
| 16 |  65 536 | 2.335 | 2.738 | **1.17×** | 0.93 ms | 1.23 ms |
| 17 | 131 072 | 3.586 | 5.742 | **1.60×** | 0.87 ms | 2.48 ms |
| 18 | 262 144 | 6.820 | 11.796 | **1.73×** | 1.25 ms | 4.97 ms |

Crossover at **k = 15/16** (k=15 is a tie, k=16 wins), rising to 1.73× at k=18 —
a touch better than the earlier run with BOINC still winding down.

**What this means per candidate** (full PRP ≈ bits(M) squarings; bits(M) ≈ N·log2 b):
- k=16 (~436k squarings): GPU ≈ 17.0 min vs CPU ≈ 19.9 min.
- k=18 (~1.75M squarings): GPU ≈ **3.3 h** vs CPU ≈ **5.7 h** — the GPU saves ~2.4 h
  per candidate.

**Where the time goes now.** The GPU dispatch stays nearly flat with k (0.9–1.2 ms);
the growing cost is the CPU-side CRT + carry (O(N), 1.2→5.0 ms). So the next speed
lever is the CPU post-processing (move CRT/carry to the GPU), more than the GPU
transform itself — four-step tiling (stage 5b) shortens an already-small part.

**Reproduce.**
```bash
ninja -C build ntt_metal
for k in 13 14 15 16 17 18; do ./build/ntt_metal --selftest bench --k $k --b 101 --reps 60; done
```
