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

## 2026-10-02 — large-k NTT engine (`ntt_cuda --resident`): GPU beats CPU at k ≥ 17

**Question.** Does the resident NTT engine (whole-GPU-per-squaring, the design in
`docs/cuda-largek-prp.md`) beat the CPU for large k?

**Hardware.** RTX 4060 (Ada, `sm_89`), CUDA 13.4, vs Intel i5-14500. **Single CPU
core** this time: at large k each candidate is one huge number and GMP is not
multithreaded for a single squaring, so GPU (whole device) vs one GMP core is the
correct per-candidate comparison. Montgomery modmul, forward-DIF/inverse-DIT NTT,
on-GPU CRT + balanced carry; base b = 9 (2 NTT primes). Per modular squaring:

| k | N = 2^k | \|M\| bits | GPU ms/sq | CPU (1 core GMP) ms/sq | speedup |
|---|--------:|-----------:|----------:|-----------------------:|:-------:|
| 14 | 16 384  | 51 936     | 1.72      | 0.22                   | 0.13×   |
| 15 | 32 768  | 103 872    | 2.17      | 0.66                   | 0.30×   |
| 16 | 65 536  | 207 744    | 2.69      | 1.73                   | 0.64×   |
| 17 | 131 072 | 415 488    | 3.29      | 3.69                   | **1.12×** |
| 18 | 262 144 | 830 976    | 3.48      | 8.56                   | **2.46×** |

**Result.** The crossover is **k = 17**: the GPU overtakes a CPU core there and by
k = 18 is ~2.5× ahead; the lead keeps growing with k (CPU per-squaring cost grows
faster than the GPU's). The full powering is validated bit-exact vs GMP
(`mpz_powm`) for k ≤ 11 (larger k take too long to oracle directly, but share the
identical, validated kernels).

**Diagnosis — overhead-bound, not compute-bound.** Adding Montgomery modmul and
removing the bit-reversal + buffer copy (DIF/DIT) barely moved the GPU numbers.
The floor (~1.5 ms/sq, nearly flat from k=12 to k=14) is **kernel-launch
overhead**: each squaring issues ~#primes × 2 transforms × log2(N) stage launches
plus the carry passes. So the way to lower the crossover below k=17 is *fewer
launches* — a multi-stage shared-memory NTT kernel (fold ~log2(tile) stages per
launch) — not faster arithmetic.

**Reliability.** `--ckpt FILE [--ckpt-int N]` checkpoints the residue + exponent
position atomically and resumes bit-exact (verified by stopping at 1500 squarings
and resuming to a GMP-matching result), so multi-hour k ≥ 17 runs survive crashes.

**Reproduce.**
```bash
cmake -G Ninja -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build-release --target ntt_cuda
for k in 14 15 16 17 18; do ./build-release/ntt_cuda --bench 60 --k $k --b 9; done
```

**Remaining work** (see `docs/cuda-largek-prp.md` §7): multi-stage shared-memory NTT
to cut launches (lowers the crossover and speeds every k); Gerbicz–Li error
checking for silent-error detection on long runs.
