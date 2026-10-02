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
