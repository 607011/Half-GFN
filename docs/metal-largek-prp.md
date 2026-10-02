# Metal PRP for large k on Apple Silicon — design

**Goal.** A GPU PRP engine for `M(b) = (b^N + 1)/2`, `N = 2^k`, running on the
**Apple Silicon** GPU via Metal, that offloads (and ideally beats) the
multi-threaded GMP CPU path for **large k** — numbers of ~10^5–10^6 decimal
digits.

This is the Apple-Silicon sibling of [`cuda-largek-prp.md`](cuda-largek-prp.md).
The *math* is identical and lives, validated against GMP, in
[`../ntt_ref.cpp`](../ntt_ref.cpp) — that CPU reference is **hardware-agnostic**
and is the correctness oracle for this engine too. Only the hardware layer below
the NTT differs. Read the CUDA doc first for the full derivation (§2 digit-base-`b`
negacyclic framing, §4 performance model, §5 reliability); this doc records only
what changes for Metal, and why.

The medium-k, one-thread-per-candidate prototype on this platform is
[`../prp_metal.mm`](../prp_metal.mm) (the Metal analogue of `prp_cuda.cu`). Like
its CUDA cousin it loses to the CPU at every size it supports; it stays only as a
convenience / cross-check oracle. This large-k engine is a separate paradigm:
**one squaring spread across the whole GPU**, not one candidate per thread.

**Status (2026-10-02): implemented and verified through stage 5a** in
[`../ntt_metal.mm`](../ntt_metal.mm). Stages 2a–4 (32-bit Montgomery modmul,
single-prime NTT, multi-prime negamul mod `b^N+1`, full `a^(M-1)` PRP powering)
all match GMP bit-for-bit; stage 5a (CPU post-processing optimisation) reaches the
**crossover at k=16** on an M2 Pro (1.65× at k=18) — see
[`../BENCHMARKS.md`](../BENCHMARKS.md). Still open: four-step tiling (§4), and
Gerbicz–Li + checkpoint (§5 stage 4). The one finding that changed the plan is in
§2.1 below.

---

## 0. What carries over unchanged from the CUDA design

Because it is math, not hardware, all of this is shared verbatim — do **not**
re-derive or re-validate it per platform:

- Residue in base `b`, exactly `N = 2^k` digits; multiply/square mod `b^N+1` is
  the length-`N` **negacyclic** convolution of the digit vectors (since
  `b^N ≡ −1`), realised by an integer NTT with a `2N`-th-root `ψ` right-angle
  weighting.
- Run the **entire** powering chain `a^(M−1)` mod `b^N+1` (the clean GFN form),
  reduce to `M = (b^N+1)/2` with a single conditional subtract **only at the
  very end**.
- Exact coefficients via **multi-prime CRT** with NTT-friendly primes
  `p ≡ 1 (mod 2N)`, `p < 2^31`, enough of them that the product exceeds
  `2·N·(b−1)²`.
- **Balanced** base-`b` carry propagation with the `b^N = −1` wrap (digit 0
  absorbs `−1`; see the carry loop in `ntt_ref.cpp` for why non-negative digits
  oscillate and balanced ones converge).
- **Gerbicz–Li error check + checkpoint/resume** are non-negotiable at 10^5–10^6
  squarings. Same discipline as the CUDA engine and the existing `prp_test`
  journal.
- The staged plan gates each step against the reference / the previous stage.

---

## 1. The decisive constraint: Apple GPUs have no fp64

Metal Shading Language has **no `double` type at all** — not slow, absent. The
classic floating-point IBDWT route (gpuOwl / genefer on datacenter cards) is not
merely suboptimal here, it is **impossible without software emulation**, which
would be catastrophic.

On the RTX 4060 the integer NTT was the *better* choice (fp64 runs at ~1:64).
On Apple Silicon it is the **only** choice. This is a feature, not a loss: the
integer NTT is **exact and bit-for-bit deterministic**, which is exactly what a
prime hunt wants — no round-off analysis, no ULP guard, reproducible residues.

Consequence: the CUDA doc's "keep a float FFT path as a future option" footnote
**does not apply on Mac**. There is one backend: integer NTT. The engine's single
`square_mod(x)` primitive has no float alternative here.

---

## 2. Two things that are *easier* on Apple Silicon than on CUDA

### 2.1 Unified memory — no PCIe, no staging

CPU and GPU share one physical address space. Metal buffers with
`MTLStorageModeShared` are directly readable and writable by both with no copy.
This deletes a whole chapter of the CUDA design:

- The CUDA doc's §6 "double-buffer so candidate *i+1*'s host setup overlaps
  candidate *i*'s kernel work" is **moot**: there is no host↔device transfer to
  hide.
- **CRT reconstruction, carry propagation, and the Gerbicz–Li check may stay on
  the CPU in v1** — operating directly on the same buffers the GPU just
  transformed, zero-copy. Only the expensive NTT butterflies go to the GPU. This
  makes the first working version dramatically smaller: port the transform,
  borrow everything else from `ntt_ref.cpp` running on the CPU half.
- **Checkpointing and candidate streaming are nearly free.** Persisting a residue
  is a CPU read of a shared buffer — no device→host copy, no stall.

Move CRT/carry onto the GPU later only if profiling says the CPU half is the
bottleneck (on an integrated GPU, memory bandwidth is shared, so it often is not
worth it early).

> **Finding (stage 5a, measured — this corrects the paragraph above).** Keeping
> CRT + carry on the CPU is the right call, but it is **not** cheap-by-default: in
> the first working engine it was **74–94% of the per-squaring time**, not the GPU
> transform. The GPU dispatch stayed flat with k (~0.7→1.5 ms); the CPU half grew
> linearly and dominated. The naive CPU post ran mpz Garner CRT **per coefficient**
> with a `modinv` per coefficient — O(N) with huge constants. The engine only
> crossed over after the CPU half was optimised hard, all without touching the GPU:
> (1) mpz Garner → pure `__int128` integer CRT (~3×); (2) precompute the
> prime-set-constant Garner inverses once instead of per coefficient (~2×); (3) a
> 2-prime fast path entirely in `u64` (`p0·p1 < 2^62`, no 128-bit divide) → crossover.
> Lesson: on unified memory, "leave it on the CPU" must still mean *a tight,
> mpz-free, constants-precomputed* CPU path — profile the post-processing first,
> it is the thing that silently costs O(N) per squaring. See `crt_balanced` /
> `CrtPlan` / the 2-prime branch in `ntt_metal.mm`, and the breakdown in
> `--selftest bench`.

### 2.2 The GPU watchdog is handled for free by the transform structure

macOS has its own GPU command-buffer timeout (it will reset the GPU and corrupt
results if a single dispatch runs too long — we already hit this in the Regime-A
Metal prototype). The four-step transform below makes **every dispatch a bounded
slice** (a few NTT stages over one tile), so the watchdog is satisfied
structurally, with no manual chunking. Same payoff the CUDA multi-launch gets
from dodging WDDM TDR.

---

## 3. The one piece that must be rewritten: modular multiply in 32-bit

`ntt_ref.cpp` does `mulmod(a,b,p) = (a*b) % p` in `u64`. On Apple GPUs, 64-bit
integer arithmetic — and **division/modulo above all** — is emulated and slow.
The NTT hot path (every butterfly's twiddle multiply) must run in **32-bit** with
no `%` operator:

- Keep NTT primes `p < 2^31`. Replace `mulmod` with **Montgomery** (or Barrett)
  reduction built on `metal::mulhi(x, y)` — MSL's native 32×32→high-32 multiply
  (`mulhi`/`madhi`), which gives a 64-bit product without a 64-bit type and
  reduces without division.
- Precompute per-prime Montgomery constants (`p`, `p' = −p^{−1} mod 2^32`,
  `R² mod p`, the twiddle tables `ω^t` already in Montgomery form) once on the
  CPU and pass them as a small constant buffer. The GPU kernel then only does
  `mul` + `mulhi` + a couple of adds per modmul.
- This is the **only** line of mathematics that changes shape. Everything else in
  `negamul` is data movement that the GPU reorganises, not new math. Validate the
  32-bit Montgomery modmul against the `u64` reference in isolation before wiring
  it into the transform.

---

## 4. Execution model: four-step / Stockham NTT, tiled to threadgroups

CUDA offers a cooperative-groups **grid-wide barrier**; Metal does **not** — the
only barrier is `threadgroup_barrier`, *within* one threadgroup. The standard GPU
solution is exactly the "multi-launch per stage" the CUDA doc recommends, made
concrete with Apple numbers:

- **Four-step (Stockham) decomposition.** Factor `N = N₁ · N₂`. Each threadgroup
  computes a sub-transform that fits in **threadgroup memory** (Apple M-series:
  **32 KB** per threadgroup, max **1024** threads). With one prime's coefficients
  as `u32`, that is ~8192-point tiles. For **k=16**, `N = 65536 = 8192 × 8` → two
  passes:
  1. Dispatch A: each threadgroup loads a tile into threadgroup memory, runs the
     local radix-2 stages with `threadgroup_barrier`, applies twiddles, writes
     back.
  2. Dispatch B: the cross-tile stages (the "step" between the two factors).
  The **dispatch boundary is the global barrier.** k=18 (`N = 262144`) factors as
  `8192 × 32` or `512 × 512`, same structure.
- **One prime per dispatch set.** The 2–3 CRT primes are independent until
  reconstruction, so run each prime's transform separately (full 32 KB available
  for its `u32` tile) and combine on the CPU (see §2.1).
- **SIMD width is 32** on Apple GPUs — same as a CUDA warp. The innermost radix
  stages use `simd_shuffle` / `simd_shuffle_xor` instead of threadgroup memory,
  exactly as a CUDA kernel would use `__shfl`.
- **Bit-reversal / weighting.** Fold the `ψ^j` negacyclic weighting and the
  `ψ^{−j}` unweighting into the load/store of the first/last pass (don't spend a
  separate dispatch on them). The `1/N` scaling folds into the inverse twiddles.

---

## 5. Staged implementation plan

Mirrors the CUDA plan but reuses our existing assets; each stage has a concrete
pass/fail gate. Status markers are as of 2026-10-02 (all in `ntt_metal.mm`, each
behind a `--selftest` mode).

1. **Oracle — DONE.** `ntt_ref.cpp` validates `a^(M−1) mod M` against GMP for
   k = 4…10, bit-for-bit. Ground truth for every Metal stage below.
2. **Single-prime Metal NTT — DONE (stages 2a, 2b).** 32-bit Montgomery modmul
   (§3, `--selftest montmul`) and the forward+inverse transform (§4, currently the
   robust **multi-dispatch** form, not yet four-step tiled; `--selftest ntt`).
   Gate met: matches `ntt_ref`'s per-prime NTT bit-for-bit, k = 4…16.
3. **Multi-prime + CRT + carry — DONE (stage 3).** 2–3 primes, CRT + balanced carry
   on the CPU over shared buffers (§2.1). `--selftest negamul`. Gate met: matches
   GMP `x·y mod (b^N+1)` for k = 4…16, bases up to ~10^9 (multiply and square).
4. **Powering — DONE as 4a (`--selftest prp`); GEC + checkpoint still TODO (4b).**
   Full `a^(M−1)` chain, reduced to M only at the end. Gate met: residue + verdict
   match GMP for k ≤ 12. Gerbicz–Li every ~1000 squarings and a resumable residue
   journal (mirror `prp_test`) are **not yet built** — required before trusting
   long large-k runs.
5. **Benchmark + crossover — DONE as 5a.** `--selftest bench` reports ms/squaring,
   GPU vs one GMP core, with a GPU-vs-CPU-post breakdown. **Crossover at k=16** on
   the M2 Pro, 1.65× at k=18; recorded in [`../BENCHMARKS.md`](../BENCHMARKS.md).
   The optimisation that got there was all CPU-side post-processing (see §2.1
   finding). Remaining perf work (**5b**): four-step tiling (§4) + batching the
   primes into one command buffer to cut the GPU half and push the crossover lower.
6. **Integrate** behind the same CLI / journal contract as `prp_test`, so the
   sieve → PRP → prove pipeline is unchanged. Route small k to CPU/`prp_metal`,
   large k to this engine.

---

## 6. Honest payoff assessment

The Apple Silicon GPU is **integrated** and shares memory bandwidth with the CPU —
it is no discrete 8 GB GDDR6 card. The NTT is bandwidth-hungry. The stage-5a
measurements (BENCHMARKS.md) now replace the earlier guesses:

- **Crossover confirmed at k=16** on an M2 Pro (one k later than the RTX 4060's
  k=15 — integrated vs. discrete, as predicted), scaling to **1.65× at k=18**. The
  predicted "higher crossover, more modest speedup" held: in absolute ms/sq the
  4060 is ~8× faster, and the Mac speedup factor is smaller (it is measured against
  a strong single M2-P core, not an i5 core).
- The real win is as much **freeing the CPU** as raw throughput: run GPU PRP and
  CPU proof (`gp` APR-CL/ECPP) concurrently, or keep BOINC fed while hunting.
- The paradigm is validated: GPU ms/sq stays ~flat with k while the GMP curve rises
  steeply — the `O(N log N)`-per-squaring complexity class (CUDA doc §4) shows up in
  the data on both the 4060 and the M2 Pro.

What makes starting cheap on *this* platform specifically: Stage 1 is free (shared
oracle), Stage 2 is small (one modmul + one tiled transform), and unified memory
(§2.1) lets Stages 2–4 keep CRT/carry/GEC on the capable CPU half at zero copy
cost — so a correct, end-to-end engine exists long before any of it is tuned.

---

## 7. Build vs. reuse

The CUDA doc's §8 survey applies (genefer = closest prior art for `b^(2^n)+1`,
reuse its transform not its test; gpuOwl = reference for GEC/checkpoint). The
Apple-specific caveat: most prior GPU-FFT prime code is **CUDA/OpenCL and
float-IBDWT**, i.e. the one approach that cannot run on Metal (§1). The integer-NTT
Metal transform (§3–§4) is largely **in-house** work — the digit-base-`b`
negacyclic framing is what keeps that tractable, and `ntt_ref.cpp` keeps it
honest at every step.
