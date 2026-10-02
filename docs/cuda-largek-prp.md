# CUDA PRP for large k — design

**Goal.** A GPU PRP engine for `M(b) = (b^N + 1)/2`, `N = 2^k`, that beats the
20-thread GMP CPU path for **large k (k ≥ 16)** — numbers of ~10^5–10^6 decimal
digits. The existing `prp_cuda.cu` (one thread per candidate) is a medium-k
prototype and loses to the CPU at every size (see [`../BENCHMARKS.md`](../BENCHMARKS.md));
this is a separate engine, not an evolution of that kernel.

For the Apple Silicon (Metal) port of this same design, see the sibling
[`metal-largek-prp.md`](metal-largek-prp.md): identical math and shared
[`../ntt_ref.cpp`](../ntt_ref.cpp) oracle, different hardware layer (integer NTT
forced by the absence of fp64, 32-bit Montgomery modmul, four-step tiled
transform, unified-memory CPU-side CRT/carry).

Status: design only. Nothing here is built yet.

---

## 1. The scale problem

| k | N = 2^k | bits of M (b≈10³) | ≈ decimal digits | squarings per PRP (b≈10³) |
|---|--------:|------------------:|-----------------:|--------------------------:|
| 10 | 1 024   | ~10 200    | ~3 000     | ~10 000     |
| 16 | 65 536  | ~650 000   | ~197 000   | ~650 000    |
| 18 | 262 144 | ~2 600 000 | ~790 000   | ~2 600 000  |

(bits scale with `log2(b)`, so larger bases push these up further.)

A strong PRP test is `a^(M-1) ≡ 1 (mod M)` with `M-1 = (b^N - 1)/2`. That is
**~bits(M) modular squarings**, each a squaring of a number with ~bits(M) bits.
At k=16 that is ~650k squarings of a ~650k-bit number. Schoolbook O(n²) per
squaring is hopeless; we need **O(N log N) per squaring**, and the whole GPU
working on **one** squaring at a time.

Paradigm shift vs. the prototype:

| | prototype (`prp_cuda.cu`) | large-k engine (this doc) |
|---|---|---|
| parallelism | one thread = one candidate | whole GPU = one squaring |
| multiply | schoolbook CIOS O(n²) | NTT convolution O(N log N) |
| candidates | thousands in flight | one at a time, streamed |
| size cap | `MAXNL=64` (2048 bits) | millions of bits |

---

## 2. Core idea: digit-base-`b`, length-`N` negacyclic convolution

Write the residue in **base `b`** with exactly `N` digits:

```
X = Σ_{j=0}^{N-1} x_j · b^j ,   0 ≤ x_j < b
```

Because `b^N ≡ -1 (mod b^N+1)`, multiplying two such vectors and reducing mod
`b^N+1` is **exactly the length-N negacyclic (sign-twisted cyclic) convolution**
of the digit vectors, followed by carry propagation in base `b`. The wrap-around
term `b^N` folds back as `−1` into digit 0.

Two consequences make this a very good fit:

1. **Transform length is exactly `N = 2^k`** — a clean power of two, independent
   of `b`. For k=16 that is a 65 536-point transform. `b` only changes the digit
   size (bits per word), never the length.
2. **The modular reduction is free** — it is baked into the negacyclic transform
   (the `X^N = −1` wrap), no separate division by `M`.

### Working modulus: `b^N+1`, reduce to `M` only at the end

`M = (b^N+1)/2` is *not* of the clean `β^N±1` form the transform likes, but
`2M = b^N+1` is. So run the **entire** powering chain modulo `b^N+1` (the GFN
form), and reduce mod `M` only once, at the very end:

```
x ≡ y (mod b^N+1)  ⟹  x ≡ y (mod M)        since M | (b^N+1)
```

Compute `r = a^(M-1) mod (b^N+1)` with the negacyclic engine, then a single
conditional subtract (`r -= M` if `r ≥ M`) gives `a^(M-1) mod M`. Compare to 1.
This sidesteps the awkward modulus entirely.

---

## 3. Transform choice: integer NTT (not float FFT) on this GPU

The RTX 4060 (Ada, consumer) has **weak fp64** (~1:64 rate). Classic fp64 IBDWT
(gpuOwl/genefer on datacenter cards) would waste the 4060. Prefer an **integer
NTT**, which runs on the 4060's strong int32 path and is **exact** — no round-off
analysis, deterministic bit-for-bit results (a big correctness advantage for a
prime hunt).

- **Primes.** Use NTT-friendly primes `p ≡ 1 (mod 2N)` so a primitive `2N`-th root
  of unity (`ψ`, for the negacyclic twist) exists. Convolution coefficients reach
  `≈ N · (b−1)²`. For N=65536 and `b` up to ~10⁶ that is ~6·10¹⁷ (~59 bits), so
  use **three ~31-bit primes + CRT** (≈93 bits headroom) — generous and safe. For
  smaller `b`, two primes suffice; pick the count from `N` and `b` at setup.
- **Negacyclic via weighting.** Weight `x_j ← x_j · ψ^j` before the length-N NTT,
  pointwise square, inverse NTT, unweight by `ψ^{−j}`. This realises convolution
  mod `X^N+1` directly (the `b^N ≡ −1` reduction).
- **Carry propagation.** After the inverse transform, release base-`b` carries
  (parallel prefix / two-pass carry) to return to `0 ≤ x_j < b`.

Keep a **float FFT path as a future option** if we later target a card with real
fp64 throughput; the engine interface (one `square_mod(x)` primitive) should not
care which transform backend is used.

Per-residue memory: `N × 4 bytes × (#primes)` ≈ 65536·4·3 ≈ **0.75 MB** at k=16 —
trivial on 8 GB. Even k=18 is ~3 MB. Memory is not the constraint; modmul
throughput is.

---

## 4. Rough performance model (why this can win)

Per squaring ≈ `#primes × (2 NTTs + pointwise + carry)` ≈ `O(#primes · N log₂N)`
modular mults. At k=16: `3 · 65536 · 16 ≈ 3M` butterflies → ~10M int modmuls per
squaring (incl. inverse + carry). A 4060 sustains ~10¹⁰–10¹¹ such ops/s, i.e.
**~0.1–1 ms per squaring**, so ~650k squarings ≈ **1–10 min per candidate**.

The CPU baseline (`prp_test`, GMP FFT, 20 threads) is also O(N log N) but shares
20 cores across candidates. The GPU wins when its modmul throughput on one
transform beats 20 Raptor Cove cores on GMP's FFT — plausible for large N, where
the GPU's width is fully used. **This must be measured, not assumed**; the model
only says the approach is in the right complexity class, unlike the prototype.

The win is *large k*. For small k the transform is too short to fill the GPU and
the CPU (or even the old prototype) stays ahead — keep a size threshold that
routes small k to the CPU/prototype and large k to this engine.

---

## 5. Correctness and reliability (non-negotiable at 10⁵–10⁶ iterations)

Long GPU runs hit silent hardware errors; a prime hunt cannot ship an unverified
residue. Required, not optional:

- **Gerbicz–Li error check (GEC).** Maintain a running check product every ~1000
  squarings, verify every ~L iterations; on mismatch, roll back to the last good
  checkpoint. Catches virtually all transient errors cheaply (~0.2–1% overhead).
- **Checkpoint / resume.** Persist the residue + iteration counter periodically
  (mirror the existing `prp_test` journal discipline), so a multi-hour candidate
  survives interruption and BOINC co-scheduling.
- **Bit-exact oracle for small N.** For k = 4…8, the NTT residue
  `a^(M-1) mod M` must equal the GMP result **bit-for-bit**. `prp_test` / the
  prototype already produce the reference — wire this into the test suite before
  trusting any large k.
- **Round-off guard** only if a float backend is ever added (max convolution
  error < 0.5 ULP). The integer-NTT path does not need it.

---

## 6. GPU execution model

- One candidate occupies the whole GPU. A length-N NTT spans many blocks, so
  inter-block synchronisation is either **multiple kernel launches per transform
  stage** or a **cooperative-groups grid sync**; start with multi-launch
  (simpler, robust), optimise later.
- **Stream candidates**: double-buffer so candidate *i+1*'s host-side setup
  (compute `M`, weights, base conversion of `a`) overlaps candidate *i*'s kernel
  work.
- **Stay under the WDDM TDR watchdog (~2 s).** Each kernel launch must be a bounded
  slice of the squaring (one or few NTT stages), never the whole powering chain —
  this falls out naturally from the multi-launch structure.

---

## 7. Staged implementation plan

1. **Host reference.** CPU negacyclic-NTT squaring mod `b^N+1` in plain C++
   (int64/__int128 or GMP for CRT). Validate `a^(M-1) mod M` against `prp_test`
   for k=4…10. This nails the math before any CUDA.
2. **Single-prime GPU NTT.** Port the length-N NTT + pointwise + carry to CUDA for
   one prime, small N. Validate against the host reference.
3. **Multi-prime + CRT.** Add the 2–3 prime CRT reconstruction; push N up to k=16.
4. **Powering + GEC + checkpoint.** Full `a^(M-1)` chain with Gerbicz–Li and
   resumable checkpoints.
5. **Benchmark vs. CPU** at k=12, 14, 16, 18; find the crossover k where the GPU
   overtakes 20-thread GMP; set the routing threshold. Record in `BENCHMARKS.md`.
6. **Integrate** behind the same CLI/journal contract as `prp_test` so the
   pipeline (sieve → PRP → prove) is unchanged.

Each stage has a concrete pass/fail gate (match the reference / beat the previous
stage), so progress is measurable.

---

## 8. Build vs. reuse — be honest about the cost

A competitive large-number GPU transform is a **multi-month** effort. Proven prior
art exists and should be studied (and possibly reused) before reinventing:

- **genefer** (Y. Gallot) — purpose-built for Generalized Fermat Numbers
  `b^(2^n)+1`, GPU (OpenCL/CUDA), IBDWT + Gerbicz. Closest existing tool to our
  exact form. **Caveat:** it runs a GFN primality/Fermat test on `b^N+1`, whereas
  we need a base-`a` strong-PRP residue of `M = (b^N+1)/2`. Reuse its **transform
  engine**, not its test — the PRP harness stays ours.
- **gpuOwl** — Mersenne-focused (fp64 IBDWT + Gerbicz–Li); excellent reference for
  the GEC and checkpoint machinery.

Recommended path: build the **host NTT reference** (stage 1) ourselves regardless —
it is small and it is the correctness oracle — then decide at stage 2 whether to
lift genefer's CUDA transform or write our own integer-NTT kernels. The
digit-base-`b` negacyclic framing in §2 is what makes an in-house engine tractable
if we go that way.
