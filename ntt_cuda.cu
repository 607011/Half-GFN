// ntt_cuda.cu -- Stage 2 of the large-k GPU PRP engine (docs/cuda-largek-prp.md).
//
// Ports the negacyclic-NTT transform to CUDA and validates it, bit-for-bit,
// against GMP (same oracle as the CPU reference ntt_ref.cpp). The GPU does the
// per-prime transform work -- weight, NTT, pointwise, inverse NTT, unweight --
// one kernel launch per NTT stage (the robust multi-launch structure from the
// design, naturally under the WDDM TDR watchdog). The multi-prime CRT and the
// balanced base-b carry are still orchestrated on the host (already verified in
// Stage 1); moving them onto the GPU is Stage 3/4.
//
// This is a correctness milestone, not a performance target: device modmul is a
// plain 64-bit `% p` (primes < 2^31 so a*b < 2^62). Montgomery/Barrett come later.
//
// Usage:
//   ntt_cuda --k K --b B [--base A]          full a^(M-1) mod M, bit-exact vs GMP
//   ntt_cuda --selftest --k K --b B [--n R]  R random negamul/square checks vs GMP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <gmp.h>
#include <cuda_runtime.h>

using u64 = uint64_t;
using i64 = int64_t;

#define CUDA_OK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(3);} } while (0)

// ---------------------------------------------------------------------------
// Host modular arithmetic (plan/setup only)
// ---------------------------------------------------------------------------
static inline u64 h_mulmod(u64 a, u64 b, u64 p) { return (a * b) % p; }
static u64 h_powmod(u64 a, u64 e, u64 p) {
    u64 r = 1 % p; a %= p;
    while (e) { if (e & 1) r = h_mulmod(r, a, p); a = h_mulmod(a, a, p); e >>= 1; }
    return r;
}
static inline u64 h_modinv(u64 a, u64 p) { return h_powmod(a, p - 2, p); }

static i64 mpz_to_i64(const mpz_t x) {           // 32-bit-safe on Windows LLP64
    int neg = mpz_sgn(x) < 0;
    mpz_t a; mpz_init(a); mpz_abs(a, x);
    uint32_t low = (uint32_t)mpz_get_ui(a);
    mpz_fdiv_q_2exp(a, a, 32);
    uint32_t high = (uint32_t)mpz_get_ui(a);
    mpz_clear(a);
    u64 v = ((u64)high << 32) | low;
    return neg ? -(i64)v : (i64)v;
}

// ---------------------------------------------------------------------------
// Device modular arithmetic
// ---------------------------------------------------------------------------
__device__ __forceinline__ u64 d_mulmod(u64 a, u64 b, u64 p) { return (a * b) % p; }

// Montgomery arithmetic, R = 2^32 (primes < 2^31). Values are held as a*R mod p;
// n0 = -p^{-1} mod 2^32. REDC(T), T < p*2^32, returns T*R^-1 mod p.
__device__ __forceinline__ u64 mont_redc(u64 T, u64 p, uint32_t n0) {
    uint32_t m = (uint32_t)T * n0;              // mod 2^32
    u64 t = (T + (u64)m * p) >> 32;             // T + m*p < 2^64
    return (t >= p) ? t - p : t;
}
__device__ __forceinline__ u64 mont_mul(u64 a, u64 b, u64 p, uint32_t n0) {
    return mont_redc(a * b, p, n0);             // a,b < p < 2^31 -> a*b < 2^62
}

// Weight x_j (already reduced to [0,p)) by psi^j, converting into the Montgomery
// domain. wj holds psi^j in Montgomery form; R2 = 2^64 mod p converts x in.
__global__ void k_weight(const u64* __restrict__ xmod, const u64* __restrict__ wj,
                         u64* __restrict__ A, int N, u64 p, uint32_t n0, u64 R2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    A[j] = mont_mul(mont_mul(xmod[j], R2, p, n0), wj[j], p, n0);
}

// Bit-reversal permutation: dst[j] = src[rev[j]]  (position j gets src[bitrev(j)]).
__global__ void k_bitperm(const u64* __restrict__ src, u64* __restrict__ dst,
                          const int* __restrict__ rev, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) dst[j] = src[rev[j]];
}

// One in-place radix-2 DIT stage (Montgomery domain). W holds twiddles omega^t in
// Montgomery form; the twiddle for position t at this stage is W[stride*t].
__global__ void k_stage(u64* __restrict__ A, const u64* __restrict__ W,
                        int N, u64 p, uint32_t n0, int stride, int half, int len) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= N / 2) return;
    int block = bid / half;
    int t = bid % half;
    int i = block * len + t;
    u64 w = W[stride * t];
    u64 u = A[i];
    u64 v = mont_mul(A[i + half], w, p, n0);
    A[i]        = (u + v) % p;
    A[i + half] = (u + p - v) % p;
}

// Forward radix-2 DIF stage (Montgomery): natural-order in, bit-reversed out, so
// no separate bit-reversal permutation is needed. Stages run len = N down to 2.
__global__ void k_stage_dif(u64* __restrict__ A, const u64* __restrict__ W,
                            int N, u64 p, uint32_t n0, int stride, int half, int len) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= N / 2) return;
    int block = bid / half;
    int t = bid % half;
    int i = block * len + t;
    u64 u = A[i], v = A[i + half];
    A[i]        = (u + v) % p;
    A[i + half] = mont_mul((u + p - v) % p, W[stride * t], p, n0);
}

// Multi-stage shared-memory kernels: one block owns a contiguous TILE of the
// array and runs several stages locally (one launch instead of log2(TILE)). The
// small-stride stages (len <= TILE) are self-contained within a tile because the
// tile base is a multiple of TILE (hence of len). Twiddle index uses the full-N
// stride N/len and the within-half position t, exactly as the global kernels.
// DIT variant: runs stages len = 2 .. TILE (used after the large-stride stages).
__global__ void k_stages_shared_dit(u64* __restrict__ A, const u64* __restrict__ W,
                                    int N, u64 p, uint32_t n0, int TILE) {
    extern __shared__ u64 sh[];
    int base = blockIdx.x * TILE;
    for (int t = threadIdx.x; t < TILE; t += blockDim.x) sh[t] = A[base + t];
    __syncthreads();
    for (int len = 2; len <= TILE; len <<= 1) {
        int half = len / 2, stride = N / len;
        for (int bid = threadIdx.x; bid < TILE / 2; bid += blockDim.x) {
            int blk = bid / half, t = bid % half, i = blk * len + t;
            u64 u = sh[i], v = mont_mul(sh[i + half], W[stride * t], p, n0);
            sh[i] = (u + v) % p; sh[i + half] = (u + p - v) % p;
        }
        __syncthreads();
    }
    for (int t = threadIdx.x; t < TILE; t += blockDim.x) A[base + t] = sh[t];
}
// DIF variant: runs stages len = TILE .. 2 (used before the large-stride stages).
__global__ void k_stages_shared_dif(u64* __restrict__ A, const u64* __restrict__ W,
                                    int N, u64 p, uint32_t n0, int TILE) {
    extern __shared__ u64 sh[];
    int base = blockIdx.x * TILE;
    for (int t = threadIdx.x; t < TILE; t += blockDim.x) sh[t] = A[base + t];
    __syncthreads();
    for (int len = TILE; len >= 2; len >>= 1) {
        int half = len / 2, stride = N / len;
        for (int bid = threadIdx.x; bid < TILE / 2; bid += blockDim.x) {
            int blk = bid / half, t = bid % half, i = blk * len + t;
            u64 u = sh[i], v = sh[i + half];
            sh[i] = (u + v) % p; sh[i + half] = mont_mul((u + p - v) % p, W[stride * t], p, n0);
        }
        __syncthreads();
    }
    for (int t = threadIdx.x; t < TILE; t += blockDim.x) A[base + t] = sh[t];
}

__global__ void k_pointwise_sq(u64* __restrict__ C, const u64* __restrict__ A,
                               int N, u64 p, uint32_t n0) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) C[j] = mont_mul(A[j], A[j], p, n0);
}
__global__ void k_pointwise_mul(u64* __restrict__ C, const u64* __restrict__ A,
                                const u64* __restrict__ B, int N, u64 p, uint32_t n0) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) C[j] = mont_mul(A[j], B[j], p, n0);
}

// After inverse transform: scale by N^-1, unweight by psi^-j, convert OUT of
// Montgomery to a plain residue in [0,p) for the CRT.
__global__ void k_unweight(u64* __restrict__ C, const u64* __restrict__ wij,
                           int N, u64 p, uint32_t n0, u64 ninv_mont) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    u64 v = mont_mul(C[j], ninv_mont, p, n0);
    v = mont_mul(v, wij[j], p, n0);
    C[j] = mont_redc(v, p, n0);                 // Montgomery -> plain
}

// ===========================================================================
// Resident (on-GPU) path: CRT and balanced carry as kernels, so the whole
// powering loop stays on the device (no host round-trip per squaring).
// ===========================================================================

// Weight an i64 digit vector (balanced, possibly negative) by psi^j, into Montgomery.
__global__ void k_weight_i64(const i64* __restrict__ x, const u64* __restrict__ wj,
                             u64* __restrict__ A, int N, u64 p, uint32_t n0, u64 R2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    i64 r = x[j] % (i64)p; if (r < 0) r += (i64)p;
    A[j] = mont_mul(mont_mul((u64)r, R2, p, n0), wj[j], p, n0);
}

// Balanced multi-prime CRT per coefficient, pure 64-bit (nvcc has no __int128 on
// the MSVC host). Garner mixed-radix digits d[i], then balance only the TOP
// digit: since the prime product exceeds 2*|coeff|max, that yields the true
// signed value. Reconstruction c = sum d[t]*W_t is done wrapping mod 2^64 --
// valid because the true |coeff| < 2^62 fits i64, so the low 64 bits are exact.
#define HGFN_MAXP 8
__global__ void k_crt(u64* const* __restrict__ dC, const u64* __restrict__ primes,
                      const u64* __restrict__ ginv, int m, i64* __restrict__ out, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    u64 d[HGFN_MAXP];
    d[0] = dC[0][j] % primes[0];
    for (int i = 1; i < m; ++i) {
        u64 pI = primes[i];
        u64 sum = 0, w = 1;                          // xi = (sum_{t<i} d[t] W_t) mod p_i
        for (int t = 0; t < i; ++t) {
            sum = (sum + (d[t] % pI) * w) % pI;
            w = (w * (primes[t] % pI)) % pI;
        }
        i64 diff = (i64)(dC[i][j] % pI) - (i64)sum; diff %= (i64)pI; if (diff < 0) diff += (i64)pI;
        d[i] = ((u64)diff * ginv[i]) % pI;          // ginv[i] = inv(W_i mod p_i)
    }
    i64 dtop = (i64)d[m - 1];
    if (dtop > (i64)(primes[m - 1] / 2)) dtop -= (i64)primes[m - 1];   // balance top
    u64 w = 1; i64 c = 0;                            // reconstruct mod 2^64
    for (int t = 0; t < m - 1; ++t) { c += (i64)((u64)d[t] * w); w *= primes[t]; }
    c += (i64)((u64)dtop * w);
    out[j] = c;
}

// One balanced-carry iteration, phase 1: split e[j] into a balanced digit and
// the carry `hi` that must move to position j+1 (with the b^N=-1 wrap at top).
__global__ void k_carry_split(const i64* __restrict__ e, i64* __restrict__ digit,
                              i64* __restrict__ hi, int N, i64 b) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    i64 v = e[j];
    i64 rem = v % b; if (rem < 0) rem += b;
    if (rem > b / 2) rem -= b;                     // balanced digit in (-b/2, b/2]
    digit[j] = rem;
    hi[j] = (v - rem) / b;                          // exact
}
// Phase 2: e[j] = digit[j] + incoming carry (hi[j-1]); position 0 gets -hi[N-1].
__global__ void k_carry_combine(i64* __restrict__ e, const i64* __restrict__ digit,
                                const i64* __restrict__ hi, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    e[j] = digit[j] + (j == 0 ? -hi[N - 1] : hi[j - 1]);
}

__global__ void k_set_unit(i64* __restrict__ d, int N) {   // d = 1
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) d[j] = (j == 0) ? 1 : 0;
}

__global__ void k_any_nonzero(const i64* __restrict__ hi, int N, int* __restrict__ flag) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N && hi[j] != 0) *flag = 1;        // benign race: all writers write 1
}

// ---------------------------------------------------------------------------
// Per-prime plan (device buffers live for the program's lifetime)
// ---------------------------------------------------------------------------
struct PrimePlan {
    u64 p, psi, psi_inv, omega, omega_inv, ninv;
    uint32_t n0;            // -p^-1 mod 2^32  (Montgomery)
    u64 R2;                 // 2^64 mod p      (Montgomery convert-in)
    u64 ninv_mont;          // ninv in Montgomery form
    u64 *dW = nullptr, *dWinv = nullptr;   // twiddle tables (device)
    u64 *dWj = nullptr, *dWij = nullptr;   // weight tables psi^j, psi^-j (device)
    int *dRev = nullptr;                    // bit-reversal permutation (device)
    u64 *dA = nullptr, *dB = nullptr, *dC = nullptr, *dX = nullptr; // scratch (device)
    u64 *dP = nullptr;                      // permutation scratch for ntt (device)
};

struct Plan {
    int k; u64 N, b;
    std::vector<u64> primes;
    std::vector<PrimePlan> pp;
    // resident-path device state
    u64*  dPrimes = nullptr;     // primes[m]
    u64*  dGinv   = nullptr;     // Garner inverses inv((p0..p_{i-1}) mod p_i)
    u64** dCptrs  = nullptr;     // device array of the m per-prime dC pointers
    i64*  dCoef   = nullptr;     // CRT output (signed coefficients)
    i64*  dDigit  = nullptr;     // carry scratch
    i64*  dHi     = nullptr;     // carry scratch
    int*  dFlag   = nullptr;     // carry-convergence flag (device)
};

static bool is_prime_u64(u64 n) {
    mpz_t z; mpz_init_set_ui(z, (unsigned long)n);
    int r = mpz_probab_prime_p(z, 40); mpz_clear(z); return r != 0;
}
static u64 find_psi(u64 p, u64 N) {
    u64 e = (p - 1) / (2 * N);
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull ^ p);
    for (;;) { u64 g = 2 + rng() % (p - 3); u64 r = h_powmod(g, e, p);
               if (h_powmod(r, N, p) == p - 1) return r; }
}

static int bitrev(int x, int bits) { int r = 0; for (int i = 0; i < bits; ++i) { r = (r << 1) | (x & 1); x >>= 1; } return r; }

// Montgomery setup for an odd prime p < 2^31, R = 2^32.
static uint32_t mont_n0(u64 p) {                 // -p^-1 mod 2^32 (Newton)
    uint32_t inv = 1;
    for (int i = 0; i < 5; ++i) inv *= (uint32_t)(2 - (uint32_t)p * inv);
    return (uint32_t)(0u - inv);
}
static u64 mont_R2(u64 p) {                        // 2^64 mod p
    u64 Rmod = ((unsigned long long)1 << 32) % p;  // 2^32 mod p
    return h_mulmod(Rmod, Rmod, p);
}
static inline u64 to_mont(u64 x, u64 p) { return h_mulmod(x, ((unsigned long long)1 << 32) % p, p); }

static Plan make_plan(int k, u64 b) {
    Plan P; P.k = k; P.N = (u64)1 << k; P.b = b;
    const u64 N = P.N;

    // choose primes p = j*2N+1 until product > 2*N*(b-1)^2
    mpz_t bound, prod; mpz_init(bound); mpz_init_set_ui(prod, 1);
    mpz_set_ui(bound, (unsigned long)(b - 1)); mpz_mul(bound, bound, bound);
    mpz_mul_ui(bound, bound, (unsigned long)N); mpz_mul_ui(bound, bound, 2);
    const u64 step = 2 * N; u64 p = 1 + step;
    while (mpz_cmp(prod, bound) <= 0) {
        for (; p < (1u << 31); p += step) { if (b >= p) continue; if (is_prime_u64(p)) break; }
        if (p >= (1u << 31)) { fprintf(stderr, "out of NTT primes < 2^31\n"); exit(1); }
        P.primes.push_back(p); mpz_mul_ui(prod, prod, (unsigned long)p); p += step;
    }
    mpz_clear(bound); mpz_clear(prod);

    int bits = k;
    std::vector<int> rev(N); for (u64 j = 0; j < N; ++j) rev[j] = bitrev((int)j, bits);

    for (u64 q : P.primes) {
        PrimePlan pp; pp.p = q;
        pp.psi = find_psi(q, N); pp.psi_inv = h_modinv(pp.psi, q);
        pp.omega = h_mulmod(pp.psi, pp.psi, q); pp.omega_inv = h_modinv(pp.omega, q);
        pp.ninv = h_modinv(N % q, q);
        pp.n0 = mont_n0(q); pp.R2 = mont_R2(q);
        pp.ninv_mont = to_mont(pp.ninv, q);
        // twiddle / weight tables, stored in Montgomery form (value * 2^32 mod p)
        std::vector<u64> W(N), Winv(N), Wj(N), Wij(N);
        u64 w = 1, wi = 1, wj = 1, wij = 1;        // plain running powers
        for (u64 j = 0; j < N; ++j) {
            W[j] = to_mont(w, q); Winv[j] = to_mont(wi, q);
            Wj[j] = to_mont(wj, q); Wij[j] = to_mont(wij, q);
            w = h_mulmod(w, pp.omega, q);    wi  = h_mulmod(wi,  pp.omega_inv, q);
            wj = h_mulmod(wj, pp.psi, q);    wij = h_mulmod(wij, pp.psi_inv, q);
        }
        size_t nb = N * sizeof(u64);
        CUDA_OK(cudaMalloc(&pp.dW, nb));   CUDA_OK(cudaMemcpy(pp.dW, W.data(), nb, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMalloc(&pp.dWinv, nb));CUDA_OK(cudaMemcpy(pp.dWinv, Winv.data(), nb, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMalloc(&pp.dWj, nb));  CUDA_OK(cudaMemcpy(pp.dWj, Wj.data(), nb, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMalloc(&pp.dWij, nb)); CUDA_OK(cudaMemcpy(pp.dWij, Wij.data(), nb, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMalloc(&pp.dRev, N * sizeof(int)));
        CUDA_OK(cudaMemcpy(pp.dRev, rev.data(), N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMalloc(&pp.dA, nb)); CUDA_OK(cudaMalloc(&pp.dB, nb));
        CUDA_OK(cudaMalloc(&pp.dC, nb)); CUDA_OK(cudaMalloc(&pp.dX, nb));
        CUDA_OK(cudaMalloc(&pp.dP, nb));
        P.pp.push_back(pp);
    }

    // resident-path setup: primes, Garner inverses, dC pointer array, scratch
    const int m = (int)P.primes.size();
    std::vector<u64> ginv(m, 0);
    for (int i = 1; i < m; ++i) {
        u64 pi = P.primes[i];
        u64 prod = 1;                                    // (p0..p_{i-1}) mod p_i
        for (int t = 0; t < i; ++t) prod = h_mulmod(prod, P.primes[t] % pi, pi);
        ginv[i] = h_modinv(prod, pi);
    }
    CUDA_OK(cudaMalloc(&P.dPrimes, m * sizeof(u64)));
    CUDA_OK(cudaMemcpy(P.dPrimes, P.primes.data(), m * sizeof(u64), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&P.dGinv, m * sizeof(u64)));
    CUDA_OK(cudaMemcpy(P.dGinv, ginv.data(), m * sizeof(u64), cudaMemcpyHostToDevice));
    std::vector<u64*> cptrs(m);
    for (int i = 0; i < m; ++i) cptrs[i] = P.pp[i].dC;
    CUDA_OK(cudaMalloc(&P.dCptrs, m * sizeof(u64*)));
    CUDA_OK(cudaMemcpy(P.dCptrs, cptrs.data(), m * sizeof(u64*), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&P.dCoef,  N * sizeof(i64)));
    CUDA_OK(cudaMalloc(&P.dDigit, N * sizeof(i64)));
    CUDA_OK(cudaMalloc(&P.dHi,    N * sizeof(i64)));
    CUDA_OK(cudaMalloc(&P.dFlag,  sizeof(int)));
    return P;
}

// NTT on device buffer `dBuf` in place. Forward uses DIF (natural -> bit-reversed),
// inverse uses DIT (bit-reversed -> natural), so the pair round-trips to natural
// order with NO explicit bit-reversal and NO extra buffer copy. The pointwise step
// in between operates on bit-reversed data, which is fine (it is elementwise).
static void ntt_device(u64* dBuf, const PrimePlan& pp, int N, bool inverse, int tpb) {
    const int TILE = (N < 2048) ? N : 2048;        // intra-tile stages fold into 1 launch
    const int nblk_tile = N / TILE;
    const size_t shb = (size_t)TILE * sizeof(u64);
    const int blocks = (N / 2 + tpb - 1) / tpb;
    if (!inverse) {
        // DIF: large-stride stages (len > TILE) global, then intra-tile stages in shared mem
        for (int len = N; len > TILE; len >>= 1) {
            int half = len / 2, stride = N / len;
            k_stage_dif<<<blocks, tpb>>>(dBuf, pp.dW, N, pp.p, pp.n0, stride, half, len);
        }
        k_stages_shared_dif<<<nblk_tile, tpb, shb>>>(dBuf, pp.dW, N, pp.p, pp.n0, TILE);
    } else {
        // DIT: intra-tile stages in shared mem, then large-stride stages global
        k_stages_shared_dit<<<nblk_tile, tpb, shb>>>(dBuf, pp.dWinv, N, pp.p, pp.n0, TILE);
        for (int len = 2 * TILE; len <= N; len <<= 1) {
            int half = len / 2, stride = N / len;
            k_stage<<<blocks, tpb>>>(dBuf, pp.dWinv, N, pp.p, pp.n0, stride, half, len);
        }
    }
}

// Negacyclic multiply/square of signed digit vectors x,y (base b) -> signed
// coefficients mod b^N+1, per prime, combined by balanced CRT on the host.
static std::vector<i64> negamul_gpu(const std::vector<i64>& x,
                                    const std::vector<i64>& y, Plan& P) {
    const int N = (int)P.N; const u64 b = P.b;
    const bool squaring = (&x == &y);
    const int m = (int)P.primes.size();
    const int tpb = 256;
    const int gridN = (N + tpb - 1) / tpb;

    std::vector<std::vector<u64>> cres((size_t)N, std::vector<u64>(m));
    std::vector<u64> hx(N), hy(N), hc(N);

    for (int pi = 0; pi < m; ++pi) {
        PrimePlan& pp = P.pp[pi]; u64 p = pp.p;
        auto to_mod = [p](i64 d) -> u64 { i64 r = d % (i64)p; if (r < 0) r += (i64)p; return (u64)r; };
        for (int j = 0; j < N; ++j) hx[j] = to_mod(x[j]);
        CUDA_OK(cudaMemcpy(pp.dX, hx.data(), N * sizeof(u64), cudaMemcpyHostToDevice));
        k_weight<<<gridN, tpb>>>(pp.dX, pp.dWj, pp.dA, N, p, pp.n0, pp.R2);
        ntt_device(pp.dA, pp, N, false, tpb);
        if (squaring) {
            k_pointwise_sq<<<gridN, tpb>>>(pp.dC, pp.dA, N, p, pp.n0);
        } else {
            for (int j = 0; j < N; ++j) hy[j] = to_mod(y[j]);
            CUDA_OK(cudaMemcpy(pp.dX, hy.data(), N * sizeof(u64), cudaMemcpyHostToDevice));
            k_weight<<<gridN, tpb>>>(pp.dX, pp.dWj, pp.dB, N, p, pp.n0, pp.R2);
            ntt_device(pp.dB, pp, N, false, tpb);
            k_pointwise_mul<<<gridN, tpb>>>(pp.dC, pp.dA, pp.dB, N, p, pp.n0);
        }
        ntt_device(pp.dC, pp, N, true, tpb);
        k_unweight<<<gridN, tpb>>>(pp.dC, pp.dWij, N, p, pp.n0, pp.ninv_mont);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaMemcpy(hc.data(), pp.dC, N * sizeof(u64), cudaMemcpyDeviceToHost));
        for (int j = 0; j < N; ++j) cres[(size_t)j][pi] = hc[j];
    }

    // balanced CRT (host) -- fast path for <=2 primes, mpz Garner otherwise
    std::vector<i64> c(N);
    for (int j = 0; j < N; ++j) {
        const std::vector<u64>& r = cres[(size_t)j];
        if (m == 1) {
            u64 p0 = P.primes[0], v = r[0] % p0;
            c[j] = (v > p0 / 2) ? (i64)v - (i64)p0 : (i64)v;
        } else if (m == 2) {
            u64 p0 = P.primes[0], p1 = P.primes[1];
            u64 inv01 = h_modinv(p0 % p1, p1);
            i64 d = (i64)(r[1] % p1) - (i64)(r[0] % p0 % p1); d %= (i64)p1; if (d < 0) d += (i64)p1;
            u64 t = h_mulmod((u64)d, inv01, p1);
            u64 v = r[0] % p0 + p0 * t, Pp = p0 * p1;
            c[j] = (v > Pp / 2) ? (i64)v - (i64)Pp : (i64)v;
        } else {
            mpz_t x_, Macc, t_, P_, half;
            mpz_init_set_ui(x_, (unsigned long)(r[0] % P.primes[0]));
            mpz_init_set_ui(Macc, (unsigned long)P.primes[0]); mpz_init(t_);
            for (int i = 1; i < m; ++i) {
                u64 pi2 = P.primes[i];
                u64 xmod = mpz_fdiv_ui(x_, (unsigned long)pi2);
                u64 inv = h_modinv(mpz_fdiv_ui(Macc, (unsigned long)pi2), pi2);
                i64 dd = (i64)(r[i] % pi2) - (i64)xmod; dd %= (i64)pi2; if (dd < 0) dd += (i64)pi2;
                u64 tt = h_mulmod((u64)dd, inv, pi2);
                mpz_mul_ui(t_, Macc, (unsigned long)tt); mpz_add(x_, x_, t_);
                mpz_mul_ui(Macc, Macc, (unsigned long)pi2);
            }
            mpz_init(P_); mpz_set(P_, Macc); mpz_init(half); mpz_fdiv_q_ui(half, P_, 2);
            if (mpz_cmp(x_, half) > 0) mpz_sub(x_, x_, P_);
            c[j] = mpz_to_i64(x_);
            mpz_clear(x_); mpz_clear(Macc); mpz_clear(t_); mpz_clear(P_); mpz_clear(half);
        }
    }

    // balanced base-b carry with b^N = -1 wrap (host)
    const i64 bb = (i64)b, hlf = bb / 2;
    std::vector<i64> d = c;
    for (int guard = 0; guard < 128; ++guard) {
        i64 carry = 0;
        for (int j = 0; j < N; ++j) {
            i64 v = d[j] + carry; i64 rem = v % bb; if (rem < 0) rem += bb;
            if (rem > hlf) rem -= bb; carry = (v - rem) / bb; d[j] = rem;
        }
        if (carry == 0) break;
        d[0] -= carry;
    }
    return d;
}

// Resident negacyclic multiply/square: digit vectors live on the device as i64.
// dX, dY -> dOut, all device pointers; squaring when dX == dY. Nothing touches
// host memory. The per-prime transform reuses the same kernels as the host path.
static void negamul_resident(const i64* dX, const i64* dY, i64* dOut, Plan& P) {
    const int N = (int)P.N; const i64 b = (i64)P.b;
    const int m = (int)P.primes.size();
    const bool squaring = (dX == dY);
    const int tpb = 256, g = (N + tpb - 1) / tpb, gh = (N / 2 + tpb - 1) / tpb;
    (void)gh;
    for (int pi = 0; pi < m; ++pi) {
        PrimePlan& pp = P.pp[pi]; u64 p = pp.p;
        k_weight_i64<<<g, tpb>>>(dX, pp.dWj, pp.dA, N, p, pp.n0, pp.R2);
        ntt_device(pp.dA, pp, N, false, tpb);
        if (squaring) {
            k_pointwise_sq<<<g, tpb>>>(pp.dC, pp.dA, N, p, pp.n0);
        } else {
            k_weight_i64<<<g, tpb>>>(dY, pp.dWj, pp.dB, N, p, pp.n0, pp.R2);
            ntt_device(pp.dB, pp, N, false, tpb);
            k_pointwise_mul<<<g, tpb>>>(pp.dC, pp.dA, pp.dB, N, p, pp.n0);
        }
        ntt_device(pp.dC, pp, N, true, tpb);
        k_unweight<<<g, tpb>>>(pp.dC, pp.dWij, N, p, pp.n0, pp.ninv_mont);
    }
    k_crt<<<g, tpb>>>(P.dCptrs, P.dPrimes, P.dGinv, m, P.dCoef, N);
    // parallel balanced carry: iterate until no carry remains. Each pass moves
    // carries one position (with the b^N=-1 wrap); converges because it is just
    // normalising a fixed residue. Cap guards against a logic error.
    // Check convergence only every CHK passes: an extra (already-converged) pass
    // is a harmless no-op (combine adds zero), so batching just trims host syncs.
    const int cap = N + 64, CHK = 8;
    for (int it = 0; it < cap; ++it) {
        k_carry_split<<<g, tpb>>>(P.dCoef, P.dDigit, P.dHi, N, b);
        if (it % CHK == CHK - 1 || it == cap - 1) {
            CUDA_OK(cudaMemset(P.dFlag, 0, sizeof(int)));
            k_any_nonzero<<<g, tpb>>>(P.dHi, N, P.dFlag);
            int flag = 0;
            CUDA_OK(cudaMemcpy(&flag, P.dFlag, sizeof(int), cudaMemcpyDeviceToHost));
            if (!flag) break;                   // all e[j] already balanced digits
        }
        k_carry_combine<<<g, tpb>>>(P.dCoef, P.dDigit, P.dHi, N);
    }
    CUDA_OK(cudaMemcpy(dOut, P.dCoef, (size_t)N * sizeof(i64), cudaMemcpyDeviceToDevice));
}

// Checkpoint format: a small header (identifying the exact problem) + the next
// exponent bit to process + the N i64 residue digits. Written atomically (tmp +
// rename) so a crash mid-write cannot corrupt a good checkpoint.
static const uint32_t CKPT_MAGIC = 0x48474E31u;   // "HGN1"
struct CkptHdr { uint32_t magic, k, base; u64 b; int64_t nextbit; int64_t N; };

static void ckpt_save(const char* path, int k, u64 b, uint32_t base,
                      int64_t nextbit, const std::vector<i64>& res) {
    std::string tmp = std::string(path) + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) { fprintf(stderr, "ckpt: cannot write %s\n", tmp.c_str()); return; }
    CkptHdr h{CKPT_MAGIC, (uint32_t)k, base, b, nextbit, (int64_t)res.size()};
    fwrite(&h, sizeof(h), 1, f);
    fwrite(res.data(), sizeof(i64), res.size(), f);
    fclose(f);
    remove(path); rename(tmp.c_str(), path);
}
static bool ckpt_load(const char* path, int k, u64 b, uint32_t base,
                      int N, std::vector<i64>& res, int64_t& nextbit) {
    FILE* f = fopen(path, "rb"); if (!f) return false;
    CkptHdr h{};
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != CKPT_MAGIC ||
        h.k != (uint32_t)k || h.b != b || h.base != base || h.N != N) { fclose(f); return false; }
    res.assign(N, 0);
    bool ok = fread(res.data(), sizeof(i64), N, f) == (size_t)N;
    fclose(f);
    nextbit = h.nextbit;
    return ok;
}

// Resident powering: res = a^E mod (b^N+1), returned as host digit vector.
// `time_ms` (if non-null) receives this run's GPU powering time. If `ckpt_path`
// is set, the residue is checkpointed every `ckpt_int` squarings and the run
// resumes from an existing matching checkpoint. `stopat` > 0 stops early after
// that many squarings (for testing resume); then `*stopped` is set and the
// returned vector is empty.
static std::vector<i64> powering_resident(const mpz_t a_red, const mpz_t E, Plan& P,
                                          double* time_ms, const char* ckpt_path,
                                          int ckpt_int, int stopat, uint32_t base,
                                          bool* stopped) {
    const int N = (int)P.N; const u64 b = P.b;
    const int tpb = 256, g = (N + tpb - 1) / tpb;
    if (stopped) *stopped = false;
    std::vector<i64> ad(N, 0);
    { mpz_t t, q; mpz_init_set(t, a_red); mpz_init(q);
      for (int j = 0; j < N; ++j) { u64 r = mpz_fdiv_q_ui(q, t, (unsigned long)b); ad[j] = (i64)r; mpz_set(t, q); }
      mpz_clear(t); mpz_clear(q); }
    i64 *dAcc, *dRes, *dTmp;
    CUDA_OK(cudaMalloc(&dAcc, N * sizeof(i64)));
    CUDA_OK(cudaMalloc(&dRes, N * sizeof(i64)));
    CUDA_OK(cudaMalloc(&dTmp, N * sizeof(i64)));
    CUDA_OK(cudaMemcpy(dAcc, ad.data(), N * sizeof(i64), cudaMemcpyHostToDevice));

    size_t bits = mpz_sizeinbase(E, 2);
    int64_t startbit = (int64_t)bits - 1;
    std::vector<i64> host(N);
    if (ckpt_path && ckpt_load(ckpt_path, P.k, b, base, N, host, startbit)) {
        CUDA_OK(cudaMemcpy(dRes, host.data(), N * sizeof(i64), cudaMemcpyHostToDevice));
        fprintf(stderr, "ckpt: resumed at bit %lld / %zu\n", (long long)startbit, bits);
    } else {
        k_set_unit<<<g, tpb>>>(dRes, N);
    }

    int sqcount = 0;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    CUDA_OK(cudaDeviceSynchronize()); cudaEventRecord(t0);
    for (int64_t i = startbit; i >= 0; --i) {
        negamul_resident(dRes, dRes, dTmp, P); std::swap(dRes, dTmp);        // square
        if (mpz_tstbit(E, (mp_bitcnt_t)i)) { negamul_resident(dRes, dAcc, dTmp, P); std::swap(dRes, dTmp); }
        ++sqcount;
        bool do_ckpt = ckpt_path && ckpt_int > 0 && (sqcount % ckpt_int == 0);
        bool do_stop = stopat > 0 && sqcount >= stopat;
        if (do_ckpt || do_stop) {
            CUDA_OK(cudaMemcpy(host.data(), dRes, N * sizeof(i64), cudaMemcpyDeviceToHost));
            ckpt_save(ckpt_path ? ckpt_path : "resume.ckpt", P.k, b, base, i - 1, host);
            if (do_stop) {
                cudaFree(dAcc); cudaFree(dRes); cudaFree(dTmp);
                if (stopped) *stopped = true;
                return {};
            }
        }
    }
    cudaEventRecord(t1); CUDA_OK(cudaEventSynchronize(t1));
    if (time_ms) { float ms = 0; cudaEventElapsedTime(&ms, t0, t1); *time_ms = ms; }
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    std::vector<i64> res(N);
    CUDA_OK(cudaMemcpy(res.data(), dRes, N * sizeof(i64), cudaMemcpyDeviceToHost));
    cudaFree(dAcc); cudaFree(dRes); cudaFree(dTmp);
    if (ckpt_path) remove(ckpt_path);        // completed -> drop the checkpoint
    return res;
}

// ---------------------------------------------------------------------------
static void digits_to_mpz(mpz_t out, const std::vector<i64>& d, u64 b) {
    mpz_set_ui(out, 0);
    for (size_t j = d.size(); j-- > 0;) {
        mpz_mul_ui(out, out, (unsigned long)b);
        if (d[j] >= 0) mpz_add_ui(out, out, (unsigned long)d[j]);
        else           mpz_sub_ui(out, out, (unsigned long)(-d[j]));
    }
}
static std::vector<i64> mpz_to_digits(const mpz_t x, u64 N, u64 b) {
    mpz_t t, q; mpz_init_set(t, x); mpz_init(q);
    std::vector<i64> d(N, 0);
    for (u64 j = 0; j < N; ++j) { u64 r = mpz_fdiv_q_ui(q, t, (unsigned long)b); d[j] = (i64)r; mpz_set(t, q); }
    mpz_clear(t); mpz_clear(q);
    return d;
}

int main(int argc, char** argv) {
    int k = -1; u64 b = 0; unsigned long base_a = 3; bool selftest = false; int reps = 20;
    bool resident = false; int bench = 0;
    const char* ckpt = nullptr; int ckpt_int = 50000, stopat = -1;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nx = [&](const char* f) { if (i + 1 >= argc) { fprintf(stderr, "missing arg %s\n", f); exit(2);} return argv[++i]; };
        if      (s == "--k") k = atoi(nx("--k"));
        else if (s == "--b") b = strtoull(nx("--b"), nullptr, 10);
        else if (s == "--base") base_a = strtoul(nx("--base"), nullptr, 10);
        else if (s == "--selftest") selftest = true;
        else if (s == "--resident") resident = true;
        else if (s == "--bench") bench = atoi(nx("--bench"));
        else if (s == "--ckpt") ckpt = nx("--ckpt");
        else if (s == "--ckpt-int") ckpt_int = atoi(nx("--ckpt-int"));
        else if (s == "--stopat") stopat = atoi(nx("--stopat"));
        else if (s == "--n") reps = atoi(nx("--n"));
        else { fprintf(stderr, "Usage: %s --k K --b B [--base A] [--resident [--ckpt FILE --ckpt-int N]] "
                               "[--selftest --n R] [--bench S]\n", argv[0]); return 2; }
    }
    if (k < 1 || b < 3 || (b % 2) == 0) { fprintf(stderr, "Need --k>=1 and odd --b>=3.\n"); return 2; }

    Plan P = make_plan(k, b);
    const u64 N = P.N;
    mpz_t BN1, M, a_mpz, ref, got, Emo;
    mpz_inits(BN1, M, a_mpz, ref, got, Emo, nullptr);
    mpz_ui_pow_ui(BN1, (unsigned long)b, (unsigned long)N); mpz_add_ui(BN1, BN1, 1);
    mpz_fdiv_q_ui(M, BN1, 2); mpz_set_ui(a_mpz, base_a);
    printf("k=%d N=%llu b=%llu base=%lu |M|=%zu bits primes=%zu (GPU transform)\n",
           k, (unsigned long long)N, (unsigned long long)b, base_a, mpz_sizeinbase(M, 2), P.primes.size());

    if (selftest) {
        std::mt19937_64 rng(12345 + (u64)k * 1000 + b);
        mpz_t X, Y, Z, Zr; mpz_inits(X, Y, Z, Zr, nullptr);
        int fails = 0;
        for (int r = 0; r < reps; ++r) {
            std::vector<i64> x(N), y(N);
            for (u64 j = 0; j < N; ++j) { x[j] = rng() % b; y[j] = rng() % b; }
            std::vector<i64> z = negamul_gpu(x, y, P);
            digits_to_mpz(X, x, b); digits_to_mpz(Y, y, b);
            mpz_mul(Z, X, Y); mpz_mod(Z, Z, BN1); digits_to_mpz(Zr, z, b); mpz_mod(Zr, Zr, BN1);
            if (mpz_cmp(Z, Zr) != 0) { if (++fails <= 3) printf("  MISMATCH mul rep %d\n", r); }
            std::vector<i64> sq = negamul_gpu(x, x, P);
            mpz_mul(Z, X, X); mpz_mod(Z, Z, BN1); digits_to_mpz(Zr, sq, b); mpz_mod(Zr, Zr, BN1);
            if (mpz_cmp(Z, Zr) != 0) { if (++fails <= 3) printf("  MISMATCH sq rep %d\n", r); }
        }
        mpz_clears(X, Y, Z, Zr, nullptr);
        printf("selftest: %d reps x2 ops -> %s\n", reps, fails == 0 ? "ALL OK" : "FAILURES");
        return fails == 0 ? 0 : 1;
    }

    if (bench > 0) {
        // GPU: time `bench` resident squarings of a random residue
        const int Ni = (int)N, tpb = 256, g = (Ni + tpb - 1) / tpb;
        std::vector<i64> rd(Ni); std::mt19937_64 rng(99);
        for (int j = 0; j < Ni; ++j) rd[j] = (i64)(rng() % b);
        i64 *dR, *dT; CUDA_OK(cudaMalloc(&dR, Ni * sizeof(i64))); CUDA_OK(cudaMalloc(&dT, Ni * sizeof(i64)));
        CUDA_OK(cudaMemcpy(dR, rd.data(), Ni * sizeof(i64), cudaMemcpyHostToDevice));
        negamul_resident(dR, dR, dT, P); std::swap(dR, dT);          // warm-up
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        CUDA_OK(cudaDeviceSynchronize()); cudaEventRecord(t0);
        for (int s = 0; s < bench; ++s) { negamul_resident(dR, dR, dT, P); std::swap(dR, dT); }
        cudaEventRecord(t1); CUDA_OK(cudaEventSynchronize(t1));
        float gms = 0; cudaEventElapsedTime(&gms, t0, t1);
        cudaFree(dR); cudaFree(dT); (void)g;

        // CPU: time `bench` single-core GMP squarings mod M
        mpz_t r; mpz_init(r); gmp_randstate_t st; gmp_randinit_default(st);
        mpz_urandomm(r, st, M);
        auto c0 = std::chrono::steady_clock::now();
        for (int s = 0; s < bench; ++s) { mpz_mul(r, r, r); mpz_mod(r, r, M); }
        auto c1 = std::chrono::steady_clock::now();
        double cms = std::chrono::duration<double, std::milli>(c1 - c0).count();
        mpz_clear(r); gmp_randclear(st);

        printf("  bench %d squarings:  GPU %.4f ms/sq   CPU(1 core GMP) %.4f ms/sq   speedup %.2fx\n",
               bench, gms / bench, cms / bench, (cms / bench) / (gms / bench));
        mpz_clears(BN1, M, a_mpz, ref, got, Emo, nullptr);
        return 0;
    }

    mpz_sub_ui(Emo, M, 1);
    mpz_powm(ref, a_mpz, Emo, M);
    mpz_t a_red; mpz_init(a_red); mpz_mod(a_red, a_mpz, BN1);
    std::vector<i64> res;
    double tms = 0;
    if (resident) {
        bool stopped = false;
        res = powering_resident(a_red, Emo, P, &tms, ckpt, ckpt_int, stopat, (uint32_t)base_a, &stopped);
        if (stopped) {
            printf("  stopped after %d squarings; checkpoint written%s\n", stopat,
                   ckpt ? "" : " (resume.ckpt)");
            mpz_clear(a_red); mpz_clears(BN1, M, a_mpz, ref, got, Emo, nullptr);
            return 0;
        }
    } else {
        std::vector<i64> acc = mpz_to_digits(a_red, N, b);
        res.assign(N, 0); res[0] = 1;
        size_t bits = mpz_sizeinbase(Emo, 2);
        for (size_t i = bits; i-- > 0;) {
            res = negamul_gpu(res, res, P);
            if (mpz_tstbit(Emo, (mp_bitcnt_t)i)) res = negamul_gpu(res, acc, P);
        }
    }
    mpz_clear(a_red);
    digits_to_mpz(got, res, b); mpz_mod(got, got, M);
    bool ok = (mpz_cmp(got, ref) == 0), prp = (mpz_cmp_ui(ref, 1) == 0);
    printf("  match vs GMP: %s   |   M is %s%s", ok ? "YES" : "NO",
           prp ? "probable prime" : "composite",
           resident ? "" : "\n");
    if (resident) {
        size_t sq = mpz_sizeinbase(Emo, 2);
        printf("   |   %s: %.1f ms  (%zu squarings, %.3f ms/sq)\n",
               "GPU resident", tms, sq, tms / (double)sq);
    }
    mpz_clears(BN1, M, a_mpz, ref, got, Emo, nullptr);
    return ok ? 0 : 1;
}
