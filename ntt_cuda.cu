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

// Weight x_j (already reduced to [0,p)) by psi^j, in natural order.
__global__ void k_weight(const u64* __restrict__ xmod, const u64* __restrict__ wj,
                         u64* __restrict__ A, int N, u64 p) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    A[j] = d_mulmod(xmod[j], wj[j], p);
}

// Bit-reversal permutation: dst[j] = src[rev[j]]  (position j gets src[bitrev(j)]).
__global__ void k_bitperm(const u64* __restrict__ src, u64* __restrict__ dst,
                          const int* __restrict__ rev, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) dst[j] = src[rev[j]];
}

// One in-place radix-2 DIT stage. W is the twiddle table (omega^t, t=0..N-1);
// the twiddle for position t at this stage is W[stride*t], stride = N/len.
__global__ void k_stage(u64* __restrict__ A, const u64* __restrict__ W,
                        int N, u64 p, int stride, int half, int len) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= N / 2) return;
    int block = bid / half;
    int t = bid % half;
    int i = block * len + t;
    u64 w = W[stride * t];
    u64 u = A[i];
    u64 v = d_mulmod(A[i + half], w, p);
    A[i]        = (u + v) % p;
    A[i + half] = (u + p - v) % p;
}

__global__ void k_pointwise_sq(u64* __restrict__ C, const u64* __restrict__ A,
                               int N, u64 p) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) C[j] = d_mulmod(A[j], A[j], p);
}
__global__ void k_pointwise_mul(u64* __restrict__ C, const u64* __restrict__ A,
                                const u64* __restrict__ B, int N, u64 p) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) C[j] = d_mulmod(A[j], B[j], p);
}

// After inverse transform: scale by N^-1 and unweight by psi^-j.
__global__ void k_unweight(u64* __restrict__ C, const u64* __restrict__ wij,
                           int N, u64 p, u64 ninv) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    C[j] = d_mulmod(d_mulmod(C[j], ninv, p), wij[j], p);
}

// ---------------------------------------------------------------------------
// Per-prime plan (device buffers live for the program's lifetime)
// ---------------------------------------------------------------------------
struct PrimePlan {
    u64 p, psi, psi_inv, omega, omega_inv, ninv;
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
        std::vector<u64> W(N), Winv(N), Wj(N), Wij(N);
        W[0] = Winv[0] = Wj[0] = Wij[0] = 1;
        for (u64 j = 1; j < N; ++j) {
            W[j]    = h_mulmod(W[j - 1], pp.omega, q);
            Winv[j] = h_mulmod(Winv[j - 1], pp.omega_inv, q);
            Wj[j]   = h_mulmod(Wj[j - 1], pp.psi, q);
            Wij[j]  = h_mulmod(Wij[j - 1], pp.psi_inv, q);
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
    return P;
}

// forward/inverse NTT on device buffer `dBuf`, in natural order in and out.
// Mirrors the CPU reference: bit-reverse permute first, then the DIT stages.
static void ntt_device(u64* dBuf, const PrimePlan& pp, int N, bool inverse, int tpb) {
    const u64* Wtab = inverse ? pp.dWinv : pp.dW;
    int gN = (N + tpb - 1) / tpb;
    k_bitperm<<<gN, tpb>>>(dBuf, pp.dP, pp.dRev, N);   // dP = bitrev(dBuf)
    for (int len = 2; len <= N; len <<= 1) {
        int half = len / 2, stride = N / len;
        int blocks = (N / 2 + tpb - 1) / tpb;
        k_stage<<<blocks, tpb>>>(pp.dP, Wtab, N, pp.p, stride, half, len);
    }
    CUDA_OK(cudaMemcpy(dBuf, pp.dP, (size_t)N * sizeof(u64), cudaMemcpyDeviceToDevice));
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
        k_weight<<<gridN, tpb>>>(pp.dX, pp.dWj, pp.dA, N, p);
        ntt_device(pp.dA, pp, N, false, tpb);
        if (squaring) {
            k_pointwise_sq<<<gridN, tpb>>>(pp.dC, pp.dA, N, p);
        } else {
            for (int j = 0; j < N; ++j) hy[j] = to_mod(y[j]);
            CUDA_OK(cudaMemcpy(pp.dX, hy.data(), N * sizeof(u64), cudaMemcpyHostToDevice));
            k_weight<<<gridN, tpb>>>(pp.dX, pp.dWj, pp.dB, N, p);
            ntt_device(pp.dB, pp, N, false, tpb);
            k_pointwise_mul<<<gridN, tpb>>>(pp.dC, pp.dA, pp.dB, N, p);
        }
        ntt_device(pp.dC, pp, N, true, tpb);
        k_unweight<<<gridN, tpb>>>(pp.dC, pp.dWij, N, p, pp.ninv);
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
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nx = [&](const char* f) { if (i + 1 >= argc) { fprintf(stderr, "missing arg %s\n", f); exit(2);} return argv[++i]; };
        if      (s == "--k") k = atoi(nx("--k"));
        else if (s == "--b") b = strtoull(nx("--b"), nullptr, 10);
        else if (s == "--base") base_a = strtoul(nx("--base"), nullptr, 10);
        else if (s == "--selftest") selftest = true;
        else if (s == "--n") reps = atoi(nx("--n"));
        else { fprintf(stderr, "Usage: %s --k K --b B [--base A] [--selftest --n R]\n", argv[0]); return 2; }
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

    mpz_sub_ui(Emo, M, 1);
    mpz_powm(ref, a_mpz, Emo, M);
    mpz_t a_red; mpz_init(a_red); mpz_mod(a_red, a_mpz, BN1);
    std::vector<i64> acc = mpz_to_digits(a_red, N, b); mpz_clear(a_red);
    std::vector<i64> res(N, 0); res[0] = 1;
    size_t bits = mpz_sizeinbase(Emo, 2);
    for (size_t i = bits; i-- > 0;) {
        res = negamul_gpu(res, res, P);
        if (mpz_tstbit(Emo, (mp_bitcnt_t)i)) res = negamul_gpu(res, acc, P);
    }
    digits_to_mpz(got, res, b); mpz_mod(got, got, M);
    bool ok = (mpz_cmp(got, ref) == 0), prp = (mpz_cmp_ui(ref, 1) == 0);
    printf("  match vs GMP: %s   |   M is %s\n", ok ? "YES" : "NO", prp ? "probable prime" : "composite");
    mpz_clears(BN1, M, a_mpz, ref, got, Emo, nullptr);
    return ok ? 0 : 1;
}
