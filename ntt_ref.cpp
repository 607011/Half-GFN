// ntt_ref.cpp -- Stage 1 CPU reference for the large-k GPU PRP engine.
//
// Validates the core math from docs/cuda-largek-prp.md on the CPU, before any
// CUDA/Metal is written:
//
//   * residue in base b, exactly N = 2^k digits;
//   * multiply/square mod (b^N + 1) == length-N NEGACYCLIC convolution of the
//     digit vectors (because b^N == -1 mod (b^N+1)), realised via an integer
//     NTT with a 2N-th-root "right-angle" weighting;
//   * exact coefficients recovered by multi-prime CRT (balanced);
//   * base-b carry propagation with the b^N = -1 wrap;
//   * the whole PRP exponentiation a^(M-1) is run mod (b^N+1), and reduced to
//     M = (b^N+1)/2 only at the very end.
//
// Ground truth is GMP (mpz_powm). This file is a CORRECTNESS ORACLE, not a
// performance target -- clarity over speed.
//
// Build: part of CMake (links GMP). Usage:
//   ntt_ref --k K --b B [--base A]        run one strong-PRP of M=(B^(2^K)+1)/2
//   ntt_ref --selftest --k K --b B [--n R]  R random negamul checks vs GMP
//
// The companion script tests/ntt_ref_sweep.sh drives k = 4..10.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <gmp.h>

using u64 = uint64_t;
using i64 = int64_t;

// ---------------------------------------------------------------------------
// 64-bit modular arithmetic for NTT primes p < 2^31 (so a*b < 2^62 fits u64).
// ---------------------------------------------------------------------------
static inline u64 mulmod(u64 a, u64 b, u64 p) { return (a * b) % p; }

static u64 powmod(u64 a, u64 e, u64 p) {
    u64 r = 1 % p;
    a %= p;
    while (e) {
        if (e & 1) r = mulmod(r, a, p);
        a = mulmod(a, a, p);
        e >>= 1;
    }
    return r;
}
static inline u64 modinv(u64 a, u64 p) { return powmod(a, p - 2, p); }  // p prime

// Extract a signed 64-bit value from an mpz. GMP's mpz_get_si/_ui return a
// 32-bit `long` on Windows (LLP64), so build the 64-bit value from two halves.
static i64 mpz_to_i64(const mpz_t x) {
    int neg = mpz_sgn(x) < 0;
    mpz_t a; mpz_init(a); mpz_abs(a, x);
    uint32_t low = (uint32_t)mpz_get_ui(a);   // low 32 bits
    mpz_fdiv_q_2exp(a, a, 32);
    uint32_t high = (uint32_t)mpz_get_ui(a);
    mpz_clear(a);
    uint64_t v = ((uint64_t)high << 32) | low;
    return neg ? -(i64)v : (i64)v;
}

// ---------------------------------------------------------------------------
// NTT prime selection: primes p = j*2N + 1 (so a primitive 2N-th root exists),
// p < 2^31, collected until the product exceeds 2*N*b^2 (room for signed,
// balanced convolution coefficients |c| <= N*(b-1)^2).
// ---------------------------------------------------------------------------
static bool is_prime_u64(u64 n) {
    mpz_t z; mpz_init_set_ui(z, (unsigned long)n);
    int r = mpz_probab_prime_p(z, 40);
    mpz_clear(z);
    return r != 0;
}

struct Plan {
    int k = 0;
    u64 N = 0;          // 2^k
    u64 b = 0;          // digit base
    std::vector<u64> primes;
    std::vector<u64> psi;     // primitive 2N-th root per prime  (psi^N = p-1)
    std::vector<u64> psi_inv; // modinv(psi)
    std::vector<u64> omega;   // psi^2 : primitive N-th root
    std::vector<u64> omega_inv;
    std::vector<u64> ninv;    // modinv(N)
    // per-prime weight tables psi^j and psi^-j, j = 0..N-1
    std::vector<std::vector<u64>> wj, wij;
};

// A 2N-th root of unity r with r^N == -1 (i.e. order exactly 2N).
static u64 find_psi(u64 p, u64 N) {
    u64 e = (p - 1) / (2 * N);
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull ^ p);
    for (;;) {
        u64 g = 2 + rng() % (p - 3);
        u64 r = powmod(g, e, p);
        if (powmod(r, N, p) == p - 1) return r;  // order divisible by 2N, not N
    }
}

static Plan make_plan(int k, u64 b) {
    Plan P;
    P.k = k; P.N = (u64)1 << k; P.b = b;
    const u64 N = P.N;

    // bound = 2 * N * (b-1)^2  (use mpz to avoid overflow)
    mpz_t bound, prod, tmp;
    mpz_init(bound); mpz_init(tmp);
    mpz_init_set_ui(prod, 1);
    mpz_set_ui(bound, (unsigned long)(b - 1));
    mpz_mul(bound, bound, bound);        // (b-1)^2
    mpz_mul_ui(bound, bound, (unsigned long)N);  // N(b-1)^2
    mpz_mul_ui(bound, bound, 2);         // *2 for the sign

    // scan p = j*2N + 1 upward, p < 2^31, keep primes until prod > bound
    const u64 step = 2 * N;
    u64 p = 1 + step;
    while (mpz_cmp(prod, bound) <= 0) {
        for (; p < (1u << 31); p += step) {
            if (b >= p) continue;           // need digits < p
            if (is_prime_u64(p)) break;
        }
        if (p >= (1u << 31)) {
            fprintf(stderr, "No more NTT primes < 2^31 for N=%llu; k/b too large "
                            "for this reference.\n", (unsigned long long)N);
            exit(1);
        }
        P.primes.push_back(p);
        mpz_mul_ui(prod, prod, (unsigned long)p);
        p += step;
    }
    mpz_clear(bound); mpz_clear(prod); mpz_clear(tmp);

    for (u64 q : P.primes) {
        u64 ps = find_psi(q, N);
        P.psi.push_back(ps);
        P.psi_inv.push_back(modinv(ps, q));
        u64 om = mulmod(ps, ps, q);
        P.omega.push_back(om);
        P.omega_inv.push_back(modinv(om, q));
        P.ninv.push_back(modinv(N % q, q));
        std::vector<u64> wj(N), wij(N);
        wj[0] = 1; wij[0] = 1;
        for (u64 j = 1; j < N; ++j) {
            wj[j]  = mulmod(wj[j - 1], ps, q);
            wij[j] = mulmod(wij[j - 1], P.psi_inv.back(), q);
        }
        P.wj.push_back(std::move(wj));
        P.wij.push_back(std::move(wij));
    }
    return P;
}

// ---------------------------------------------------------------------------
// Iterative radix-2 NTT. `root` is a primitive n-th root of unity mod p
// (pass omega for forward, omega_inv for inverse; caller scales by n^-1).
// ---------------------------------------------------------------------------
static void ntt(std::vector<u64>& a, u64 p, u64 root) {
    const int n = (int)a.size();
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        u64 wlen = powmod(root, (u64)(n / len), p);
        for (int i = 0; i < n; i += len) {
            u64 w = 1;
            for (int t = 0; t < len / 2; ++t) {
                u64 u = a[i + t];
                u64 v = mulmod(a[i + t + len / 2], w, p);
                a[i + t]           = (u + v) % p;
                a[i + t + len / 2] = (u + p - v) % p;
                w = mulmod(w, wlen, p);
            }
        }
    }
}

// Balanced CRT of residues r[i] mod primes[i] into a signed i64 coefficient.
// Fast u64 path for <=2 primes (product < 2^62); mpz fallback otherwise.
static i64 crt_balanced(const std::vector<u64>& r, const Plan& P) {
    const size_t m = P.primes.size();
    if (m == 1) {
        u64 p0 = P.primes[0];
        u64 v = r[0] % p0;
        return (v > p0 / 2) ? (i64)v - (i64)p0 : (i64)v;
    }
    if (m == 2) {
        u64 p0 = P.primes[0], p1 = P.primes[1];
        u64 inv01 = modinv(p0 % p1, p1);
        i64 d = (i64)(r[1] % p1) - (i64)(r[0] % p0 % p1);
        d %= (i64)p1; if (d < 0) d += (i64)p1;
        u64 t = mulmod((u64)d, inv01, p1);
        u64 v = r[0] % p0 + p0 * t;            // < p0*p1 < 2^62
        // P = p0*p1
        u64 Pp = p0 * p1;
        return (v > Pp / 2) ? (i64)v - (i64)Pp : (i64)v;
    }
    // >=3 primes: exact mpz Garner, then balance.
    mpz_t x, Macc, t, P_, half;
    mpz_init_set_ui(x, (unsigned long)(r[0] % P.primes[0]));
    mpz_init_set_ui(Macc, (unsigned long)P.primes[0]);
    mpz_init(t);
    for (size_t i = 1; i < m; ++i) {
        u64 pi = P.primes[i];
        u64 xmod = mpz_fdiv_ui(x, (unsigned long)pi);
        u64 inv = modinv(mpz_fdiv_ui(Macc, (unsigned long)pi), pi);
        i64 dd = (i64)(r[i] % pi) - (i64)xmod;
        dd %= (i64)pi; if (dd < 0) dd += (i64)pi;
        u64 tt = mulmod((u64)dd, inv, pi);
        mpz_mul_ui(t, Macc, (unsigned long)tt);
        mpz_add(x, x, t);
        mpz_mul_ui(Macc, Macc, (unsigned long)pi);
    }
    mpz_init(P_); mpz_set(P_, Macc);
    mpz_init(half); mpz_fdiv_q_ui(half, P_, 2);
    if (mpz_cmp(x, half) > 0) { mpz_sub(x, x, P_); }
    i64 out = mpz_to_i64(x);   // |coeff| <= N(b-1)^2 < 2^62; 32-bit-safe extract
    mpz_clear(x); mpz_clear(Macc); mpz_clear(t); mpz_clear(P_); mpz_clear(half);
    return out;
}

// ---------------------------------------------------------------------------
// Negacyclic multiply of digit vectors x,y (base b, length N) -> digits mod b^N+1.
// For squaring pass y == x.
// ---------------------------------------------------------------------------
static std::vector<i64> negamul(const std::vector<i64>& x,
                                const std::vector<i64>& y, const Plan& P) {
    const u64 N = P.N, b = P.b;
    const size_t m = P.primes.size();
    const bool squaring = (&x == &y);

    // signed coefficients, one CRT per position
    std::vector<std::vector<u64>> cres(N, std::vector<u64>(m));
    for (size_t pi = 0; pi < m; ++pi) {
        u64 p = P.primes[pi];
        auto to_mod = [p](i64 d) -> u64 {           // signed digit -> [0,p)
            i64 r = d % (i64)p; if (r < 0) r += (i64)p; return (u64)r;
        };
        std::vector<u64> A(N), B;
        for (u64 j = 0; j < N; ++j)
            A[j] = mulmod(to_mod(x[j]), P.wj[pi][j], p);    // weight by psi^j
        ntt(A, p, P.omega[pi]);
        if (!squaring) {
            B.assign(N, 0);
            for (u64 j = 0; j < N; ++j)
                B[j] = mulmod(to_mod(y[j]), P.wj[pi][j], p);
            ntt(B, p, P.omega[pi]);
        }
        std::vector<u64> C(N);
        for (u64 j = 0; j < N; ++j)
            C[j] = squaring ? mulmod(A[j], A[j], p) : mulmod(A[j], B[j], p);
        ntt(C, p, P.omega_inv[pi]);                         // inverse transform
        for (u64 j = 0; j < N; ++j) {
            u64 v = mulmod(C[j], P.ninv[pi], p);            // /N
            v = mulmod(v, P.wij[pi][j], p);                 // unweight by psi^-j
            cres[j][pi] = v;
        }
    }

    // CRT -> signed coefficients
    std::vector<i64> c(N);
    for (u64 j = 0; j < N; ++j) c[j] = crt_balanced(cres[j], P);

    // Balanced base-b carry with the b^N = -1 wrap. Balanced digits
    // (-b/2, b/2] are essential: the residue -1 == b^N is not representable
    // with N non-negative base-b digits (max is b^N-1), which makes a [0,b)
    // carry oscillate. Balanced, -1 is simply digit[0] = -1 and it converges.
    const i64 bb = (i64)b, half = bb / 2;   // b odd -> half = (b-1)/2
    std::vector<i64> d = c;
    for (int guard = 0; guard < 128; ++guard) {
        i64 carry = 0;
        for (u64 j = 0; j < N; ++j) {
            i64 v = d[j] + carry;
            i64 rem = v % bb; if (rem < 0) rem += bb;
            if (rem > half) rem -= bb;       // fold into (-b/2, b/2]
            carry = (v - rem) / bb;          // exact: rem == v (mod b)
            d[j] = rem;
        }
        if (carry == 0) break;
        d[0] -= carry;                       // b^N == -1
    }
    return d;
}

// ---------------------------------------------------------------------------
// digit <-> mpz helpers
// ---------------------------------------------------------------------------
static void digits_to_mpz(mpz_t out, const std::vector<i64>& d, u64 b) {
    mpz_set_ui(out, 0);
    for (size_t j = d.size(); j-- > 0;) {        // Horner, high -> low
        mpz_mul_ui(out, out, (unsigned long)b);
        if (d[j] >= 0) mpz_add_ui(out, out, (unsigned long)d[j]);
        else           mpz_sub_ui(out, out, (unsigned long)(-d[j]));  // balanced
    }
}
static std::vector<i64> mpz_to_digits(const mpz_t x, u64 N, u64 b) {
    mpz_t t, q; mpz_init_set(t, x); mpz_init(q);
    std::vector<i64> d(N, 0);
    for (u64 j = 0; j < N; ++j) {
        u64 r = mpz_fdiv_q_ui(q, t, (unsigned long)b);  // q = floor(t/b), returns t mod b
        d[j] = (i64)r;
        mpz_set(t, q);
    }
    mpz_clear(t); mpz_clear(q);
    return d;
}

// ---------------------------------------------------------------------------
// PRP: compute a^(M-1) mod M via negacyclic powering mod (b^N+1), reduce at end.
// Returns true iff residue == 1 (probable prime).  `ref_match` set from GMP.
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int k = -1; u64 b = 0; unsigned long base_a = 3;
    bool selftest = false; int reps = 20;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nx = [&](const char* f) { if (i + 1 >= argc) { fprintf(stderr, "missing arg for %s\n", f); exit(2);} return argv[++i]; };
        if      (s == "--k") k = atoi(nx("--k"));
        else if (s == "--b") b = strtoull(nx("--b"), nullptr, 10);
        else if (s == "--base") base_a = strtoul(nx("--base"), nullptr, 10);
        else if (s == "--selftest") selftest = true;
        else if (s == "--n") reps = atoi(nx("--n"));
        else { fprintf(stderr, "Usage: %s --k K --b B [--base A] [--selftest --n R]\n", argv[0]); return 2; }
    }
    if (k < 1 || b < 3 || (b % 2) == 0) {
        fprintf(stderr, "Need --k >= 1 and odd --b >= 3.\n"); return 2;
    }

    Plan P = make_plan(k, b);
    const u64 N = P.N;

    // modulus mpz: BN1 = b^N + 1,  M = BN1/2
    mpz_t BN1, M, a_mpz, ref, got, Emo;
    mpz_init(BN1); mpz_init(M); mpz_init(a_mpz); mpz_init(ref); mpz_init(got); mpz_init(Emo);
    mpz_ui_pow_ui(BN1, (unsigned long)b, (unsigned long)N);  mpz_add_ui(BN1, BN1, 1);
    mpz_fdiv_q_ui(M, BN1, 2);
    mpz_set_ui(a_mpz, base_a);

    printf("k=%d  N=%llu  b=%llu  base=%lu  |M|=%zu bits  NTT primes=%zu\n",
           k, (unsigned long long)N, (unsigned long long)b, base_a,
           mpz_sizeinbase(M, 2), P.primes.size());

    if (selftest) {
        // random residues: negamul vs GMP (x*y mod BN1)
        std::mt19937_64 rng(12345 + (u64)k * 1000 + b);
        mpz_t X, Y, Z, Zr;
        mpz_init(X); mpz_init(Y); mpz_init(Z); mpz_init(Zr);
        int fails = 0;
        for (int r = 0; r < reps; ++r) {
            std::vector<i64> x(N), y(N);
            for (u64 j = 0; j < N; ++j) { x[j] = rng() % b; y[j] = rng() % b; }
            std::vector<i64> z = negamul(x, y, P);
            digits_to_mpz(X, x, b); digits_to_mpz(Y, y, b);
            mpz_mul(Z, X, Y); mpz_mod(Z, Z, BN1);     // ground truth
            digits_to_mpz(Zr, z, b); mpz_mod(Zr, Zr, BN1);
            if (mpz_cmp(Z, Zr) != 0) { ++fails; if (fails <= 3) printf("  MISMATCH at rep %d\n", r); }
            // also exercise squaring path
            std::vector<i64> s = negamul(x, x, P);
            mpz_mul(Z, X, X); mpz_mod(Z, Z, BN1);
            digits_to_mpz(Zr, s, b); mpz_mod(Zr, Zr, BN1);
            if (mpz_cmp(Z, Zr) != 0) { ++fails; if (fails <= 3) printf("  SQ MISMATCH at rep %d\n", r); }
        }
        mpz_clear(X); mpz_clear(Y); mpz_clear(Z); mpz_clear(Zr);
        printf("selftest: %d reps x2 ops -> %s\n", reps, fails == 0 ? "ALL OK" : "FAILURES");
        return fails == 0 ? 0 : 1;
    }

    // E = M - 1 = (b^N - 1)/2
    mpz_sub_ui(Emo, M, 1);

    // GMP oracle
    mpz_powm(ref, a_mpz, Emo, M);

    // NTT powering: result = a^E mod BN1, left-to-right binary
    mpz_t a_red; mpz_init(a_red); mpz_mod(a_red, a_mpz, BN1);
    std::vector<i64> acc = mpz_to_digits(a_red, N, b);
    mpz_clear(a_red);
    // start result = 1
    std::vector<i64> res(N, 0); res[0] = 1;
    size_t bits = mpz_sizeinbase(Emo, 2);
    for (size_t i = bits; i-- > 0;) {
        res = negamul(res, res, P);                 // square
        if (mpz_tstbit(Emo, (mp_bitcnt_t)i)) res = negamul(res, acc, P);  // *a
    }
    digits_to_mpz(got, res, b);
    mpz_mod(got, got, M);                           // reduce to M only at the end

    bool ok = (mpz_cmp(got, ref) == 0);
    bool prp = (mpz_cmp_ui(ref, 1) == 0);
    gmp_printf("  NTT a^(M-1) mod M = %Zd\n  GMP reference      = %Zd\n", got, ref);
    printf("  match vs GMP: %s   |   M is %s\n",
           ok ? "YES" : "NO", prp ? "probable prime" : "composite");

    mpz_clears(BN1, M, a_mpz, ref, got, Emo, nullptr);
    return ok ? 0 : 1;
}
