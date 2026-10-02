// prp_cuda.cu  --  CUDA port of prp_metal.mm: GPU-accelerated strong PRP test
// (Miller-Rabin) for M(b) = (b^N + 1)/2, N = 2^k.  Regime A: one GPU thread per
// candidate, CIOS Montgomery in 32-bit limbs. For NVIDIA GPUs (e.g. RTX 4060,
// Ada, sm_89).
//
// Same arithmetic as the (verified) Metal version; only the Metal API is replaced
// by the CUDA runtime. The CPU (GMP) prepares M and the Montgomery constants; the
// GPU runs the strong test to several bases (M-1 = d*2^s, then a^d and squarings),
// staying in the Montgomery domain. A CPU reference runs the same test and the two
// are compared for equality.
//
// NOTE: this file is built and verified ON the NVIDIA machine (the Mac used for
// development has no CUDA). It has not been compiled here -- build on the target.
//
// Build (Windows/Linux with CUDA Toolkit >= 11.8 and GMP):
//   cmake -G Ninja -B build && ninja -C build prp_cuda
//   (CMake enables the prp_cuda target only when a CUDA compiler is found.)
//
// Usage (same as prp_metal):
//   ./prp_cuda [--bases "3 5 7"] [--limit N] [--exp N] [candidate-file]
//
// Prototype limit: MAXNL limbs (see below). Like the Metal prototype this targets
// small/medium k with many candidates; large k (tens of thousands of digits) needs
// an FFT approach, not one-thread-per-candidate.

#include <cuda_runtime.h>
#include <gmp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>

using Clock = std::chrono::steady_clock;
static double secs_since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

// Fixed maximum limb count compiled into the kernel (per-thread arrays). Keeping
// it modest bounds per-thread local memory AND per-thread runtime, so a single
// kernel launch stays well under the Windows WDDM TDR watchdog (~2 s). 64 limbs
// ~= 2048-bit M. (A per-NL build via NVRTC, like Metal's, is a later optimization.)
#ifndef MAXNL
#define MAXNL 64
#endif

static void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// Device: CIOS Montgomery multiplication  out = a*b*R^-1 mod m, R = 2^(32*n).
// a, b, m < m, each with n (<= MAXNL) 32-bit limbs, little-endian. n0 = -m^-1 mod 2^32.
// ---------------------------------------------------------------------------
__device__ static void montmul(unsigned int* out,
                               const unsigned int* a,
                               const unsigned int* b,
                               const unsigned int* m,
                               unsigned int n0, int n) {
    unsigned int t[MAXNL + 2];
    for (int i = 0; i < n + 2; ++i) {
        t[i] = 0u;
    }
    for (int i = 0; i < n; ++i) {
        unsigned long long C = 0;
        for (int j = 0; j < n; ++j) {
            unsigned long long s = (unsigned long long)t[j]
                                 + (unsigned long long)a[j] * (unsigned long long)b[i] + C;
            t[j] = (unsigned int)s;
            C = s >> 32;
        }
        unsigned long long s = (unsigned long long)t[n] + C;
        t[n] = (unsigned int)s;
        t[n + 1] = (unsigned int)(s >> 32);

        unsigned int mm = (unsigned int)((unsigned long long)t[0] * (unsigned long long)n0);
        C = ((unsigned long long)t[0] + (unsigned long long)mm * (unsigned long long)m[0]) >> 32;
        for (int j = 1; j < n; ++j) {
            unsigned long long s2 = (unsigned long long)t[j]
                                  + (unsigned long long)mm * (unsigned long long)m[j] + C;
            t[j - 1] = (unsigned int)s2;
            C = s2 >> 32;
        }
        unsigned long long s3 = (unsigned long long)t[n] + C;
        t[n - 1] = (unsigned int)s3;
        t[n] = t[n + 1] + (unsigned int)(s3 >> 32);
        t[n + 1] = 0u;
    }

    // Final conditional subtraction: if t >= m, subtract m once.
    bool ge = (t[n] != 0u);
    if (!ge) {
        for (int j = n - 1; j >= 0; --j) {
            if (t[j] != m[j]) { ge = t[j] > m[j]; break; }
            if (j == 0) { ge = true; }   // t == m -> result 0
        }
    }
    if (ge) {
        unsigned long long borrow = 0;
        for (int j = 0; j < n; ++j) {
            unsigned long long d = (unsigned long long)t[j] - (unsigned long long)m[j] - borrow;
            out[j] = (unsigned int)d;
            borrow = (d >> 63) & 1ull;
        }
    } else {
        for (int j = 0; j < n; ++j) { out[j] = t[j]; }
    }
}

__device__ static bool bnequal(const unsigned int* a, const unsigned int* b, int n) {
    for (int j = 0; j < n; ++j) {
        if (a[j] != b[j]) { return false; }
    }
    return true;
}

// Count trailing / leading zeros of a 32-bit word (CUDA intrinsics).
__device__ static inline int ctz32(unsigned int x) { return __ffs((int)x) - 1; }   // x != 0
__device__ static inline int clz32(unsigned int x) { return __clz((int)x); }        // x != 0

// ---------------------------------------------------------------------------
// Strong Miller-Rabin to several bases; PRP only if all bases pass. Early-out on
// the first failing base. (Warps are SIMT, so the early-out helps less than on a
// CPU, but the test is still correct.)
// ---------------------------------------------------------------------------
__global__ void miller_rabin(const unsigned int* Ms,       // ncand * n
                             const unsigned int* oneMonts, // ncand * n  (R mod M)
                             const unsigned int* aMs,       // nbases * ncand * n  (a*R mod M)
                             const unsigned int* n0s,       // ncand
                             unsigned char* out,            // ncand
                             unsigned int ncand,
                             unsigned int offset,
                             unsigned int nbases,
                             int n) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cand = offset + gid;
    if (cand >= ncand) { return; }
    unsigned int base = cand * n;

    unsigned int m[MAXNL], om[MAXNL], r[MAXNL], am[MAXNL], mm1[MAXNL], tmp[MAXNL];
    for (int j = 0; j < n; ++j) {
        m[j]  = Ms[base + j];
        om[j] = oneMonts[base + j];
    }
    unsigned int n0 = n0s[cand];

    // mm1 = M - om : Montgomery form of M-1.
    {
        unsigned long long borrow = 0;
        for (int j = 0; j < n; ++j) {
            unsigned long long d = (unsigned long long)m[j] - (unsigned long long)om[j] - borrow;
            mm1[j] = (unsigned int)d;
            borrow = (d >> 63) & 1ull;
        }
    }

    // M-1 = d * 2^s. M odd -> E := M-1 is M with bit 0 cleared.
    int s;
    unsigned int low = m[0] & ~1u;
    if (low != 0u) {
        s = ctz32(low);
    } else {
        int lj = 1;
        while (lj < n && m[lj] == 0u) { ++lj; }
        s = lj * 32 + ctz32(m[lj]);
    }
    int topbit = -1;
    for (int j = n - 1; j >= 0 && topbit < 0; --j) {
        if (m[j] != 0u) { topbit = j * 32 + (31 - clz32(m[j])); }
    }

    bool prp = true;
    for (unsigned int bi = 0; bi < nbases && prp; ++bi) {
        unsigned int abase = (bi * ncand + cand) * n;
        for (int j = 0; j < n; ++j) {
            am[j] = aMs[abase + j];
            r[j]  = om[j];
        }
        // x = a^d mod M (Montgomery): scan bits of E from topbit down to s.
        for (int i = topbit; i >= s; --i) {
            montmul(tmp, r, r, m, n0, n);
            for (int j = 0; j < n; ++j) { r[j] = tmp[j]; }
            unsigned int bit = (m[(unsigned)i >> 5] >> ((unsigned)i & 31u)) & 1u;
            if (bit != 0u) {
                montmul(tmp, r, am, m, n0, n);
                for (int j = 0; j < n; ++j) { r[j] = tmp[j]; }
            }
        }
        bool pass = bnequal(r, om, n) || bnequal(r, mm1, n);
        for (int it = 1; it < s && !pass; ++it) {
            montmul(tmp, r, r, m, n0, n);
            for (int j = 0; j < n; ++j) { r[j] = tmp[j]; }
            if (bnequal(r, mm1, n)) { pass = true; break; }
            if (bnequal(r, om, n))  { break; }
        }
        if (!pass) { prp = false; }
    }
    out[cand] = prp ? 1 : 0;
}

// ---------------------------------------------------------------------------
// GMP helper: export x to n 32-bit limbs (little-endian).
// ---------------------------------------------------------------------------
static void to_limbs(const mpz_t x, uint32_t* dst, int n) {
    for (int j = 0; j < n; ++j) { dst[j] = 0u; }
    size_t count = 0;
    mpz_export(dst, &count, -1, 4, -1, 0, x);
    (void)count;
}

static uint32_t mont_n0(uint32_t m0) {
    uint32_t inv = 1u;
    for (int i = 0; i < 5; ++i) { inv = inv * (2u - m0 * inv); }
    return (uint32_t)(0u - inv);
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string candfile;
    long exp_override = -1;
    long limit = 0;
    std::vector<unsigned long> bases;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* n) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", n); exit(2); }
            return argv[++i];
        };
        if (a == "--exp") {
            exp_override = atol(next("--exp").c_str());
        } else if (a == "--bases") {
            std::istringstream is(next("--bases"));
            unsigned long v;
            while (is >> v) { bases.push_back(v); }
        } else if (a == "--limit") {
            limit = atol(next("--limit").c_str());
        } else if (a == "-h" || a == "--help") {
            printf("Usage: %s [--bases \"3 5 7\"] [--limit N] [--exp N] [candidate-file]\n", argv[0]);
            return 0;
        } else if (a[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", a.c_str());
            return 2;
        } else {
            candfile = a;
        }
    }
    if (candfile.empty()) { candfile = "kand.txt"; }
    if (bases.empty()) { bases = {3}; }

    std::ifstream in(candfile);
    if (!in) { fprintf(stderr, "Cannot open %s\n", candfile.c_str()); return 1; }
    unsigned long exp = (exp_override > 0) ? (unsigned long)exp_override : 0;
    std::vector<unsigned long> cands;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '#') {
            if (exp == 0) {
                size_t c = line.find('^');
                if (c != std::string::npos) { exp = strtoul(line.c_str() + c + 1, nullptr, 10); }
            }
            continue;
        }
        if (line.empty()) { continue; }
        unsigned long b = strtoul(line.c_str(), nullptr, 10);
        if (b > 0) { cands.push_back(b); }
    }
    in.close();
    if (exp == 0) { fprintf(stderr, "Exponent not readable -- pass --exp N.\n"); return 1; }
    if (cands.empty()) { fprintf(stderr, "No candidates.\n"); return 1; }
    if (limit > 0 && (long)cands.size() > limit) { cands.resize(limit); }

    const size_t ncand = cands.size();

    // ---- Size: n = limbs for the largest M ----
    mpz_t M, t;
    mpz_inits(M, t, nullptr);
    size_t max_bits = 0;
    for (unsigned long b : cands) {
        mpz_ui_pow_ui(M, b, exp);
        mpz_add_ui(M, M, 1);
        mpz_fdiv_q_2exp(M, M, 1);
        size_t bits = mpz_sizeinbase(M, 2);
        if (bits > max_bits) { max_bits = bits; }
    }
    const int n = (int)((max_bits + 31) / 32);
    if (n > MAXNL) {
        fprintf(stderr, "M too large for this prototype (%d limbs > MAXNL=%d). "
                        "Rebuild with a larger MAXNL or choose a smaller k/bmax.\n", n, MAXNL);
        return 1;
    }
    printf("Candidates: %zu, exponent N=%lu, largest M ~%zu bits -> n=%d limbs\n",
           ncand, exp, max_bits, n);

    // ---- CPU: prepare M, n0, R mod M, and a*R mod M per base (parallel) ----
    const uint32_t nbases = (uint32_t)bases.size();
    std::string basestr;
    for (size_t i = 0; i < bases.size(); ++i) {
        basestr += (i ? " " : "") + std::to_string(bases[i]);
    }
    std::vector<uint32_t> hostM((size_t)ncand * n);
    std::vector<uint32_t> hostOne((size_t)ncand * n);
    std::vector<uint32_t> hostAm((size_t)ncand * nbases * n);
    std::vector<uint32_t> hostN0(ncand);

    auto t_setup = Clock::now();
    int setup_threads = (int)std::thread::hardware_concurrency();
    if (setup_threads < 1) { setup_threads = 1; }
    std::atomic<size_t> setup_next{0};
    auto setupworker = [&]() {
        mpz_t lM, R, one_mont, am, abig;
        mpz_inits(lM, R, one_mont, am, abig, nullptr);
        mpz_setbit(R, (mp_bitcnt_t)n * 32);
        for (;;) {
            size_t idx = setup_next.fetch_add(1, std::memory_order_relaxed);
            if (idx >= ncand) { break; }
            mpz_ui_pow_ui(lM, cands[idx], exp);
            mpz_add_ui(lM, lM, 1);
            mpz_fdiv_q_2exp(lM, lM, 1);
            to_limbs(lM, &hostM[idx * n], n);
            hostN0[idx] = mont_n0(mpz_get_ui(lM) & 0xffffffffu);
            mpz_mod(one_mont, R, lM);
            to_limbs(one_mont, &hostOne[idx * n], n);
            for (uint32_t bi = 0; bi < nbases; ++bi) {
                mpz_set_ui(abig, bases[bi]);
                mpz_mul(am, abig, R);
                mpz_mod(am, am, lM);
                to_limbs(am, &hostAm[((size_t)bi * ncand + idx) * n], n);
            }
        }
        mpz_clears(lM, R, one_mont, am, abig, nullptr);
    };
    {
        std::vector<std::thread> sp;
        for (int tt = 0; tt < setup_threads; ++tt) { sp.emplace_back(setupworker); }
        for (auto& th : sp) { th.join(); }
    }
    double setup_secs = secs_since(t_setup);

    // ---- Upload to the GPU ----
    unsigned int *dMs, *dOne, *dAm, *dN0;
    unsigned char* dOut;
    cuda_check(cudaMalloc(&dMs,  hostM.size()   * 4), "malloc Ms");
    cuda_check(cudaMalloc(&dOne, hostOne.size() * 4), "malloc One");
    cuda_check(cudaMalloc(&dAm,  hostAm.size()  * 4), "malloc Am");
    cuda_check(cudaMalloc(&dN0,  hostN0.size()  * 4), "malloc N0");
    cuda_check(cudaMalloc(&dOut, ncand), "malloc Out");
    cuda_check(cudaMemcpy(dMs,  hostM.data(),   hostM.size()   * 4, cudaMemcpyHostToDevice), "cpy Ms");
    cuda_check(cudaMemcpy(dOne, hostOne.data(), hostOne.size() * 4, cudaMemcpyHostToDevice), "cpy One");
    cuda_check(cudaMemcpy(dAm,  hostAm.data(),  hostAm.size()  * 4, cudaMemcpyHostToDevice), "cpy Am");
    cuda_check(cudaMemcpy(dN0,  hostN0.data(),  hostN0.size()  * 4, cudaMemcpyHostToDevice), "cpy N0");
    cuda_check(cudaMemset(dOut, 0, ncand), "memset Out");

    // ---- Launch in chunks: each launch stays short (Windows TDR ~2 s) ----
    const unsigned int TPB = 128;
    const unsigned int CHUNK = 8192;
    auto t_gpu = Clock::now();
    for (unsigned int off = 0; off < (unsigned int)ncand; off += CHUNK) {
        unsigned int cnt = (unsigned int)std::min<size_t>(CHUNK, ncand - off);
        unsigned int blocks = (cnt + TPB - 1) / TPB;
        miller_rabin<<<blocks, TPB>>>(dMs, dOne, dAm, dN0, dOut,
                                      (unsigned int)ncand, off, nbases, n);
        cuda_check(cudaGetLastError(), "launch");
        cuda_check(cudaDeviceSynchronize(), "sync");
    }
    double gpu_secs = secs_since(t_gpu);

    std::vector<unsigned char> gpu_res(ncand);
    cuda_check(cudaMemcpy(gpu_res.data(), dOut, ncand, cudaMemcpyDeviceToHost), "cpy Out");
    cudaFree(dMs); cudaFree(dOne); cudaFree(dAm); cudaFree(dN0); cudaFree(dOut);

    // ---- CPU reference: same strong test over all bases, all cores ----
    std::vector<unsigned char> cpu_res(ncand, 0);
    int nthreads = (int)std::thread::hardware_concurrency();
    if (nthreads < 1) { nthreads = 1; }
    std::atomic<size_t> nexti{0};
    auto t_cpu = Clock::now();
    auto cpuworker = [&]() {
        mpz_t m, mm1, d, r, ab;
        mpz_inits(m, mm1, d, r, ab, nullptr);
        for (;;) {
            size_t i = nexti.fetch_add(1, std::memory_order_relaxed);
            if (i >= ncand) { break; }
            mpz_ui_pow_ui(m, cands[i], exp);
            mpz_add_ui(m, m, 1);
            mpz_fdiv_q_2exp(m, m, 1);
            mpz_sub_ui(mm1, m, 1);
            unsigned long s = mpz_scan1(mm1, 0);
            mpz_fdiv_q_2exp(d, mm1, s);
            bool prp = true;
            for (uint32_t bi = 0; bi < nbases && prp; ++bi) {
                mpz_set_ui(ab, bases[bi]);
                mpz_powm(r, ab, d, m);
                bool pass = (mpz_cmp_ui(r, 1) == 0) || (mpz_cmp(r, mm1) == 0);
                for (unsigned long it = 1; it < s && !pass; ++it) {
                    mpz_powm_ui(r, r, 2, m);
                    if (mpz_cmp(r, mm1) == 0) { pass = true; break; }
                    if (mpz_cmp_ui(r, 1) == 0) { break; }
                }
                if (!pass) { prp = false; }
            }
            cpu_res[i] = prp ? 1 : 0;
        }
        mpz_clears(m, mm1, d, r, ab, nullptr);
    };
    {
        std::vector<std::thread> pool;
        for (int tt = 0; tt < nthreads; ++tt) { pool.emplace_back(cpuworker); }
        for (auto& th : pool) { th.join(); }
    }
    double cpu_secs = secs_since(t_cpu);

    size_t mism = 0, gpu_prp = 0, cpu_prp = 0;
    for (size_t i = 0; i < ncand; ++i) {
        if (gpu_res[i]) { ++gpu_prp; }
        if (cpu_res[i]) { ++cpu_prp; }
        if (gpu_res[i] != cpu_res[i]) { ++mism; }
    }

    printf("\n--- Correctness ---\n");
    printf("GPU PRP: %zu   CPU PRP: %zu   mismatches: %zu  (%s)\n",
           gpu_prp, cpu_prp, mism, mism == 0 ? "OK" : "ERROR");

    printf("\n--- Performance (bases %s, strong Miller-Rabin) ---\n", basestr.c_str());
    printf("CPU setup (GMP, M + Montgomery constants): %.3f s\n", setup_secs);
    printf("GPU kernel (compute only):                 %.3f s\n", gpu_secs);
    printf("GPU total (setup + kernel):                %.3f s\n", setup_secs + gpu_secs);
    printf("CPU total (%2d threads, GMP):               %.3f s\n", nthreads, cpu_secs);
    if (gpu_secs > 0) { printf("\nSpeedup kernel vs CPU:   %.1fx\n", cpu_secs / gpu_secs); }
    if (setup_secs + gpu_secs > 0) { printf("Speedup total vs CPU:    %.1fx\n", cpu_secs / (setup_secs + gpu_secs)); }

    mpz_clears(M, t, nullptr);
    return 0;
}
