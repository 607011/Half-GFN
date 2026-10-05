// prp_metal.mm  --  GPU-accelerated strong PRP test (Miller-Rabin) via
// Apple Metal (Regime A).
//
// Idea (see README): one GPU thread per candidate b. The CPU (GMP) computes
// M = (b^N+1)/2 and the Montgomery constants; the GPU does only the expensive
// part: the strong Miller-Rabin test to base a with Montgomery multiplication in
// 32-bit limbs (decompose M-1 = d*2^s, then a^d and the squarings).
//
// This is a PROTOTYPE for "many medium-sized candidates": the fixed limb count
// NL is determined at runtime from the largest M and compiled into the kernel
// source. For very large k (tens of thousands of digits) an FFT-based approach
// is needed instead (cf. genefer) -- deliberately not covered here.
//
// For a fair comparison it also runs the same strong test on the CPU (GMP,
// multithreaded) and checks the GPU result against the CPU result for equality.
//
// Build (macOS only): see CMakeLists.txt (target prp_metal), or:
//   clang++ -std=c++17 -O3 -ObjC++ -fobjc-arc prp_metal.mm \
//     -I$(brew --prefix gmp)/include -L$(brew --prefix gmp)/lib -lgmp \
//     -framework Metal -framework Foundation -o prp_metal

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

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

// ---------------------------------------------------------------------------
// The Metal kernel. NL (the limb count) is substituted at runtime via #define
// so all arrays are sized exactly (best GPU occupancy).
// Each thread tests one candidate with a strong Miller-Rabin test.
// ---------------------------------------------------------------------------
static const char* kKernelTemplate = R"METAL(
#include <metal_stdlib>
using namespace metal;

// CIOS Montgomery multiplication: out = a * b * R^-1 mod m,  R = 2^(32*NL).
// a, b, m < m. n0 = -m^-1 mod 2^32. All 32-bit limbs, little-endian.
static void montmul(thread uint* out,
                    thread const uint* a,
                    thread const uint* b,
                    thread const uint* m,
                    uint n0) {
    uint t[NL + 2];
    for (uint i = 0; i < NL + 2; ++i) {
        t[i] = 0u;
    }
    for (uint i = 0; i < NL; ++i) {
        // t += a * b[i]
        ulong C = 0;
        for (uint j = 0; j < NL; ++j) {
            ulong s = (ulong)t[j] + (ulong)a[j] * (ulong)b[i] + C;
            t[j] = (uint)s;
            C = s >> 32;
        }
        ulong s = (ulong)t[NL] + C;
        t[NL] = (uint)s;
        t[NL + 1] = (uint)(s >> 32);

        // choose mm so the lowest limb becomes 0, then >> 32
        uint mm = (uint)((ulong)t[0] * (ulong)n0);   // mod 2^32 implicit
        C = ((ulong)t[0] + (ulong)mm * (ulong)m[0]) >> 32;
        for (uint j = 1; j < NL; ++j) {
            ulong s2 = (ulong)t[j] + (ulong)mm * (ulong)m[j] + C;
            t[j - 1] = (uint)s2;
            C = s2 >> 32;
        }
        ulong s3 = (ulong)t[NL] + C;
        t[NL - 1] = (uint)s3;
        t[NL] = t[NL + 1] + (uint)(s3 >> 32);
        t[NL + 1] = 0u;
    }

    // Final conditional subtraction: if t >= m, subtract m once.
    bool ge = (t[NL] != 0u);
    if (!ge) {
        for (int j = NL - 1; j >= 0; --j) {
            if (t[j] != m[j]) {
                ge = t[j] > m[j];
                break;
            }
            if (j == 0) {
                ge = true;   // t == m  ->  result 0
            }
        }
    }
    if (ge) {
        ulong borrow = 0;
        for (uint j = 0; j < NL; ++j) {
            ulong d = (ulong)t[j] - (ulong)m[j] - borrow;
            out[j] = (uint)d;
            borrow = (d >> 63) & 1ul;   // 1 if it underflowed
        }
    } else {
        for (uint j = 0; j < NL; ++j) {
            out[j] = t[j];
        }
    }
}

// Equality of two NL-limb numbers.
static bool bnequal(thread const uint* a, thread const uint* b) {
    for (uint j = 0; j < NL; ++j) {
        if (a[j] != b[j]) { return false; }
    }
    return true;
}

// Strong Miller-Rabin test to several bases (via Montgomery). A candidate counts
// as PRP only if it passes ALL bases; if it fails one, we stop immediately (most
// candidates are composite and fail the first base -> the other bases cost almost
// nothing).
kernel void miller_rabin(device const uint*  Ms       [[buffer(0)]],   // ncand * NL
                         device const uint*  oneMonts [[buffer(1)]],   // ncand * NL  (R mod M)
                         device const uint*  aMs      [[buffer(2)]],   // nbases * ncand * NL  (a*R mod M)
                         device const uint*  n0s      [[buffer(3)]],   // ncand
                         device uchar*       out      [[buffer(4)]],   // ncand
                         constant uint&      ncand    [[buffer(5)]],
                         constant uint&      offset   [[buffer(6)]],   // first candidate of the block
                         constant uint&      nbases   [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
    const uint cand = offset + gid;
    if (cand >= ncand) {
        return;
    }
    const uint base = cand * NL;

    thread uint m[NL];     // modulus M
    thread uint om[NL];    // Montgomery form of 1    (= R mod M)
    thread uint r[NL];     // running value, in Montgomery form
    thread uint am[NL];    // Montgomery form of the current base (= a*R mod M)
    thread uint mm1[NL];   // Montgomery form of M-1  (= M - om)
    thread uint tmp[NL];
    for (uint j = 0; j < NL; ++j) {
        m[j]  = Ms[base + j];
        om[j] = oneMonts[base + j];
    }
    uint n0 = n0s[cand];

    // mm1 = M - om : Montgomery form of M-1, since (M-1)*R = -R (mod M).
    {
        ulong borrow = 0;
        for (uint j = 0; j < NL; ++j) {
            ulong d = (ulong)m[j] - (ulong)om[j] - borrow;
            mm1[j] = (uint)d;
            borrow = (d >> 63) & 1ul;
        }
    }

    // M-1 = d * 2^s. M odd -> E := M-1 is M with bit 0 cleared.
    // s = lowest set bit of E; topbit = highest bit of M.
    // (Depends only on M -> once per candidate, not per base.)
    uint s;
    uint low = m[0] & ~1u;             // bit 0 of E is 0
    if (low != 0u) {
        s = ctz(low);
    } else {
        uint lj = 1;
        while (lj < NL && m[lj] == 0u) { ++lj; }
        s = lj * 32 + ctz(m[lj]);
    }
    int topbit = -1;
    for (int j = NL - 1; j >= 0 && topbit < 0; --j) {
        if (m[j] != 0u) {
            topbit = j * 32 + (31 - (int)clz(m[j]));
        }
    }

    bool prp = true;
    for (uint bi = 0; bi < nbases && prp; ++bi) {
        const uint abase = (bi * ncand + cand) * NL;
        for (uint j = 0; j < NL; ++j) {
            am[j] = aMs[abase + j];
            r[j]  = om[j];             // start: Montgomery form of 1
        }

        // x = a^d mod M (Montgomery): scan bits of E from topbit down to s.
        for (int i = topbit; i >= (int)s; --i) {
            montmul(tmp, r, r, m, n0);
            for (uint j = 0; j < NL; ++j) { r[j] = tmp[j]; }
            uint bit = (m[(uint)i >> 5] >> ((uint)i & 31u)) & 1u;
            if (bit != 0u) {
                montmul(tmp, r, am, m, n0);
                for (uint j = 0; j < NL; ++j) { r[j] = tmp[j]; }
            }
        }

        // Strong test (in Montgomery form): x == 1 or x == M-1 ?
        bool pass = bnequal(r, om) || bnequal(r, mm1);
        for (uint it = 1; it < s && !pass; ++it) {
            montmul(tmp, r, r, m, n0);                 // x = x^2
            for (uint j = 0; j < NL; ++j) { r[j] = tmp[j]; }
            if (bnequal(r, mm1)) { pass = true; break; } // found -1
            if (bnequal(r, om))  { break; }              // 1 before -1 -> composite
        }
        if (!pass) {
            prp = false;   // this base witnesses: composite -> stop
        }
    }
    out[cand] = prp ? 1 : 0;
}
)METAL";

// ---------------------------------------------------------------------------
// GMP helper: export M to NL 32-bit limbs (little-endian).
// ---------------------------------------------------------------------------
static void to_limbs(const mpz_t x, uint32_t* dst, int NL) {
    for (int j = 0; j < NL; ++j) {
        dst[j] = 0u;
    }
    size_t count = 0;
    mpz_export(dst, &count, -1 /*least significant first*/, 4 /*word size*/,
               -1 /*little endian within word*/, 0, x);
    (void)count;  // stays <= NL, the rest is already 0
}

// n0 = -M^-1 mod 2^32  (Montgomery constant, only the lowest limb is needed)
static uint32_t mont_n0(uint32_t m0) {
    // m0 odd -> invertible. Newton iteration mod 2^32.
    uint32_t inv = 1u;
    for (int i = 0; i < 5; ++i) {   // converges for 2,4,8,...,32 bits
        inv = inv * (2u - m0 * inv);
    }
    return (uint32_t)(0u - inv);    // -m0^-1 mod 2^32
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string candfile;
    long exp_override = -1;
    long limit = 0;
    bool verbose = false;
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
        } else if (a == "-v" || a == "--verbose") {
            verbose = true;
        } else if (a == "-h" || a == "--help") {
            printf("Usage: %s [--bases \"2 3 5 ...\"] [--limit N] [--exp N] [-v|--verbose] [candidate-file]\n"
                   "  --bases: default is the first 13 primes (2 3 5 ... 41).\n"
                   "  -v:      print the Metal device, kernel size and dispatch detail.\n", argv[0]);
            return 0;
        } else if (a[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", a.c_str());
            return 2;
        } else {
            candfile = a;
        }
    }
    if (candfile.empty()) { candfile = "kand.txt"; }
    if (bases.empty()) {
        // Default: first 13 primes (see prp_test.cpp for the rationale; composites
        // fail base 2 first, so the extra bases are near-free via early-out).
        bases = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};
    }

    // Read candidates (exponent from the header).
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

    // ---- Determine the size: NL = limbs for the largest M ----
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
    const int NL = (int)((max_bits + 31) / 32);
    const int MAXNL = 128;   // prototype limit (4096 bits)
    if (NL > MAXNL) {
        fprintf(stderr, "M too large for this prototype (%d limbs > %d). "
                        "Choose a smaller k/bmax (FFT approach needed).\n", NL, MAXNL);
        return 1;
    }
    printf("Candidates: %zu, exponent N=%lu, largest M ~%zu bits -> NL=%d limbs\n",
           ncand, exp, max_bits, NL);

    // ---- CPU: prepare M, n0, R mod M, and a*R mod M per base ----
    const uint32_t nbases = (uint32_t)bases.size();
    std::string basestr;
    for (size_t i = 0; i < bases.size(); ++i) {
        basestr += (i ? " " : "") + std::to_string(bases[i]);
    }
    std::vector<uint32_t> hostM((size_t)ncand * NL);
    std::vector<uint32_t> hostOne((size_t)ncand * NL);
    std::vector<uint32_t> hostAm((size_t)ncand * nbases * NL);   // base-major
    std::vector<uint32_t> hostN0(ncand);

    // Setup is independent per candidate -> parallelize across all cores.
    auto t_setup = Clock::now();
    int setup_threads = (int)std::thread::hardware_concurrency();
    if (setup_threads < 1) { setup_threads = 1; }
    std::atomic<size_t> setup_next{0};
    auto setupworker = [&]() {
        mpz_t lM, R, one_mont, am, abig;
        mpz_inits(lM, R, one_mont, am, abig, nullptr);
        mpz_setbit(R, (mp_bitcnt_t)NL * 32);      // R = 2^(32*NL)
        for (;;) {
            size_t idx = setup_next.fetch_add(1, std::memory_order_relaxed);
            if (idx >= ncand) { break; }
            mpz_ui_pow_ui(lM, cands[idx], exp);
            mpz_add_ui(lM, lM, 1);
            mpz_fdiv_q_2exp(lM, lM, 1);           // M = (b^N+1)/2
            to_limbs(lM, &hostM[idx * NL], NL);
            hostN0[idx] = mont_n0(mpz_get_ui(lM) & 0xffffffffu);

            mpz_mod(one_mont, R, lM);             // R mod M  (Montgomery 1)
            to_limbs(one_mont, &hostOne[idx * NL], NL);

            for (uint32_t bi = 0; bi < nbases; ++bi) {
                mpz_set_ui(abig, bases[bi]);
                mpz_mul(am, abig, R);
                mpz_mod(am, am, lM);              // a*R mod M
                to_limbs(am, &hostAm[((size_t)bi * ncand + idx) * NL], NL);
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

    // ---- Set up Metal ----
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "No Metal device.\n"); return 1; }

        std::string src = std::string("#define NL ") + std::to_string(NL) + "\n" + kKernelTemplate;
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                               options:nil
                                                 error:&err];
        if (!lib) { fprintf(stderr, "Kernel compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"miller_rabin"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "Pipeline: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        if (verbose) {
            fprintf(stderr,
                    "[metal] device: %s\n"
                    "[metal] kernel NL=%d limbs (%zu-bit M)  bases: %s\n"
                    "[metal] setup threads: %d  maxThreadsPerThreadgroup: %lu\n",
                    dev.name.UTF8String, NL, max_bits, basestr.c_str(),
                    setup_threads, (unsigned long)pso.maxTotalThreadsPerThreadgroup);
            fflush(stderr);
        }

        auto mkbuf = [&](const void* p, size_t bytes) {
            return [dev newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> bM   = mkbuf(hostM.data(),   hostM.size()   * 4);
        id<MTLBuffer> bOne = mkbuf(hostOne.data(), hostOne.size() * 4);
        id<MTLBuffer> bAm  = mkbuf(hostAm.data(),  hostAm.size()  * 4);
        id<MTLBuffer> bN0  = mkbuf(hostN0.data(),  hostN0.size()  * 4);
        id<MTLBuffer> bOut = [dev newBufferWithLength:ncand options:MTLResourceStorageModeShared];
        uint32_t nc = (uint32_t)ncand;
        id<MTLBuffer> bNc  = mkbuf(&nc, 4);
        id<MTLBuffer> bNb  = mkbuf(&nbases, 4);

        // ---- Run the kernel and measure the time ----
        // Dispatch in chunks: each command buffer stays short so no single run
        // hits the GPU watchdog (which would abort threads and corrupt results).
        NSUInteger tptg = pso.maxTotalThreadsPerThreadgroup;
        if (tptg > 256) { tptg = 256; }
        // Chunk size: small enough that a single command buffer does not hit the
        // GPU watchdog -- but we do NOT wait per chunk; we enqueue them all
        // (serial queue) and wait only at the end. That keeps the GPU busy
        // without per-chunk CPU synchronization stalls.
        const uint32_t CHUNK = 8192;

        auto t_gpu = Clock::now();
        std::vector<id<MTLCommandBuffer>> cbs;
        for (uint32_t off = 0; off < (uint32_t)ncand; off += CHUNK) {
            uint32_t cnt = (uint32_t)std::min<size_t>(CHUNK, ncand - off);
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bM offset:0 atIndex:0];
            [enc setBuffer:bOne offset:0 atIndex:1];
            [enc setBuffer:bAm offset:0 atIndex:2];
            [enc setBuffer:bN0 offset:0 atIndex:3];
            [enc setBuffer:bOut offset:0 atIndex:4];
            [enc setBuffer:bNc offset:0 atIndex:5];
            [enc setBytes:&off length:4 atIndex:6];
            [enc setBuffer:bNb offset:0 atIndex:7];
            [enc dispatchThreads:MTLSizeMake(cnt, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
            [enc endEncoding];
            [cb commit];
            cbs.push_back(cb);
        }
        [cbs.back() waitUntilCompleted];   // serial queue -> all finished
        for (id<MTLCommandBuffer> cb : cbs) {
            if (cb.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "GPU block failed (status %ld): %s\n",
                        (long)cb.status,
                        cb.error ? cb.error.localizedDescription.UTF8String : "(no detail)");
                return 1;
            }
        }
        double gpu_secs = secs_since(t_gpu);

        const uint8_t* gout = (const uint8_t*)bOut.contents;
        std::vector<uint8_t> gpu_res(gout, gout + ncand);

        // ---- CPU reference: same strong test over all bases, all cores ----
        std::vector<uint8_t> cpu_res(ncand, 0);
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
                mpz_fdiv_q_2exp(m, m, 1);              // M = (b^N+1)/2
                mpz_sub_ui(mm1, m, 1);                 // M-1
                unsigned long s = mpz_scan1(mm1, 0);   // M-1 = d * 2^s
                mpz_fdiv_q_2exp(d, mm1, s);
                bool prp = true;
                for (uint32_t bi = 0; bi < nbases && prp; ++bi) {
                    mpz_set_ui(ab, bases[bi]);
                    mpz_powm(r, ab, d, m);             // x = a^d mod M
                    bool pass = (mpz_cmp_ui(r, 1) == 0) || (mpz_cmp(r, mm1) == 0);
                    for (unsigned long it = 1; it < s && !pass; ++it) {
                        mpz_powm_ui(r, r, 2, m);       // x = x^2 mod M
                        if (mpz_cmp(r, mm1) == 0) { pass = true; break; }
                        if (mpz_cmp_ui(r, 1) == 0) { break; }
                    }
                    if (!pass) { prp = false; }
                }
                cpu_res[i] = prp ? 1 : 0;
            }
            mpz_clears(m, mm1, d, r, ab, nullptr);
        };
        std::vector<std::thread> pool;
        for (int tt = 0; tt < nthreads; ++tt) { pool.emplace_back(cpuworker); }
        for (auto& th : pool) { th.join(); }
        double cpu_secs = secs_since(t_cpu);

        // ---- Correctness: GPU must match the CPU exactly ----
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
        if (gpu_secs > 0) {
            printf("\nSpeedup kernel vs CPU:   %.1fx\n", cpu_secs / gpu_secs);
        }
        if (setup_secs + gpu_secs > 0) {
            printf("Speedup total vs CPU:    %.1fx\n", cpu_secs / (setup_secs + gpu_secs));
        }
    }

    mpz_clears(M, t, nullptr);
    return 0;
}
