// prp_metal.mm  --  GPU-beschleunigter starker PRP-Test (Miller-Rabin) via
// Apple Metal (Regime A).
//
// Idee (siehe README): ein GPU-Thread pro Kandidat b. Die CPU (GMP) berechnet
// M = (b^N+1)/2 und die Montgomery-Konstanten; die GPU rechnet nur das Teure:
// den starken Miller-Rabin-Test zu Basis a mit Montgomery-Multiplikation in
// 32-Bit-Limbs (Zerlegung M-1 = d*2^s, dann a^d und die Quadrierungen).
//
// Dies ist ein PROTOTYP fuer "viele mittelgrosse Kandidaten": die feste Limb-Zahl
// NL wird zur Laufzeit aus dem groessten M bestimmt und in den Kernel-Quelltext
// einkompiliert. Fuer sehr grosse k (zehntausende Stellen) ist stattdessen ein
// FFT-basierter Ansatz noetig (vgl. genefer) -- hier bewusst nicht abgedeckt.
//
// Enthaelt zum fairen Vergleich denselben starken Test auf der CPU (GMP,
// multithreaded) und prueft GPU- gegen CPU-Ergebnis auf Gleichheit.
//
// Build (nur macOS): siehe CMakeLists.txt (Target prp_metal), oder:
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
// Der Metal-Kernel. NL (Limb-Zahl) wird zur Laufzeit per #define eingesetzt,
// damit alle Felder exakt passend dimensioniert sind (beste Belegung der GPU).
// Jeder Thread testet einen Kandidaten: a^(M-1) mod M == 1 ?
// ---------------------------------------------------------------------------
static const char* kKernelTemplate = R"METAL(
#include <metal_stdlib>
using namespace metal;

// CIOS-Montgomery-Multiplikation: out = a * b * R^-1 mod m,  R = 2^(32*NL).
// a, b, m < m. n0 = -m^-1 mod 2^32. Alles 32-Bit-Limbs, little-endian.
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

        // mm so waehlen, dass das niederwertigste Limb 0 wird, dann >> 32
        uint mm = (uint)((ulong)t[0] * (ulong)n0);   // mod 2^32 implizit
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

    // Finale bedingte Subtraktion: ist t >= m, einmal m abziehen.
    bool ge = (t[NL] != 0u);
    if (!ge) {
        for (int j = NL - 1; j >= 0; --j) {
            if (t[j] != m[j]) {
                ge = t[j] > m[j];
                break;
            }
            if (j == 0) {
                ge = true;   // t == m  ->  Ergebnis 0
            }
        }
    }
    if (ge) {
        ulong borrow = 0;
        for (uint j = 0; j < NL; ++j) {
            ulong d = (ulong)t[j] - (ulong)m[j] - borrow;
            out[j] = (uint)d;
            borrow = (d >> 63) & 1ul;   // 1, falls unterlaufen
        }
    } else {
        for (uint j = 0; j < NL; ++j) {
            out[j] = t[j];
        }
    }
}

// Gleichheit zweier NL-Limb-Zahlen.
static bool bnequal(thread const uint* a, thread const uint* b) {
    for (uint j = 0; j < NL; ++j) {
        if (a[j] != b[j]) { return false; }
    }
    return true;
}

// Starker Miller-Rabin-Test zu mehreren Basen (via Montgomery). Ein Kandidat
// gilt nur dann als PRP, wenn er ALLE Basen besteht; faellt er bei einer durch,
// brechen wir sofort ab (die meisten Kandidaten sind zusammengesetzt und fallen
// schon bei der ersten Basis -> die weiteren Basen kosten fast nichts).
kernel void miller_rabin(device const uint*  Ms       [[buffer(0)]],   // ncand * NL
                         device const uint*  oneMonts [[buffer(1)]],   // ncand * NL  (R mod M)
                         device const uint*  aMs      [[buffer(2)]],   // nbases * ncand * NL  (a*R mod M)
                         device const uint*  n0s      [[buffer(3)]],   // ncand
                         device uchar*       out      [[buffer(4)]],   // ncand
                         constant uint&      ncand    [[buffer(5)]],
                         constant uint&      offset   [[buffer(6)]],   // erster Kandidat des Blocks
                         constant uint&      nbases   [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
    const uint cand = offset + gid;
    if (cand >= ncand) {
        return;
    }
    const uint base = cand * NL;

    thread uint m[NL];     // Modulus M
    thread uint om[NL];    // Montgomery-Form der 1   (= R mod M)
    thread uint r[NL];     // laufender Wert, in Montgomery-Form
    thread uint am[NL];    // Montgomery-Form der aktuellen Basis (= a*R mod M)
    thread uint mm1[NL];   // Montgomery-Form von M-1 (= M - om)
    thread uint tmp[NL];
    for (uint j = 0; j < NL; ++j) {
        m[j]  = Ms[base + j];
        om[j] = oneMonts[base + j];
    }
    uint n0 = n0s[cand];

    // mm1 = M - om : Montgomery-Form von M-1, denn (M-1)*R = -R (mod M).
    {
        ulong borrow = 0;
        for (uint j = 0; j < NL; ++j) {
            ulong d = (ulong)m[j] - (ulong)om[j] - borrow;
            mm1[j] = (uint)d;
            borrow = (d >> 63) & 1ul;
        }
    }

    // M-1 = d * 2^s. M ungerade -> E := M-1 ist M mit geloeschtem Bit 0.
    // s = niedrigstes gesetztes Bit von E; topbit = hoechstes Bit von M.
    // (Haengt nur von M ab -> einmal pro Kandidat, nicht pro Basis.)
    uint s;
    uint low = m[0] & ~1u;             // Bit 0 von E ist 0
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
            r[j]  = om[j];             // Start: Montgomery-Form der 1
        }

        // x = a^d mod M (Montgomery): Bits von E von topbit hinab bis s scannen.
        for (int i = topbit; i >= (int)s; --i) {
            montmul(tmp, r, r, m, n0);
            for (uint j = 0; j < NL; ++j) { r[j] = tmp[j]; }
            uint bit = (m[(uint)i >> 5] >> ((uint)i & 31u)) & 1u;
            if (bit != 0u) {
                montmul(tmp, r, am, m, n0);
                for (uint j = 0; j < NL; ++j) { r[j] = tmp[j]; }
            }
        }

        // Starker Test (in Montgomery-Form): x == 1 oder x == M-1 ?
        bool pass = bnequal(r, om) || bnequal(r, mm1);
        for (uint it = 1; it < s && !pass; ++it) {
            montmul(tmp, r, r, m, n0);                 // x = x^2
            for (uint j = 0; j < NL; ++j) { r[j] = tmp[j]; }
            if (bnequal(r, mm1)) { pass = true; break; } // -1 gefunden
            if (bnequal(r, om))  { break; }              // 1 vor -1 -> zusammengesetzt
        }
        if (!pass) {
            prp = false;   // diese Basis bezeugt: zusammengesetzt -> Abbruch
        }
    }
    out[cand] = prp ? 1 : 0;
}
)METAL";

// ---------------------------------------------------------------------------
// GMP-Helfer: M in NL 32-Bit-Limbs (little-endian) exportieren.
// ---------------------------------------------------------------------------
static void to_limbs(const mpz_t x, uint32_t* dst, int NL) {
    for (int j = 0; j < NL; ++j) {
        dst[j] = 0u;
    }
    size_t count = 0;
    mpz_export(dst, &count, -1 /*least significant first*/, 4 /*word size*/,
               -1 /*little endian within word*/, 0, x);
    (void)count;  // bleibt <= NL, Rest ist bereits 0
}

// n0 = -M^-1 mod 2^32  (Montgomery-Konstante, nur niederwertigstes Limb noetig)
static uint32_t mont_n0(uint32_t m0) {
    // m0 ungerade -> invertierbar. Newton-Iteration mod 2^32.
    uint32_t inv = 1u;
    for (int i = 0; i < 5; ++i) {   // konvergiert fuer 2,4,8,...,32 Bit
        inv = inv * (2u - m0 * inv);
    }
    return (uint32_t)(0u - inv);    // -m0^-1 mod 2^32
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
            if (i + 1 >= argc) { fprintf(stderr, "Fehlender Wert fuer %s\n", n); exit(2); }
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
            printf("Aufruf: %s [--bases \"3 5 7\"] [--limit N] [--exp N] [kandidatendatei]\n", argv[0]);
            return 0;
        } else if (a[0] == '-') {
            fprintf(stderr, "Unbekannte Option: %s\n", a.c_str());
            return 2;
        } else {
            candfile = a;
        }
    }
    if (candfile.empty()) { candfile = "kand.txt"; }
    if (bases.empty()) { bases = {3}; }

    // Kandidaten lesen (Exponent aus Header).
    std::ifstream in(candfile);
    if (!in) { fprintf(stderr, "Kann %s nicht oeffnen\n", candfile.c_str()); return 1; }
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
    if (exp == 0) { fprintf(stderr, "Exponent unlesbar -- --exp N angeben.\n"); return 1; }
    if (cands.empty()) { fprintf(stderr, "Keine Kandidaten.\n"); return 1; }
    if (limit > 0 && (long)cands.size() > limit) { cands.resize(limit); }

    const size_t ncand = cands.size();

    // ---- Groesse bestimmen: NL = Limbs fuer das groesste M ----
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
    const int MAXNL = 128;   // Prototyp-Grenze (4096 Bit)
    if (NL > MAXNL) {
        fprintf(stderr, "M zu gross fuer diesen Prototyp (%d Limbs > %d). "
                        "Kleineres k/bmax waehlen (FFT-Ansatz noetig).\n", NL, MAXNL);
        return 1;
    }
    printf("Kandidaten: %zu, Exponent N=%lu, groesstes M ~%zu Bit -> NL=%d Limbs\n",
           ncand, exp, max_bits, NL);

    // ---- CPU: M, n0, R mod M, und je Basis a*R mod M vorbereiten ----
    const uint32_t nbases = (uint32_t)bases.size();
    std::string basestr;
    for (size_t i = 0; i < bases.size(); ++i) {
        basestr += (i ? " " : "") + std::to_string(bases[i]);
    }
    std::vector<uint32_t> hostM((size_t)ncand * NL);
    std::vector<uint32_t> hostOne((size_t)ncand * NL);
    std::vector<uint32_t> hostAm((size_t)ncand * nbases * NL);   // basis-major
    std::vector<uint32_t> hostN0(ncand);

    // Setup ist pro Kandidat unabhaengig -> ueber alle Kerne parallelisieren.
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

            mpz_mod(one_mont, R, lM);             // R mod M  (Montgomery-1)
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

    // ---- Metal aufsetzen ----
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "Kein Metal-Device.\n"); return 1; }

        std::string src = std::string("#define NL ") + std::to_string(NL) + "\n" + kKernelTemplate;
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                               options:nil
                                                 error:&err];
        if (!lib) { fprintf(stderr, "Kernel-Compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"miller_rabin"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "Pipeline: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

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

        // ---- Kernel ausfuehren und Zeit messen ----
        // In Bloecken dispatchen: jeder Command-Buffer bleibt kurz, damit kein
        // einzelner Lauf in den GPU-Watchdog laeuft (das wuerde Threads abbrechen
        // und Ergebnisse verfaelschen).
        NSUInteger tptg = pso.maxTotalThreadsPerThreadgroup;
        if (tptg > 256) { tptg = 256; }
        // Blockgroesse: klein genug, dass ein einzelner Command-Buffer nicht in
        // den GPU-Watchdog laeuft -- aber wir warten NICHT pro Block, sondern
        // reihen alle ein (serielle Queue) und warten erst am Ende. So laeuft die
        // GPU ohne CPU-Synchronisationspausen durch.
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
        [cbs.back() waitUntilCompleted];   // serielle Queue -> alle fertig
        for (id<MTLCommandBuffer> cb : cbs) {
            if (cb.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "GPU-Block fehlgeschlagen (Status %ld): %s\n",
                        (long)cb.status,
                        cb.error ? cb.error.localizedDescription.UTF8String : "(kein Detail)");
                return 1;
            }
        }
        double gpu_secs = secs_since(t_gpu);

        const uint8_t* gout = (const uint8_t*)bOut.contents;
        std::vector<uint8_t> gpu_res(gout, gout + ncand);

        // ---- CPU-Referenz: gleicher starker Test ueber alle Basen, alle Kerne ----
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

        // ---- Korrektheit: GPU muss exakt der CPU entsprechen ----
        size_t mism = 0, gpu_prp = 0, cpu_prp = 0;
        for (size_t i = 0; i < ncand; ++i) {
            if (gpu_res[i]) { ++gpu_prp; }
            if (cpu_res[i]) { ++cpu_prp; }
            if (gpu_res[i] != cpu_res[i]) { ++mism; }
        }

        printf("\n--- Korrektheit ---\n");
        printf("GPU PRP: %zu   CPU PRP: %zu   Abweichungen: %zu  (%s)\n",
               gpu_prp, cpu_prp, mism, mism == 0 ? "OK" : "FEHLER");

        printf("\n--- Performance (Basen %s, starker Miller-Rabin) ---\n", basestr.c_str());
        printf("CPU-Setup (GMP, M + Montgomery-Konstanten): %.3f s\n", setup_secs);
        printf("GPU-Kernel (nur Rechnung):                  %.3f s\n", gpu_secs);
        printf("GPU gesamt (Setup + Kernel):                %.3f s\n", setup_secs + gpu_secs);
        printf("CPU gesamt (%2d Threads, GMP):               %.3f s\n", nthreads, cpu_secs);
        if (gpu_secs > 0) {
            printf("\nSpeedup Kernel vs CPU:   %.1fx\n", cpu_secs / gpu_secs);
        }
        if (setup_secs + gpu_secs > 0) {
            printf("Speedup gesamt vs CPU:   %.1fx\n", cpu_secs / (setup_secs + gpu_secs));
        }
    }

    mpz_clears(M, t, nullptr);
    return 0;
}
