// ntt_metal.mm -- Large-k NTT PRP engine for Apple Silicon (Metal).
//
// See docs/metal-largek-prp.md for the full design. The math is shared with the
// CPU oracle ntt_ref.cpp and the CUDA sibling; this file is the Apple-specific
// hardware layer: integer NTT (no fp64 on Apple GPUs), 32-bit Montgomery modmul
// via mulhi (no 64-bit arithmetic on the GPU hot path), four-step tiled
// transform, unified-memory CPU-side CRT/carry.
//
// Built incrementally, each stage gated against ntt_ref.cpp / GMP:
//   Stage 2a (this commit): 32-bit Montgomery modmul on the GPU, verified
//                           against the u64 reference (a*b) % p.  --selftest montmul
//
// Build (macOS only): see CMakeLists.txt (target ntt_metal), or:
//   clang++ -std=c++17 -O3 -ObjC++ -fobjc-arc ntt_metal.mm \
//     -I$(brew --prefix gmp)/include -L$(brew --prefix gmp)/lib -lgmp \
//     -framework Metal -framework Foundation -o ntt_metal

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <gmp.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>

using u32 = uint32_t;
using u64 = uint64_t;

// ---------------------------------------------------------------------------
// Host-side modular helpers (plain u64; these are setup/reference, not hot path)
// ---------------------------------------------------------------------------

// -p^-1 mod 2^32, for odd p (Montgomery n0). Newton's iteration doubles the
// number of correct low bits each step; 5 steps cover 32 bits.
static u32 mont_n0(u32 p) {
    u32 inv = p;                       // inv == p^-1 mod 2^(2^1) to start (p odd)
    for (int i = 0; i < 5; ++i) {
        inv *= 2u - p * inv;           // mod 2^32 implicit
    }
    return (u32)(0u - inv);            // -p^-1 mod 2^32
}

// R^2 mod p, with R = 2^32. (R mod p)^2 mod p; R mod p < p < 2^31 so the square
// fits in u64.
static u32 mont_r2(u32 p) {
    u64 r = ((u64)1 << 32) % p;
    return (u32)((r * r) % p);
}

static bool is_prime_u64(u64 n) {
    mpz_t z; mpz_init_set_ui(z, (unsigned long)n);
    int r = mpz_probab_prime_p(z, 40);
    mpz_clear(z);
    return r != 0;
}

// Smallest few NTT-friendly primes p = j*2N + 1 (p < 2^31) above a floor.
static std::vector<u32> ntt_primes(u64 N, int count) {
    std::vector<u32> out;
    const u64 step = 2 * N;
    for (u64 p = 1 + step; p < ((u64)1 << 31) && (int)out.size() < count; p += step) {
        if (is_prime_u64(p)) out.push_back((u32)p);
    }
    return out;
}

// ---------------------------------------------------------------------------
// The Metal kernels (runtime-compiled). 32-bit only: 32x32->64 via mul + mulhi.
// ---------------------------------------------------------------------------
static const char* kKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

// Montgomery REDC for a single 32-bit prime p, R = 2^32.
// Given T = hi*2^32 + lo with T < p*R, returns T * R^-1 mod p in [0, p).
inline uint redc(uint hi, uint lo, uint p, uint n0) {
    uint m     = lo * n0;              // (T mod R) * n0  mod R
    uint mp_lo = m * p;                // low 32 of m*p
    uint mp_hi = mulhi(m, p);          // high 32 of m*p
    // low word of (T + m*p) is 0 by construction; carry out == (lo != 0).
    uint carry = (lo + mp_lo) < lo ? 1u : 0u;
    uint t = hi + mp_hi + carry;       // (T + m*p) / R,  t < 2p
    if (t >= p) t -= p;
    return t;
}

// a, b in the Montgomery domain -> a*b in the Montgomery domain.
inline uint mont_mul(uint a, uint b, uint p, uint n0) {
    return redc(mulhi(a, b), a * b, p, n0);
}

// Self-test: OUT[i] = (A[i] * B[i]) mod p, computed entirely in the Montgomery
// domain (toMont, multiply, fromMont), to exercise the exact modmul the NTT uses.
kernel void mont_test(device const uint* A  [[buffer(0)]],
                      device const uint* B  [[buffer(1)]],
                      device       uint* OUT [[buffer(2)]],
                      constant uint& p       [[buffer(3)]],
                      constant uint& n0      [[buffer(4)]],
                      constant uint& r2      [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    uint a  = A[gid], b = B[gid];
    uint am = redc(mulhi(a, r2), a * r2, p, n0);   // toMont(a) = REDC(a * R^2)
    uint bm = redc(mulhi(b, r2), b * r2, p, n0);   // toMont(b)
    uint pm = mont_mul(am, bm, p, n0);             // (a*b) in Montgomery domain
    OUT[gid] = redc(0u, pm, p, n0);                // fromMont -> normal domain
}
)METAL";

// ---------------------------------------------------------------------------
// Stage 2a self-test: GPU Montgomery modmul vs. u64 (a*b)%p reference.
// ---------------------------------------------------------------------------
static int selftest_montmul(int k) {
    const u64 N = (u64)1 << k;
    std::vector<u32> primes = ntt_primes(N, 4);
    if (primes.empty()) {
        fprintf(stderr, "no NTT primes for k=%d\n", k);
        return 2;
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kKernelSource]
                                               options:nil error:&err];
        if (!lib) { fprintf(stderr, "kernel compile failed: %s\n",
                            err.localizedDescription.UTF8String); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"mont_test"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pipeline failed: %s\n",
                            err.localizedDescription.UTF8String); return 2; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        const u32 n = 1u << 20;        // ~1M random pairs per prime
        std::mt19937 rng(0xC0FFEEu);

        printf("Metal device: %s\n", dev.name.UTF8String);
        printf("montmul self-test: k=%d  N=%llu  %zu primes  %u pairs each\n",
               k, (unsigned long long)N, primes.size(), n);

        int total_fail = 0;
        for (u32 p : primes) {
            u32 n0 = mont_n0(p), r2 = mont_r2(p);

            id<MTLBuffer> bA = [dev newBufferWithLength:n * sizeof(u32)
                                                options:MTLResourceStorageModeShared];
            id<MTLBuffer> bB = [dev newBufferWithLength:n * sizeof(u32)
                                                options:MTLResourceStorageModeShared];
            id<MTLBuffer> bO = [dev newBufferWithLength:n * sizeof(u32)
                                                options:MTLResourceStorageModeShared];
            u32* A = (u32*)bA.contents;
            u32* B = (u32*)bB.contents;
            for (u32 i = 0; i < n; ++i) {
                // include edge values 0, 1, p-1 up front, random after
                if      (i == 0) { A[i] = 0;     B[i] = p - 1; }
                else if (i == 1) { A[i] = p - 1; B[i] = p - 1; }
                else if (i == 2) { A[i] = 1;     B[i] = p - 1; }
                else             { A[i] = rng() % p; B[i] = rng() % p; }
            }

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bA offset:0 atIndex:0];
            [enc setBuffer:bB offset:0 atIndex:1];
            [enc setBuffer:bO offset:0 atIndex:2];
            [enc setBytes:&p  length:sizeof(u32) atIndex:3];
            [enc setBytes:&n0 length:sizeof(u32) atIndex:4];
            [enc setBytes:&r2 length:sizeof(u32) atIndex:5];
            NSUInteger tptg = pso.maxTotalThreadsPerThreadgroup;
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const u32* O = (const u32*)bO.contents;
            int fail = 0;
            for (u32 i = 0; i < n; ++i) {
                u32 want = (u32)(((u64)A[i] * B[i]) % p);
                if (O[i] != want) {
                    if (fail < 3) {
                        printf("  MISMATCH p=%u  %u*%u mod p: got %u want %u\n",
                               p, A[i], B[i], O[i], want);
                    }
                    ++fail;
                }
            }
            printf("  p=%-10u n0=%u r2=%u  -> %s (%d/%u)\n",
                   p, n0, r2, fail == 0 ? "OK" : "FAIL", fail, n);
            total_fail += fail;
        }
        printf("montmul self-test: %s\n", total_fail == 0 ? "ALL OK" : "FAILURES");
        return total_fail == 0 ? 0 : 1;
    }
}

int main(int argc, char** argv) {
    int k = 16;
    std::string mode;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nx = [&](const char* f) {
            if (i + 1 >= argc) { fprintf(stderr, "missing arg for %s\n", f); exit(2); }
            return argv[++i];
        };
        if      (s == "--selftest") mode = nx("--selftest");
        else if (s == "--k")        k = atoi(nx("--k"));
        else { fprintf(stderr, "Usage: %s --selftest montmul [--k K]\n", argv[0]); return 2; }
    }
    if (mode == "montmul") return selftest_montmul(k);
    fprintf(stderr, "Usage: %s --selftest montmul [--k K]\n", argv[0]);
    return 2;
}
