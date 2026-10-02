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

// Plain u64 modular arithmetic for p < 2^31 (products fit in u64). Setup only.
static inline u64 mulmod(u64 a, u64 b, u64 p) { return (a * b) % p; }
static u64 powmod(u64 a, u64 e, u64 p) {
    u64 r = 1 % p; a %= p;
    while (e) { if (e & 1) r = mulmod(r, a, p); a = mulmod(a, a, p); e >>= 1; }
    return r;
}
static inline u64 modinv(u64 a, u64 p) { return powmod(a, p - 2, p); }

// Host Montgomery conversion, R = 2^32:  toMont(x) = x*R mod p.
static inline u32 to_mont(u32 x, u32 p) { return (u32)((((u64)x) << 32) % p); }

// A 2N-th root of unity psi with psi^N == -1 (order exactly 2N).  (cf. ntt_ref)
static u32 find_psi(u32 p, u64 N) {
    u64 e = (p - 1) / (2 * N);
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull ^ p);
    for (;;) {
        u64 g = 2 + rng() % (p - 3);
        u64 r = powmod(g, e, p);
        if (powmod(r, N, p) == p - 1) return (u32)r;
    }
}

// Iterative radix-2 NTT in the normal domain (CPU reference, matches ntt_ref).
static void ntt_cpu(std::vector<u32>& a, u32 p, u32 root) {
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
                a[i + t]           = (u32)((u + v) % p);
                a[i + t + len / 2] = (u32)((u + p - v) % p);
                w = mulmod(w, wlen, p);
            }
        }
    }
}

// CPU reference: single-prime negacyclic squaring of digit vector x (length N,
// entries < p) mod (X^N + 1), reduced mod p. Returns the N result coefficients.
static std::vector<u32> negasq_cpu(const std::vector<u32>& x, u32 p, u64 N) {
    u32 psi = find_psi(p, N);
    u32 omega = (u32)mulmod(psi, psi, p);
    u32 omega_inv = (u32)modinv(omega, p);
    u32 psi_inv = (u32)modinv(psi, p);
    u32 ninv = (u32)modinv(N % p, p);
    std::vector<u32> A(N);
    u64 w = 1;
    for (u64 j = 0; j < N; ++j) { A[j] = (u32)mulmod(x[j], w, p); w = mulmod(w, psi, p); }
    ntt_cpu(A, p, omega);
    for (u64 j = 0; j < N; ++j) A[j] = (u32)mulmod(A[j], A[j], p);
    ntt_cpu(A, p, omega_inv);
    u64 wi = 1;
    for (u64 j = 0; j < N; ++j) {
        A[j] = (u32)mulmod(A[j], ninv, p);
        A[j] = (u32)mulmod(A[j], wi, p);
        wi = mulmod(wi, psi_inv, p);
    }
    return A;
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

// modular add/sub for values in [0,p), p < 2^31 (so a+b and a+p-b fit in 32 bits).
inline uint addm(uint a, uint b, uint p) { uint s = a + b;     return s >= p ? s - p : s; }
inline uint subm(uint a, uint b, uint p) { uint s = a + p - b; return s >= p ? s - p : s; }

// X[g] *= W[g]   (both already in the Montgomery domain) -- psi weighting.
kernel void k_weight(device uint* X [[buffer(0)]],
                     device const uint* W [[buffer(1)]],
                     constant uint& p [[buffer(2)]], constant uint& n0 [[buffer(3)]],
                     uint g [[thread_position_in_grid]]) {
    X[g] = mont_mul(X[g], W[g], p, n0);
}

// Bit-reversal gather: OUT[g] = IN[reverse_ln_bits(g)].
kernel void k_bitrev(device const uint* IN [[buffer(0)]],
                     device uint* OUT [[buffer(1)]],
                     constant uint& ln [[buffer(2)]],
                     uint g [[thread_position_in_grid]]) {
    uint r = 0;
    for (uint b = 0; b < ln; ++b) { r = (r << 1) | ((g >> b) & 1u); }
    OUT[g] = IN[r];
}

// One in-place Cooley-Tukey radix-2 stage. One thread per butterfly (N/2 total).
// W is the twiddle table W[t] = root^t (Montgomery); twiddle = W[(n/len)*offset].
kernel void k_stage(device uint* X [[buffer(0)]],
                    device const uint* W [[buffer(1)]],
                    constant uint& p [[buffer(2)]], constant uint& n0 [[buffer(3)]],
                    constant uint& len [[buffer(4)]], constant uint& n [[buffer(5)]],
                    uint g [[thread_position_in_grid]]) {
    uint hlen   = len >> 1;
    uint block  = g / hlen;
    uint offset = g - block * hlen;
    uint i = block * len + offset;
    uint j = i + hlen;
    uint w = W[(n / len) * offset];
    uint u = X[i];
    uint v = mont_mul(X[j], w, p, n0);
    X[i] = addm(u, v, p);
    X[j] = subm(u, v, p);
}

// Pointwise square (Montgomery domain).
kernel void k_sq(device uint* X [[buffer(0)]],
                 constant uint& p [[buffer(1)]], constant uint& n0 [[buffer(2)]],
                 uint g [[thread_position_in_grid]]) {
    uint a = X[g];
    X[g] = mont_mul(a, a, p, n0);
}

// Finalize: X[g] = fromMont( X[g] * ninv * Wij[g] ).  ninv in Montgomery domain.
kernel void k_final(device uint* X [[buffer(0)]],
                    device const uint* WIJ [[buffer(1)]],
                    constant uint& p [[buffer(2)]], constant uint& n0 [[buffer(3)]],
                    constant uint& ninv [[buffer(4)]],
                    uint g [[thread_position_in_grid]]) {
    uint v = mont_mul(X[g], ninv, p, n0);
    v = mont_mul(v, WIJ[g], p, n0);
    X[g] = redc(0u, v, p, n0);
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

// ---------------------------------------------------------------------------
// Stage 2b self-test: single-prime negacyclic squaring on the GPU (weight ->
// bit-reverse -> forward NTT -> pointwise square -> bit-reverse -> inverse NTT
// -> 1/N + unweight), one dispatch per Cooley-Tukey stage, verified bit-for-bit
// against the CPU reference negasq_cpu (same math mod p).
// ---------------------------------------------------------------------------
static int selftest_ntt(int k) {
    const u64 N = (u64)1 << k;
    std::vector<u32> primes = ntt_primes(N, 1);
    if (primes.empty()) { fprintf(stderr, "no NTT prime for k=%d\n", k); return 2; }
    const u32 p = primes[0];
    const u32 n0 = mont_n0(p);
    const u32 ln = (u32)k;

    // host roots / tables (normal domain, then converted to Montgomery)
    const u32 psi = find_psi(p, N);
    const u32 omega = (u32)mulmod(psi, psi, p);
    const u32 omega_inv = (u32)modinv(omega, p);
    const u32 psi_inv = (u32)modinv(psi, p);
    const u32 ninv_mont = to_mont((u32)modinv(N % p, p), p);

    std::vector<u32> Wfwd(N / 2), Winv(N / 2), WJ(N), WIJ(N);
    { u64 w = 1; for (u64 t = 0; t < N / 2; ++t) { Wfwd[t] = to_mont((u32)w, p); w = mulmod(w, omega, p); } }
    { u64 w = 1; for (u64 t = 0; t < N / 2; ++t) { Winv[t] = to_mont((u32)w, p); w = mulmod(w, omega_inv, p); } }
    { u64 w = 1; for (u64 j = 0; j < N; ++j) { WJ[j]  = to_mont((u32)w, p); w = mulmod(w, psi, p); } }
    { u64 w = 1; for (u64 j = 0; j < N; ++j) { WIJ[j] = to_mont((u32)w, p); w = mulmod(w, psi_inv, p); } }

    // random digit vector in [0,p)
    std::vector<u32> x(N);
    std::mt19937 rng(0xABCDEF ^ (unsigned)k);
    for (u64 j = 0; j < N; ++j) x[j] = rng() % p;

    std::vector<u32> ref = negasq_cpu(x, p, N);   // ground truth (this prime, mod p)

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kKernelSource]
                                               options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 2; }
        auto pso_for = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!ps) { fprintf(stderr, "pso %s failed: %s\n", name, err.localizedDescription.UTF8String); exit(2); }
            return ps;
        };
        id<MTLComputePipelineState> psWeight = pso_for("k_weight");
        id<MTLComputePipelineState> psBitrev = pso_for("k_bitrev");
        id<MTLComputePipelineState> psStage  = pso_for("k_stage");
        id<MTLComputePipelineState> psSq     = pso_for("k_sq");
        id<MTLComputePipelineState> psFinal  = pso_for("k_final");
        id<MTLCommandQueue> q = [dev newCommandQueue];

        auto buf = [&](const void* src, size_t bytes) {
            return src ? [dev newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared]
                       : [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        };
        std::vector<u32> xm(N);
        for (u64 j = 0; j < N; ++j) xm[j] = to_mont(x[j], p);
        id<MTLBuffer> bX    = buf(xm.data(),  N * sizeof(u32));
        id<MTLBuffer> bT    = buf(nullptr,    N * sizeof(u32));
        id<MTLBuffer> bWf   = buf(Wfwd.data(), Wfwd.size() * sizeof(u32));
        id<MTLBuffer> bWi   = buf(Winv.data(), Winv.size() * sizeof(u32));
        id<MTLBuffer> bWJ   = buf(WJ.data(),  N * sizeof(u32));
        id<MTLBuffer> bWIJ  = buf(WIJ.data(), N * sizeof(u32));

        id<MTLCommandBuffer> cb = [q commandBuffer];
        // one encoder per dispatch; Metal hazard-tracks shared buffers between
        // encoders in a command buffer, so this serialises the transform stages.
        auto dispatch = [&](id<MTLComputePipelineState> ps, NSUInteger threads,
                            void (^setup)(id<MTLComputeCommandEncoder>)) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            setup(enc);
            NSUInteger t = ps.maxTotalThreadsPerThreadgroup;
            if (t > threads) t = threads;
            [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(t, 1, 1)];
            [enc endEncoding];
        };

        // 1. weight by psi^j
        dispatch(psWeight, N, ^(id<MTLComputeCommandEncoder> e) {
            [e setBuffer:bX offset:0 atIndex:0]; [e setBuffer:bWJ offset:0 atIndex:1];
            [e setBytes:&p length:4 atIndex:2]; [e setBytes:&n0 length:4 atIndex:3];
        });
        // 2. bit-reverse bX -> bT
        dispatch(psBitrev, N, ^(id<MTLComputeCommandEncoder> e) {
            [e setBuffer:bX offset:0 atIndex:0]; [e setBuffer:bT offset:0 atIndex:1];
            [e setBytes:&ln length:4 atIndex:2];
        });
        // 3. forward stages on bT
        u32 nn = (u32)N;
        for (u32 len = 2; len <= (u32)N; len <<= 1) {
            u32 L = len;
            dispatch(psStage, N / 2, ^(id<MTLComputeCommandEncoder> e) {
                [e setBuffer:bT offset:0 atIndex:0]; [e setBuffer:bWf offset:0 atIndex:1];
                [e setBytes:&p length:4 atIndex:2]; [e setBytes:&n0 length:4 atIndex:3];
                [e setBytes:&L length:4 atIndex:4]; [e setBytes:&nn length:4 atIndex:5];
            });
        }
        // 4. pointwise square
        dispatch(psSq, N, ^(id<MTLComputeCommandEncoder> e) {
            [e setBuffer:bT offset:0 atIndex:0];
            [e setBytes:&p length:4 atIndex:1]; [e setBytes:&n0 length:4 atIndex:2];
        });
        // 5. bit-reverse bT -> bX
        dispatch(psBitrev, N, ^(id<MTLComputeCommandEncoder> e) {
            [e setBuffer:bT offset:0 atIndex:0]; [e setBuffer:bX offset:0 atIndex:1];
            [e setBytes:&ln length:4 atIndex:2];
        });
        // 6. inverse stages on bX
        for (u32 len = 2; len <= (u32)N; len <<= 1) {
            u32 L = len;
            dispatch(psStage, N / 2, ^(id<MTLComputeCommandEncoder> e) {
                [e setBuffer:bX offset:0 atIndex:0]; [e setBuffer:bWi offset:0 atIndex:1];
                [e setBytes:&p length:4 atIndex:2]; [e setBytes:&n0 length:4 atIndex:3];
                [e setBytes:&L length:4 atIndex:4]; [e setBytes:&nn length:4 atIndex:5];
            });
        }
        // 7. 1/N + unweight + fromMont
        dispatch(psFinal, N, ^(id<MTLComputeCommandEncoder> e) {
            [e setBuffer:bX offset:0 atIndex:0]; [e setBuffer:bWIJ offset:0 atIndex:1];
            [e setBytes:&p length:4 atIndex:2]; [e setBytes:&n0 length:4 atIndex:3];
            [e setBytes:&ninv_mont length:4 atIndex:4];
        });

        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "GPU error: %s\n", cb.error.localizedDescription.UTF8String); return 2; }

        const u32* G = (const u32*)bX.contents;
        int fail = 0;
        for (u64 j = 0; j < N; ++j) {
            if (G[j] != ref[j]) {
                if (fail < 5) printf("  MISMATCH j=%llu got %u want %u\n",
                                     (unsigned long long)j, G[j], ref[j]);
                ++fail;
            }
        }
        printf("Metal device: %s\n", dev.name.UTF8String);
        printf("ntt self-test: k=%d N=%llu p=%u  single-prime negacyclic square\n",
               k, (unsigned long long)N, p);
        printf("  GPU vs CPU reference: %s (%d/%llu mismatches)\n",
               fail == 0 ? "OK" : "FAIL", fail, (unsigned long long)N);
        return fail == 0 ? 0 : 1;
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
    if (mode == "ntt")     return selftest_ntt(k);
    fprintf(stderr, "Usage: %s --selftest {montmul|ntt} [--k K]\n", argv[0]);
    return 2;
}
