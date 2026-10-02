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
#include <chrono>

using u32 = uint32_t;
using u64 = uint64_t;
using i64 = int64_t;
using i128 = __int128;          // host coefficient path (Apple Silicon native)

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

// Pointwise multiply A[g] *= B[g] (Montgomery domain).
kernel void k_mul(device uint* A [[buffer(0)]],
                  device const uint* B [[buffer(1)]],
                  constant uint& p [[buffer(2)]], constant uint& n0 [[buffer(3)]],
                  uint g [[thread_position_in_grid]]) {
    A[g] = mont_mul(A[g], B[g], p, n0);
}

// Load digits into the Montgomery domain and weight by psi^j in one pass.
// IN holds normal-domain digits < p; WJ holds psi^j already in Montgomery form.
// toMont(x) = REDC(x * R2); result = toMont(x) * WJ[g].
kernel void k_load_weight(device const uint* IN [[buffer(0)]],
                          device uint* OUT [[buffer(1)]],
                          device const uint* WJ [[buffer(2)]],
                          constant uint& p [[buffer(3)]], constant uint& n0 [[buffer(4)]],
                          constant uint& r2 [[buffer(5)]],
                          uint g [[thread_position_in_grid]]) {
    uint x  = IN[g];
    uint xm = redc(mulhi(x, r2), x * r2, p, n0);   // toMont
    OUT[g]  = mont_mul(xm, WJ[g], p, n0);
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

// ---------------------------------------------------------------------------
// Stage 3 host helpers: NTT prime set by bound, balanced CRT, balanced base-b
// carry with the b^N = -1 wrap, digit <-> mpz. (CRT/carry mirror ntt_ref.cpp.)
// ---------------------------------------------------------------------------
static std::vector<u32> ntt_primes_bound(u64 N, u64 b) {
    std::vector<u32> out;
    mpz_t bound, prod;
    mpz_init(bound); mpz_init_set_ui(prod, 1);
    mpz_set_ui(bound, (unsigned long)(b - 1));
    mpz_mul(bound, bound, bound);
    mpz_mul_ui(bound, bound, (unsigned long)N);
    mpz_mul_ui(bound, bound, 2);                 // 2 * N * (b-1)^2
    const u64 step = 2 * N;
    for (u64 p = 1 + step; p < ((u64)1 << 31) && mpz_cmp(prod, bound) <= 0; p += step) {
        if (b >= p) continue;
        if (is_prime_u64(p)) { out.push_back((u32)p); mpz_mul_ui(prod, prod, (unsigned long)p); }
    }
    mpz_clear(bound); mpz_clear(prod);
    return out;
}

// Extract a signed 128-bit value from an mpz (little-endian 32-bit chunks, so it
// is independent of the platform's `long` width). |coeff| <= N*(b-1)^2 fits in
// i128 for every realistic k/b (e.g. k=18, b~10^7 -> ~10^19 << 2^127).
static i128 mpz_to_i128(const mpz_t x) {
    int neg = mpz_sgn(x) < 0;
    mpz_t t; mpz_init(t); mpz_abs(t, x);
    unsigned __int128 v = 0;
    for (int sh = 0; sh < 128 && mpz_sgn(t) != 0; sh += 32) {
        uint32_t limb = (uint32_t)mpz_get_ui(t);
        v |= ((unsigned __int128)limb) << sh;
        mpz_fdiv_q_2exp(t, t, 32);
    }
    mpz_clear(t);
    return neg ? -(i128)v : (i128)v;
}

// Balanced CRT of residues r[i] mod primes[i] -> signed coefficient (mpz Garner).
static i128 crt_balanced(const std::vector<u32>& r, const std::vector<u32>& primes) {
    const size_t m = primes.size();
    mpz_t x, Macc, t, P_, half;
    mpz_init_set_ui(x, r[0] % primes[0]);
    mpz_init_set_ui(Macc, primes[0]);
    mpz_init(t);
    for (size_t i = 1; i < m; ++i) {
        u64 pi = primes[i];
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
    if (mpz_cmp(x, half) > 0) mpz_sub(x, x, P_);
    i128 out = mpz_to_i128(x);
    mpz_clear(x); mpz_clear(Macc); mpz_clear(t); mpz_clear(P_); mpz_clear(half);
    return out;
}

// Balanced base-b carry with the b^N = -1 wrap. Coefficients arrive as i128
// (they can exceed i64); the resulting digits are in (-b/2, b/2] and fit i64.
static std::vector<i64> carry_balanced(const std::vector<i128>& c, u64 b) {
    const i128 bb = (i128)b, half = bb / 2;
    std::vector<i128> d = c;
    const u64 N = d.size();
    for (int guard = 0; guard < 128; ++guard) {
        i128 carry = 0;
        for (u64 j = 0; j < N; ++j) {
            i128 v = d[j] + carry;
            i128 rem = v % bb; if (rem < 0) rem += bb;
            if (rem > half) rem -= bb;
            carry = (v - rem) / bb;
            d[j] = rem;
        }
        if (carry == 0) break;
        d[0] -= carry;                 // b^N == -1
    }
    std::vector<i64> out(N);
    for (u64 j = 0; j < N; ++j) out[j] = (i64)d[j];   // digits fit i64
    return out;
}

static void digits_to_mpz(mpz_t out, const std::vector<i64>& d, u64 b) {
    mpz_set_ui(out, 0);
    for (size_t j = d.size(); j-- > 0;) {
        mpz_mul_ui(out, out, (unsigned long)b);
        if (d[j] >= 0) mpz_add_ui(out, out, (unsigned long)d[j]);
        else           mpz_sub_ui(out, out, (unsigned long)(-d[j]));
    }
}

// Base-b digits (non-negative, length N) of a non-negative mpz x < b^N.
static std::vector<i64> mpz_to_digits(const mpz_t x, u64 N, u64 b) {
    mpz_t t, q; mpz_init_set(t, x); mpz_init(q);
    std::vector<i64> d(N, 0);
    for (u64 j = 0; j < N; ++j) {
        u64 r = mpz_fdiv_q_ui(q, t, (unsigned long)b);
        d[j] = (i64)r;
        mpz_set(t, q);
    }
    mpz_clear(t); mpz_clear(q);
    return d;
}

// ---------------------------------------------------------------------------
// Stage 3 self-test: full multi-prime negacyclic multiply/square mod (b^N+1) on
// the GPU (per-prime NTT channels) + CPU-side balanced CRT and carry, verified
// against GMP x*y mod (b^N+1).
// ---------------------------------------------------------------------------
struct PrimeGPU {
    u32 p, n0, r2, ninv_mont;
    id<MTLBuffer> bWf, bWi, bWJ, bWIJ;
};

// ---------------------------------------------------------------------------
// The engine: per-prime GPU NTT channels built once; negamul() runs one
// negacyclic multiply/square mod (b^N+1) and returns balanced base-b digits.
// Inputs/outputs are signed (balanced) digits so results feed straight back in.
// ---------------------------------------------------------------------------
struct Engine {
    int k = 0; u64 N = 0, b = 0;
    std::vector<u32> primes;
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> q = nil;
    id<MTLComputePipelineState> psLW, psBR, psST, psSQ, psMUL, psFIN;
    std::vector<PrimeGPU> G;
    id<MTLBuffer> bInX, bInY, bW, bFx, bFy, bT;
    u32 ln = 0, nn = 0;

    bool init(int k_, u64 b_) {
        k = k_; N = (u64)1 << k_; b = b_; ln = (u32)k_; nn = (u32)N;
        if ((b & 1) == 0 || b < 3) { fprintf(stderr, "need odd b >= 3\n"); return false; }
        primes = ntt_primes_bound(N, b);
        if (primes.empty()) { fprintf(stderr, "no NTT primes for k=%d b=%llu\n",
                                      k, (unsigned long long)b); return false; }
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return false; }
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kKernelSource]
                                               options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return false; }
        auto pso_for = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!ps) { fprintf(stderr, "pso %s: %s\n", name, err.localizedDescription.UTF8String); exit(2); }
            return ps;
        };
        psLW = pso_for("k_load_weight"); psBR = pso_for("k_bitrev");
        psST = pso_for("k_stage");       psSQ = pso_for("k_sq");
        psMUL = pso_for("k_mul");        psFIN = pso_for("k_final");
        q = [dev newCommandQueue];

        auto mkbuf = [&](const void* src, size_t bytes) {
            return src ? [dev newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared]
                       : [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        };
        for (u32 p : primes) {
            PrimeGPU g; g.p = p; g.n0 = mont_n0(p); g.r2 = mont_r2(p);
            u32 psi = find_psi(p, N);
            u32 omega = (u32)mulmod(psi, psi, p);
            u32 omega_inv = (u32)modinv(omega, p);
            u32 psi_inv = (u32)modinv(psi, p);
            g.ninv_mont = to_mont((u32)modinv(N % p, p), p);
            std::vector<u32> Wf(N / 2), Wi(N / 2), WJ(N), WIJ(N);
            { u64 w = 1; for (u64 t = 0; t < N / 2; ++t) { Wf[t] = to_mont((u32)w, p); w = mulmod(w, omega, p); } }
            { u64 w = 1; for (u64 t = 0; t < N / 2; ++t) { Wi[t] = to_mont((u32)w, p); w = mulmod(w, omega_inv, p); } }
            { u64 w = 1; for (u64 j = 0; j < N; ++j) { WJ[j]  = to_mont((u32)w, p); w = mulmod(w, psi, p); } }
            { u64 w = 1; for (u64 j = 0; j < N; ++j) { WIJ[j] = to_mont((u32)w, p); w = mulmod(w, psi_inv, p); } }
            g.bWf = mkbuf(Wf.data(), Wf.size() * 4);
            g.bWi = mkbuf(Wi.data(), Wi.size() * 4);
            g.bWJ = mkbuf(WJ.data(), N * 4);
            g.bWIJ = mkbuf(WIJ.data(), N * 4);
            G.push_back(g);
        }
        bInX = mkbuf(nullptr, N * 4); bInY = mkbuf(nullptr, N * 4);
        bW = mkbuf(nullptr, N * 4);
        bFx = mkbuf(nullptr, N * 4); bFy = mkbuf(nullptr, N * 4);
        bT = mkbuf(nullptr, N * 4);
        return true;
    }

    // One negacyclic multiply (squaring when squaring==true) of signed base-b
    // digit vectors mod (b^N+1); returns balanced base-b digits.
    std::vector<i64> negamul(const std::vector<i64>& x, const std::vector<i64>& y,
                             bool squaring) {
        std::vector<i64> result;
        @autoreleasepool {
            std::vector<std::vector<u32>> cres(primes.size(), std::vector<u32>(N));
            id<MTLCommandBuffer> cb = [q commandBuffer];
            auto disp = [&](id<MTLComputePipelineState> ps, NSUInteger threads,
                            void (^setup)(id<MTLComputeCommandEncoder>)) {
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps]; setup(e);
                NSUInteger t = ps.maxTotalThreadsPerThreadgroup; if (t > threads) t = threads;
                [e dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(t,1,1)];
                [e endEncoding];
            };
            // reduce signed digits mod p into [0,p), upload, weight, transform
            auto forward = [&](PrimeGPU& g, const std::vector<i64>& src,
                               id<MTLBuffer> bIn, id<MTLBuffer> dst) {
                u32* in = (u32*)bIn.contents; u32 pr = g.p;
                for (u64 j = 0; j < N; ++j) { i64 r = src[j] % (i64)pr; if (r < 0) r += pr; in[j] = (u32)r; }
                id<MTLBuffer> WJ = g.bWJ, Wf = g.bWf; u32 p = g.p, n0 = g.n0, r2 = g.r2;
                disp(psLW, N, ^(id<MTLComputeCommandEncoder> e){
                    [e setBuffer:bIn offset:0 atIndex:0]; [e setBuffer:bW offset:0 atIndex:1];
                    [e setBuffer:WJ offset:0 atIndex:2];
                    [e setBytes:&p length:4 atIndex:3]; [e setBytes:&n0 length:4 atIndex:4];
                    [e setBytes:&r2 length:4 atIndex:5];
                });
                disp(psBR, N, ^(id<MTLComputeCommandEncoder> e){
                    [e setBuffer:bW offset:0 atIndex:0]; [e setBuffer:dst offset:0 atIndex:1];
                    [e setBytes:&ln length:4 atIndex:2];
                });
                for (u32 len = 2; len <= nn; len <<= 1) { u32 L = len;
                    disp(psST, N/2, ^(id<MTLComputeCommandEncoder> e){
                        [e setBuffer:dst offset:0 atIndex:0]; [e setBuffer:Wf offset:0 atIndex:1];
                        [e setBytes:&p length:4 atIndex:2]; [e setBytes:&n0 length:4 atIndex:3];
                        [e setBytes:&L length:4 atIndex:4]; [e setBytes:&nn length:4 atIndex:5];
                    });
                }
            };
            for (size_t pi = 0; pi < G.size(); ++pi) {
                PrimeGPU& g = G[pi]; u32 p = g.p, n0 = g.n0, ninv = g.ninv_mont;
                id<MTLBuffer> Wi = g.bWi, WIJ = g.bWIJ;
                forward(g, x, bInX, bFx);
                if (squaring) {
                    disp(psSQ, N, ^(id<MTLComputeCommandEncoder> e){
                        [e setBuffer:bFx offset:0 atIndex:0];
                        [e setBytes:&p length:4 atIndex:1]; [e setBytes:&n0 length:4 atIndex:2];
                    });
                } else {
                    forward(g, y, bInY, bFy);
                    disp(psMUL, N, ^(id<MTLComputeCommandEncoder> e){
                        [e setBuffer:bFx offset:0 atIndex:0]; [e setBuffer:bFy offset:0 atIndex:1];
                        [e setBytes:&p length:4 atIndex:2]; [e setBytes:&n0 length:4 atIndex:3];
                    });
                }
                disp(psBR, N, ^(id<MTLComputeCommandEncoder> e){
                    [e setBuffer:bFx offset:0 atIndex:0]; [e setBuffer:bT offset:0 atIndex:1];
                    [e setBytes:&ln length:4 atIndex:2];
                });
                for (u32 len = 2; len <= nn; len <<= 1) { u32 L = len;
                    disp(psST, N/2, ^(id<MTLComputeCommandEncoder> e){
                        [e setBuffer:bT offset:0 atIndex:0]; [e setBuffer:Wi offset:0 atIndex:1];
                        [e setBytes:&p length:4 atIndex:2]; [e setBytes:&n0 length:4 atIndex:3];
                        [e setBytes:&L length:4 atIndex:4]; [e setBytes:&nn length:4 atIndex:5];
                    });
                }
                disp(psFIN, N, ^(id<MTLComputeCommandEncoder> e){
                    [e setBuffer:bT offset:0 atIndex:0]; [e setBuffer:WIJ offset:0 atIndex:1];
                    [e setBytes:&p length:4 atIndex:2]; [e setBytes:&n0 length:4 atIndex:3];
                    [e setBytes:&ninv length:4 atIndex:4];
                });
                [cb commit]; [cb waitUntilCompleted];
                if (cb.error) { fprintf(stderr, "GPU error: %s\n",
                               cb.error.localizedDescription.UTF8String); exit(2); }
                memcpy(cres[pi].data(), bT.contents, N * 4);
                cb = [q commandBuffer];
            }
            std::vector<i128> c(N);
            std::vector<u32> col(primes.size());
            for (u64 j = 0; j < N; ++j) {
                for (size_t pi = 0; pi < primes.size(); ++pi) col[pi] = cres[pi][j];
                c[j] = crt_balanced(col, primes);
            }
            result = carry_balanced(c, b);
        }
        return result;
    }
};

static int selftest_negamul(int k, u64 b, int reps) {
    Engine eng;
    if (!eng.init(k, b)) return 2;
    const u64 N = eng.N;

    printf("Metal device: %s\n", eng.dev.name.UTF8String);
    printf("negamul self-test: k=%d N=%llu b=%llu  %zu NTT primes  %d reps\n",
           k, (unsigned long long)N, (unsigned long long)b, eng.primes.size(), reps);

    mpz_t BN1, X, Y, Z, GOT; mpz_inits(BN1, X, Y, Z, GOT, nullptr);
    mpz_ui_pow_ui(BN1, (unsigned long)b, (unsigned long)N); mpz_add_ui(BN1, BN1, 1);
    std::mt19937 rng(0x5EED ^ (unsigned)k ^ (unsigned)b);
    int fails = 0;
    for (int r = 0; r < reps; ++r) {
        std::vector<i64> x(N), y(N);
        for (u64 j = 0; j < N; ++j) { x[j] = rng() % (u32)b; y[j] = rng() % (u32)b; }
        digits_to_mpz(X, x, b); digits_to_mpz(Y, y, b);

        std::vector<i64> zm = eng.negamul(x, y, false);        // multiply path
        mpz_mul(Z, X, Y); mpz_mod(Z, Z, BN1);
        digits_to_mpz(GOT, zm, b); mpz_mod(GOT, GOT, BN1);
        if (mpz_cmp(Z, GOT) != 0) { ++fails; if (fails <= 3) printf("  MUL mismatch rep %d\n", r); }

        std::vector<i64> zs = eng.negamul(x, x, true);         // square path
        mpz_mul(Z, X, X); mpz_mod(Z, Z, BN1);
        digits_to_mpz(GOT, zs, b); mpz_mod(GOT, GOT, BN1);
        if (mpz_cmp(Z, GOT) != 0) { ++fails; if (fails <= 3) printf("  SQ mismatch rep %d\n", r); }
    }
    mpz_clears(BN1, X, Y, Z, GOT, nullptr);
    printf("  GPU negamul vs GMP: %s (%d failures over %d reps x2 ops)\n",
           fails == 0 ? "OK" : "FAIL", fails, reps);
    return fails == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Stage 4: full strong-PRP powering a^(M-1) mod M on the GPU.  Run the whole
// chain mod (b^N+1) with the NTT engine (left-to-right binary), reduce to
// M = (b^N+1)/2 only at the end, and compare the residue AND the prime/composite
// verdict to GMP's mpz_powm.
// ---------------------------------------------------------------------------
static int selftest_prp(int k, u64 b, unsigned long base_a) {
    Engine eng;
    if (!eng.init(k, b)) return 2;
    const u64 N = eng.N;

    mpz_t BN1, M, E, a_mpz, a_red, ref, got;
    mpz_inits(BN1, M, E, a_mpz, a_red, ref, got, nullptr);
    mpz_ui_pow_ui(BN1, (unsigned long)b, (unsigned long)N); mpz_add_ui(BN1, BN1, 1);
    mpz_fdiv_q_ui(M, BN1, 2);              // M = (b^N+1)/2
    mpz_sub_ui(E, M, 1);                   // E = M-1 = (b^N-1)/2
    mpz_set_ui(a_mpz, base_a);
    mpz_mod(a_red, a_mpz, BN1);

    printf("Metal device: %s\n", eng.dev.name.UTF8String);
    printf("prp self-test: k=%d N=%llu b=%llu base=%lu  |M|=%zu bits  %zu NTT primes\n",
           k, (unsigned long long)N, (unsigned long long)b, base_a,
           mpz_sizeinbase(M, 2), eng.primes.size());

    // GPU powering, residue in balanced base-b digits.
    std::vector<i64> acc = mpz_to_digits(a_red, N, b);
    std::vector<i64> res(N, 0); res[0] = 1;
    size_t bits = mpz_sizeinbase(E, 2);
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = bits; i-- > 0;) {
        res = eng.negamul(res, res, true);
        if (mpz_tstbit(E, (mp_bitcnt_t)i)) res = eng.negamul(res, acc, false);
    }
    double gpu_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    digits_to_mpz(got, res, b); mpz_mod(got, got, M);   // reduce to M at the very end

    auto t1 = std::chrono::steady_clock::now();
    mpz_powm(ref, a_mpz, E, M);                          // GMP oracle (1 core)
    double cpu_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    bool ok = (mpz_cmp(got, ref) == 0);
    bool prp = (mpz_cmp_ui(ref, 1) == 0);
    printf("  GPU residue vs GMP: %s   |   M is %s\n",
           ok ? "MATCH" : "MISMATCH", prp ? "probable prime" : "composite");
    printf("  timing: GPU %.3f s   CPU/GMP(1 core) %.4f s   -> GPU is %.1fx %s\n",
           gpu_s, cpu_s, gpu_s > cpu_s ? gpu_s / cpu_s : cpu_s / gpu_s,
           gpu_s > cpu_s ? "SLOWER" : "faster");

    mpz_clears(BN1, M, E, a_mpz, a_red, ref, got, nullptr);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    int k = 16;
    u64 b = 10001;
    int reps = 8;
    unsigned long base_a = 3;
    std::string mode;
    const char* usage = "Usage: %s --selftest {montmul|ntt|negamul|prp} [--k K] [--b B] [--base A] [--reps R]\n";
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nx = [&](const char* f) {
            if (i + 1 >= argc) { fprintf(stderr, "missing arg for %s\n", f); exit(2); }
            return argv[++i];
        };
        if      (s == "--selftest") mode = nx("--selftest");
        else if (s == "--k")        k = atoi(nx("--k"));
        else if (s == "--b")        b = strtoull(nx("--b"), nullptr, 10);
        else if (s == "--base")     base_a = strtoul(nx("--base"), nullptr, 10);
        else if (s == "--reps")     reps = atoi(nx("--reps"));
        else { fprintf(stderr, usage, argv[0]); return 2; }
    }
    if (mode == "montmul")  return selftest_montmul(k);
    if (mode == "ntt")      return selftest_ntt(k);
    if (mode == "negamul")  return selftest_negamul(k, b, reps);
    if (mode == "prp")      return selftest_prp(k, b, base_a);
    fprintf(stderr, usage, argv[0]);
    return 2;
}
