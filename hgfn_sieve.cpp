// hgfn_sieve.cpp
//
// Sieve for halved generalized Fermat numbers  M(b) = (b^N + 1) / 2
// with N = 2^k and odd base b in [bmin, bmax].
//
// C++ port of ___attic/hgfn_sieve.py, tuned for throughput:
//   * mulmod/powmod via __int128 (no overflow up to p ~ 1.8e19)
//   * odd bases only in a byte array (index i <-> b = b0 + 2i)
//   * multithreaded over the sieve primes (std::thread, no OpenMP needed)
//
// Mathematical core (see the Python original in ___attic/):
//   If p is an odd prime divisor of b^N + 1, then b has order 2N modulo p,
//   so p = 1 (mod 2N). It is enough to consider primes p = 2N*j + 1.
//   x^N = -1 (mod p) has exactly N solutions: the odd powers of a primitive
//   2N-th root of unity r. Every base b congruent to one of them has the
//   factor p and is struck out.
//
// Example:
//   ./hgfn_sieve --k 15 --bmin 3 --bmax 1000001 --plimit 1e9 --out kand.txt
//
// Build:
//   cmake -G Ninja -B build && ninja -C build
//   (or directly: clang++ -O3 -std=c++17 -pthread hgfn_sieve.cpp -o hgfn_sieve)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <filesystem>

using u64 = uint64_t;

// 128-bit helper for mulmod. GCC/Clang have a native __int128; MSVC does not,
// so there we fall back to the x64 wide-multiply/divide intrinsics below.
#if defined(__SIZEOF_INT128__)
#define HGFN_HAVE_U128 1
using u128 = unsigned __int128;
#elif defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>  // _umul128, _udiv128
#endif

// Set by the SIGINT/SIGTERM handler: next block boundary -> checkpoint + exit.
static volatile std::sig_atomic_t g_stop = 0;
static void on_signal(int) {
    g_stop = 1;
}

// ---------------------------------------------------------------------------
// Modular arithmetic
// ---------------------------------------------------------------------------
static inline u64 mulmod(u64 a, u64 b, u64 m) {
#if defined(HGFN_HAVE_U128)
    return (u64)((u128)a * b % m);
#elif defined(_MSC_VER) && defined(_M_X64)
    // 128-bit product, then 128-by-64 divide for the remainder. _udiv128
    // requires the high dividend < divisor, so reduce it mod m first.
    u64 hi;
    u64 lo = _umul128(a, b, &hi);
    u64 rem;
    _udiv128(hi % m, lo, m, &rem);
    return rem;
#else
    // Portable fallback: double-and-add mulmod (no 128-bit type needed).
    u64 r = 0;
    a %= m;
    while (b) {
        if (b & 1) {
            r = (r + a) % m;
        }
        a = (a << 1) % m;
        b >>= 1;
    }
    return r;
#endif
}

static inline u64 powmod(u64 a, u64 e, u64 m) {
    u64 r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1) {
            r = mulmod(r, a, m);
        }
        a = mulmod(a, a, m);
        e >>= 1;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Deterministic Miller-Rabin (correct for all 64-bit n)
// ---------------------------------------------------------------------------
static const u64 MR_BASES[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};

static bool is_prime(u64 n) {
    if (n < 2) {
        return false;
    }
    for (u64 q : MR_BASES) {
        if (n % q == 0) {
            return n == q;
        }
    }
    u64 d = n - 1;
    int s = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        ++s;
    }
    for (u64 a : MR_BASES) {
        u64 x = powmod(a, d, n);
        if (x == 1 || x == n - 1) {
            continue;
        }
        bool composite = true;
        for (int i = 0; i < s - 1; ++i) {
            x = mulmod(x, x, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        if (composite) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Primitive 2N-th root of unity: r with r^N = -1 (mod p)
// ---------------------------------------------------------------------------
static u64 primitive_2N_root(u64 p, u64 N) {
    u64 e = (p - 1) / (2 * N);
    for (u64 a = 2;; ++a) {
        u64 r = powmod(a, e, p);
        if (powmod(r, N, p) == p - 1) {
            return r;
        }
    }
}

// ---------------------------------------------------------------------------
// Strike algebraically composite bases up front:  b = c^e (e >= 3 odd).
// Then b^N + 1 is divisible by c^N + 1. Overflow-safe via a pre-multiply
// divide check, so no 128-bit type is needed here.
// ---------------------------------------------------------------------------
static void strike_odd_powers(std::vector<uint8_t>& alive, int64_t b0, int64_t bmax) {
    const u64 limit = (u64)bmax;
    for (int e = 3; ; e += 2) {
        // 3^e <= bmax ?  (overflow-safe: stop before t*3 could exceed bmax)
        u64 t = 1;
        bool too_big = false;
        for (int i = 0; i < e; ++i) {
            if (t > limit / 3) {
                too_big = true;
                break;
            }
            t *= 3;
        }
        if (too_big) {
            break;
        }

        for (int64_t c = 3; ; c += 2) {
            u64 b = 1;
            bool over = false;
            for (int i = 0; i < e; ++i) {
                if (b > limit / (u64)c) {
                    over = true;
                    break;
                }
                b *= (u64)c;
            }
            if (over) {
                break;
            }
            int64_t bi = (int64_t)b;
            if (bi >= b0) {
                alive[(size_t)((bi - b0) / 2)] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Strike every affected base for a single sieve prime p.
// (Writes 0 only -> benign under concurrency: idempotent.)
// ---------------------------------------------------------------------------
static void strike_prime(uint8_t* alive, size_t count, u64 p, u64 N, int64_t b0) {
    u64 r  = primitive_2N_root(p, N);
    u64 r2 = mulmod(r, r, p);
    u64 x  = r;
    const int64_t twop = (int64_t)(2 * p);
    for (u64 i = 0; i < N; ++i) {
        // b must be odd -> pin the residue class modulo 2p
        int64_t y = (x & 1) ? (int64_t)x : (int64_t)(x + p);
        int64_t off = ((y - b0) % twop + twop) % twop;
        int64_t first = b0 + off;              // smallest b >= b0 with b = y (mod 2p)
        size_t idx = (size_t)((first - b0) / 2);
        for (; idx < count; idx += p) {
            alive[idx] = 0;
        }
        x = mulmod(x, r2, p);
    }
}

// ---------------------------------------------------------------------------
// Checkpoint: parameters + progress (next_j) + the full alive array.
// Format (little-endian, same machine):
//   magic[8]="HGFNCK01", int k, int64 bmin, int64 bmax,
//   u64 count, u64 next_j, u64 primes_used, then count bytes of alive.
// ---------------------------------------------------------------------------
static const char CKPT_MAGIC[8] = {'H', 'G', 'F', 'N', 'C', 'K', '0', '1'};

static bool save_checkpoint(const std::string& path, int k, int64_t bmin,
                            int64_t bmax, u64 next_j, u64 primes_used,
                            const std::vector<uint8_t>& alive) {
    std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "  [warning: checkpoint %s not writable]\n", tmp.c_str());
        return false;
    }
    u64 count = alive.size();
    bool ok = true;
    ok = ok && fwrite(CKPT_MAGIC, 1, 8, f) == 8;
    ok = ok && fwrite(&k, sizeof(k), 1, f) == 1;
    ok = ok && fwrite(&bmin, sizeof(bmin), 1, f) == 1;
    ok = ok && fwrite(&bmax, sizeof(bmax), 1, f) == 1;
    ok = ok && fwrite(&count, sizeof(count), 1, f) == 1;
    ok = ok && fwrite(&next_j, sizeof(next_j), 1, f) == 1;
    ok = ok && fwrite(&primes_used, sizeof(primes_used), 1, f) == 1;
    ok = ok && fwrite(alive.data(), 1, count, f) == count;
    // Force to disk, then rename atomically.
    ok = ok && fflush(f) == 0;
    fclose(f);
    if (!ok) {
        fprintf(stderr, "  [warning: checkpoint write failed]\n");
        remove(tmp.c_str());
        return false;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "  [warning: checkpoint rename failed]\n");
        return false;
    }
    return true;
}

// Load a checkpoint if present AND the parameters match. On mismatch:
// error message + exit (so we never overwrite someone else's checkpoint).
static bool load_checkpoint(const std::string& path, int k, int64_t bmin,
                            int64_t bmax, std::vector<uint8_t>& alive,
                            u64& next_j, u64& primes_used) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;   // no checkpoint -> start fresh
    }
    char magic[8];
    int fk = 0;
    int64_t fbmin = 0, fbmax = 0;
    u64 fcount = 0, fnext = 0, fprimes = 0;
    bool ok = true;
    ok = ok && fread(magic, 1, 8, f) == 8;
    ok = ok && memcmp(magic, CKPT_MAGIC, 8) == 0;
    ok = ok && fread(&fk, sizeof(fk), 1, f) == 1;
    ok = ok && fread(&fbmin, sizeof(fbmin), 1, f) == 1;
    ok = ok && fread(&fbmax, sizeof(fbmax), 1, f) == 1;
    ok = ok && fread(&fcount, sizeof(fcount), 1, f) == 1;
    ok = ok && fread(&fnext, sizeof(fnext), 1, f) == 1;
    ok = ok && fread(&fprimes, sizeof(fprimes), 1, f) == 1;
    if (!ok) {
        fclose(f);
        fprintf(stderr, "Checkpoint %s is corrupt -- aborting.\n", path.c_str());
        exit(1);
    }
    if (fk != k || fbmin != bmin || fbmax != bmax || fcount != alive.size()) {
        fclose(f);
        fprintf(stderr,
                "Checkpoint %s does not match this invocation.\n"
                "  checkpoint: k=%d bmin=%lld bmax=%lld count=%llu\n"
                "  invocation: k=%d bmin=%lld bmax=%lld count=%llu\n"
                "Choose a different --checkpoint name or delete the file.\n",
                path.c_str(), fk, (long long)fbmin, (long long)fbmax,
                (unsigned long long)fcount, k, (long long)bmin, (long long)bmax,
                (unsigned long long)alive.size());
        exit(1);
    }
    if (fread(alive.data(), 1, fcount, f) != fcount) {
        fclose(f);
        fprintf(stderr, "Checkpoint %s: alive array incomplete -- aborting.\n", path.c_str());
        exit(1);
    }
    fclose(f);
    next_j = fnext;
    primes_used = fprimes;
    return true;
}

// ---------------------------------------------------------------------------
// The sieve itself (multithreaded, block by block, resumable)
// ---------------------------------------------------------------------------
struct SieveResult {
    std::vector<int64_t> survivors;
    u64 primes_used;
    double secs;
    bool interrupted;
};

static SieveResult sieve(int k, int64_t bmin, int64_t bmax, u64 plimit,
                         int nthreads, const std::string& ckpt_path,
                         double ckpt_interval, const std::string& pause_file,
                         double report_every = 10.0) {
    const u64 N = (u64)1 << k;
    const u64 step = 2 * N;

    int64_t b0 = (bmin % 2 == 1) ? bmin : bmin + 1;
    size_t count = (size_t)((bmax - b0) / 2 + 1);
    std::vector<uint8_t> alive(count, 1);

    // Candidates p = step*j + 1, j = 1 .. jmax
    u64 jmax = (plimit >= 1) ? (plimit - 1) / step : 0;
    u64 next_j = 1;
    u64 primes_start = 0;

    bool resumed = false;
    if (!ckpt_path.empty()) {
        resumed = load_checkpoint(ckpt_path, k, bmin, bmax, alive, next_j, primes_start);
    }
    if (resumed) {
        fprintf(stderr, "Checkpoint loaded: resuming at j=%llu (p=%llu), "
                        "%llu sieve primes already done.\n",
                (unsigned long long)next_j, (unsigned long long)(step * next_j + 1),
                (unsigned long long)primes_start);
    } else {
        // Fresh start: strike algebraically composite bases.
        strike_odd_powers(alive, b0, bmax);
    }

    auto t_start = std::chrono::steady_clock::now();
    auto t_last_report = t_start;
    auto t_last_ckpt = t_start;

    std::atomic<u64> primes_used{primes_start};
    uint8_t* alive_ptr = alive.data();

    const u64 CHUNK = (u64)1 << 16;   // j-candidates per block (sync/stop granularity)
    bool interrupted = false;

    u64 j = next_j;
    while (j <= jmax) {
        // Pause? (at a block boundary, like checkpoint/stop) Triggered by the
        // existence of the pause file -- cross-platform and usable in the
        // background too. Write a checkpoint first, just in case.
        if (!pause_file.empty() && std::filesystem::exists(pause_file)) {
            if (!ckpt_path.empty()) {
                if (save_checkpoint(ckpt_path, k, bmin, bmax, j, primes_used.load(), alive)) {
                    fprintf(stderr, "  [checkpoint before pause: next_j=%llu]\n",
                            (unsigned long long)j);
                }
            }
            fprintf(stderr, "Paused (file '%s' present) -- remove it to resume.\n",
                    pause_file.c_str());
            fflush(stderr);
            while (std::filesystem::exists(pause_file) && !g_stop) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            if (!g_stop) {
                fprintf(stderr, "Resumed.\n");
                fflush(stderr);
            }
            // Do not count the pause against the report/checkpoint intervals.
            auto resume_now = std::chrono::steady_clock::now();
            t_last_report = resume_now;
            t_last_ckpt = resume_now;
        }

        if (g_stop) {   // stop (possibly during the pause) -> leave the loop
            break;
        }

        u64 chunk_end = j + CHUNK;
        if (chunk_end > jmax + 1) {
            chunk_end = jmax + 1;
        }

        auto worker = [&](int tid) {
            for (u64 jj = j + (u64)tid; jj < chunk_end; jj += nthreads) {
                u64 p = step * jj + 1;
                if (!is_prime(p)) {
                    continue;
                }
                strike_prime(alive_ptr, count, p, N, b0);
                primes_used.fetch_add(1, std::memory_order_relaxed);
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t) {
            pool.emplace_back(worker, t);
        }
        for (auto& th : pool) {
            th.join();
        }

        j = chunk_end;   // all j < chunk_end are done now -> clean boundary

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_start).count();

        if (std::chrono::duration<double>(now - t_last_report).count() >= report_every) {
            t_last_report = now;
            fprintf(stderr, "p up to %16llu  primes: %10llu  time: %6.0f s\n",
                    (unsigned long long)(step * (j - 1) + 1),
                    (unsigned long long)primes_used.load(), elapsed);
            fflush(stderr);
        }

        bool time_to_ckpt =
            std::chrono::duration<double>(now - t_last_ckpt).count() >= ckpt_interval;
        if (!ckpt_path.empty() && (time_to_ckpt || g_stop)) {
            if (save_checkpoint(ckpt_path, k, bmin, bmax, j, primes_used.load(), alive)) {
                fprintf(stderr, "  [checkpoint saved: next_j=%llu]\n",
                        (unsigned long long)j);
                fflush(stderr);
            }
            t_last_ckpt = now;
        }

        if (g_stop) {   // stop after a completed block -> loop top breaks out
            continue;
        }
    }

    if (g_stop) {
        interrupted = true;
        fprintf(stderr, "Stop requested -- ");
        if (ckpt_path.empty()) {
            fprintf(stderr, "no --checkpoint set, progress is lost.\n");
        } else {
            fprintf(stderr, "checkpoint written, the same command resumes.\n");
        }
    }

    double secs = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t_start).count();

    std::vector<int64_t> survivors;
    if (!interrupted) {
        for (size_t i = 0; i < count; ++i) {
            if (alive[i]) {
                survivors.push_back(b0 + 2 * (int64_t)i);
            }
        }
    }

    return {std::move(survivors), primes_used.load(), secs, interrupted};
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int k = -1;
    int64_t bmin = 3, bmax = -1;
    double plimit_d = 1e8;
    std::string out = "candidates.txt";
    std::string ckpt_path;
    double ckpt_interval = 60.0;
    std::string pause_file;
    int nthreads = (int)std::thread::hardware_concurrency();
    if (nthreads < 1) {
        nthreads = 1;
    }

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", name);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--k") {
            k = atoi(next("--k"));
        } else if (a == "--bmin") {
            bmin = atoll(next("--bmin"));
        } else if (a == "--bmax") {
            bmax = atoll(next("--bmax"));
        } else if (a == "--plimit") {
            plimit_d = atof(next("--plimit"));
        } else if (a == "--out") {
            out = next("--out");
        } else if (a == "--threads") {
            nthreads = atoi(next("--threads"));
        } else if (a == "--checkpoint") {
            ckpt_path = next("--checkpoint");
        } else if (a == "--checkpoint-interval") {
            ckpt_interval = atof(next("--checkpoint-interval"));
        } else if (a == "--pause-file") {
            pause_file = next("--pause-file");
        } else if (a == "-h" || a == "--help") {
            printf("Usage: %s --k K --bmax BMAX [--bmin 3] [--plimit 1e8] "
                   "[--out candidates.txt] [--threads N]\n"
                   "       [--checkpoint FILE] [--checkpoint-interval SEC] [--pause-file FILE]\n"
                   "  If the checkpoint file exists, the run resumes automatically.\n"
                   "  Ctrl-C writes a checkpoint at the next block boundary and exits.\n"
                   "  --pause-file: while the file exists, the sieve pauses at a block\n"
                   "                boundary (remove it to resume).\n",
                   argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (k < 0 || bmax < 0) {
        fprintf(stderr, "Error: --k and --bmax are required. (--help for usage)\n");
        return 2;
    }
    if (nthreads < 1) {
        nthreads = 1;
    }
    u64 plimit = (u64)plimit_d;

    // Catch Ctrl-C / kill cleanly -> checkpoint at the next block boundary.
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    SieveResult res = sieve(k, bmin, bmax, plimit, nthreads, ckpt_path, ckpt_interval,
                            pause_file);

    if (res.interrupted) {
        printf("\nAborted after %.1f s (%llu sieve primes). "
               "No candidate file written.\n",
               res.secs, (unsigned long long)res.primes_used);
        if (!ckpt_path.empty()) {
            printf("The same command resumes from checkpoint %s.\n", ckpt_path.c_str());
        }
        return 130;
    }

    int64_t total = (bmax - bmin) / 2 + 1;
    printf("\nN = 2^%d = %llu, bases %lld..%lld  (%d threads)\n",
           k, (unsigned long long)((u64)1 << k), (long long)bmin, (long long)bmax, nthreads);
    printf("%llu sieve primes up to %llu in %.1f s\n",
           (unsigned long long)res.primes_used, (unsigned long long)plimit, res.secs);
    printf("%zu of ~%lld odd bases survive (%.2f %%)\n",
           res.survivors.size(), (long long)total,
           100.0 * res.survivors.size() / (total > 0 ? total : 1));

    FILE* f = fopen(out.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Cannot write %s\n", out.c_str());
        return 1;
    }
    fprintf(f, "# (b^%llu+1)/2, sieved up to p = %llu\n",
            (unsigned long long)((u64)1 << k), (unsigned long long)plimit);
    for (int64_t b : res.survivors) {
        fprintf(f, "%lld\n", (long long)b);
    }
    fclose(f);
    printf("Candidates written to %s\n", out.c_str());
    return 0;
}
