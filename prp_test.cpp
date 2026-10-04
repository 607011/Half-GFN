// prp_test.cpp
//
// ARM-native PRP test (strong Fermat / Miller-Rabin) of the sieve survivors
// for M(b) = (b^N + 1)/2 with N = 2^k.  Replaces pfgw: only GMP as a dependency,
// portable C code -> runs natively on Apple Silicon (no x86/Rosetta, no gwnum
// assembler).
//
// Second step of the solo workflow: hgfn_sieve produces the candidate bases,
// this program tests each M(b) for (probable) primality.
//
// The exponent N is read automatically from the candidate file's header line
// ("# (b^N+1)/2, ...").
//
// Test: strong PRP test (Miller-Rabin) to one or more bases a.
//   Write M-1 = d * 2^s.  a witnesses "composite" if
//   a^d != 1 (mod M) AND a^(d*2^i) != -1 (mod M) for all 0<=i<s.
//   If M survives all bases -> "PRP" (probable prime; not a proof).
//   An a with 1 < gcd(a,M) < M proves "composite" (a real factor).
//
// Important: passing a PRP test is NOT a primality proof. It is the usual, very
// reliable pre-filter; the final proof (APR-CL/ECPP) is then run only for the
// few survivors (prove.sh).
//
// Build (Homebrew GMP on Apple Silicon):
//   clang++ -O3 -std=c++17 -I/opt/homebrew/include prp_test.cpp \
//           -L/opt/homebrew/lib -lgmp -pthread -o prp_test
//   (or: cmake -G Ninja -B build && ninja -C build prp_test)
//
// Usage:
//   ./prp_test [options] [candidate-file]        (default: kand.txt)
// Options:
//   --exp N        override the exponent (otherwise from the header)
//   --bases "a b"  PRP bases, space-separated (default: first 13 primes, 2..41)
//   --limit N      test only the first N candidates (0 = all)
//   --out FILE     write the PRP bases here (default prp.txt)
//   --threads N    threads (default: all cores)
//   -v, --verbose  also report composite candidates

#include <gmp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_set>

// Set by the SIGINT/SIGTERM handler: threads stop pulling new candidates;
// in-flight tests still finish and are written to the journal.
static volatile std::sig_atomic_t g_stop = 0;
static void on_signal(int) {
    g_stop = 1;
}

// ---------------------------------------------------------------------------
// Strong PRP test (Miller-Rabin) for an already-computed M to base a.
// Precondition: M odd and > 2. Uses precomputed d, s with M-1 = d*2^s.
// Returns: true = "probable prime to base a", false = "composite".
// ---------------------------------------------------------------------------
static bool strong_prp(const mpz_t M, const mpz_t Mm1, const mpz_t d,
                       unsigned long s, unsigned long a) {
    mpz_t base, x, gcd;
    mpz_inits(base, x, gcd, nullptr);
    mpz_set_ui(base, a);

    // gcd(a, M): if a real factor turns up, M is composite.
    mpz_gcd(gcd, base, M);
    if (mpz_cmp_ui(gcd, 1) != 0) {
        bool eq = (mpz_cmp(gcd, M) == 0);      // a is a multiple of M -> useless
        mpz_clears(base, x, gcd, nullptr);
        return eq;  // gcd==M (a>=M and divisible): inconclusive -> treat as "passed"
    }

    mpz_powm(x, base, d, M);                    // x = a^d mod M
    if (mpz_cmp_ui(x, 1) == 0 || mpz_cmp(x, Mm1) == 0) {
        mpz_clears(base, x, gcd, nullptr);
        return true;
    }
    for (unsigned long i = 1; i < s; ++i) {
        mpz_powm_ui(x, x, 2, M);                // x = x^2 mod M
        if (mpz_cmp(x, Mm1) == 0) {
            mpz_clears(base, x, gcd, nullptr);
            return true;
        }
        if (mpz_cmp_ui(x, 1) == 0) {            // 1 before -1 -> composite
            break;
        }
    }
    mpz_clears(base, x, gcd, nullptr);
    return false;
}

// ---------------------------------------------------------------------------
// Test a candidate b. Returns true if M(b) passes all bases (PRP).
// ---------------------------------------------------------------------------
static bool test_candidate(unsigned long b, unsigned long exp,
                           const std::vector<unsigned long>& bases) {
    mpz_t M, Mm1, d;
    mpz_inits(M, Mm1, d, nullptr);

    // M = (b^exp + 1) / 2
    mpz_ui_pow_ui(M, b, exp);
    mpz_add_ui(M, M, 1);
    mpz_fdiv_q_2exp(M, M, 1);                   // /2 (b odd -> b^exp+1 even)

    // M-1 = d * 2^s
    mpz_sub_ui(Mm1, M, 1);
    unsigned long s = mpz_scan1(Mm1, 0);        // number of trailing twos
    mpz_fdiv_q_2exp(d, Mm1, s);

    bool prp = true;
    for (unsigned long a : bases) {
        if (!strong_prp(M, Mm1, d, s, a)) {
            prp = false;
            break;
        }
    }
    mpz_clears(M, Mm1, d, nullptr);
    return prp;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string candfile, out = "prp.txt";
    std::string journal_path;
    long exp_override = -1;
    long limit = 0;
    int nthreads = (int)std::thread::hardware_concurrency();
    if (nthreads < 1) {
        nthreads = 1;
    }
    bool verbose = false;
    std::vector<unsigned long> bases;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* n) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", n);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--exp") {
            exp_override = atol(next("--exp").c_str());
        } else if (a == "--bases") {
            std::istringstream is(next("--bases"));
            unsigned long v;
            while (is >> v) {
                bases.push_back(v);
            }
        } else if (a == "--limit") {
            limit = atol(next("--limit").c_str());
        } else if (a == "--out") {
            out = next("--out");
        } else if (a == "--threads") {
            nthreads = atoi(next("--threads").c_str());
        } else if (a == "--verbose" || a == "-v") {
            verbose = true;
        } else if (a == "--journal") {
            journal_path = next("--journal");
        } else if (a == "-h" || a == "--help") {
            printf("Usage: %s [--exp N] [--bases \"2 3 5 ...\"] [--limit N] "
                   "[--out prp.txt] [--threads N] [-v|--verbose]\n"
                   "       [--journal FILE] [candidate-file]\n"
                   "  --bases: default is the first 13 primes (2 3 5 ... 41).\n"
                   "  -v:      also report composite candidates.\n"
                   "  --journal: every tested base is logged immediately; running\n"
                   "             again with the same journal skips those bases\n"
                   "             (resume). Ctrl-C exits cleanly.\n", argv[0]);
            return 0;
        } else if (a[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", a.c_str());
            return 2;
        } else {
            candfile = a;
        }
    }
    if (candfile.empty()) {
        candfile = "kand.txt";
    }
    if (bases.empty()) {
        // Default: the first 13 primes as strong-PRP bases. Composites almost
        // always fail the first base (2), so the extra bases cost next to nothing
        // (early-out) yet make a false "probable prime" vanishingly unlikely before
        // the proof stage -- and are the strongest practical claim at large k where
        // a full proof is infeasible. NB: these are *not* a deterministic witness
        // set at these sizes (that only holds for n < 3.3e24); they are 13
        // probabilistic rounds.
        bases = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};
    }
    if (nthreads < 1) {
        nthreads = 1;
    }

    // Read the candidate file
    std::ifstream in(candfile);
    if (!in) {
        fprintf(stderr, "Cannot open %s\n", candfile.c_str());
        return 1;
    }

    unsigned long exp = 0;
    if (exp_override > 0) {
        exp = (unsigned long)exp_override;
    }

    std::vector<unsigned long> cands;
    std::string line;
    while (std::getline(in, line)) {
        // Header:  "# (b^EXP+1)/2, ..."  -> extract the exponent
        if (!line.empty() && line[0] == '#') {
            if (exp == 0) {
                size_t caret = line.find('^');
                if (caret != std::string::npos) {
                    exp = strtoul(line.c_str() + caret + 1, nullptr, 10);
                }
            }
            continue;
        }
        if (line.empty()) {
            continue;
        }
        unsigned long b = strtoul(line.c_str(), nullptr, 10);
        if (b > 0) {
            cands.push_back(b);
        }
    }
    in.close();

    if (exp == 0) {
        fprintf(stderr, "Exponent not readable from header -- please pass --exp N.\n");
        return 1;
    }
    if (cands.empty()) {
        fprintf(stderr, "No candidates in %s.\n", candfile.c_str());
        return 1;
    }
    if (limit > 0 && (long)cands.size() > limit) {
        cands.resize(limit);
    }
    size_t total_all = cands.size();

    // --- Load the journal (resume): skip bases that were already tested ---
    // Journal line:  "<base> <0|1>"  (1 = was PRP). The PRP hits from the
    // journal go straight into prp_bases so the final output is complete.
    std::unordered_set<unsigned long> done_set;
    std::vector<unsigned long> prp_bases;
    if (!journal_path.empty()) {
        std::ifstream jin(journal_path);
        if (jin) {
            std::string jline;
            while (std::getline(jin, jline)) {
                if (jline.empty() || jline[0] == '#') {
                    continue;
                }
                std::istringstream is(jline);
                unsigned long jb = 0;
                int jr = 0;
                if (is >> jb >> jr) {           // silently skip torn/garbled lines
                    done_set.insert(jb);
                    if (jr == 1) {
                        prp_bases.push_back(jb);
                    }
                }
            }
            jin.close();
        }
    }

    // Filter candidates: drop those already done per the journal.
    if (!done_set.empty()) {
        std::vector<unsigned long> todo;
        todo.reserve(cands.size());
        for (unsigned long b : cands) {
            if (done_set.find(b) == done_set.end()) {
                todo.push_back(b);
            }
        }
        cands.swap(todo);
    }

    std::string basestr;
    for (size_t i = 0; i < bases.size(); ++i) {
        basestr += (i ? " " : "") + std::to_string(bases[i]);
    }

    // Catch Ctrl-C / kill cleanly -> still journal in-flight tests, then exit.
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    printf("Candidates:  %zu total", total_all);
    if (!journal_path.empty()) {
        printf(", %zu done per journal, %zu to test", done_set.size(), cands.size());
    }
    printf("  (from %s)\n", candfile.c_str());
    printf("Test:        M(b) = (b^%lu+1)/2, strong PRP to base(s) %s\n", exp, basestr.c_str());
    printf("Threads:     %d\n", nthreads);
    if (!journal_path.empty()) {
        printf("Journal:     %s\n", journal_path.c_str());
    }
    printf("Result ->    %s\n", out.c_str());
    printf("--------------------------------------------------------------------------\n");
    fflush(stdout);

    // Open the journal for appending (creates it if needed).
    FILE* jf = nullptr;
    if (!journal_path.empty()) {
        jf = fopen(journal_path.c_str(), "a");
        if (!jf) {
            fprintf(stderr, "Warning: journal %s not writable -- running without a journal.\n",
                    journal_path.c_str());
        }
    }
    const bool journaling = (jf != nullptr);

    auto t0 = std::chrono::steady_clock::now();
    std::atomic<size_t> next_idx{0};
    std::atomic<size_t> done{0};
    std::mutex mtx;

    auto worker = [&]() {
        for (;;) {
            if (g_stop) {                       // no more new candidates
                break;
            }
            size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (idx >= cands.size()) {
                break;
            }
            unsigned long b = cands[idx];
            bool prp = test_candidate(b, exp, bases);
            size_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;

            std::lock_guard<std::mutex> lg(mtx);
            if (jf) {                           // persist the result immediately
                fprintf(jf, "%lu %d\n", b, prp ? 1 : 0);
                fflush(jf);
            }
            if (prp) {
                prp_bases.push_back(b);
                double t = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0).count();
                printf("[%zu/%zu, %.0fs]  (%lu^%lu+1)/2  is PRP!\n",
                       d, cands.size(), t, b, exp);
                fflush(stdout);
            } else if (verbose) {
                printf("[%zu/%zu]  (%lu^%lu+1)/2  composite\n",
                       d, cands.size(), b, exp);
                fflush(stdout);
            }
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t) {
        pool.emplace_back(worker);
    }
    for (auto& th : pool) {
        th.join();
    }
    if (jf) {
        fclose(jf);
    }

    double secs = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0).count();

    if (g_stop != 0) {
        printf("--------------------------------------------------------------------------\n");
        printf("Aborted after %.1f s (%zu bases tested in this run).\n",
               secs, done.load());
        if (journaling) {
            printf("Progress is in the journal %s -- the same command resumes. "
                   "%s is written only on a complete run.\n",
                   journal_path.c_str(), out.c_str());
        } else {
            printf("Without a (writable) --journal the progress is lost.\n");
        }
        return 130;
    }

    std::sort(prp_bases.begin(), prp_bases.end());
    prp_bases.erase(std::unique(prp_bases.begin(), prp_bases.end()), prp_bases.end());

    std::ofstream of(out);
    of << "# PRP bases for (b^" << exp << "+1)/2, base(s) " << basestr << "\n";
    for (unsigned long b : prp_bases) {
        of << b << "\n";
    }
    of.close();

    printf("--------------------------------------------------------------------------\n");
    printf("Done in %.1f s. %zu bases tested in this run, %zu PRP total -> %s\n",
           secs, done.load(), prp_bases.size(), out.c_str());
    return 0;
}
