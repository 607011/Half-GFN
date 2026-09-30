// hgfn_sieve.cpp
//
// Sieb fuer halbierte verallgemeinerte Fermat-Zahlen  M(b) = (b^N + 1) / 2
// mit N = 2^k und ungerader Basis b in [bmin, bmax].
//
// C++-Portierung von hgfn_sieve.py, optimiert auf Durchsatz:
//   * mulmod/powmod ueber __int128 (kein Overflow bis p ~ 1.8e19)
//   * nur ungerade Basen im Bit-... aeh Byte-Array (Index i <-> b = b0 + 2i)
//   * Multithreading ueber die Siebprimzahlen (std::thread, kein OpenMP noetig)
//
// Mathematischer Kern (siehe Python-Original):
//   Ist p ein ungerader Primteiler von b^N + 1, dann hat b mod p die Ordnung
//   2N, also p = 1 (mod 2N). Es genuegt, Primzahlen p = 2N*j + 1 zu betrachten.
//   x^N = -1 (mod p) hat genau N Loesungen: die ungeraden Potenzen einer
//   primitiven 2N-ten Einheitswurzel r. Jede Basis b, die zu einer davon
//   kongruent ist, hat den Faktor p und wird gestrichen.
//
// Aufruf (Beispiel):
//   ./hgfn_sieve --k 15 --bmin 3 --bmax 1000001 --plimit 1e9 --out kand.txt
//
// Kompilieren:
//   clang++ -O3 -std=c++17 -pthread hgfn_sieve.cpp -o hgfn_sieve
//   (oder: make)

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

using u64 = uint64_t;
using u128 = unsigned __int128;

// Wird vom SIGINT/SIGTERM-Handler gesetzt: naechster Blockrand -> Checkpoint + Ende.
static volatile std::sig_atomic_t g_stop = 0;
static void on_signal(int) {
    g_stop = 1;
}

// ---------------------------------------------------------------------------
// Modulare Arithmetik
// ---------------------------------------------------------------------------
static inline u64 mulmod(u64 a, u64 b, u64 m) {
    return (u64)((u128)a * b % m);
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
// Deterministischer Miller-Rabin (korrekt fuer alle 64-Bit-n)
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
// Primitive 2N-te Einheitswurzel: r mit r^N = -1 (mod p)
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
// Algebraisch zusammengesetzte Basen vorab streichen:  b = c^e (e >= 3 ungerade)
// Dann ist b^N + 1 durch c^N + 1 teilbar. Overflow-sicher via __int128.
// ---------------------------------------------------------------------------
static void strike_odd_powers(std::vector<uint8_t>& alive, int64_t b0, int64_t bmax) {
    for (int e = 3; ; e += 2) {
        // 3^e <= bmax ?  (overflow-sicher)
        u128 t = 1;
        bool too_big = false;
        for (int i = 0; i < e; ++i) {
            t *= 3;
            if (t > (u128)bmax) {
                too_big = true;
                break;
            }
        }
        if (too_big) {
            break;
        }

        for (int64_t c = 3; ; c += 2) {
            u128 b = 1;
            bool over = false;
            for (int i = 0; i < e; ++i) {
                b *= (u128)c;
                if (b > (u128)bmax) {
                    over = true;
                    break;
                }
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
// Streicht fuer eine einzelne Siebprimzahl p alle betroffenen Basen.
// (Schreibt nur 0-Werte -> nebenlaeufig unkritisch: idempotent.)
// ---------------------------------------------------------------------------
static void strike_prime(uint8_t* alive, size_t count, u64 p, u64 N, int64_t b0) {
    u64 r  = primitive_2N_root(p, N);
    u64 r2 = mulmod(r, r, p);
    u64 x  = r;
    const int64_t twop = (int64_t)(2 * p);
    for (u64 i = 0; i < N; ++i) {
        // b muss ungerade sein -> Restklasse modulo 2p festlegen
        int64_t y = (x & 1) ? (int64_t)x : (int64_t)(x + p);
        int64_t off = ((y - b0) % twop + twop) % twop;
        int64_t first = b0 + off;              // kleinstes b >= b0 mit b = y (mod 2p)
        size_t idx = (size_t)((first - b0) / 2);
        for (; idx < count; idx += p) {
            alive[idx] = 0;
        }
        x = mulmod(x, r2, p);
    }
}

// ---------------------------------------------------------------------------
// Checkpoint: Parameter + Fortschritt (next_j) + das komplette alive-Array.
// Format (little-endian, gleiche Maschine):
//   magic[8]="HGFNCK01", int k, int64 bmin, int64 bmax,
//   u64 count, u64 next_j, u64 primes_used, dann count Bytes alive.
// ---------------------------------------------------------------------------
static const char CKPT_MAGIC[8] = {'H', 'G', 'F', 'N', 'C', 'K', '0', '1'};

static bool save_checkpoint(const std::string& path, int k, int64_t bmin,
                            int64_t bmax, u64 next_j, u64 primes_used,
                            const std::vector<uint8_t>& alive) {
    std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "  [Warnung: Checkpoint %s nicht schreibbar]\n", tmp.c_str());
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
    // Auf Platte zwingen, dann atomar umbenennen.
    ok = ok && fflush(f) == 0;
    fclose(f);
    if (!ok) {
        fprintf(stderr, "  [Warnung: Checkpoint-Schreiben fehlgeschlagen]\n");
        remove(tmp.c_str());
        return false;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "  [Warnung: Checkpoint-Rename fehlgeschlagen]\n");
        return false;
    }
    return true;
}

// Laedt Checkpoint, wenn vorhanden UND Parameter passen. Bei Nichtpassung:
// Fehlermeldung + exit (um einen fremden Checkpoint nicht zu ueberschreiben).
static bool load_checkpoint(const std::string& path, int k, int64_t bmin,
                            int64_t bmax, std::vector<uint8_t>& alive,
                            u64& next_j, u64& primes_used) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;   // kein Checkpoint -> frisch starten
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
        fprintf(stderr, "Checkpoint %s ist beschaedigt -- Abbruch.\n", path.c_str());
        exit(1);
    }
    if (fk != k || fbmin != bmin || fbmax != bmax || fcount != alive.size()) {
        fclose(f);
        fprintf(stderr,
                "Checkpoint %s passt nicht zu diesem Aufruf.\n"
                "  Checkpoint: k=%d bmin=%lld bmax=%lld count=%llu\n"
                "  Aufruf:     k=%d bmin=%lld bmax=%lld count=%llu\n"
                "Anderen --checkpoint-Namen waehlen oder Datei loeschen.\n",
                path.c_str(), fk, (long long)fbmin, (long long)fbmax,
                (unsigned long long)fcount, k, (long long)bmin, (long long)bmax,
                (unsigned long long)alive.size());
        exit(1);
    }
    if (fread(alive.data(), 1, fcount, f) != fcount) {
        fclose(f);
        fprintf(stderr, "Checkpoint %s: alive-Array unvollstaendig -- Abbruch.\n", path.c_str());
        exit(1);
    }
    fclose(f);
    next_j = fnext;
    primes_used = fprimes;
    return true;
}

// ---------------------------------------------------------------------------
// Das eigentliche Sieb (multithreaded, blockweise, resumbar)
// ---------------------------------------------------------------------------
struct SieveResult {
    std::vector<int64_t> survivors;
    u64 primes_used;
    double secs;
    bool interrupted;
};

static SieveResult sieve(int k, int64_t bmin, int64_t bmax, u64 plimit,
                         int nthreads, const std::string& ckpt_path,
                         double ckpt_interval, double report_every = 10.0) {
    const u64 N = (u64)1 << k;
    const u64 step = 2 * N;

    int64_t b0 = (bmin % 2 == 1) ? bmin : bmin + 1;
    size_t count = (size_t)((bmax - b0) / 2 + 1);
    std::vector<uint8_t> alive(count, 1);

    // Kandidaten p = step*j + 1, j = 1 .. jmax
    u64 jmax = (plimit >= 1) ? (plimit - 1) / step : 0;
    u64 next_j = 1;
    u64 primes_start = 0;

    bool resumed = false;
    if (!ckpt_path.empty()) {
        resumed = load_checkpoint(ckpt_path, k, bmin, bmax, alive, next_j, primes_start);
    }
    if (resumed) {
        fprintf(stderr, "Checkpoint geladen: weiter ab j=%llu (p=%llu), "
                        "bereits %llu Siebprimzahlen.\n",
                (unsigned long long)next_j, (unsigned long long)(step * next_j + 1),
                (unsigned long long)primes_start);
    } else {
        // Frischer Start: algebraisch zusammengesetzte Basen streichen.
        strike_odd_powers(alive, b0, bmax);
    }

    auto t_start = std::chrono::steady_clock::now();
    auto t_last_report = t_start;
    auto t_last_ckpt = t_start;

    std::atomic<u64> primes_used{primes_start};
    uint8_t* alive_ptr = alive.data();

    const u64 CHUNK = (u64)1 << 16;   // j-Kandidaten pro Block (Sync-/Stop-Granularitaet)
    bool interrupted = false;

    u64 j = next_j;
    while (j <= jmax) {
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

        j = chunk_end;   // alle j < chunk_end sind jetzt erledigt -> sauberer Rand

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_start).count();

        if (std::chrono::duration<double>(now - t_last_report).count() >= report_every) {
            t_last_report = now;
            fprintf(stderr, "p bis %16llu  Primzahlen: %10llu  Zeit: %6.0f s\n",
                    (unsigned long long)(step * (j - 1) + 1),
                    (unsigned long long)primes_used.load(), elapsed);
            fflush(stderr);
        }

        bool time_to_ckpt =
            std::chrono::duration<double>(now - t_last_ckpt).count() >= ckpt_interval;
        if (!ckpt_path.empty() && (time_to_ckpt || g_stop)) {
            if (save_checkpoint(ckpt_path, k, bmin, bmax, j, primes_used.load(), alive)) {
                fprintf(stderr, "  [Checkpoint gespeichert: next_j=%llu]\n",
                        (unsigned long long)j);
                fflush(stderr);
            }
            t_last_ckpt = now;
        }

        if (g_stop) {
            interrupted = true;
            fprintf(stderr, "Abbruch angefordert -- ");
            if (ckpt_path.empty()) {
                fprintf(stderr, "kein --checkpoint gesetzt, Fortschritt geht verloren.\n");
            } else {
                fprintf(stderr, "Checkpoint geschrieben, gleicher Aufruf setzt fort.\n");
            }
            break;
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
// Kommandozeile
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int k = -1;
    int64_t bmin = 3, bmax = -1;
    double plimit_d = 1e8;
    std::string out = "kandidaten.txt";
    std::string ckpt_path;
    double ckpt_interval = 60.0;
    int nthreads = (int)std::thread::hardware_concurrency();
    if (nthreads < 1) {
        nthreads = 1;
    }

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "Fehlender Wert fuer %s\n", name);
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
        } else if (a == "-h" || a == "--help") {
            printf("Aufruf: %s --k K --bmax BMAX [--bmin 3] [--plimit 1e8] "
                   "[--out kandidaten.txt] [--threads N]\n"
                   "        [--checkpoint DATEI] [--checkpoint-interval SEK]\n"
                   "  Existiert die Checkpoint-Datei, wird automatisch fortgesetzt.\n"
                   "  Strg-C schreibt am naechsten Blockrand einen Checkpoint und endet.\n",
                   argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unbekanntes Argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (k < 0 || bmax < 0) {
        fprintf(stderr, "Fehler: --k und --bmax sind erforderlich. (--help fuer Hilfe)\n");
        return 2;
    }
    if (nthreads < 1) {
        nthreads = 1;
    }
    u64 plimit = (u64)plimit_d;

    // Strg-C / kill sauber abfangen -> Checkpoint am naechsten Blockrand.
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    SieveResult res = sieve(k, bmin, bmax, plimit, nthreads, ckpt_path, ckpt_interval);

    if (res.interrupted) {
        printf("\nAbgebrochen nach %.1f s (%llu Siebprimzahlen). "
               "Keine Kandidatendatei geschrieben.\n",
               res.secs, (unsigned long long)res.primes_used);
        if (!ckpt_path.empty()) {
            printf("Gleicher Aufruf setzt beim Checkpoint %s fort.\n", ckpt_path.c_str());
        }
        return 130;
    }

    int64_t total = (bmax - bmin) / 2 + 1;
    printf("\nN = 2^%d = %llu, Basen %lld..%lld  (%d Threads)\n",
           k, (unsigned long long)((u64)1 << k), (long long)bmin, (long long)bmax, nthreads);
    printf("%llu Siebprimzahlen bis %llu in %.1f s\n",
           (unsigned long long)res.primes_used, (unsigned long long)plimit, res.secs);
    printf("%zu von ca. %lld ungeraden Basen ueberleben (%.2f %%)\n",
           res.survivors.size(), (long long)total,
           100.0 * res.survivors.size() / (total > 0 ? total : 1));

    FILE* f = fopen(out.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Kann %s nicht schreiben\n", out.c_str());
        return 1;
    }
    fprintf(f, "# (b^%llu+1)/2, gesiebt bis p = %llu\n",
            (unsigned long long)((u64)1 << k), (unsigned long long)plimit);
    for (int64_t b : res.survivors) {
        fprintf(f, "%lld\n", (long long)b);
    }
    fclose(f);
    printf("Kandidaten geschrieben nach %s\n", out.c_str());
    return 0;
}
