// prp_test.cpp
//
// ARM-nativer PRP-Test (starker Fermat-/Miller-Rabin-Test) der Sieb-Ueberlebenden
// fuer M(b) = (b^N + 1)/2 mit N = 2^k.  Ersetzt pfgw: nur GMP als Abhaengigkeit,
// portabler C-Code -> laeuft nativ auf Apple Silicon (kein x86/Rosetta, kein
// gwnum-Assembler).
//
// Zweiter Schritt des Solo-Workflows: hgfn_sieve liefert die Kandidatenbasen,
// dieses Programm testet jedes M(b) auf (probable) Primalitaet.
//
// Der Exponent N wird automatisch aus der Header-Zeile der Kandidatendatei
// gelesen ("# (b^N+1)/2, ...").
//
// Test: starker PRP-Test (Miller-Rabin) zu einer oder mehreren Basen a.
//   Schreibe M-1 = d * 2^s.  a ist Zeuge fuer "zusammengesetzt", falls
//   a^d != 1 (mod M) UND a^(d*2^i) != -1 (mod M) fuer alle 0<=i<s.
//   Ueberlebt M alle Basen -> "PRP" (probable prime; kein Beweis).
//   Ein a mit 1 < gcd(a,M) < M beweist "zusammengesetzt" (echter Faktor).
//
// Wichtig: Ein bestandener PRP-Test ist KEIN Primzahlbeweis. Er ist der
// uebliche, sehr zuverlaessige Vorfilter; den finalen Beweis (APR-CL/ECPP)
// macht man danach nur noch fuer die wenigen Ueberlebenden (prove.sh).
//
// Kompilieren (Homebrew-GMP auf Apple Silicon):
//   clang++ -O3 -std=c++17 -I/opt/homebrew/include prp_test.cpp \
//           -L/opt/homebrew/lib -lgmp -pthread -o prp_test
//   (oder: make prp_test)
//
// Aufruf:
//   ./prp_test [Optionen] [kandidatendatei]        (Default: kand.txt)
// Optionen:
//   --exp N        Exponent ueberschreiben (sonst aus Header)
//   --bases "a b"  PRP-Basen, space-separiert (Default "3")
//   --limit N      nur die ersten N Kandidaten testen (0 = alle)
//   --out FILE     gefundene PRP-Basen hierhin (Default prp.txt)
//   --threads N    Threads (Default: alle Kerne)
//   --verbose      auch zusammengesetzte Kandidaten melden

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

// Wird vom SIGINT/SIGTERM-Handler gesetzt: Threads ziehen keine neuen
// Kandidaten mehr, laufende Tests werden noch fertig ins Journal geschrieben.
static volatile std::sig_atomic_t g_stop = 0;
static void on_signal(int) {
    g_stop = 1;
}

// ---------------------------------------------------------------------------
// Starker PRP-Test (Miller-Rabin) fuer bereits berechnetes M zu Basis a.
// Voraussetzung: M ungerade und > 2. Nutzt vorberechnetes d, s mit M-1=d*2^s.
// Rueckgabe: true = "probable prime zu Basis a", false = "zusammengesetzt".
// ---------------------------------------------------------------------------
static bool strong_prp(const mpz_t M, const mpz_t Mm1, const mpz_t d,
                       unsigned long s, unsigned long a) {
    mpz_t base, x, gcd;
    mpz_inits(base, x, gcd, nullptr);
    mpz_set_ui(base, a);

    // gcd(a, M): faellt ein echter Faktor auf, ist M zusammengesetzt.
    mpz_gcd(gcd, base, M);
    if (mpz_cmp_ui(gcd, 1) != 0) {
        bool eq = (mpz_cmp(gcd, M) == 0);      // a Vielfaches von M -> unbrauchbar
        mpz_clears(base, x, gcd, nullptr);
        return eq;  // gcd==M (a>=M und teilbar): nicht aussagekraeftig -> "bestanden"
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
        if (mpz_cmp_ui(x, 1) == 0) {            // 1 vor -1 -> zusammengesetzt
            break;
        }
    }
    mpz_clears(base, x, gcd, nullptr);
    return false;
}

// ---------------------------------------------------------------------------
// Testet einen Kandidaten b. Liefert true, wenn M(b) alle Basen besteht (PRP).
// ---------------------------------------------------------------------------
static bool test_candidate(unsigned long b, unsigned long exp,
                           const std::vector<unsigned long>& bases) {
    mpz_t M, Mm1, d;
    mpz_inits(M, Mm1, d, nullptr);

    // M = (b^exp + 1) / 2
    mpz_ui_pow_ui(M, b, exp);
    mpz_add_ui(M, M, 1);
    mpz_fdiv_q_2exp(M, M, 1);                   // /2 (b ungerade -> b^exp+1 gerade)

    // M-1 = d * 2^s
    mpz_sub_ui(Mm1, M, 1);
    unsigned long s = mpz_scan1(Mm1, 0);        // Zahl der Zweierpotenzen
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
                fprintf(stderr, "Fehlender Wert fuer %s\n", n);
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
        } else if (a == "--verbose") {
            verbose = true;
        } else if (a == "--journal") {
            journal_path = next("--journal");
        } else if (a == "-h" || a == "--help") {
            printf("Aufruf: %s [--exp N] [--bases \"3 5 7\"] [--limit N] "
                   "[--out prp.txt] [--threads N] [--verbose]\n"
                   "        [--journal DATEI] [kandidatendatei]\n"
                   "  --journal: jede getestete Basis wird sofort protokolliert; ein\n"
                   "             erneuter Aufruf mit gleichem Journal ueberspringt sie\n"
                   "             (Wiederaufsetzen). Strg-C beendet sauber.\n", argv[0]);
            return 0;
        } else if (a[0] == '-') {
            fprintf(stderr, "Unbekannte Option: %s\n", a.c_str());
            return 2;
        } else {
            candfile = a;
        }
    }
    if (candfile.empty()) {
        candfile = "kand.txt";
    }
    if (bases.empty()) {
        bases = {3};
    }
    if (nthreads < 1) {
        nthreads = 1;
    }

    // Kandidatendatei einlesen
    std::ifstream in(candfile);
    if (!in) {
        fprintf(stderr, "Kann %s nicht oeffnen\n", candfile.c_str());
        return 1;
    }

    unsigned long exp = 0;
    if (exp_override > 0) {
        exp = (unsigned long)exp_override;
    }

    std::vector<unsigned long> cands;
    std::string line;
    while (std::getline(in, line)) {
        // Header:  "# (b^EXP+1)/2, ..."  -> Exponent extrahieren
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
        fprintf(stderr, "Exponent nicht aus Header lesbar -- bitte --exp N angeben.\n");
        return 1;
    }
    if (cands.empty()) {
        fprintf(stderr, "Keine Kandidaten in %s.\n", candfile.c_str());
        return 1;
    }
    if (limit > 0 && (long)cands.size() > limit) {
        cands.resize(limit);
    }
    size_t total_all = cands.size();

    // --- Journal laden (Wiederaufsetzen): bereits getestete Basen ueberspringen ---
    // Journalzeile:  "<basis> <0|1>"  (1 = war PRP). Die PRP-Treffer aus dem
    // Journal wandern direkt in prp_bases, damit die Endausgabe vollstaendig ist.
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
                if (is >> jb >> jr) {           // torn/kaputte Zeilen still ueberspringen
                    done_set.insert(jb);
                    if (jr == 1) {
                        prp_bases.push_back(jb);
                    }
                }
            }
            jin.close();
        }
    }

    // Kandidaten filtern: bereits im Journal erledigte raus.
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

    // Strg-C / kill sauber abfangen -> laufende Tests noch journalieren, dann Ende.
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    printf("Kandidaten:  %zu gesamt", total_all);
    if (!journal_path.empty()) {
        printf(", %zu laut Journal erledigt, %zu zu testen", done_set.size(), cands.size());
    }
    printf("  (aus %s)\n", candfile.c_str());
    printf("Test:        M(b) = (b^%lu+1)/2, starker PRP zu Basis(en) %s\n", exp, basestr.c_str());
    printf("Threads:     %d\n", nthreads);
    if (!journal_path.empty()) {
        printf("Journal:     %s\n", journal_path.c_str());
    }
    printf("Ergebnis ->  %s\n", out.c_str());
    printf("--------------------------------------------------------------------------\n");
    fflush(stdout);

    // Journal zum Anhaengen oeffnen (legt Datei bei Bedarf an).
    FILE* jf = nullptr;
    if (!journal_path.empty()) {
        jf = fopen(journal_path.c_str(), "a");
        if (!jf) {
            fprintf(stderr, "Warnung: Journal %s nicht schreibbar -- Lauf ohne Journal.\n",
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
            if (g_stop) {                       // kein neuer Kandidat mehr
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
            if (jf) {                           // Ergebnis sofort dauerhaft festhalten
                fprintf(jf, "%lu %d\n", b, prp ? 1 : 0);
                fflush(jf);
            }
            if (prp) {
                prp_bases.push_back(b);
                double t = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0).count();
                printf("[%zu/%zu, %.0fs]  (%lu^%lu+1)/2  ist PRP!\n",
                       d, cands.size(), t, b, exp);
                fflush(stdout);
            } else if (verbose) {
                printf("[%zu/%zu]  (%lu^%lu+1)/2  zusammengesetzt\n",
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
        printf("Abgebrochen nach %.1f s (%zu Basen in diesem Lauf getestet).\n",
               secs, done.load());
        if (journaling) {
            printf("Fortschritt im Journal %s -- gleicher Aufruf setzt fort. "
                   "%s entsteht erst bei vollstaendigem Durchlauf.\n",
                   journal_path.c_str(), out.c_str());
        } else {
            printf("Ohne (schreibbares) --journal geht der Fortschritt verloren.\n");
        }
        return 130;
    }

    std::sort(prp_bases.begin(), prp_bases.end());
    prp_bases.erase(std::unique(prp_bases.begin(), prp_bases.end()), prp_bases.end());

    std::ofstream of(out);
    of << "# PRP-Basen fuer (b^" << exp << "+1)/2, Basis(en) " << basestr << "\n";
    for (unsigned long b : prp_bases) {
        of << b << "\n";
    }
    of.close();

    printf("--------------------------------------------------------------------------\n");
    printf("Fertig in %.1f s. %zu Basen in diesem Lauf getestet, %zu PRP gesamt -> %s\n",
           secs, done.load(), prp_bases.size(), out.c_str());
    return 0;
}
