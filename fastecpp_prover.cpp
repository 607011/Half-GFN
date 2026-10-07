// fastecpp_prover.cpp
//
// Deterministic primality proof of M = (b^N + 1) / 2 via Andreas Enge's CM
// library (a fastECPP implementation), linked directly -- no subprocess.
//
// We call cm_ecpp() with check=true, so CM builds an ECPP certificate AND
// verifies it in-process (cm_pari_ecpp_check); the function's return value is
// the verification result. A proof is reported only when that check passes --
// we never trust an unverified certificate.
//
// Composite handling: with trust=false CM aborts the whole process via exit(1)
// on a composite input. To keep this usable as a library front-end we instead
// pre-screen with a strong probable-prime test ourselves and only hand CM a
// number that already looks prime (trust=true). Input is expected to be PRP-
// filtered upstream anyway; this just makes an accidental composite a clean
// rejection instead of a hard exit.
//
// Build: the "fastecpp_prover" CMake target (needs libcm + PARI/mpfrcx/mpc/
// mpfr/gmp). Build CM once with ./build_cm.sh. The actual CM call lives in the
// C shim fastecpp_cm.c (CM's headers are not C++-safe).

#include <gmp.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>

// Defined in fastecpp_cm.c (C linkage): run CM's fastECPP on N and verify.
extern "C" int fe_cm_ecpp_prove(mpz_srcptr N, const char* modpoldir,
                                char* tmpdir, int verbose);

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --exp N --base B [options]\n"
        "   or: %s --in FILE   (FILE holds one decimal integer M)\n"
        "\n"
        "Proves M = (b^N + 1) / 2 prime via CM's fastECPP (linked), and\n"
        "verifies the certificate in-process. Input must be PRP-filtered.\n"
        "\n"
        "Options:\n"
        "  --modpoldir DIR    CM modular-polynomial data dir (default: autodetect\n"
        "                     ../cm/_install/share/cm, or $CM_MODPOLDIR)\n"
        "  --workdir DIR      scratch/checkpoint dir for resumable proofs\n"
        "  --mr-rounds K      pre-screen Miller-Rabin rounds (default 25)\n"
        "  --trust            skip the pre-screen (input is known prime)\n"
        "  -v, --verbose      verbose CM output\n",
        prog, prog);
    std::exit(2);
}

static bool dir_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static void compute_M(mpz_t out, unsigned long base, unsigned long exp) {
    mpz_t tmp;
    mpz_init(tmp);
    mpz_ui_pow_ui(tmp, base, exp);  // base^exp
    mpz_add_ui(out, tmp, 1);        // base^exp + 1
    mpz_divexact_ui(out, out, 2);   // (base^exp + 1) / 2
    mpz_clear(tmp);
}

// CM needs the directory holding its modular-polynomial data (df/af/mf). It is
// compiled into the ecpp binary, but when we link the library we must supply it.
static std::string autodetect_modpoldir() {
    if (const char* env = std::getenv("CM_MODPOLDIR")) {
        if (env[0] != '\0') {
            return env;
        }
    }
    const std::array<const char*, 3> candidates = {
        "cm/_install/share/cm",
        "../cm/_install/share/cm",
        "/usr/local/share/cm",
    };
    for (const char* c : candidates) {
        if (dir_exists(c)) {
            return c;
        }
    }
    return "/usr/local/share/cm";  // CM's own default; may still work for tiny inputs
}

int main(int argc, char** argv) {
    unsigned long exp = 0;
    unsigned long base = 0;
    std::string in_file;
    std::string modpoldir;
    std::string workdir;
    int mr_rounds = 25;
    bool trust = false;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--exp") {
            if (i + 1 >= argc) { usage(argv[0]); }
            exp = std::strtoul(argv[++i], nullptr, 10);
        } else if (a == "--base") {
            if (i + 1 >= argc) { usage(argv[0]); }
            base = std::strtoul(argv[++i], nullptr, 10);
        } else if (a == "--in") {
            if (i + 1 >= argc) { usage(argv[0]); }
            in_file = argv[++i];
        } else if (a == "--modpoldir") {
            if (i + 1 >= argc) { usage(argv[0]); }
            modpoldir = argv[++i];
        } else if (a == "--workdir") {
            if (i + 1 >= argc) { usage(argv[0]); }
            workdir = argv[++i];
        } else if (a == "--mr-rounds") {
            if (i + 1 >= argc) { usage(argv[0]); }
            mr_rounds = std::atoi(argv[++i]);
        } else if (a == "--trust") {
            trust = true;
        } else if (a == "-v" || a == "--verbose") {
            verbose = true;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", a.c_str());
            usage(argv[0]);
        }
    }

    mpz_t M;
    mpz_init(M);

    if (!in_file.empty()) {
        std::ifstream f(in_file);
        if (!f) {
            std::fprintf(stderr, "cannot read --in file: %s\n", in_file.c_str());
            mpz_clear(M);
            return 2;
        }
        std::string s;
        f >> s;
        if (mpz_set_str(M, s.c_str(), 10) != 0) {
            std::fprintf(stderr, "file does not hold a decimal integer: %s\n", in_file.c_str());
            mpz_clear(M);
            return 2;
        }
    } else if (exp != 0 && base != 0) {
        compute_M(M, base, exp);
    } else {
        mpz_clear(M);
        usage(argv[0]);
    }

    const std::size_t digits = mpz_sizeinbase(M, 10);

    // Pre-screen so a composite is rejected cleanly instead of CM's exit(1).
    if (!trust) {
        if (mpz_probab_prime_p(M, mr_rounds) == 0) {
            std::fprintf(stderr, "[FASTECPP] %zu-digit M is composite -- not proved\n", digits);
            mpz_clear(M);
            return 1;
        }
    }

    if (modpoldir.empty()) {
        modpoldir = autodetect_modpoldir();
    }
    if (!dir_exists(modpoldir)) {
        std::fprintf(stderr,
            "[FASTECPP] warning: modpoldir '%s' not found; large proofs will fail.\n"
            "           Build CM (./build_cm.sh) or pass --modpoldir.\n",
            modpoldir.c_str()); // TODO: why not write to std::cerr?
    }

    // Optional scratch/checkpoint dir: CM stores discriminant-independent
    // precomputations and polynomial-factoring checkpoints here, which makes a
    // long proof resumable. NULL tmpdir simply disables that.
    char* tmpdir = nullptr;
    std::string tmpbuf;
    if (!workdir.empty()) {
        mkdir(workdir.c_str(), 0755);  // ignore EEXIST
        tmpbuf = workdir;
        tmpdir = tmpbuf.data();
    }

    std::fprintf(stderr, "[FASTECPP] proving %zu-digit M (modpoldir=%s)...\n",
                 digits, modpoldir.c_str());

    // The shim runs cm_ecpp with check=true, so success means the certificate
    // was built AND verified in-process (we pre-screened, so trust=true there).
    const bool proved = fe_cm_ecpp_prove(M, modpoldir.c_str(), tmpdir,
                                         verbose ? 1 : 0) != 0;
    mpz_clear(M);

    if (!proved) {
        std::fprintf(stderr, "[FASTECPP] certificate did NOT verify -- not proved\n");
        return 1;
    }

    if (base != 0) {
        std::printf("PROVED base=%lu exp=%lu digits=%zu\n", base, exp, digits);
    } else {
        std::printf("PROVED digits=%zu\n", digits);
    }
    return 0;
}
