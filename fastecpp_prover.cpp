// fastecpp_prover.cpp
//
// Drives Andreas Enge's CM "ecpp" program (a fastECPP implementation) to
// produce a *deterministic* primality proof of M = (b^N + 1) / 2, and then
// verifies the resulting certificate independently with CM's "ecpp-check".
//
// Design notes:
//   * Input is assumed to be already PRP-filtered (Miller-Rabin). By default
//     we still let ecpp run its own cheap initial test so an accidental
//     composite is rejected fast instead of sending the ECPP downrun into a
//     long spin. Pass --trust (ecpp "-t") to skip that pre-test when the
//     caller is certain; it only saves one Miller-Rabin and never weakens or
//     strengthens the resulting proof.
//   * A proof is reported ONLY when the independent ecpp-check confirms a
//     valid certificate. We never treat a bare exit code as "proved".
//   * CM writes its certificate (CM format + a .primo twin) plus checkpoint
//     and scratch files into CM_ECPP_TMPDIR. We point that at a work dir so
//     a proof can be interrupted and resumed.
//
// Build: see the "fastecpp_prover" target in the Makefile (needs GMP).

#include <gmp.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --exp N --base B [options]\n"
        "   or: %s --in FILE   (FILE holds one decimal integer M)\n"
        "\n"
        "Proves M = (b^N + 1) / 2 prime via CM's fastECPP, then verifies the\n"
        "certificate with ecpp-check. Input must already be PRP-filtered.\n"
        "\n"
        "Options:\n"
        "  --ecpp PATH        path to CM 'ecpp' binary (default: autodetect)\n"
        "  --ecpp-check PATH  path to CM 'ecpp-check' (default: next to ecpp)\n"
        "  --workdir DIR      scratch/checkpoint dir (CM_ECPP_TMPDIR)\n"
        "  --cert PATH        keep the certificate at PATH (default: workdir)\n"
        "  --keep-cert        do not delete the certificate on success\n"
        "  --trust            skip ecpp's initial pre-test (input is prime)\n"
        "  -v, --verbose      pass -v to ecpp\n",
        prog, prog);
    std::exit(2);
}

static bool file_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

// Shell-quote a single argument for use inside std::system().
static std::string shq(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

static void compute_M(mpz_t out, unsigned long base, unsigned long exp) {
    mpz_t tmp;
    mpz_init(tmp);
    mpz_ui_pow_ui(tmp, base, exp);  // base^exp
    mpz_add_ui(out, tmp, 1);        // base^exp + 1
    mpz_divexact_ui(out, out, 2);   // (base^exp + 1) / 2
    mpz_clear(tmp);
}

// Try to locate the CM ecpp binary relative to common build locations.
static std::string autodetect_ecpp() {
    if (const char* env = std::getenv("CM_ECPP")) {
        if (env[0] != '\0') {
            return env;
        }
    }
    // Prefer an installed, self-contained build (correct CM_MODPOLDIR, static
    // libcm) over the in-tree libtool wrapper.
    const std::array<const char*, 6> candidates = {
        "cm/_install/bin/ecpp",
        "../cm/_install/bin/ecpp",
        "cm/src/ecpp",
        "../cm/src/ecpp",
        "./ecpp",
        "/usr/local/bin/ecpp",
    };
    for (const char* c : candidates) {
        if (file_exists(c)) {
            return c;
        }
    }
    return "ecpp";  // last resort: rely on PATH
}

// Derive the ecpp-check path from the ecpp path (same directory).
static std::string sibling_check(const std::string& ecpp) {
    std::size_t slash = ecpp.find_last_of('/');
    if (slash == std::string::npos) {
        return "ecpp-check";
    }
    return ecpp.substr(0, slash + 1) + "ecpp-check";
}

int main(int argc, char** argv) {
    unsigned long exp = 0;
    unsigned long base = 0;
    std::string in_file;
    std::string ecpp;
    std::string ecpp_check;
    std::string workdir;
    std::string cert;
    bool keep_cert = false;
    bool verbose = false;
    bool trust = false;

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
        } else if (a == "--ecpp") {
            if (i + 1 >= argc) { usage(argv[0]); }
            ecpp = argv[++i];
        } else if (a == "--ecpp-check") {
            if (i + 1 >= argc) { usage(argv[0]); }
            ecpp_check = argv[++i];
        } else if (a == "--workdir") {
            if (i + 1 >= argc) { usage(argv[0]); }
            workdir = argv[++i];
        } else if (a == "--cert") {
            if (i + 1 >= argc) { usage(argv[0]); }
            cert = argv[++i];
        } else if (a == "--keep-cert") {
            keep_cert = true;
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
            std::fprintf(stderr, "file does not contain a decimal integer: %s\n", in_file.c_str());
            mpz_clear(M);
            return 2;
        }
    } else if (exp != 0 && base != 0) {
        compute_M(M, base, exp);
    } else {
        mpz_clear(M);
        usage(argv[0]);
    }

    if (ecpp.empty()) {
        ecpp = autodetect_ecpp();
    }
    if (ecpp_check.empty()) {
        ecpp_check = sibling_check(ecpp);
    }

    // Scratch / checkpoint directory. CM writes the certificate, its .primo
    // twin, and (large) intermediate factor/class-polynomial files here.
    if (workdir.empty()) {
        const char* env = std::getenv("CM_ECPP_TMPDIR");
        if (env && env[0] != '\0') {
            workdir = env;
        } else {
            workdir = "fastecpp_work";
        }
    }
    mkdir(workdir.c_str(), 0755);  // ignore EEXIST

    if (cert.empty()) {
        std::ostringstream c;
        c << workdir << "/cert";
        if (base != 0) {
            c << "_" << base << "_" << exp;
        }
        c << ".out";
        cert = c.str();
    }

    const std::size_t digits = mpz_sizeinbase(M, 10);
    std::fprintf(stderr, "[FASTECPP] M has %zu decimal digits; ecpp=%s\n",
                 digits, ecpp.c_str());

    // Write M to a file so we never build a multi-hundred-kB argv string.
    const std::string nfile = workdir + "/N.txt";
    {
        std::ofstream nf(nfile, std::ios::trunc);
        if (!nf) {
            std::fprintf(stderr, "cannot write %s\n", nfile.c_str());
            mpz_clear(M);
            return 2;
        }
        char* s = mpz_get_str(nullptr, 10, M);
        nf << s << "\n";
        std::free(s);
    }
    mpz_clear(M);

    // Phase 1: prove.  -t trusts (skip pre-test; we are PRP-filtered),
    //                  -c self-checks, -f writes the certificate.
    // CM reads the number from -n; we feed it from the file via a safe
    // expansion so the shell -- not our process -- builds the argument.
    std::ostringstream prove;
    prove << "CM_ECPP_TMPDIR=" << shq(workdir) << " "
          << shq(ecpp) << " -c"
          << (trust ? " -t" : "")
          << (verbose ? " -v" : "")
          << " -f " << shq(cert)
          << " -n \"$(cat " << shq(nfile) << ")\"";
    std::fprintf(stderr, "[FASTECPP] proving...\n");
    int rc = std::system(prove.str().c_str());
    if (rc != 0) {
        std::fprintf(stderr, "[FASTECPP] ecpp exited non-zero (rc=%d)\n", rc);
        return 1;
    }
    if (!file_exists(cert)) {
        std::fprintf(stderr, "[FASTECPP] no certificate produced at %s\n", cert.c_str());
        return 1;
    }

    // Phase 2: independent verification. Only "valid ECPP certificate in the
    // CM format" (ecpp-check res==1) counts as a proof.
    std::ostringstream check;
    check << shq(ecpp_check) << " -f " << shq(cert);
    std::fprintf(stderr, "[FASTECPP] verifying certificate...\n");
    FILE* pipe = popen(check.str().c_str(), "r");
    if (!pipe) {
        std::fprintf(stderr, "[FASTECPP] cannot run ecpp-check\n");
        return 1;
    }
    std::string out;
    std::array<char, 4096> buf;
    std::size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), pipe)) > 0) {
        out.append(buf.data(), n);
    }
    int crc = pclose(pipe);
    std::fputs(out.c_str(), stderr);

    const bool valid =
        crc == 0 &&
        out.find("valid ECPP certificate in the CM format") != std::string::npos;
    if (!valid) {
        std::fprintf(stderr, "[FASTECPP] certificate did NOT verify -- not proved\n");
        return 1;
    }

    if (!keep_cert) {
        std::remove(cert.c_str());
        std::remove((cert + ".primo").c_str());
    }
    std::remove(nfile.c_str());

    if (base != 0) {
        std::printf("PROVED base=%lu exp=%lu digits=%zu\n", base, exp, digits);
    } else {
        std::printf("PROVED digits=%zu\n", digits);
    }
    return 0;
}
