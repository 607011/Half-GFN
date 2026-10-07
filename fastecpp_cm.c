/* fastecpp_cm.c -- C shim around CM's cm_ecpp().
 *
 * CM's public headers (cm.h -> cm-arith.h) are C-only: cm-arith.h closes an
 * `extern "C"` block it never opens, which is fine under C (the brace is
 * behind `#if defined(__cplusplus)`) but a hard error when included from C++.
 * So we include cm.h here, compiled as C, and expose one small C-linkage entry
 * point that the C++ driver (fastecpp_prover.cpp) declares and calls. The C++
 * side never includes any CM header.
 */

#include <cm.h>
#include <gmp.h>

/* Prove N prime with CM's fastECPP and verify the certificate in-process.
 * N is trusted to be (probable-)prime -- the caller pre-screens, so CM does
 * not take its own exit(1)-on-composite path. check=true makes cm_ecpp return
 * the certificate verification result.
 * Returns 1 if proved and verified, 0 otherwise. */
int fe_cm_ecpp_prove(mpz_srcptr N, const char *modpoldir, char *tmpdir,
                     int verbose)
{
    int proved;
    cm_pari_init();
    proved = cm_ecpp(N, modpoldir, /*filename=*/NULL, tmpdir,
                     /*print=*/false, /*trust=*/true, /*check=*/true,
                     /*phases=*/0, verbose ? true : false, /*debug=*/false)
             ? 1 : 0;
    cm_pari_clear();
    return proved;
}
