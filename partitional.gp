/* partitional.gp -- exploration script for "Partitional" primes.
 *
 * Definition (Oliver Lau): with p(k) the integer partition function,
 *   n! !-ish "partitorial"   Pi(n) = p(0) * p(1) * ... * p(n)
 *   Partitional number        n⊢   = Pi(n) ± 1
 *
 * Key fact: Pi(n) is a product of small partition numbers, so it is fully
 * factored (just factor each p(k)) and carries only small prime factors.
 * Therefore a Partitional prime is provable by the classical N∓1 tests, NOT
 * ECPP:
 *   Pi(n)+1 is prime  <=>  N-1 = Pi(n) fully factored  -> Pocklington/BLS (N-1)
 *   Pi(n)-1 is prime  <=>  N+1 = Pi(n) fully factored  -> Lucas/BLS      (N+1)
 * Both are one (modular) exponentiation per prime factor: fast, O(N) memory,
 * and scale far beyond what ECPP can reach.
 *
 * The N-1 / N+1 provers below were validated against PARI's isprime() for all
 * Partitional primes with n <= 50 (both variants); a proof is reported only
 * when the BLS witness check passes.
 *
 * Usage:
 *   gp -q partitional.gp                 # runs scan(DEFAULT_NMAX) at the end
 *   gp -q                                # interactive, then:
 *     \r partitional.gp
 *     scan(300)                          # scan n = 2..300
 *     prove(67)                          # prove both variants for one n, verbose
 *     report(300)                        # list only the primes found
 *
 * Caveat: prtfac() factors each p(k) in full. For large n some p(k) become big
 * and may not factor quickly; a production tool would use BLS with the factored
 * part F > N^(1/3) (and ECPP only on the leftover). Out of scope for this quick
 * explorer.
 */

DEFAULT_NMAX = 120;

/* Pi(n) = product of partition numbers p(0..n). */
prt(n) = prod(k = 0, n, numbpart(k));

/* Distinct prime factors of Pi(n), obtained by factoring each small p(k). */
prtfac(n) =
{
   my(f = Map(), fk, p, e);
   for (k = 2, n,
      fk = factor(numbpart(k));
      for (i = 1, #fk~,
         p = fk[i, 1]; e = fk[i, 2];
         mapput(f, p, if (mapisdefined(f, p), mapget(f, p), 0) + e)
      )
   );
   vecsort([kk | kk <- Set(Mat(f)[, 1])]);
};

/* Lucas sequences U_k, V_k mod N (double-and-add). N must be odd. */
lucas(P, Q, k, N) =
{
   my(u = Mod(1, N), v = Mod(P, N), qk = Mod(Q, N),
      Pm = Mod(P, N), Qm = Mod(Q, N), D = Mod(P^2 - 4*Q, N), b);
   if (k == 0, return([Mod(0, N), Mod(2, N)]));
   b = binary(k);
   for (i = 2, #b,
      my(u2 = u*v, v2 = v*v - 2*qk);
      u = u2; v = v2; qk = qk*qk;
      if (b[i] == 1,
         my(nu = (Pm*u + v) / 2, nv = (D*u + Pm*v) / 2);
         u = nu; v = nv; qk = qk*Qm
      )
   );
   [u, v];
};

QLIST = [-1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8];

/* Pocklington/BLS (N-1) for N = Pi(n)+1, with N-1 = Pi(n) fully factored.
 * Per prime q | N-1, find a base a with a^(N-1) = 1 and gcd(a^((N-1)/q)-1,N)=1.
 * Returns 1 (proved prime) or 0 (no witness found). */
blsN1(n) =
{
   my(N = prt(n) + 1, qs = prtfac(n), q, a, ok);
   for (i = 1, #qs,
      q = qs[i]; ok = 0;
      for (a = 2, 3000,
         if (Mod(a, N)^(N - 1) != 1, next);
         if (gcd(lift(Mod(a, N)^((N - 1)/q)) - 1, N) == 1, ok = 1; break)
      );
      if (!ok, return(0))
   );
   1;
};

/* A Lucas base sequence for N: (P,Q) with (D/N) = -1 and N | U_{N+1}.
 * Returns [P, Q] or 0 if none found in range (then N is composite, or the
 * search bound is too small). */
lucasbase(N) =
{
   my(P, Q, D);
   for (P = 1, 60,
      for (qi = 1, #QLIST,
         Q = QLIST[qi];
         if (gcd(Q, N) != 1, next);
         D = P^2 - 4*Q;
         if (kronecker(D, N) != -1, next);
         if (lucas(P, Q, N + 1, N)[1] == 0, return([P, Q]))
      )
   );
   0;
};

/* Lucas/BLS (N+1) for N = Pi(n)-1, with N+1 = Pi(n) fully factored.
 * Find one base sequence ((D/N) = -1, N | U_{N+1}); per prime q | N+1 require
 * gcd(U_{(N+1)/q}, N) = 1, reusing the base and only searching an alternate
 * sequence for a prime the base fails on (per-prime sequences are a valid BLS
 * N+1 form). Returns 1 (proved prime) or 0. */
blsP1(n) =
{
   my(N = prt(n) - 1, qs = prtfac(n), q, base, bP, bQ, found, P, Q, D);
   base = lucasbase(N);
   if (base == 0, return(0));
   bP = base[1]; bQ = base[2];
   for (iq = 1, #qs,
      q = qs[iq];
      if (gcd(lift(lucas(bP, bQ, (N + 1)/q, N)[1]), N) == 1, next);
      found = 0;   /* base failed this q: search an alternate sequence for it */
      for (P = 1, 60,
         for (qi = 1, #QLIST,
            Q = QLIST[qi];
            if (gcd(Q, N) != 1, next);
            D = P^2 - 4*Q;
            if (kronecker(D, N) != -1, next);
            if (lucas(P, Q, N + 1, N)[1] != 0, next);
            if (gcd(lift(lucas(P, Q, (N + 1)/q, N)[1]), N) == 1, found = 1; break(2))
         )
      );
      if (!found, return(0))
   );
   1;
};

/* Prove both variants for a single n, with timing. Verbose. */
prove(n) =
{
   my(Np = prt(n) + 1, Nm = prt(n) - 1, t, el);
   print("n = ", n, "   Pi(n) has ", #Str(prt(n)), " digits, ",
         #prtfac(n), " distinct prime factors (max ", vecmax(prtfac(n)), ")");
   if (ispseudoprime(Np),
      t = gettime(); my(r = blsN1(n)); el = gettime();
      print("  Pi(n)+1  (", #Str(Np), " digits): PRP, N-1 BLS -> ",
            if (r, "PROVED PRIME", "no witness"), "   [", el, " ms]"),
      print("  Pi(n)+1  (", #Str(Np), " digits): composite"));
   if (ispseudoprime(Nm),
      t = gettime(); my(r = blsP1(n)); el = gettime();
      print("  Pi(n)-1  (", #Str(Nm), " digits): PRP, N+1 BLS -> ",
            if (r, "PROVED PRIME", "no witness"), "   [", el, " ms]"),
      print("  Pi(n)-1  (", #Str(Nm), " digits): composite"));
};

/* Scan n = 2..nmax: one row per n, marking proven primes of either variant. */
scan(nmax) =
{
   my(np = 0, nm = 0);
   print("  n | digits |  Pi(n)+1  |  Pi(n)-1");
   print("----+--------+-----------+----------");
   for (n = 2, nmax,
      my(P = prt(n), dp = #Str(P + 1),
         sp = "    .    ", sm = "    .   ");
      if (ispseudoprime(P + 1),
         sp = if (blsN1(n), "PRIME(N-1)", "PRP only  "); np++);
      if (ispseudoprime(P - 1),
         sm = if (blsP1(n), "PRIME(N+1)", "PRP only "); nm++);
      if (ispseudoprime(P + 1) || ispseudoprime(P - 1),
         printf("%3d | %6d | %s | %s\n", n, dp, sp, sm))
   );
   print("----+--------+-----------+----------");
   print("Partitional primes found up to n=", nmax,
         ":  ", np, " of form Pi(n)+1,  ", nm, " of form Pi(n)-1.");
};

/* List only the primes found, with their digit sizes. */
report(nmax) =
{
   print("Proven Partitional primes (n, variant, digits):");
   for (n = 2, nmax,
      my(P = prt(n));
      if (ispseudoprime(P + 1) && blsN1(n),
         print("  n=", n, "  Pi(n)+1  ", #Str(P + 1), " digits"));
      if (ispseudoprime(P - 1) && blsP1(n),
         print("  n=", n, "  Pi(n)-1  ", #Str(P - 1), " digits"))
   );
};

scan(DEFAULT_NMAX);
