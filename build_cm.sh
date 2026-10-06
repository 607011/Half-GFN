#!/usr/bin/env bash
#
# build_cm.sh - fetch and build Andreas Enge's CM library (fastECPP) so that
# fastecpp_prover can use its `ecpp` / `ecpp-check` binaries.
#
# CM is not in Homebrew; we build it from source into a sibling directory and
# install it into a user-owned prefix (no sudo, no writing to /usr/local).
# The install is built STATIC so the `ecpp` binary is self-contained and only
# depends on the Homebrew math dylibs (gmp/mpfr/mpc/mpfrcx/pari).
#
# Result: ../cm/_install/bin/{ecpp,ecpp-check}, which fastecpp_prover
# autodetects. Override with --ecpp / $CM_ECPP if you install elsewhere.
#
# Usage: ./build_cm.sh            # clone+build+install next to this repo
#        CM_DIR=/path ./build_cm.sh
set -euo pipefail

CM_REPO="https://gitlab.inria.fr/enge/cm.git"
CM_DIR="${CM_DIR:-$(cd "$(dirname "$0")/.." && pwd)/cm}"
PREFIX="${CM_PREFIX:-$CM_DIR/_install}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1" >&2; exit 1; }; }
need git
need brew
need autoreconf
need make

# CM's dependencies. mpfrcx is effectively CM-only; the rest are common.
echo ">> ensuring Homebrew dependencies"
for f in gmp mpfr libmpc mpfrcx pari; do
    brew list --versions "$f" >/dev/null 2>&1 || brew install "$f"
done

if [ ! -d "$CM_DIR/.git" ]; then
    echo ">> cloning CM into $CM_DIR"
    git clone --depth 1 "$CM_REPO" "$CM_DIR"
fi

cd "$CM_DIR"
echo ">> bootstrapping"
autoreconf -i

echo ">> configuring (static, prefix=$PREFIX)"
./configure --prefix="$PREFIX" --disable-shared --enable-static \
    --with-gmp="$(brew --prefix gmp)" \
    --with-mpfr="$(brew --prefix mpfr)" \
    --with-mpc="$(brew --prefix libmpc)" \
    --with-mpfrcx="$(brew --prefix mpfrcx)" \
    --with-pari="$(brew --prefix pari)"

# Build/install only lib, src and data -- the doc target needs `makeinfo`
# (texinfo), which we do not require.
echo ">> building"
make -C lib -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
make -C lib install
make -C src install
make -C data install

echo
echo ">> done:"
"$PREFIX/bin/ecpp" -h >/dev/null 2>&1 || true
ls -l "$PREFIX/bin/ecpp" "$PREFIX/bin/ecpp-check"
echo ">> fastecpp_prover will autodetect $PREFIX/bin/ecpp"
