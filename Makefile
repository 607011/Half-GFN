CXX      ?= clang++
CXXFLAGS ?= -O3 -std=c++17 -march=native -pthread -Wall

# GMP-Prefix automatisch ermitteln (Homebrew), sonst Standardpfade.
GMP_PREFIX ?= $(shell brew --prefix gmp 2>/dev/null)
ifneq ($(GMP_PREFIX),)
GMP_INC := -I$(GMP_PREFIX)/include
GMP_LIB := -L$(GMP_PREFIX)/lib
endif

all: hgfn_sieve prp_test

hgfn_sieve: hgfn_sieve.cpp
	$(CXX) $(CXXFLAGS) hgfn_sieve.cpp -o hgfn_sieve

prp_test: prp_test.cpp
	$(CXX) $(CXXFLAGS) $(GMP_INC) prp_test.cpp $(GMP_LIB) -lgmp -o prp_test

clean:
	rm -f hgfn_sieve prp_test

.PHONY: all clean
