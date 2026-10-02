cmake_minimum_required(VERSION 3.16)

project(hgfn_sieve
    VERSION 1.0.0
    DESCRIPTION "ARM-native prime-hunting toolchain for (b^2^k+1)/2"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Standard / defaults
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# __int128 is a GNU/Clang builtin -> keep compiler extensions enabled.
set(CMAKE_CXX_EXTENSIONS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optional: tune for the building machine (fast, but not portable binaries).
option(HGFN_NATIVE "Optimize for the host CPU (-march=native)" ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

# GMP has no official CMake config; try pkg-config first, then a manual search.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(GMP IMPORTED_TARGET gmp)
endif()

if(TARGET PkgConfig::GMP)
    add_library(GMP::GMP ALIAS PkgConfig::GMP)
else()
    find_path(GMP_INCLUDE_DIR gmp.h
        HINTS ENV GMP_DIR
        PATH_SUFFIXES include)
    find_library(GMP_LIBRARY NAMES gmp libgmp
        HINTS ENV GMP_DIR
        PATH_SUFFIXES lib)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(GMP
        REQUIRED_VARS GMP_LIBRARY GMP_INCLUDE_DIR)
    add_library(GMP::GMP UNKNOWN IMPORTED)
    set_target_properties(GMP::GMP PROPERTIES
        IMPORTED_LOCATION "${GMP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}")
endif()

# ---------------------------------------------------------------------------
# Shared compile options
# ---------------------------------------------------------------------------
add_library(hgfn_flags INTERFACE)
if(MSVC)
    target_compile_options(hgfn_flags INTERFACE /W3)
    # fopen() is ISO C and used portably (fopen_s is MSVC-only); silence the
    # MSVC-specific C4996 deprecation warning instead of forking the code.
    target_compile_definitions(hgfn_flags INTERFACE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(hgfn_flags INTERFACE -Wall)
    if(HGFN_NATIVE)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-march=native" HGFN_HAS_MARCH_NATIVE)
        if(HGFN_HAS_MARCH_NATIVE)
            target_compile_options(hgfn_flags INTERFACE -march=native)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
# Stage 1: the sieve (threads only).
add_executable(hgfn_sieve hgfn_sieve.cpp)
target_link_libraries(hgfn_sieve PRIVATE hgfn_flags Threads::Threads)

# Stage 2: the strong-PRP filter (threads + GMP).
add_executable(prp_test prp_test.cpp)
target_link_libraries(prp_test PRIVATE hgfn_flags Threads::Threads GMP::GMP)

# Stage 1 reference for the large-k GPU engine: CPU negacyclic-NTT squaring
# mod (b^N+1), validated against GMP. CPU-only; see docs/cuda-largek-prp.md.
add_executable(ntt_ref ntt_ref.cpp)
target_link_libraries(ntt_ref PRIVATE hgfn_flags GMP::GMP)

# Optional: GPU-accelerated PRP test via Apple Metal (macOS only, Regime A).
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        add_executable(prp_metal prp_metal.mm)
        set_source_files_properties(prp_metal.mm PROPERTIES
            COMPILE_OPTIONS "-fobjc-arc")
        target_link_libraries(prp_metal PRIVATE
            hgfn_flags GMP::GMP ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

        # Large-k NTT PRP engine for Apple Silicon (see docs/metal-largek-prp.md).
        # Work in progress; built incrementally, each stage gated vs. ntt_ref.
        add_executable(ntt_metal ntt_metal.mm)
        set_source_files_properties(ntt_metal.mm PROPERTIES
            COMPILE_OPTIONS "-fobjc-arc")
        target_link_libraries(ntt_metal PRIVATE
            hgfn_flags GMP::GMP ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    else()
        message(STATUS "Metal/Foundation not found -- skipping prp_metal.")
    endif()
endif()

# Optional: CUDA PRP test for NVIDIA GPUs (e.g. RTX 4060, Ada = sm_89), Regime A.
# Built only when a CUDA compiler is present, so the macOS build is unaffected.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_executable(prp_cuda prp_cuda.cu)
    target_link_libraries(prp_cuda PRIVATE GMP::GMP)
    target_compile_features(prp_cuda PRIVATE cxx_std_17)
    # Default to Ada (RTX 4060); override with -DCMAKE_CUDA_ARCHITECTURES=...
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(prp_cuda PROPERTIES CUDA_ARCHITECTURES "89")
    endif()
else()
    message(STATUS "No CUDA compiler found -- skipping prp_cuda (build it on the NVIDIA box).")
endif()

# Stage 3 (prove.sh) needs PARI/GP's `gp` binary at runtime, not at build time.
find_program(GP_EXECUTABLE gp)
if(NOT GP_EXECUTABLE)
    message(STATUS "PARI/GP (gp) not found -- prove.sh needs it at runtime "
                   "(install: brew install pari / apt install pari-gp).")
endif()

install(TARGETS hgfn_sieve prp_test RUNTIME DESTINATION bin)
install(PROGRAMS prove.sh DESTINATION bin)
