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

# Stage 3 (prove.sh) needs PARI/GP's `gp` binary at runtime, not at build time.
find_program(GP_EXECUTABLE gp)
if(NOT GP_EXECUTABLE)
    message(STATUS "PARI/GP (gp) not found -- prove.sh needs it at runtime "
                   "(install: brew install pari / apt install pari-gp).")
endif()

install(TARGETS hgfn_sieve prp_test RUNTIME DESTINATION bin)
install(PROGRAMS prove.sh DESTINATION bin)
