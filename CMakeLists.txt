cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(lyra-w4afp8 LANGUAGES CXX CUDA)

# CMake
cmake_policy(SET CMP0169 OLD)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Python
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# CXX
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Torch
find_package(Torch REQUIRED)

# cutlass
include(FetchContent)
FetchContent_Declare(
    repo-cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass
    GIT_TAG        664c4f7b3ed1959414905025728eef5568209479
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-cutlass)

# ccache option
option(ENABLE_CCACHE "Whether to use ccache" ON)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND AND ENABLE_CCACHE AND DEFINED ENV{CCACHE_DIR})
    message(STATUS "Building with CCACHE enabled")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "ccache")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/csrc
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/tools/util/include
)

set(CUDA_FLAGS
    "-DNDEBUG"
    "-DOPERATOR_NAMESPACE=lyra-w4afp8"
    "-O3"
    "-Xcompiler"
    "-fPIC"
    "-gencode=arch=compute_90,code=sm_90"
    "-std=c++17"
    "-DFLASHINFER_ENABLE_F16"
    "-DCUTE_USE_PACKED_TUPLE=1"
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
    "-DCUTLASS_VERSIONS_GENERATED"
    "-DCUTLASS_TEST_LEVEL=0"
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "--threads=32"
    "-use_fast_math"
    # Suppress warnings
    "-Xcompiler=-Wconversion"
    "-Xcompiler=-fno-strict-aliasing"
)

# Enable gencode below SM90
option(ENABLE_BELOW_SM90 "Enable below SM90" ON)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ENABLE_BELOW_SM90 OFF)
    message(STATUS "For aarch64, disable gencode below SM90 by default")
endif()
if (ENABLE_BELOW_SM90)
    list(APPEND CUDA_FLAGS
        "-gencode=arch=compute_75,code=sm_75"
        "-gencode=arch=compute_80,code=sm_80"
        "-gencode=arch=compute_89,code=sm_89"
    )
endif()

message("CUDA_VERSION ${CUDA_VERSION}")
if (NOT "${CUDA_VERSION}" VERSION_GREATER_EQUAL 12.0)
    message(STATUS "Not building Machete kernels as CUDA Compiler version is "
                    "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                    "later if you intend on running w4a16 quantized models on Hopper.")
endif()

option(ENABLE_SM90A  "Enable SM90A"  ON)
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" OR ENABLE_SM90A)
    list(APPEND CUDA_FLAGS
        "-gencode=arch=compute_90a,code=sm_90a"
    )
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.0")
    list(APPEND CUDA_FLAGS
        "-DCOMPILE_HOPPER_TMA_GROUPED_GEMMS"
    )
endif()

string(REPLACE "-D__CUDA_NO_HALF_OPERATORS__"       "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF_CONVERSIONS__"     "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_BFLOAT16_CONVERSIONS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF2_OPERATORS__"      "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")


# For the Machete kernels we automatically generate sources for various
# preselected input type pairs and schedules.
# Generate sources: python3 scripts/generate_kernel.py
set(MACHETE_GEN_SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_kernel.py)
file(MD5 ${MACHETE_GEN_SCRIPT} MACHETE_GEN_SCRIPT_HASH)

message(STATUS "Machete generation script hash: ${MACHETE_GEN_SCRIPT_HASH}")
message(STATUS "Last run machete generate script hash: $CACHE{MACHETE_GEN_SCRIPT_HASH}")

if (NOT DEFINED CACHE{MACHETE_GEN_SCRIPT_HASH}
    OR NOT $CACHE{MACHETE_GEN_SCRIPT_HASH} STREQUAL ${MACHETE_GEN_SCRIPT_HASH})
    execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/csrc/:${repo-cutlass_SOURCE_DIR}/python/:$PYTHONPATH
        ${Python_EXECUTABLE} ${MACHETE_GEN_SCRIPT}
    RESULT_VARIABLE machete_generation_result
    OUTPUT_VARIABLE machete_generation_output
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
    ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
    )

    if (NOT machete_generation_result EQUAL 0)
    message(FATAL_ERROR "Machete generation failed."
                        " Result: \"${machete_generation_result}\""
                        "\nCheck the log for details: "
                        "${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log")
    else()
    set(MACHETE_GEN_SCRIPT_HASH ${MACHETE_GEN_SCRIPT_HASH}
        CACHE STRING "Last run machete generate script hash" FORCE)
    message(STATUS "Machete generation completed successfully.")
    endif()
else()
    message(STATUS "Machete generation script has not changed, skipping generation.")
endif()

# Add machete generated sources
file(GLOB MACHETE_GEN_SOURCES "csrc/generated/*.cu")
list(APPEND SOURCES ${MACHETE_GEN_SOURCES})

list(APPEND SOURCES csrc/machete/machete_pytorch.cu)
list(APPEND SOURCES csrc/moe_kernel/moe_align_kernel.cu)

list(APPEND SOURCES csrc/kernel_extension.cc)

message(STATUS "Building Machete kernels for archs: cuda")


Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

target_compile_options(common_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_CXX11_ABI STREQUAL "0")
    message(STATUS "Using old C++ ABI (-D_GLIBCXX_USE_CXX11_ABI=0)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
else()
    message(STATUS "Using new C++11 ABI (-D_GLIBCXX_USE_CXX11_ABI=1)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
endif()

target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)

target_compile_definitions(common_ops PRIVATE
    FLASHATTENTION_DISABLE_BACKWARD
    FLASHATTENTION_DISABLE_DROPOUT
    FLASHATTENTION_DISABLE_UNEVEN_K
)

install(TARGETS common_ops LIBRARY DESTINATION lyra_w4afp8)

