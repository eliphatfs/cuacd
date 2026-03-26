cmake_minimum_required(VERSION 3.24)
project(coacd_gpu LANGUAGES C)

# We only need nvcc for offline fatbin compilation, not for linking.
# The shared library links against libcuda (driver API), NOT libcudart.

find_package(CUDAToolkit REQUIRED)

# C host source directory (passed from setup.py or defaults to ../csrc)
set(COACD_GPU_CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../csrc" CACHE PATH
    "Path to csrc/ directory containing C host code")

# --------------------------------------------------------------------------
# Step 1: Compile kernels.cu → fatbin
# --------------------------------------------------------------------------

set(FATBIN_ARCHS "80;86;89;90" CACHE STRING
    "SM architectures to embed as cubins in the fatbin")

set(NVCC_GENCODE_FLAGS "")
set(_highest_arch "")
foreach(arch ${FATBIN_ARCHS})
    list(APPEND NVCC_GENCODE_FLAGS
        -gencode=arch=compute_${arch},code=sm_${arch})
    set(_highest_arch ${arch})
endforeach()
list(APPEND NVCC_GENCODE_FLAGS
    -gencode=arch=compute_${_highest_arch},code=compute_${_highest_arch})

set(FATBIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/kernels.fatbin")
set(FATBIN_C    "${CMAKE_CURRENT_BINARY_DIR}/kernels_fatbin.c")

add_custom_command(
    OUTPUT ${FATBIN_FILE}
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu
        --fatbin
        -O3 --use_fast_math
        ${NVCC_GENCODE_FLAGS}
        -o ${FATBIN_FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu
    COMMENT "Compiling CUDA kernels to fatbin"
    VERBATIM
)

add_custom_command(
    OUTPUT ${FATBIN_C}
    COMMAND ${CMAKE_COMMAND} -E echo
        "// Auto-generated - do not edit" > ${FATBIN_C}
    COMMAND ${CUDAToolkit_BIN_DIR}/bin2c
        --name coacd_kernels_fatbin
        ${FATBIN_FILE} >> ${FATBIN_C}
    DEPENDS ${FATBIN_FILE}
    COMMENT "Embedding fatbin as C array"
    VERBATIM
)

# --------------------------------------------------------------------------
# Step 2: Build shared library
# --------------------------------------------------------------------------

add_library(coacd_gpu SHARED
    ${COACD_GPU_CSRC_DIR}/coacd_gpu.c
    ${FATBIN_C}
)

target_include_directories(coacd_gpu PUBLIC
    $<BUILD_INTERFACE:${COACD_GPU_CSRC_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(coacd_gpu PRIVATE CUDA::cuda_driver)

set_target_properties(coacd_gpu PROPERTIES
    C_STANDARD 11
    POSITION_INDEPENDENT_CODE ON
    VERSION   0.1.0
    SOVERSION 0
)

if(NOT WIN32)
    set_target_properties(coacd_gpu PROPERTIES
        C_VISIBILITY_PRESET hidden
    )
endif()

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------

install(TARGETS coacd_gpu
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES ${COACD_GPU_CSRC_DIR}/coacd_gpu.h DESTINATION include)
