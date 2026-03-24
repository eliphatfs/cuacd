cmake_minimum_required(VERSION 3.24)
project(coacd_gpu LANGUAGES C)

# We only need nvcc for offline fatbin compilation, not for linking.
# The shared library links against libcuda (driver API), NOT libcudart.

find_package(CUDAToolkit REQUIRED)

# --------------------------------------------------------------------------
# Step 1: Compile kernels.cu → fatbin
# --------------------------------------------------------------------------

set(FATBIN_ARCHS "80;86;89;90" CACHE STRING
    "SM architectures to embed as cubins in the fatbin")

# Build the nvcc flags: one -gencode per arch, plus PTX for the highest
set(NVCC_GENCODE_FLAGS "")
set(_highest_arch "")
foreach(arch ${FATBIN_ARCHS})
    list(APPEND NVCC_GENCODE_FLAGS
        -gencode=arch=compute_${arch},code=sm_${arch})
    set(_highest_arch ${arch})
endforeach()
# Embed PTX for the highest arch (forward compat with future GPUs)
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

# Step 2: Convert fatbin to a C array so it can be linked into the .so
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
# Step 3: Build shared library
# --------------------------------------------------------------------------

add_library(coacd_gpu SHARED
    coacd_gpu.c
    ${FATBIN_C}
)

target_include_directories(coacd_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link only against the CUDA driver library (libcuda), NOT cudart
target_link_libraries(coacd_gpu PRIVATE CUDA::cuda_driver)

set_target_properties(coacd_gpu PROPERTIES
    C_STANDARD 11
    POSITION_INDEPENDENT_CODE ON
    # Version info for soname
    VERSION   0.1.0
    SOVERSION 0
)

if(NOT WIN32)
    # Hide all symbols except the COACD_GPU_API ones
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
install(FILES coacd_gpu.h DESTINATION include)
