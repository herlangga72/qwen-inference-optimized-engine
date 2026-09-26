function(ggml_add_cpu_backend_features cpu_name arch)
    # The feature detection code is compiled as a separate target so that
    # it can be built without the architecture flags
    # Since multiple variants of the CPU backend may be included in the same
    # build, using set_source_files_properties() to set the arch flags is not possible
    set(GGML_CPU_FEATS_NAME ${cpu_name}-feats)
    add_library(${GGML_CPU_FEATS_NAME} OBJECT ggml-cpu/arch/${arch}/cpu-feats.cpp)
    target_include_directories(${GGML_CPU_FEATS_NAME} PRIVATE . ../include)
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE ${ARGN})
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
    set_target_properties(${GGML_CPU_FEATS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # Disable LTO for the feature detection code to prevent cross-module optimization
    # from inlining architecture-specific instructions into the score function.
    # Without this, LTO can cause SIGILL when loading backends on older CPUs
    # (e.g., loading power10 backend on power9 crashes before feature check runs).
    target_compile_options(${GGML_CPU_FEATS_NAME} PRIVATE -fno-lto)
    target_link_libraries(${cpu_name} PRIVATE ${GGML_CPU_FEATS_NAME})
endfunction()

function(ggml_add_cpu_backend_variant_impl tag_name)
    if (tag_name)
        set(GGML_CPU_NAME ggml-cpu-${tag_name})
    else()
        set(GGML_CPU_NAME ggml-cpu)
    endif()

    ggml_add_backend_library(${GGML_CPU_NAME})

    list (APPEND GGML_CPU_SOURCES
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/repack.cpp
        ggml-cpu/repack.h
        ggml-cpu/quants.c
        ggml-cpu/quants.h
        ggml-cpu/traits.cpp
        ggml-cpu/traits.h
        ggml-cpu/amx/amx.cpp
        ggml-cpu/amx/amx.h
        ggml-cpu/amx/mmq.cpp
        ggml-cpu/amx/mmq.h
        ggml-cpu/tiled/tiled.cpp
        ggml-cpu/tiled/tiled.h
        ggml-cpu/tiled/tiled-kernel.cpp
        ggml-cpu/tiled/tiled-kernel.h
        ggml-cpu/ggml-cpu-impl.h
        ggml-cpu/common.h
        ggml-cpu/binary-ops.h
        ggml-cpu/binary-ops.cpp
        ggml-cpu/unary-ops.h
        ggml-cpu/unary-ops.cpp
        ggml-cpu/simd-mappings.h
        ggml-cpu/vec.h
        ggml-cpu/vec.cpp
        ggml-cpu/ops.h
        ggml-cpu/ops.cpp
        )

    target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)
    target_include_directories(${GGML_CPU_NAME} PRIVATE . ggml-cpu)

    if (GGML_OPENMP_ENABLED)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_OPENMP)
        target_link_libraries(${GGML_CPU_NAME} PRIVATE ${GGML_OPENMP_TARGET_C} ${GGML_OPENMP_TARGET_CXX})
    endif()

    if (GGML_LLAMAFILE)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_LLAMAFILE)

        list(APPEND GGML_CPU_SOURCES
                    ggml-cpu/llamafile/sgemm.cpp
                    ggml-cpu/llamafile/sgemm.h)
    endif()

    if (GGML_SYSTEM_ARCH STREQUAL "x86")
        message(STATUS "x86 detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/x86/quants.c
            ggml-cpu/arch/x86/repack.cpp
            )

        if (MSVC)
            # instruction set detection for MSVC only
            if (GGML_NATIVE)
                include(ggml-cpu/cmake/FindSIMD.cmake)
            endif ()
            if (GGML_AVX512)
                list(APPEND ARCH_FLAGS /arch:AVX512)
                # /arch:AVX512 includes: __AVX512F__, __AVX512CD__, __AVX512BW__, __AVX512DQ__, and __AVX512VL__
                # MSVC has no compile-time flags enabling specific
                # AVX512 extensions, neither it defines the
                # macros corresponding to the extensions.
                # Do it manually.
                list(APPEND ARCH_DEFINITIONS GGML_AVX512)
                if (GGML_AVX512_VBMI)
                    list(APPEND ARCH_DEFINITIONS __AVX512VBMI__)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512vbmi)
                    endif()
                endif()
                if (GGML_AVX512_VNNI)
                    list(APPEND ARCH_DEFINITIONS __AVX512VNNI__ GGML_AVX512_VNNI)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512vnni)
                    endif()
                endif()
                if (GGML_AVX512_BF16)
                    list(APPEND ARCH_DEFINITIONS __AVX512BF16__ GGML_AVX512_BF16)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512bf16)
                    endif()
                endif()
                if (GGML_AMX_TILE)
                    list(APPEND ARCH_DEFINITIONS __AMX_TILE__ GGML_AMX_TILE)
                endif()
                if (GGML_AMX_INT8)
                    list(APPEND ARCH_DEFINITIONS __AMX_INT8__ GGML_AMX_INT8)
                endif()
                if (GGML_AMX_BF16)
                    list(APPEND ARCH_DEFINITIONS __AMX_BF16__ GGML_AMX_BF16)
                endif()
            elseif (GGML_AVX2)
                list(APPEND ARCH_FLAGS /arch:AVX2)
                list(APPEND ARCH_DEFINITIONS GGML_AVX2 GGML_FMA GGML_F16C)
            elseif (GGML_AVX)
                list(APPEND ARCH_FLAGS /arch:AVX)
                list(APPEND ARCH_DEFINITIONS GGML_AVX)
            elseif (GGML_SSE42)
                list(APPEND ARCH_FLAGS /arch:SSE4.2)
                list(APPEND ARCH_DEFINITIONS GGML_SSE42)
            endif()
            if (GGML_AVX_VNNI)
                list(APPEND ARCH_DEFINITIONS __AVXVNNI__ GGML_AVX_VNNI)
            endif()
            if (GGML_BMI2)
                # MSVC does not define macro __BMI2__
                list(APPEND ARCH_DEFINITIONS __BMI2__ GGML_BMI2)
            endif()
        else ()
            if (GGML_NATIVE)
                list(APPEND ARCH_FLAGS -march=native)
            else ()
                if (GGML_SSE42)
                    list(APPEND ARCH_FLAGS -msse4.2)
                    list(APPEND ARCH_DEFINITIONS GGML_SSE42)
                endif()
                if (GGML_F16C)
                    list(APPEND ARCH_FLAGS -mf16c)
                    list(APPEND ARCH_DEFINITIONS GGML_F16C)
                endif()
                if (GGML_FMA)
                    list(APPEND ARCH_FLAGS -mfma)
                    list(APPEND ARCH_DEFINITIONS GGML_FMA)
                endif()
                if (GGML_BMI2)
                    list(APPEND ARCH_FLAGS -mbmi2)
                    list(APPEND ARCH_DEFINITIONS GGML_BMI2)
                endif()
                if (GGML_AVX)
                    list(APPEND ARCH_FLAGS -mavx)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX)
                endif()
                if (GGML_AVX2)
                    list(APPEND ARCH_FLAGS -mavx2)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX2)
                endif()
                if (GGML_AVX_VNNI)
                    list(APPEND ARCH_FLAGS -mavxvnni)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX_VNNI)
                endif()
                if (GGML_AVX512)
                    list(APPEND ARCH_FLAGS -mavx512f)
                    list(APPEND ARCH_FLAGS -mavx512cd)
                    list(APPEND ARCH_FLAGS -mavx512vl)
                    list(APPEND ARCH_FLAGS -mavx512dq)
                    list(APPEND ARCH_FLAGS -mavx512bw)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512)
                endif()
                if (GGML_AVX512_VBMI)
                    list(APPEND ARCH_FLAGS -mavx512vbmi)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_VBMI)
                endif()
                if (GGML_AVX512_VNNI)
                    list(APPEND ARCH_FLAGS -mavx512vnni)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_VNNI)
                endif()
                if (GGML_AVX512_BF16)
                    list(APPEND ARCH_FLAGS -mavx512bf16)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_BF16)
                endif()
                if (GGML_AMX_TILE)
                    list(APPEND ARCH_FLAGS -mamx-tile)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_TILE)
                endif()
                if (GGML_AMX_INT8)
                    list(APPEND ARCH_FLAGS -mamx-int8)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_INT8)
                endif()
                if (GGML_AMX_BF16)
                    list(APPEND ARCH_FLAGS -mamx-bf16)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_BF16)
                endif()
            endif()
        endif()

        if (GGML_BACKEND_DL)
            if (GGML_NATIVE)
                # the feature check relies on ARCH_DEFINITIONS, but it is not set with GGML_NATIVE
                message(FATAL_ERROR "GGML_NATIVE is not compatible with GGML_BACKEND_DL, consider using GGML_CPU_ALL_VARIANTS")
            endif()
            ggml_add_cpu_backend_features(${GGML_CPU_NAME} x86 ${ARCH_DEFINITIONS})
        endif()
    else()
        message(WARNING "Unknown CPU architecture. Falling back to generic implementations.")
        list(APPEND ARCH_FLAGS -DGGML_CPU_GENERIC)
    endif()

    if (GGML_CPU_REPACK)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_REPACK)
    endif()

    message(STATUS "Adding CPU backend variant ${GGML_CPU_NAME}: ${ARCH_FLAGS} ${ARCH_DEFINITIONS}")
    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})
    target_compile_options(${GGML_CPU_NAME} PRIVATE ${ARCH_FLAGS})
    target_compile_definitions(${GGML_CPU_NAME} PRIVATE ${ARCH_DEFINITIONS})

    if (EMSCRIPTEN)
        set_target_properties(${GGML_CPU_NAME} PROPERTIES COMPILE_FLAGS "-msimd128")
    endif()

  if (CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
      # The compiler automatically enables "-ffast-math" which can cause NaNs in tests due to "-fassociative-math"
      target_compile_options(${GGML_CPU_NAME} PRIVATE "$<$<OR:$<COMPILE_LANG_AND_ID:C,IntelLLVM>,$<COMPILE_LANG_AND_ID:CXX,IntelLLVM>>:$<$<BOOL:${WIN32}>:/clang:>-fno-associative-math>")
  endif()

endfunction()
