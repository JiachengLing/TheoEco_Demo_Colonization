cmake_minimum_required(VERSION 3.20)
project(CProj_TheoEco LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


option(METACOMM_OPENMP "Enable OpenMP for metacomm" OFF)

add_library(metacomm SHARED metacomm.cpp)

if (METACOMM_OPENMP)
    find_package(OpenMP REQUIRED)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(metacomm PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if (MINGW)
    target_link_options(metacomm PRIVATE
            -Wl,--export-all-symbols
            -Wl,--enable-auto-import
    )
endif()

if (MSVC AND METACOMM_OPENMP)
    target_compile_options(metacomm PRIVATE /openmp:llvm)
endif()

add_custom_command(TARGET metacomm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory E:/TheoEcoFramework/CProj_TheoEco/python
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:metacomm> E:/TheoEcoFramework/CProj_TheoEco/python/
)
