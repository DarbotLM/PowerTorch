cmake_minimum_required(VERSION 3.20)
project(PowerTorchDirectML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(PowerTorch.DirectML.Native SHARED
    src/powertorch_directml.cpp
    include/powertorch_directml.h
)

target_include_directories(PowerTorch.DirectML.Native
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_definitions(PowerTorch.DirectML.Native
    PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0A00
)

target_link_libraries(PowerTorch.DirectML.Native
    PRIVATE
        d3d12
        dxgi
        directml
        runtimeobject
)

if(MSVC)
    target_compile_options(PowerTorch.DirectML.Native PRIVATE /W4 /WX)
endif()
