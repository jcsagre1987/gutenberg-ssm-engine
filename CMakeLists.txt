cmake_minimum_required(VERSION 3.20)
project(SSM_Engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows target version & Standalone Asio settings
add_compile_definitions(_WIN32_WINNT=0x0A00)
add_compile_definitions(ASIO_STANDALONE)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/onnxruntime/include
    ${CMAKE_SOURCE_DIR}/asio/asio/include
)

link_directories(${CMAKE_SOURCE_DIR}/onnxruntime/lib)

add_executable(ssm_engine main.cpp)
target_link_libraries(ssm_engine PRIVATE onnxruntime)