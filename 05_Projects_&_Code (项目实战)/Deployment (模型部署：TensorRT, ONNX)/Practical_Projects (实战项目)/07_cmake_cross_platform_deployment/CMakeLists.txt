cmake_minimum_required(VERSION 3.20)

project(vision_deploy_template LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(vision_core
    src/inference.cpp
    src/postprocess.cpp
)

target_include_directories(vision_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(vision_app
    src/main.cpp
)

target_link_libraries(vision_app PRIVATE vision_core)

install(TARGETS vision_core vision_app
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
