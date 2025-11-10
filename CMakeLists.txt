cmake_minimum_required(VERSION 3.10)
project(SVD_Convolution)

# Configurar estándar C++
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buscar OpenCV
find_package(OpenCV REQUIRED)

# Incluir directorios de OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Crear ejecutable
add_executable(svd_separable_conv main.cpp)

# Enlazar con OpenCV
target_link_libraries(svd_separable_conv ${OpenCV_LIBS})

# Opciones de compilación
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(svd_separable_conv PRIVATE -Wall -Wextra -O3)
endif()

# Mensaje de configuración
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")