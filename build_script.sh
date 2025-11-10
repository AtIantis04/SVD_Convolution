#!/bin/bash

# Script de compilación y ejecución para SVD Convolution Demo

echo "=== Compilación de SVD Convolution Demo ==="
echo ""

# Colores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar si OpenCV está instalado
echo "Verificando dependencias..."
if ! pkg-config --exists opencv4; then
    echo -e "${RED}Error: OpenCV no encontrado${NC}"
    echo "Instala OpenCV 4.x:"
    echo "  Ubuntu/Debian: sudo apt-get install libopencv-dev"
    echo "  macOS: brew install opencv"
    exit 1
fi

OPENCV_VERSION=$(pkg-config --modversion opencv4)
echo -e "${GREEN}OpenCV $OPENCV_VERSION encontrado${NC}"
echo ""

# Crear directorio de build
if [ ! -d "build" ]; then
    echo "Creando directorio build..."
    mkdir build
fi

cd build

# Configurar con CMake
echo "Configurando con CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo -e "${RED}Error en configuración de CMake${NC}"
    exit 1
fi

echo ""
echo "Compilando..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)

if [ $? -ne 0 ]; then
    echo -e "${RED}Error en compilación${NC}"
    exit 1
fi

echo -e "${GREEN}Compilación exitosa${NC}"
echo ""

# Verificar si hay imagen de prueba
cd ..
if [ ! -f "input.jpg" ] && [ ! -f "input.png" ]; then
    echo -e "${YELLOW}Advertencia: No se encontró imagen de entrada${NC}"
    echo "Coloca una imagen llamada 'input.jpg' o 'input.png' en este directorio"
    echo "O ejecuta: ./build/svd_convolution ruta/a/tu/imagen.jpg"
else
    echo -e "${GREEN}Imagen de entrada encontrada${NC}"
    echo ""
    echo "Ejecutando programa..."
    echo "----------------------------------------"
    
    if [ -f "input.jpg" ]; then
        ./build/svd_convolution input.jpg
    else
        ./build/svd_convolution input.png
    fi
    
    echo "----------------------------------------"
    echo ""
    
    # Verificar resultados
    if [ -f "out_ref2d.png" ] && [ -f "out_sep.png" ]; then
        echo -e "${GREEN}Resultados generados exitosamente:${NC}"
        echo "  - out_ref2d.png (convolución 2D)"
        echo "  - out_sep.png (convolución separable)"
        echo "  - out_absdiff.png (diferencia absoluta)"
    else
        echo -e "${RED}Error: No se generaron los archivos de salida${NC}"
    fi
fi

echo ""
echo "=== Proceso completado ==="
echo ""
echo "Para ejecutar manualmente:"
echo "  ./build/svd_convolution [ruta_imagen]"
