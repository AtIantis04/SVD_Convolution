# Script de configuración completa para Windows
# Ejecutar con: powershell -ExecutionPolicy Bypass -File setup.ps1

Write-Host "=== Configuración del Proyecto SVD Convolution ===" -ForegroundColor Cyan
Write-Host ""

# Verificar que estamos en el directorio correcto
$currentDir = Get-Location
Write-Host "Directorio actual: $currentDir" -ForegroundColor Yellow
Write-Host ""

# Crear main.cpp
Write-Host "Creando main.cpp..." -ForegroundColor Green
$mainCpp = @'
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace cv;
using namespace std;
using namespace chrono;

// Función para crear un kernel gaussiano
Mat createGaussianKernel(int size, double sigma) {
    Mat kernel = getGaussianKernel(size, sigma, CV_64F);
    Mat kernel2D = kernel * kernel.t();
    return kernel2D;
}

// Función para realizar convolución 2D tradicional
Mat convolve2D(const Mat& image, const Mat& kernel) {
    Mat result;
    filter2D(image, result, CV_64F, kernel, Point(-1, -1), 0, BORDER_REPLICATE);
    return result;
}

// Función para realizar convolución separable usando vectores 1D
Mat convolveSeparable(const Mat& image, const Mat& kernelH, const Mat& kernelV) {
    Mat temp, result;
    
    // Convolución horizontal
    filter2D(image, temp, CV_64F, kernelH, Point(-1, -1), 0, BORDER_REPLICATE);
    
    // Convolución vertical
    filter2D(temp, result, CV_64F, kernelV, Point(-1, -1), 0, BORDER_REPLICATE);
    
    return result;
}

// Función para calcular MSE
double calculateMSE(const Mat& img1, const Mat& img2) {
    Mat diff;
    absdiff(img1, img2, diff);
    diff = diff.mul(diff);
    Scalar sum = cv::sum(diff);
    double mse = sum[0] / (img1.rows * img1.cols);
    return mse;
}

// Función para calcular diferencia máxima
double calculateMaxDiff(const Mat& img1, const Mat& img2) {
    Mat diff;
    absdiff(img1, img2, diff);
    double minVal, maxVal;
    minMaxLoc(diff, &minVal, &maxVal);
    return maxVal;
}

int main(int argc, char** argv) {
    cout << "=== Optimización de Convolución 2D mediante SVD ===" << endl << endl;
    
    // 1. CARGAR IMAGEN
    string imagePath = (argc > 1) ? argv[1] : "input.jpg";
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    
    if (image.empty()) {
        cerr << "Error: No se pudo cargar la imagen '" << imagePath << "'" << endl;
        cerr << "Uso: " << argv[0] << " [ruta_imagen]" << endl;
        return -1;
    }
    
    cout << "Imagen cargada: " << image.cols << "x" << image.rows << " píxeles" << endl;
    
    // Convertir a double para mayor precisión
    Mat imageDouble;
    image.convertTo(imageDouble, CV_64F);
    
    // 2. CREAR KERNEL GAUSSIANO 15x15
    int kernelSize = 15;
    double sigma = 3.0;
    Mat kernel = createGaussianKernel(kernelSize, sigma);
    
    cout << "Kernel gaussiano: " << kernelSize << "x" << kernelSize 
         << " (sigma=" << sigma << ")" << endl << endl;
    
    // 3. PARTE 1: CONVOLUCIÓN 2D TRADICIONAL
    cout << "--- PARTE 1: Convolución 2D Tradicional ---" << endl;
    auto start2D = high_resolution_clock::now();
    Mat result2D = convolve2D(imageDouble, kernel);
    auto end2D = high_resolution_clock::now();
    auto duration2D = duration_cast<milliseconds>(end2D - start2D);
    
    cout << "Tiempo de ejecución: " << duration2D.count() << " ms" << endl;
    
    // Guardar resultado
    Mat result2D_8U;
    result2D.convertTo(result2D_8U, CV_8U);
    imwrite("out_ref2d.png", result2D_8U);
    cout << "Resultado guardado: out_ref2d.png" << endl << endl;
    
    // 4. PARTE 2: CONVOLUCIÓN SEPARABLE CON SVD
    cout << "--- PARTE 2: Convolución Separable con SVD ---" << endl;
    
    // Realizar SVD del kernel
    Mat w, u, vt;
    SVD::compute(kernel, w, u, vt);
    
    cout << "SVD calculada. Valores singulares:" << endl;
    for (int i = 0; i < min(5, (int)w.rows); i++) {
        cout << "  σ_" << i << " = " << w.at<double>(i, 0);
        if (i > 0) {
            double ratio = w.at<double>(i, 0) / w.at<double>(0, 0);
            cout << " (" << fixed << setprecision(4) << ratio * 100 << "%)";
        }
        cout << endl;
    }
    cout << endl;
    
    // Tomar el primer valor singular y vectores
    double sigma1 = w.at<double>(0, 0);
    Mat u1 = u.col(0);  // Primer vector singular izquierdo
    Mat v1 = vt.row(0); // Primer vector singular derecho
    
    // Construir vectores 1D: kernelV = sqrt(σ1) * u1, kernelH = sqrt(σ1) * v1^T
    double sqrtSigma1 = sqrt(sigma1);
    Mat kernelV = sqrtSigma1 * u1;
    Mat kernelH = sqrtSigma1 * v1.t();
    
    // Reshape para usar en filter2D
    kernelV = kernelV.reshape(1, kernelSize); // Columna
    kernelH = kernelH.reshape(1, 1);          // Fila
    
    cout << "Vectores 1D creados:" << endl;
    cout << "  kernelH: 1x" << kernelH.cols << endl;
    cout << "  kernelV: " << kernelV.rows << "x1" << endl << endl;
    
    // Aplicar convolución separable
    auto startSep = high_resolution_clock::now();
    Mat resultSep = convolveSeparable(imageDouble, kernelH, kernelV);
    auto endSep = high_resolution_clock::now();
    auto durationSep = duration_cast<milliseconds>(endSep - startSep);
    
    cout << "Tiempo de ejecución: " << durationSep.count() << " ms" << endl;
    
    // Guardar resultado
    Mat resultSep_8U;
    resultSep.convertTo(resultSep_8U, CV_8U);
    imwrite("out_sep.png", resultSep_8U);
    cout << "Resultado guardado: out_sep.png" << endl << endl;
    
    // 5. COMPARACIÓN
    cout << "--- COMPARACIÓN DE RESULTADOS ---" << endl;
    
    double mse = calculateMSE(result2D, resultSep);
    double maxDiff = calculateMaxDiff(result2D, resultSep);
    
    cout << "MSE (Mean Squared Error): " << scientific << mse << endl;
    cout << "Diferencia máxima: " << fixed << setprecision(6) << maxDiff << endl;
    cout << "PSNR: ";
    if (mse > 0) {
        double psnr = 10 * log10((255.0 * 255.0) / mse);
        cout << psnr << " dB" << endl;
    } else {
        cout << "Infinito (imágenes idénticas)" << endl;
    }
    cout << endl;
    
    // Calcular speedup
    double speedup = (double)duration2D.count() / durationSep.count();
    cout << "Speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
    cout << "Reducción de tiempo: " << (1.0 - 1.0/speedup) * 100 << "%" << endl << endl;
    
    // Guardar imagen de diferencia absoluta
    Mat absDiff;
    absdiff(result2D, resultSep, absDiff);
    
    // Normalizar para visualización
    double minDiff, maxDiffVis;
    minMaxLoc(absDiff, &minDiff, &maxDiffVis);
    Mat absDiffVis;
    if (maxDiffVis > 0) {
        absDiff.convertTo(absDiffVis, CV_8U, 255.0 / maxDiffVis);
    } else {
        absDiffVis = Mat::zeros(absDiff.size(), CV_8U);
    }
    
    imwrite("out_absdiff.png", absDiffVis);
    cout << "Diferencia absoluta guardada: out_absdiff.png" << endl;
    
    // 6. ANÁLISIS DE COMPLEJIDAD
    cout << endl << "--- ANÁLISIS DE COMPLEJIDAD COMPUTACIONAL ---" << endl;
    int N = image.rows;
    int M = image.cols;
    int K = kernelSize;
    
    long long ops2D = (long long)N * M * K * K;
    long long opsSep = (long long)N * M * 2 * K;
    
    cout << "Imagen: " << N << "x" << M << ", Kernel: " << K << "x" << K << endl;
    cout << "Operaciones convolución 2D: O(N*M*K²) = " << ops2D << endl;
    cout << "Operaciones separable: O(N*M*2K) = " << opsSep << endl;
    cout << "Reducción teórica: " << fixed << setprecision(2) 
         << (double)ops2D / opsSep << "x" << endl;
    cout << "Factor de reducción: K/2 = " << K << "/2 = " << K/2 << endl;
    
    cout << endl << "=== Programa completado exitosamente ===" << endl;
    
    return 0;
}
'@

Set-Content -Path "main.cpp" -Value $mainCpp -Encoding UTF8
Write-Host "✓ main.cpp creado" -ForegroundColor Green

# Crear CMakeLists.txt
Write-Host "Creando CMakeLists.txt..." -ForegroundColor Green
$cmakeLists = @'
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
add_executable(svd_convolution main.cpp)

# Enlazar con OpenCV
target_link_libraries(svd_convolution ${OpenCV_LIBS})

# Opciones de compilación
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(svd_convolution PRIVATE -Wall -Wextra -O3)
endif()

# Mensaje de configuración
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
'@

Set-Content -Path "CMakeLists.txt" -Value $cmakeLists -Encoding UTF8
Write-Host "✓ CMakeLists.txt creado" -ForegroundColor Green

# Crear build.bat
Write-Host "Creando build.bat..." -ForegroundColor Green
$buildBat = @'
@echo off
echo === Compilacion de SVD Convolution Demo ===
echo.

if not exist "build" mkdir build
cd build

echo Configurando con CMake...
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Intentando con Visual Studio...
    cmake .. -G "Visual Studio 17 2022" -A x64
)

if %ERRORLEVEL% NEQ 0 (
    echo Error en configuracion
    pause
    exit /b 1
)

echo.
echo Compilando...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Error en compilacion
    pause
    exit /b 1
)

echo.
echo [OK] Compilacion exitosa
cd ..

if exist "input.jpg" (
    echo Ejecutando con input.jpg...
    if exist "build\Release\svd_convolution.exe" (
        build\Release\svd_convolution.exe input.jpg
    ) else (
        build\svd_convolution.exe input.jpg
    )
) else (
    echo.
    echo Coloca una imagen 'input.jpg' en este directorio
    echo O ejecuta: build\svd_convolution.exe tu_imagen.jpg
)

echo.
pause
'@

Set-Content -Path "build.bat" -Value $buildBat -Encoding UTF8
Write-Host "✓ build.bat creado" -ForegroundColor Green

Write-Host ""
Write-Host "=== Archivos creados exitosamente ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Archivos en el directorio:" -ForegroundColor Yellow
Get-ChildItem -Name | Where-Object { $_ -match '\.(cpp|txt|bat)$' }

Write-Host ""
Write-Host "Siguiente paso:" -ForegroundColor Yellow
Write-Host "1. Coloca una imagen 'input.jpg' en este directorio"
Write-Host "2. Ejecuta: .\build.bat"
Write-Host ""
Write-Host "O compila manualmente con:" -ForegroundColor Yellow
Write-Host "  mkdir build"
Write-Host "  cd build"
Write-Host "  cmake .."
Write-Host "  cmake --build . --config Release"
Write-Host ""
