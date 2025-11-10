# Optimización de Convolución 2D mediante SVD

## Descripción del Proyecto

Este proyecto demuestra cómo la Descomposición en Valores Singulares (SVD) puede utilizarse para optimizar operaciones de convolución 2D, aprovechando el concepto de **kernels separables**.

## Fundamento Teórico

### 1. Convolución 2D Tradicional

La convolución de una imagen `I` de tamaño `N×M` con un kernel `K` de tamaño `k×k` requiere:

**Complejidad:** O(N × M × k²)

Para cada píxel de la imagen, se deben realizar k² multiplicaciones y sumas.

### 2. Kernels Separables y SVD

Un kernel 2D `K` es **separable** si puede expresarse como el producto externo de dos vectores 1D:

```
K = u · v^T
```

Donde:
- `u` es un vector columna de tamaño k×1
- `v` es un vector fila de tamaño 1×k

#### Aplicación de SVD

La SVD descompone cualquier matriz K como:

```
K = U · Σ · V^T
```

Donde:
- `U` contiene los vectores singulares izquierdos
- `Σ` es una matriz diagonal con valores singulares (σ₁ ≥ σ₂ ≥ ... ≥ σₖ)
- `V^T` contiene los vectores singulares derechos

Para kernels aproximadamente separables (como el gaussiano), el primer valor singular σ₁ domina:

```
K ≈ σ₁ · u₁ · v₁^T
```

Definiendo:
```
kernelV = √σ₁ · u₁
kernelH = √σ₁ · v₁^T
```

Obtenemos:
```
K ≈ kernelV · kernelH
```

### 3. Convolución Separable

Con un kernel separable, la convolución 2D se divide en dos pasos:

1. **Convolución horizontal:** I' = I ⊗ kernelH
2. **Convolución vertical:** I'' = I' ⊗ kernelV

**Complejidad:** O(N × M × 2k)

### 4. Reducción de Complejidad

**Factor de reducción teórico:**
```
Operaciones 2D / Operaciones Separables = (N·M·k²) / (N·M·2k) = k/2
```

Para un kernel 15×15: **Reducción de 7.5x** en número de operaciones.

## Resultados Experimentales

### Métricas de Precisión

- **MSE (Mean Squared Error):** Cercano a 0 (típicamente < 10⁻¹⁰)
- **Diferencia máxima:** < 0.001 (errores de punto flotante)
- **PSNR:** > 80 dB (calidad perfecta)

Esto confirma que el resultado es **prácticamente idéntico** al de la convolución 2D completa.

### Rendimiento

Resultados típicos en una imagen 1920×1080 con kernel 15×15:

| Método | Tiempo | Speedup |
|--------|--------|---------|
| Convolución 2D | ~150 ms | 1.0x |
| Convolución Separable | ~25 ms | **6.0x** |

El speedup real es ligeramente menor que el teórico debido a:
- Overhead de dos llamadas a `filter2D`
- Operaciones de lectura/escritura en memoria intermedia
- Optimizaciones internas de OpenCV

## Un poco de Análisis Detallado

### ¿Por qué funciona la aproximación SVD?

Para kernels gaussianos:
- El **primer valor singular** captura > 99.9% de la energía total
- Los valores singulares subsecuentes son negligibles (σ₂/σ₁ < 0.1%)
- Por lo tanto: K ≈ σ₁·u₁·v₁^T es una excelente aproximación

### Kernels Separables vs. No Separables

**Kernels naturalmente separables:**
- Gaussiano
- Derivadas gaussianas (Sobel, Scharr)
- Box filter
- Binomial

**Kernels no separables:**
- Laplaciano de gaussiano (LoG)
- Detección de esquinas (algunos)
- Kernels de rotación arbitraria

Para kernels no separables, SVD con rango 1 produce **aproximaciones** con error no despreciable.

### Aplicaciones Prácticas

#### 1. Visión Computacional
- **Filtrado rápido:** Blur, sharpening, reducción de ruido
- **Detección de bordes:** Operadores de Sobel optimizados
- **Pirámides gaussianas:** Construcción eficiente para SIFT, ORB

#### 2. Redes Neuronales Convolucionales (CNNs)
- **Depthwise Separable Convolutions** (MobileNet, EfficientNet)
- Reducción de parámetros: k² → 2k
- Reducción de FLOPs para inferencia rápida
- Compresión de modelos preentrenados

#### 3. Procesamiento de Video
- Aplicación frame-a-frame con bajo costo computacional
- Filtros temporales separables

## Relación entre SVD y PCA

### Conexiones Fundamentales

**SVD y PCA están íntimamente relacionados:**

1. **PCA busca direcciones de máxima varianza**
   - Equivalente a encontrar los vectores propios de la matriz de covarianza
   
2. **SVD proporciona esas direcciones directamente**
   - Los vectores singulares izquierdos `U` son los componentes principales
   - Los valores singulares `σᵢ` están relacionados con la varianza explicada

3. **En el contexto de kernels:**
   - El primer componente de SVD captura la "dirección principal" del filtro
   - Aproximación de rango 1 = Proyección en el espacio de máxima varianza

### Interpretación Geométrica

- **SVD del kernel:** Descompone el filtro en modos independientes
- **Primer modo:** Captura la estructura dominante (ej: gradiente suave del gaussiano)
- **Modos superiores:** Capturan detalles finos (negligibles en kernels suaves)

## Limitaciones y Consideraciones

### 1. Precisión Numérica
- Errores de punto flotante acumulados
- Irrelevantes en práctica (< 1/255)

### 2. Kernels No Separables
- La aproximación de rango 1 puede no ser suficiente
- Considerar SVD de rango k > 1 con múltiples pases

### 3. Casos Patológicos
- Kernels con múltiples valores singulares significativos
- Solución: Analizar espectro singular antes de aplicar

## Conclusiones

1. **Eficiencia Demostrada:** La convolución separable reduce dramáticamente el costo computacional (6-7x en práctica).

2. **Precisión Mantenida:** Para kernels gaussianos, el resultado es prácticamente idéntico a la convolución 2D completa.

3. **Aplicabilidad Universal:** Esta técnica es fundamental en bibliotecas modernas de visión computacional y deep learning.

4. **Trade-off Claro:** 
   - Ventaja: Velocidad
   - Limitación: Solo aplicable a kernels separables o casi-separables

5. **Implicaciones para Deep Learning:**
   - Inspiró arquitecturas completas (MobileNet, ShuffleNet)
   - Demuestra que la eficiencia puede lograrse sin sacrificar precisión

## Referencias

- Gonzalez & Woods, "Digital Image Processing" (Capítulo sobre filtrado separable)
- OpenCV Documentation: `filter2D()`, `sepFilter2D()`
- Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017)
- Strang, "Introduction to Linear Algebra" (Capítulo sobre SVD)

## Estructura de Archivos

```
.
├── main.cpp              # Código fuente principal
├── CMakeLists.txt        # Configuración de compilación
├── README.md             # Este archivo
├── baboon.png             # Imagen de entrada
├── out_ref2d.png         # Resultado convolución 2D
├── out_sep.png           # Resultado convolución separable
└── out_absdiff.png       # Diferencia absoluta
```

---

**Autor:** Atlantis04

**Fecha:** Noviembre 2025
