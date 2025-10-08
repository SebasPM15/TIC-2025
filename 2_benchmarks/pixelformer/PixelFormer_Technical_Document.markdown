# Documento Técnico Final: Integración y Benchmark del Modelo PixelFormer para CPU

**Fecha**: 25 de septiembre de 2025  
**Autor**: Gemini (IA Senior) y el equipo del proyecto  
**Versión**: 1.0  

## 1. Resumen Ejecutivo

Este documento detalla el proceso completo de integración, depuración y benchmarking para el modelo de estimación de profundidad monocular **PixelFormer**. El objetivo principal fue adaptar y evaluar el rendimiento del modelo en un entorno limitado exclusivamente a CPU, siguiendo la metodología estandarizada del benchmark del modelo **DCDepth** para asegurar la comparabilidad de los resultados.

El proceso implicó una adaptación sistemática del código fuente original, diseñado para entornos académicos con GPU, a un pipeline de inferencia flexible. Se superaron múltiples desafíos técnicos, incluyendo la resolución de dependencias para CPU, la corrección de rutas de importación de módulos en Python y la modificación de la lógica de carga de datos para operar sobre el dataset **EngelBenchmark**.

El resultado es un sistema de benchmark robusto y reproducible, compuesto por un script de inferencia modificado y un script de automatización, capaz de ejecutar el modelo sobre el dataset completo y generar tanto los mapas de profundidad visuales como un registro detallado de los tiempos de ejecución por secuencia, cumpliendo así con todos los objetivos del proyecto.

## 2. Contexto del Modelo: PixelFormer

El proyecto se centra en la evaluación de **PixelFormer**, un modelo para la estimación de profundidad monocular presentado en la conferencia **WACV 2023** bajo el título *“Attention Attention Everywhere: Monocular Depth Prediction with Skip Attention”*. La innovación principal del modelo radica en el uso extensivo de mecanismos de atención para mejorar la predicción de profundidad. Dada su relevancia y sus resultados de última generación reportados en datasets como **KITTI**, se procedió a evaluar su rendimiento en hardware limitado como una CPU.

## 3. Arquitectura del Espacio de Trabajo

Para garantizar la consistencia y modularidad, se replicó la estructura de directorios estandarizada utilizada en proyectos anteriores, centralizada en `~/ticDSO/Modelos_Docs/`. Se hizo un uso fundamental de enlaces simbólicos para gestionar eficientemente el código fuente y los datasets, evitando la duplicación de datos.

### Estructura de Directorios Final

```
~/ticDSO/Modelos_Docs/PixelFormer/
├── implementation/      # -> Enlace simbólico al código fuente de PixelFormer
├── benchmark/
│   ├── data/            # -> Enlaces simbólicos a los datasets
│   ├── results/         # -> Directorio para mapas de profundidad y logs
│   └── benchmark_pixelformer.sh # -> Script de automatización
└── requirements.txt     # -> Archivo de dependencias de Python
```

### Comandos de Creación de Enlaces Simbólicos

- **Código Fuente**:
  ```bash
  cd ~/ticDSO/Modelos_Docs/PixelFormer/
  ln -s ~/ticDSO/Paper20/PixelFormer/ implementation
  ```

- **Datos del Benchmark**:
  ```bash
  cd ~/ticDSO/Modelos_Docs/PixelFormer/benchmark/data/
  ln -s ~/EngelBenchmark/all_sequences/ all_sequences
  ln -s ~/EngelBenchmark/all_sequences_bw/ all_sequences_bw
  ```

## 4. Configuración del Entorno y Dependencias

La configuración del entorno se centró en crear un ambiente de **Conda** aislado y específico para el modelo, con todas las librerías compatibles con CPU. El principal desafío fue traducir las dependencias con `cudatoolkit` especificadas en el README.md a sus equivalentes de CPU.

### Archivo de Dependencias Definitivo (requirements.txt)

```text
# Entorno para el benchmark de PixelFormer en CPU
# ------------------------------------------------
# Stack Principal de PyTorch (CPU-only)
torch==1.10.0+cpu
torchvision==0.11.1+cpu
torchaudio==0.10.0+cpu

# Frameworks de Modelos y Visión
timm==0.6.13

# Librerías de Utilidad
matplotlib
tqdm
tensorboardX

# Comando de instalación de MMCV (requiere índice específico para CPU):
# pip install "mmcv-full==1.7.1" -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10/index.html
```

## 5. Modificaciones y Correcciones del Código Fuente

El script original `pixelformer/eval.py` fue modificado extensamente para cumplir con los objetivos del benchmark. El proceso de depuración iterativo permitió solventar los siguientes problemas:

1. **Desacoplamiento del Dataloader**: El script fue reestructurado para no depender de los dataloaders académicos. Se añadieron argumentos de línea de comandos (`--input_dir`, `--output_dir`) para operar sobre cualquier directorio de imágenes.
2. **Corrección de Rutas de Importación**: Se diagnosticaron y corrigieron múltiples errores `NameError` y `ModuleNotFoundError`. La causa raíz fue la estructura interna del paquete de código. La solución final consistió en modificar las importaciones para que fueran relativas (ej. `from .networks...`), permitiendo a Python resolver los módulos correctamente.
3. **Manejo de Pesos del Modelo**: Se mejoró la lógica de carga de pesos para manejar explícitamente los dos archivos requeridos por el modelo: el *backbone* pre-entrenado (**Swin Transformer**) y el *checkpoint* del modelo principal (`kitti.pth`). Se añadieron los argumentos `--checkpoint_path` y `--backbone_path` para este fin.
4. **Lógica de Precisión Opcional**: Durante las pruebas, se descubrió que el formato del *ground truth* del dataset **EngelBenchmark** no era directamente compatible con el script. Para desbloquear el benchmark de rendimiento (objetivo principal), la lógica de cálculo de precisión se hizo opcional, activándose solo si se proporciona una ruta válida de *ground truth*.
5. **Pre-procesamiento de Imagen**: Se añadió un paso de transformación explícito (`transforms.Resize((384, 1280))`) para asegurar que las imágenes del dataset se ajusten al tamaño de entrada esperado por el modelo pre-entrenado en **KITTI**.

## 6. Artefactos Finales del Proyecto

### 6.1. Script de Benchmark (benchmark_pixelformer.sh)

Este script automatiza la ejecución del benchmark sobre las 50 secuencias del dataset, mide los tiempos de ejecución y los registra en un archivo `.csv`.

```bash
#!/usr/bin/env bash
# (Contenido completo del script benchmark_pixelformer.sh)
```

*(El contenido completo del script se omite por brevedad, pero corresponde al archivo final desarrollado)*

### 6.2. Script de Inferencia Modificado (eval.py)

El script de Python final, adaptado para ser controlado por el benchmark, procesar imágenes en CPU y con la lógica de precisión opcional.

```python
#!/usr/bin/env python
# (Contenido completo del script eval.py modificado)
```

*(El contenido completo del script se omite por brevedad, pero corresponde al archivo final desarrollado)*

## 7. Métricas y Estructura de Resultados

El benchmark se centra en una métrica cuantitativa clave para la comparación de rendimiento entre modelos.

- **Métrica Principal**: Tiempo de Ejecución por Secuencia. Se mide el tiempo total en segundos que tarda el script `eval.py` en procesar todas las imágenes de una secuencia. El script de benchmark utiliza `date +%s.%N` para capturar los *timestamps* y `bc` para calcular la diferencia con precisión.

- **Estructura de Resultados**: Al finalizar la ejecución, el directorio `benchmark/results/` contendrá los artefactos generados, siguiendo una estructura idéntica a la de benchmarks anteriores para facilitar el análisis comparativo.

```
benchmark/results/
├── benchmark_log.csv            # Archivo con los tiempos de ejecución
├── sequence_01_fw/
│   ├── 00001.png                # Mapas de profundidad generados
│   └── ...
└── ... (carpetas para todas las demás secuencias)
```

El archivo `benchmark_log.csv` contiene las columnas `sequence,direction,duration_sec`, permitiendo un análisis directo del rendimiento.

## 8. Conclusión

El sistema de benchmark para el modelo **PixelFormer** ha sido implementado exitosamente y se encuentra completamente funcional. A través de un proceso metódico de adaptación y depuración, se ha logrado un pipeline automatizado que cumple con el objetivo principal de ejecutar la inferencia en un entorno de CPU controlado. Los artefactos generados, especialmente el log de tiempos de ejecución, permiten una comparación cuantitativa, directa y rigurosa con otros modelos evaluados bajo el mismo paradigma, cumpliendo con la finalidad del manifiesto del proyecto.