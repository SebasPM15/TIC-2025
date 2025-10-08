# Documento Técnico Final: Benchmark y Adaptación del Modelo DCDepth para CPU

**Fecha:** 25 de septiembre de 2025  
**Autor:** Gemini (IA Senior) y el equipo del proyecto.  
**Versión:** 2.1 (Edición Final con Métricas y Resultados)  

## 1. Resumen Ejecutivo

Este documento detalla el proceso completo de configuración, depuración y ejecución de un benchmark para el modelo de estimación de profundidad monocular DCDepth. El objetivo principal fue analizar el rendimiento del modelo en un entorno exclusivo de CPU, simulando las restricciones de hardware de dispositivos de bajo consumo. Las pruebas se ejecutaron sobre el dataset EngelBenchmark, procesando secuencias de imágenes tanto en avance (fw) como en retroceso (bw). El proceso implicó una compleja configuración de entorno, una depuración sistemática del código fuente para resolver múltiples desafíos (dependencias, rutas hardcodeadas, errores de sintaxis y lógica) y la mejora del script de benchmark para generar métricas cuantitativas de rendimiento. El resultado es un sistema robusto y reproducible, capaz de ejecutar el benchmark completo y generar un registro detallado de los tiempos de procesamiento por secuencia.

## 2. Contexto del Modelo: DCDepth

El proyecto se centra en la implementación y evaluación de DCDepth, un framework para la estimación de profundidad monocular presentado en la conferencia NeurIPS 2024.

- **Problema:** Estimar la profundidad a partir de una única imagen 2D.
- **Innovación:** DCDepth transforma parches de la imagen al dominio de la frecuencia mediante la Transformada de Coseno Discreta (DCT). El modelo predice los coeficientes de frecuencia en lugar de los valores de píxeles directamente.
- **Estrategia:** El modelo sigue una estrategia progresiva, prediciendo primero los coeficientes de baja frecuencia (estructura general) y luego refinando con los de alta frecuencia (detalles).
- **Relevancia:** El paper original reporta resultados de última generación (state-of-the-art), lo que justifica la investigación de su rendimiento en hardware limitado como una CPU.

## 3. Arquitectura del Espacio de Trabajo

Para mantener un entorno limpio y modular, se definió una estructura de directorios centralizada en `~/ticDSO/Modelos_Docs/`. Se hizo un uso extensivo de enlaces simbólicos para evitar la duplicación de código y datos pesados, una práctica recomendada para proyectos de gran escala.

- **Código Fuente:** El directorio `implementation/` apunta a la ubicación original del código.
- **Datos del Benchmark:** Los datos del `EngelBenchmark` se enlazan al directorio `benchmark/data/`.

```
~/ticDSO/Modelos_Docs/DCDepth/
├── implementation/  # -> Enlace simbólico al código fuente original
├── benchmark/
│   ├── data/        # -> Enlaces simbólicos a los datasets
│   ├── results/     # -> Directorio para los mapas de profundidad generados
│   └── benchmark_dcdepth.sh # -> Script de automatización
└── requirements.txt # -> Archivo de dependencias de Python
```

### 3.1. Enlaces Simbólicos (Symbolic Links)

- **Código Fuente:**

  ```bash
  cd ~/ticDSO/Modelos_Docs/DCDepth/
  ln -s ~/ticDSO/Implementacion\ paper\ 9/DCDepth/ implementation
  ```

- **Datos del Benchmark:**

  ```bash
  cd ~/ticDSO/Modelos_Docs/DCDepth/benchmark/data/
  ln -s ~/EngelBenchmark/all_sequences/ all_sequences
  ln -s ~/EngelBenchmark/all_sequences_bw/ all_sequences_bw
  ```

## 4. Configuración y Depuración del Entorno

La configuración del entorno fue el primer gran desafío, superado mediante un proceso metódico.

### 4.1. Desafío Inicial: Limitaciones de Espacio y Dependencias

El entorno de ejecución falló inicialmente por falta de espacio en el disco principal. La solución estructural fue migrar toda la instalación de Conda a un disco secundario con mayor capacidad. Posteriormente, múltiples conflictos entre las versiones de las librerías de Python requirieron la creación de un script de instalación controlado y reproducible para garantizar la estabilidad.

### 4.2. Descubrimiento de Dependencias Ocultas

Durante la ejecución inicial, se encontraron varias dependencias que no estaban documentadas y que fueron añadidas progresivamente al entorno: `pandas`, `Pillow`, `mmcv-full` y `easydict`.

### 4.3. Script de Instalación de Entorno Final

El siguiente script representa el procedimiento final y exitoso para configurar el entorno `dcdepth_env`, incluyendo todas las dependencias descubiertas:

```bash
# SCRIPT FINAL PARA LA CONFIGURACIÓN DEL ENTORNO 'dcdepth_env'
conda activate dcdepth_env
pip uninstall -y torch torchvision torchaudio timm mmcv mmengine torchmetrics opencv-python triton pandas Pillow easydict
pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu
pip install timm==0.6.13 mmengine==0.10.4
pip install numpy==1.26.4 scipy==1.14.0 matplotlib==3.9.0 pandas Pillow easydict tqdm fsspec lightning-utilities
pip install torchmetrics==0.11.4
pip install "opencv-python<4.11"
pip install "mmcv-full==1.7.1" -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13.1/index.html
```

## 5. Modificaciones y Correcciones del Código Fuente

El código original requirió varias modificaciones críticas para funcionar en el entorno de benchmark.

### 5.1. Adaptación Inicial del Script `test.py`

El script `test.py` fue modificado para aceptar directorios de entrada y salida personalizados (`--input_dir`, `--output_dir`), desacoplándolo de los dataloaders académicos. Todas las operaciones de PyTorch fueron forzadas a ejecutarse en `cpu` para cumplir los objetivos del benchmark.

### 5.2. Corrección de Errores Críticos

Durante las pruebas, se identificaron y solucionaron los siguientes errores:

1. **Ruta Hardcodeada:** El modelo intentaba cargar los pesos de un backbone (`Swin Transformer`) desde una ruta absoluta del sistema del desarrollador original. **Solución:** Se descargaron los pesos y se modificó el archivo de configuración `.yaml` para usar una ruta relativa.
2. **Incompatibilidad de Dimensiones:** El modelo esperaba imágenes de tamaño `352x1216`, pero el dataset `EngelBenchmark` las proveía en `1024x1280`. **Solución:** Se añadió una transformación `transforms.Resize((352, 1216))` en el pipeline de pre-procesamiento de `test.py`.
3. **Estructura de Directorios:** El script no encontraba las imágenes de las secuencias, ya que estaban en una subcarpeta `images/`. **Solución:** Se modificó la lógica de búsqueda de archivos en `test.py` para apuntar explícitamente a la subcarpeta `images/`.
4. **Error de Tipeo:** Un error de sintaxis (`add-argument` en lugar de `add_argument`) en la función `parse_args` impedía la ejecución. **Solución:** Se corrigió el nombre del método.

## 6. Artefactos Finales del Proyecto

A continuación se presentan los scripts y archivos de configuración en su versión final y funcional.

### 6.1. Script de Benchmark Mejorado (`benchmark_dcdepth.sh`)

El script de automatización fue mejorado para medir el tiempo de ejecución de cada secuencia y guardar los resultados en un archivo `.csv` para su posterior análisis:

```bash
#!/usr/bin/env bash
set -euo pipefail
# --- 0) Helpers
abspath() { cd "$1" >/dev/null 2>&1 && pwd -P; }
echo "⚙️ Cargando configuración..."
# --- 1) Ubicaciones base
SCRIPT_DIR="$(abspath "$(dirname "$0")")"
PROJECT_ROOT="$(abspath "$SCRIPT_DIR/..")"
IMPLEMENTATION_DIR="$PROJECT_ROOT/implementation/DCDepth"
# --- 2) Configuración del modelo
CONFIG_FILE="dct_eigen_pff"
CHECKPOINT_NAME="dcdepth_eigen.pth"
CHECKPOINT_PATH="$IMPLEMENTATION_DIR/checkpoints/$CHECKPOINT_NAME"
# --- 3) Datos y resultados
DATA_DIR="$SCRIPT_DIR/data"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"
# --- 4) CONFIGURACIÓN DEL BENCHMARK ---
SEQUENCES=({1..50}) # Rango completo de secuencias
LOG_FILE="$RESULTS_DIR/benchmark_log.csv"
# --- 5) INICIO DEL BENCHMARK ---
echo "🚀 Iniciando benchmark para DCDepth en CPU..."
echo "Los resultados numéricos se guardarán en: $LOG_FILE"
echo "sequence,direction,duration_sec" > "$LOG_FILE"
for seq_num in "${SEQUENCES[@]}"; do
  SEQ_ID="$(printf '%02d' "$seq_num")"
  for direction in fw bw; do
    if [[ "$direction" == "bw" ]]; then
      INPUT_DATA_PATH="$DATA_DIR/all_sequences_bw/sequence_${SEQ_ID}"
    else
      INPUT_DATA_PATH="$DATA_DIR/all_sequences/sequence_${SEQ_ID}"
    fi
    OUTPUT_SEQ_DIR="$RESULTS_DIR/sequence_${SEQ_ID}_${direction}"
    if [[ -d "$INPUT_DATA_PATH" ]]; then
      echo "----------------------------------------------------"
      echo "Procesando: sequence_${SEQ_ID} (${direction})..."
      mkdir -p "$OUTPUT_SEQ_DIR"
      CMD=( python "test.py" "$CONFIG_FILE" "$CHECKPOINT_PATH" --input_dir "$INPUT_DATA_PATH" --output_dir "$OUTPUT_SEQ_DIR" )
      start_time=$(date +%s.%N)
      ( cd "$IMPLEMENTATION_DIR" && "${CMD[@]}" )
      end_time=$(date +%s.%N)
      duration=$(echo "$end_time - $start_time" | bc)
      echo "${SEQ_ID},${direction},${duration}" >> "$LOG_FILE"
      echo "✅ Secuencia finalizada. Duración: ${duration} segundos."
    else
      echo "⚠️ Advertencia: Directorio no encontrado, saltando: $INPUT_DATA_PATH"
    fi
  done
done
echo ""
echo "🎉 Benchmark finalizado."
echo "Resultados numéricos guardados en: $LOG_FILE"
echo "--- RESUMEN DE TIEMPOS (segundos) ---"
cat "$LOG_FILE"
echo "------------------------------------"
```

### 6.2. Archivo de Dependencias Definitivo (`requirements.txt`)

Este archivo contiene la lista completa de paquetes de Python necesarios para recrear el entorno `dcdepth_env` de forma fiable:

```
# Entorno para el benchmark de DCDepth en CPU
# --- Stack Principal de PyTorch ---
# Instalar usando: pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt
torch==1.13.1+cpu
torchvision==0.14.1+cpu
torchaudio==0.13.1+cpu
# --- Frameworks de Modelos y Visión ---
timm==0.6.13
mmengine==0.10.4
# Para MMCV, usar un comando de instalación aparte por su índice específico:
# pip install "mmcv-full==1.7.1" -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13.1/index.html
# --- Librerías Científicas y de Utilidad ---
numpy==1.26.4
scipy==1.14.0
matplotlib==3.9.0
opencv-python<4.11
pandas
Pillow
easydict
torchmetrics==0.11.4
tqdm
fsspec
lightning-utilities
```

## 7. Métricas de Rendimiento para Comparación

Para cumplir con el objetivo de comparar el rendimiento de DCDepth con otros modelos, el benchmark se centra en la siguiente métrica cuantitativa:

- **Métrica Principal: Tiempo de Ejecución por Secuencia**
  - **Descripción:** Se mide el tiempo total, en segundos y con precisión de nanosegundos, que tarda el script `test.py` en procesar la totalidad de las imágenes de una secuencia completa.
  - **Método de Recolección:** El script `benchmark_dcdepth.sh` utiliza el comando `date +%s.%N` antes y después de la llamada al proceso de Python. La diferencia entre ambos timestamps se calcula con `bc` para obtener la duración real del procesamiento.
  - **Utilidad Comparativa:** Este dato, registrado para cada secuencia en direcciones `fw` y `bw`, es la base para la comparación de rendimiento. Permite calcular el tiempo promedio por imagen (`duración / N_imágenes`), la desviación estándar y comparar directamente la velocidad de DCDepth en CPU frente a otros frameworks.

## 8. Estructura de Resultados Esperada

Una vez que el script `benchmark_dcdepth.sh` finalice su ejecución completa sobre las 50 secuencias, la estructura del directorio `benchmark/results/` será la siguiente:

```
benchmark/results/
├── benchmark_log.csv # Archivo principal con las métricas de tiempo
│
├── sequence_01_fw/
│ ├── 00001.png # Mapa de profundidad para la imagen 1
│ ├── 00002.png
│ └── ... (imágenes hasta el final de la secuencia)
│
├── sequence_01_bw/
│ ├── 00001.png
│ └── ...
│
├── sequence_02_fw/
│ └── ...
│
├── ... (carpetas para las secuencias 02 a 50 en ambas direcciones)
│
└── sequence_50_bw/
    └── ...
```

- **`benchmark_log.csv`**: Este archivo es el resultado cuantitativo clave. Contendrá tres columnas: `sequence`, `direction`, y `duration_sec`, permitiendo un análisis sencillo en cualquier software de hoja de cálculo o scripting.
- **Directorios `sequence_XX_direction`**: Cada uno de estos directorios contendrá los mapas de profundidad generados por el model, guardados como imágenes `.png`. Los nombres de los archivos de salida corresponderán directamente a los nombres de los archivos de imagen de entrada.

## 9. Conclusión

El sistema de benchmark para el modelo DCDepth está ahora completo, robusto y es completamente funcional. A través de un proceso iterativo de depuración y mejora, se ha logrado un pipeline automatizado que no solo ejecuta la inferencia del modelo en un entorno de CPU controlado, sino que también genera los artefactos visuales (mapas de profundidad) y las métricas cuantitativas (tiempos de ejecución) necesarias para una comparación académica rigurosa. El proyecto está listo para la ejecución final del benchmark a gran escala.