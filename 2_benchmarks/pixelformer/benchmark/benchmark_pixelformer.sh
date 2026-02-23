#!/usr/bin/env bash
set -euo pipefail

# --- 0) Helpers ---
abspath() { cd "$1" >/dev/null 2>&1 && pwd -P; }

echo "⚙️  Cargando configuración para PixelFormer..."

# --- 1) Ubicaciones base ---
SCRIPT_DIR="$(abspath "$(dirname "$0")")"
PROJECT_ROOT="$(abspath "$SCRIPT_DIR/..")"
IMPLEMENTATION_DIR="$PROJECT_ROOT/implementation"

# --- 2) Configuración del modelo ---
CHECKPOINT_NAME="kitti.pth"
CHECKPOINT_PATH="$IMPLEMENTATION_DIR/checkpoints/$CHECKPOINT_NAME"
BACKBONE_NAME="swin_large_patch4_window12_384_22k.pth"
BACKBONE_PATH="$IMPLEMENTATION_DIR/checkpoints/$BACKBONE_NAME"

# --- 3) Datos y resultados ---
DATA_DIR="$SCRIPT_DIR/data"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

# --- 4) Configuración del Benchmark ---
# Se procesarán solo las primeras 10 secuencias.
SEQUENCES=({1..10})
LOG_FILE="$RESULTS_DIR/benchmark_log.csv"

# --- 5) INICIO DEL BENCHMARK (Modo Secuencial) ---
echo "🚀 Iniciando benchmark de RENDIMIENTO para PixelFormer en CPU (Modo Secuencial)..."
echo "Los resultados se guardarán en: $LOG_FILE"

# Crear el encabezado del archivo de log
echo "sequence,direction,duration_sec" > "$LOG_FILE"

# Bucle principal que procesa una secuencia a la vez
for seq_num in "${SEQUENCES[@]}"; do
  # Se usa %02d para asegurar el formato de dos dígitos con cero a la izquierda (ej. 1 -> 01)
  SEQ_ID="$(printf '%02d' "$seq_num")"

  for direction in fw bw; do
    BASE_SEQ_PATH="$DATA_DIR/all_sequences${direction:+_bw}/sequence_${SEQ_ID}"
    INPUT_IMG_PATH="$BASE_SEQ_PATH/images"
    OUTPUT_SEQ_DIR="$RESULTS_DIR/sequence_${SEQ_ID}_${direction}"

    if [[ -d "$INPUT_IMG_PATH" ]]; then
      echo "----------------------------------------------------"
      echo "Procesando: sequence_${SEQ_ID} (${direction})..."
      mkdir -p "$OUTPUT_SEQ_DIR"

      CMD=(
        python -m "pixelformer.eval"
        --checkpoint_path "$CHECKPOINT_PATH"
        --backbone_path "$BACKBONE_PATH"
        --input_dir "$INPUT_IMG_PATH"
        --output_dir "$OUTPUT_SEQ_DIR"
        --encoder "large07"
        --eigen_crop
        # --post_process # Desactivado para casi duplicar la velocidad
      )

      start_time=$(date +%s.%N)
      # Ejecuta el comando y muestra la salida en la terminal
      ( cd "$IMPLEMENTATION_DIR" && "${CMD[@]}" )
      end_time=$(date +%s.%N)
      duration=$(echo "$end_time - $start_time" | bc)

      # Guarda el resultado en el archivo CSV
      echo "${SEQ_ID},${direction},${duration}" >> "$LOG_FILE"
      echo "✅ Secuencia finalizada. Duración: ${duration} segundos."
    else
      echo "⚠️  Advertencia: Directorio de imágenes no encontrado, saltando: $INPUT_IMG_PATH"
    fi
  done
done

echo ""
echo "🎉 Benchmark finalizado."
echo "Resultados numéricos guardados en: $LOG_FILE"
echo "--- RESUMEN DE TIEMPOS (segundos) ---"
cat "$LOG_FILE"
echo "------------------------------------"