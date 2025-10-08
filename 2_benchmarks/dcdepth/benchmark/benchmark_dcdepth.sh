#!/usr/bin/env bash
set -euo pipefail

# --- 0) Helpers
abspath() { cd "$1" >/dev/null 2>&1 && pwd -P; }

echo "⚙️  Cargando configuración..."

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
#    - SEQUENCES: Define aquí sobre qué secuencias correr.
#      Ejemplo para una prueba rápida: SEQUENCES=(1 2 5)
#      Ejemplo para el benchmark completo: SEQUENCES=({1..50})
#    - LOG_FILE: Archivo donde se guardarán los tiempos.
SEQUENCES=(1 5 10 15 20 25 30 35 40 45) # <-- PROCESARÁ EXACTAMENTE ESTAS 10 SECUENCIAS
LOG_FILE="$RESULTS_DIR/benchmark_log.csv"

# --- 5) INICIO DEL BENCHMARK ---
echo "🚀 Iniciando benchmark para DCDepth en CPU..."
echo "Los resultados numéricos se guardarán en: $LOG_FILE"

# Crear el encabezado del archivo de log
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

      # --- MEDICIÓN DE TIEMPO ---
      start_time=$(date +%s.%N)
      
      # Ejecuta el comando de python mostrando su salida para poder depurar errores.
      ( cd "$IMPLEMENTATION_DIR" && "${CMD[@]}" )
      
      end_time=$(date +%s.%N)
      # Calcula la duración usando 'bc' para manejar decimales
      duration=$(echo "$end_time - $start_time" | bc)
      # --- FIN DE MEDICIÓN ---

      # Guardar el resultado en el archivo CSV
      echo "${SEQ_ID},${direction},${duration}" >> "$LOG_FILE"

      echo "✅ Secuencia finalizada. Duración: ${duration} segundos."
    else
      echo "⚠️  Advertencia: Directorio no encontrado, saltando: $INPUT_DATA_PATH"
    fi
  done
done

echo ""
echo "🎉 Benchmark finalizado."
echo "Resultados numéricos guardados en:"
echo "   ---> $LOG_FILE <---"

# Muestra un resumen de los resultados
echo ""
echo "--- RESUMEN DE TIEMPOS (segundos) ---"
cat "$LOG_FILE"
echo "------------------------------------"