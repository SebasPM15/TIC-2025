# Archivo: ~/Documentos/PixelFormer/benchmark_pixel.py
import time
import os
import torch
import numpy as np
from tqdm import tqdm # Barra de progreso
from infer_flask_pixelformer import DepthService, MODEL_PATH, BACKBONE_PATH

# --- CONFIGURACIÓN ---
IMG_DIR = os.path.expanduser("~/Documentos/benchmark_images")
ITERATIONS = 50 # Repetir para tener promedio estable

def run_benchmark():
    print(f"=== INICIANDO BENCHMARK: PIXELFORMER ===")
    print(f"Dispositivo: CPU (Esto tardará un poco...)")
    
    # 1. Cargar el Modelo (Tal cual lo hace tu servidor)
    print("-> Cargando Servicio y Modelo...")
    service = DepthService(MODEL_PATH, BACKBONE_PATH)
    
    # 2. Obtener imágenes
    images = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    if not images:
        print("ERROR: No hay imágenes en ~/Documentos/benchmark_images")
        return
    
    print(f"-> Imágenes encontradas: {len(images)}")
    
    # 3. Warmup (Calentamiento)
    # Importante para cargar cachés de PyTorch y asignar memoria
    print("-> Calentando motores (Warmup)...")
    with open(images[0], 'rb') as f:
        _ = service.predict_depth(f)

    # 4. Loop de Prueba
    times = []
    print(f"-> Ejecutando {ITERATIONS} inferencias...")
    
    for i in tqdm(range(ITERATIONS)):
        img_path = images[i % len(images)] # Cíclico si hay pocas fotos
        
        start = time.time()
        with open(img_path, 'rb') as f:
            _ = service.predict_depth(f)
        end = time.time()
        
        times.append(end - start)

    # 5. Resultados
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print("\n" + "="*40)
    print(f"RESULTADOS: PIXELFORMER (CPU)")
    print(f"Total Inferencia: {len(times)} iteraciones")
    print(f"Tiempo Promedio:  {avg_time:.4f} segundos/frame")
    print(f"FPS Estimado:     {fps:.4f} FPS")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_benchmark()