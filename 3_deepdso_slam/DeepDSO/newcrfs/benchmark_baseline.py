# Archivo: ~/Documentos/cnn-dso/DeepDSO/newcrfs/benchmark_baseline.py
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
import time
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms

# --- IMPORTS DEL SISTEMA ORIGINAL ---
monodepth2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'monodepth2'))
if monodepth2_path not in sys.path:
    sys.path.insert(0, monodepth2_path)

import networks
from layers import disp_to_depth

# --- CONFIGURACIÓN ---
IMG_DIR = os.path.expanduser("~/Documentos/benchmark_images")
ITERATIONS = 50 
MODEL_NAME = "mono+stereo_640x192" # El que configuraste en tu guía

def load_model():
    """Lógica extraída de tu infer_flask.py original"""
    model_path = os.path.join(monodepth2_path, "models", MODEL_NAME)
    print("-> Cargando modelo Monodepth2...")
    
    encoder_path = os.path.join(model_path, "encoder.pth")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to('cpu')
    encoder.eval()

    depth_decoder_path = os.path.join(model_path, "depth.pth")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to('cpu')
    depth_decoder.eval()
    
    return encoder, depth_decoder, feed_width, feed_height

def run_benchmark():
    print("=== INICIANDO BENCHMARK: BASELINE (MONODEPTH2) ===")
    
    # 1. Cargar Modelo
    encoder, depth_decoder, feed_width, feed_height = load_model()
    
    # 2. Obtener imágenes
    images = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    if not images:
        print("ERROR: No hay imágenes en ~/Documentos/benchmark_images")
        return

    # 3. Warmup
    print("-> Warmup...")
    img_warmup = pil.open(images[0]).convert('RGB')
    input_image = img_warmup.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    with torch.no_grad():
        features = encoder(input_image)
        outputs = depth_decoder(features)

    # 4. Loop
    times = []
    print("-> Ejecutando iteraciones...")
    for i in range(ITERATIONS):
        img_path = images[i % len(images)]
        
        # Medimos TODO el proceso: Carga, Resize e Inferencia
        start = time.time()
        
        input_image = pil.open(img_path).convert('RGB')
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        
        with torch.no_grad():
            features = encoder(input_image)
            outputs = depth_decoder(features)
            
        end = time.time()
        times.append(end - start)
        
        if i % 10 == 0:
            print("Iter {}/{}".format(i, ITERATIONS))

    # 5. Resultados
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print("\n" + "="*40)
    print("RESULTADOS: MONODEPTH2 (BASELINE)")
    print("Tiempo Promedio:  {:.4f} segundos/frame".format(avg_time))
    print("FPS Estimado:     {:.4f} FPS".format(fps))
    print("="*40 + "\n")

if __name__ == "__main__":
    run_benchmark()