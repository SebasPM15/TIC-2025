# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import PIL.Image as pil
import cv2
import torch
from torchvision import transforms
from flask import Flask, jsonify

#--- Añadir Monodepth2 al path
monodepth2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'monodepth2'))
if monodepth2_path not in sys.path:
    sys.path.insert(0, monodepth2_path)

import networks
from layers import disp_to_depth

# Configuración del Servidor Flask
app = Flask(__name__)

#--- Carga del Modelo Monodepth2
model_name = "mono+stereo_640x192"
model_path = os.path.join(monodepth2_path, "models", model_name)

def load_model():
    print("==> Cargando modelo Monodepth2 en CPU...")
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

    print("==> ¡Modelo cargado exitosamente!")
    return encoder, depth_decoder, feed_width, feed_height

encoder, depth_decoder, feed_width, feed_height = load_model()

# Ruta del Servidor
@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build', 'test.jpg'))
        if not os.path.exists(image_path):
            return jsonify({"error": "No se encontró test.jpg en la carpeta build"}), 400

        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size

        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        with torch.no_grad():
            input_image = input_image.to('cpu')
            features = encoder(input_image)
            outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        _, depth = disp_to_depth(disp, 0.1, 100)

        depth_resized = torch.nn.functional.interpolate(
            depth, (original_height, original_width), mode="bilinear", align_corners=False)

        depth_numpy = depth_resized.squeeze().cpu().numpy()

        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build', 'depthcrfs.txt'))
        f = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
        f.write('mat1', depth_numpy)
        f.release()

        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)