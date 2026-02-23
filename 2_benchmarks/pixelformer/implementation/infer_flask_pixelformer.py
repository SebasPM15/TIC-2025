import os
import sys
import time
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import OrderedDict
import base64
from io import BytesIO

# --- 1. CONFIGURACIÓN DE RUTAS ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PIXELFORMER_SRC_PATH = os.path.join(SCRIPT_DIR, "pixelformer")
sys.path.append(PIXELFORMER_SRC_PATH)

try:
    from networks.PixelFormer import PixelFormer
except ImportError as e:
    print(f"Error Crítico: No se pudo importar PixelFormer.")
    print(f"Detalle del error: {e}")
    sys.exit(1)

# --- 2. CONFIGURACIÓN DEL MODELO ---
MODEL_PATH = os.path.join(SCRIPT_DIR, "pretrained/checkpoints/nyu.pth")
BACKBONE_PATH = os.path.join(SCRIPT_DIR, "pretrained/backbone/swin_large_patch4_window7_224_22k.pth")

# --- FUNCIÓN DE UTILIDAD: Limpiar llaves 'module.' ---
def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict

# --- 3. CLASE DE SERVICIO ---
class DepthService:
    def __init__(self, model_path, backbone_path):
        self.device = torch.device("cpu")
        print(f"INFO: Configurando dispositivo: {self.device}")
        print("INFO: Cargando modelo PixelFormer (Versión LARGE07)...")
        
        try:
            self.model = PixelFormer(version='large07', inv_depth=False, max_depth=10.0, pretrained=backbone_path)
            
            print(f"INFO: Cargando pesos de NYU desde {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            state_dict = clean_state_dict(state_dict)
            self.model.load_state_dict(state_dict, strict=True)
            
            self.model.to(self.device)
            self.model.eval()
            print("INFO: ¡Modelo PixelFormer cargado exitosamente!")

        except Exception as e:
            print(f"ERROR FATAL al cargar el modelo: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Transformaciones para NYU
        self.transform = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def predict_depth(self, image_source):
        """
        Predice el mapa de profundidad desde una imagen.
        
        Args:
            image_source: Puede ser un stream, PIL Image, o bytes
            
        Returns:
            dict con estructura dinBody compatible con Android
        """
        # Convertir el input a PIL Image
        if isinstance(image_source, bytes):
            img = Image.open(BytesIO(image_source)).convert('RGB')
        elif hasattr(image_source, 'read'):  # Es un stream
            img = Image.open(image_source).convert('RGB')
        else:  # Ya es PIL Image
            img = image_source.convert('RGB')
        
        # Guardar dimensiones originales
        original_width, original_height = img.size
        
        # Preprocesar
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Inferencia
        with torch.no_grad():
            start_time = time.time()
            pred_depth = self.model(img_tensor)
            inference_time = time.time() - start_time
        
        # Procesar salida
        depth_map = pred_depth.squeeze().cpu().numpy()
        
        # Normalizar para visualización (0-255)
        depth_normalized = ((depth_map - depth_map.min()) / 
                           (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        
        # Convertir a base64 para transmisión eficiente
        depth_img = Image.fromarray(depth_normalized, mode='L')
        buffered = BytesIO()
        depth_img.save(buffered, format="PNG")
        depth_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "status": "success",
            "model_info": {
                "name": "PixelFormer",
                "version": "large07",
                "backbone": "swin_transformer"
            },
            "timing": {
                "inference_ms": round(inference_time * 1000, 2),
                "timestamp": time.time()
            },
            "image_info": {
                "original_width": original_width,
                "original_height": original_height,
                "depth_width": depth_map.shape[1],
                "depth_height": depth_map.shape[0]
            },
            "depth_data": {
                "format": "base64_png",
                "encoding": "grayscale",
                "data": depth_base64,
                "min_depth": float(depth_map.min()),
                "max_depth": float(depth_map.max()),
                "mean_depth": float(depth_map.mean())
            }
        }

# --- 4. INICIALIZACIÓN ---
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: No se encuentra el modelo en {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(BACKBONE_PATH):
    print(f"ERROR: No se encuentra el backbone en {BACKBONE_PATH}")
    sys.exit(1)

depth_service = DepthService(MODEL_PATH, BACKBONE_PATH)

# --- 5. SERVIDOR FLASK ---
app = Flask(__name__)
CORS(app)  # Permitir CORS para requests desde Android

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de health check"""
    return jsonify({
        "status": "healthy",
        "model": "PixelFormer Large07",
        "ready": True
    })

@app.route('/api/v1/info', methods=['GET'])
def info():
    """Información del modelo"""
    return jsonify({
        "model": {
            "name": "PixelFormer",
            "version": "large07",
            "architecture": "Swin Transformer + Pixel Decoder",
            "trained_on": "NYU Depth V2",
            "max_depth": 10.0,
            "input_size": [480, 640]
        },
        "server": {
            "version": "2.0",
            "framework": "PyTorch + Flask",
            "device": "CPU"
        },
        "endpoints": {
            "predict": "/api/v1/predict",
            "health": "/health",
            "info": "/api/v1/info"
        }
    })

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    Endpoint principal de predicción.
    
    Acepta:
    - multipart/form-data con campo 'image' (archivo)
    - application/json con campo 'image_base64' (string base64)
    
    Retorna:
    - JSON con estructura dinBody compatible con Android
    """
    start_time = time.time()
    
    try:
        # Opción 1: Imagen como archivo multipart
        if 'image' in request.files:
            file = request.files['image']
            print(f"INFO: Procesando imagen desde multipart (nombre: {file.filename})")
            result = depth_service.predict_depth(file.stream)
            
        # Opción 2: Imagen como base64 en JSON
        elif request.is_json and 'image_base64' in request.json:
            image_base64 = request.json['image_base64']
            print("INFO: Procesando imagen desde base64...")
            image_bytes = base64.b64decode(image_base64)
            result = depth_service.predict_depth(image_bytes)
            
        else:
            return jsonify({
                "status": "error",
                "error": {
                    "code": "MISSING_IMAGE",
                    "message": "Se requiere 'image' (multipart) o 'image_base64' (JSON)"
                }
            }), 400
        
        # Añadir tiempo total de request
        total_time = time.time() - start_time
        result['timing']['total_ms'] = round(total_time * 1000, 2)
        
        print(f"INFO: ✓ Predicción exitosa en {total_time:.2f}s")
        return jsonify(result)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "status": "error",
            "error": {
                "code": "PREDICTION_FAILED",
                "message": str(e)
            }
        }), 500

@app.route('/api/v1/predict_raw', methods=['POST'])
def predict_raw():
    """
    Endpoint alternativo que devuelve el array de profundidad completo.
    ⚠️ ADVERTENCIA: Puede generar JSONs muy grandes (varios MB)
    """
    if 'image' not in request.files:
        return jsonify({"error": "Falta 'image'"}), 400
    
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img_tensor = depth_service.transform(img).unsqueeze(0).to(depth_service.device)

        with torch.no_grad():
            start_time = time.time()
            pred_depth = depth_service.model(img_tensor)
            inference_time = time.time() - start_time
        
        depth_map = pred_depth.squeeze().cpu().numpy().tolist()
        
        return jsonify({
            "status": "success",
            "model": "PixelFormer (Large07)",
            "inference_time_ms": round(inference_time * 1000, 2),
            "depth_map": depth_map  # ⚠️ Array completo
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("INFO: Servidor PixelFormer para Android iniciado")
    print("INFO: Endpoints disponibles:")
    print("  - GET  /health          -> Health check")
    print("  - GET  /api/v1/info     -> Info del modelo")
    print("  - POST /api/v1/predict  -> Predicción (recomendado)")
    print("  - POST /api/v1/predict_raw -> Array completo (DEBUG)")
    print("=" * 60)
    print(f"INFO: Escuchando en puerto 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)