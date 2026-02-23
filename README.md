# TIC-2025 — Repositorio de Componentes de SLAM con Estimación de Profundidad

Este repositorio contiene los componentes desarrollados e integrados en el proyecto TIC 2025, enfocado en sistemas de SLAM visual enriquecidos con estimación de profundidad mediante redes neuronales convolucionales (CNN).

---

## Estructura del Repositorio

```
TIC-2025/
├── 2_benchmarks/
│   ├── dcdepth/                  # Benchmark DCDepth
│   │   ├── implementation/       # Código fuente del modelo DCDepth
│   │   └── checkpoints/
│   │       └── dcdepth_eigen.pth # Pesos preentrenados (KITTI Eigen split)
│   └── pixelformer/
│       └── implementation/       # Código fuente de PixelFormer
│           ├── pixelformer/networks/
│           │   ├── PixelFormer.py
│           │   ├── PQI.py
│           │   └── SAM.py
│           ├── pretrained/checkpoints/
│           │   └── nyu.pth       # Pesos preentrenados (NYU Depth V2)
│           └── infer_flask_pixelformer.py
│
└── 3_deepdso_slam/
    ├── client_cpp/               # Implementación original del cliente DSO (C++)
    ├── client_cpp_mateo/         # Implementación DeepDSO (componente Mateo)
    ├── server_python/            # Servidor de profundidad original (Python)
    ├── server_python_mateo/      # Servidor monodepth2 (componente Mateo)
    └── third_party/              # Dependencias externas
        ├── eigen3-tf/            # Biblioteca Eigen (versión TensorFlow)
        ├── Pangolin/             # Visualizador 3D
        └── tensorflow/           # TensorFlow C++ bindings
```

---

## Componentes

### 2_benchmarks — Modelos de Estimación de Profundidad

#### DCDepth
Implementación del modelo DCDepth para estimación monocular de profundidad, evaluado sobre el split Eigen del dataset KITTI.

- **Dataset:** KITTI
- **Checkpoint:** `2_benchmarks/dcdepth/implementation/checkpoints/dcdepth_eigen.pth`

#### PixelFormer
Implementación de PixelFormer, un transformer para estimación de profundidad que incorpora mecanismos de atención a nivel de píxel.

- **Dataset:** NYU Depth V2
- **Checkpoint:** `2_benchmarks/pixelformer/implementation/pretrained/checkpoints/nyu.pth`
- **Inferencia:** `infer_flask_pixelformer.py` expone un servidor Flask para inferencia en tiempo real.

---

### 3_deepdso_slam — Sistema DeepDSO

Sistema de SLAM visual directo (basado en DSO) integrado con estimación de profundidad por CNN mediante una arquitectura cliente-servidor.

#### Arquitectura

```
┌─────────────────────┐         ┌──────────────────────────┐
│   Cliente C++ (DSO) │ ──────▶ │ Servidor Python (CNN)    │
│  client_cpp /       │  HTTP   │  server_python /         │
│  client_cpp_mateo   │ ◀────── │  server_python_mateo     │
└─────────────────────┘         └──────────────────────────┘
```

El cliente DSO envía frames al servidor Python, que responde con mapas de profundidad estimados. Estos se integran al sistema de odometría visual para mejorar la inicialización y el tracking.

#### client_cpp (original)
Código C++ del sistema DSO original, base del proyecto.

#### client_cpp_mateo (DeepDSO)
Versión extendida de DSO con integración de profundidad CNN. Incluye:
- Módulo de comunicación con el servidor de profundidad
- Integración de prior de profundidad en la optimización
- Visor Pangolin para visualización 3D en tiempo real

**Compilación:**
```bash
cd 3_deepdso_slam/client_cpp_mateo
mkdir build && cd build
cmake ..
make -j4
```

**Ejecución:**
```bash
./build/bin/dso_dataset \
  files=<path_to_images> \
  calib=<path_to_calib> \
  preset=0 mode=1
```

#### server_python / server_python_mateo (monodepth2)
Servidor Flask que carga un modelo de estimación de profundidad y responde a requests del cliente C++.

**Instalación:**
```bash
pip install -r requirements.txt
```

**Ejecución:**
```bash
python server.py --model_name mono+stereo_640x192
```

---

## Dependencias

| Dependencia | Versión | Uso |
|---|---|---|
| OpenCV | ≥ 3.4 | Procesamiento de imágenes |
| Eigen3 | incluida en `third_party/` | Álgebra lineal (C++) |
| Pangolin | incluida en `third_party/` | Visualización 3D |
| PyTorch | ≥ 1.7 | Inferencia de modelos CNN |
| Flask | ≥ 1.1 | Servidor HTTP Python |

---

## Ramas

| Rama | Descripción |
|---|---|
| `main` | Código base del proyecto |
| `componente-tic-Mateo` | Integración de DeepDSO, monodepth2 y dependencias (componente Mateo) |

---

## Notas

- Los archivos de pesos (`.pth`) **no están incluidos** en el repositorio por su tamaño. Deben descargarse por separado y ubicarse en las rutas indicadas arriba.
- Los artefactos de compilación (`build/`, `*.o`, `*.a`) están excluidos por `.gitignore`.

---

## Autores

Proyecto TIC 2025 — Escuela Politécnica Nacional