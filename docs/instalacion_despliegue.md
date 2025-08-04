# Guía de instalación y despliegue

## 1. Requisitos de sistema
- **Python** ≥3.8 (probado con 3.10).
- **GPU NVIDIA** con CUDA 11.8 y **cuDNN** correspondiente.
- Drivers y herramientas de compilación como `git`, `cmake` y `make`.

## 2. Preparar el entorno de Python
1. Crear y activar un entorno virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```
2. Instalar el paquete y dependencias definidas en `pyproject.toml`:
   ```bash
   pip install .
   ```
3. Para incorporar RAFT y OpenFace desde los extras del proyecto:
   ```bash
   pip install .[raft,openface]
   ```

## 3. Configurar CUDA y cuDNN
1. Instalar los controladores NVIDIA y el **CUDA Toolkit 11.8** desde el sitio oficial.
2. Descargar e instalar la versión de **cuDNN** compatible con CUDA 11.8.
3. Verificar la instalación:
   ```bash
   nvcc --version   # Debe mostrar 11.8
   ```
4. Instalar PyTorch con soporte CUDA adecuado siguiendo las instrucciones de [pytorch.org](https://pytorch.org/).

## 4. Descarga y preparación de datasets
1. Reunir los corpora necesarios (p.ej. LSA64, PHOENIX‑Weather‑2014T, CoL‑SLTD).
2. Ubicar los vídeos en `data/videos/` y generar un `meta.csv` con las glosas.
3. Extraer landmarks y flujo óptico para todos los vídeos:
   ```bash
   python track_lsa_4.py --input_dir data/videos --output_file data/data.h5 --no_window
   ```

## 5. Instalación de RAFT y OpenFace
1. Ejecutar los scripts incluidos:
   ```bash
   bash scripts/install_raft.sh
   bash scripts/install_openface.sh
   ```
2. Exportar las variables de entorno necesarias:
   ```bash
   export RAFT_DIR=/ruta/a/RAFT
   export RAFT_CHECKPOINT=$RAFT_DIR/models/raft-sintel.pth
   export OPENFACE_BIN=/ruta/a/OpenFace/build/bin/FeatureExtraction
   ```

## 6. Entrenamiento del modelo
1. Entrenar el reconocedor con CTC:
   ```bash
   python train.py --h5_file data/data.h5 --csv_file meta.csv --model stgcn --epochs 50
   ```
2. (Opcional) Distilación para un modelo liviano:
   ```bash
   python distill.py --h5_file data/data.h5 --csv_file meta.csv --teacher_ckpt checkpoints/epoch_10.pt --model stgcn
   ```
3. (Opcional) Entrenar un modelo de lenguaje para el beam search:
   ```bash
   python train_lm.py --csv meta.csv --out checkpoints/lm.pt
   ```

## 7. Despliegue en producción
### 7.1 Docker
1. Construir la imagen:
   ```bash
   docker build -t recon-server .
   ```
2. Ejecutar el contenedor:
   ```bash
   docker run -p 8000:8000 -p 50051:50051 recon-server
   ```

### 7.2 Kubernetes
1. Publicar la imagen en un registro accesible por el clúster.
2. Aplicar los manifiestos:
   ```bash
   kubectl apply -f k8s/
   ```
3. (Opcional) Habilitar escalado horizontal con `k8s/hpa.yaml`.

### 7.3 Despliegue en la nube
- Crear una instancia o servicio con GPU (AWS EC2, GCP Compute, etc.).
- Instalar Docker o usar un servicio gestionado (Cloud Run, ECS, etc.).
- Ejecutar la imagen `recon-server` y exponer los puertos 8000 (HTTP) y 50051 (gRPC).

## 8. Conectar el cliente web
1. Iniciar el backend:
   ```bash
   uvicorn server.app:app
   ```
2. Servir la carpeta `frontend`:
   ```bash
   cd frontend
   python -m http.server 3000
   ```
3. Abrir `http://localhost:3000` y permitir el acceso a la cámara o cargar un vídeo. El cliente enviará fragmentos al backend (`ws://localhost:8000/ws`) y mostrará las glosas transcritas en tiempo real.

---
Esta guía resume los pasos para preparar el entorno, entrenar modelos y desplegar el servidor de transcripción en producción.
