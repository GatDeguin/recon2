Readme
---

## Instalaci\xC3\xB3n y Entorno

Para trabajar con el proyecto se recomienda crear un entorno virtual de Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Instalaci\xC3\xB3n del paquete
pip install .
```

Si se desean funcionalidades opcionales como c\xC3\xA1lculo de flujo \xC3\xB3ptico con RAFT o extracci\xC3\xB3n de OpenFace, instale con:

```bash
pip install .[raft,openface]
```

### Dependencias externas RAFT y OpenFace

Para disponer de flujo \xC3\xB3ptico y atributos faciales es necesario instalar
los proyectos [RAFT](https://github.com/princeton-vl/RAFT) y
[OpenFace](https://github.com/TadasBaltrusaitis/OpenFace).
En este repositorio se incluyen scripts que automatizan la instalaci\xC3\xB3n:

```bash
# Instalar RAFT (optical flow)
bash scripts/install_raft.sh

# Compilar OpenFace
bash scripts/install_openface.sh
```

Instalaci\xC3\xB3n manual de RAFT (probado con commit `3fa0bb0`):

```bash
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
pip install -r requirements.txt
bash download_models.sh
export RAFT_DIR=$(pwd)
export RAFT_CHECKPOINT=$(pwd)/models/raft-sintel.pth
```

Instalaci\xC3\xB3n manual de OpenFace (probado con tag `OpenFace_2.2.0`):

```bash
git clone https://github.com/TadasBaltrusaitis/OpenFace.git --branch OpenFace_2.2.0 --depth 1
cd OpenFace
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
export OPENFACE_BIN=$(pwd)/bin/FeatureExtraction
```

### Prerrequisitos de GPU

Se necesita una GPU NVIDIA con CUDA 11.8 para entrenar y acelerar la inferencia.
Instale PyTorch con la versi\xC3\xB3n de CUDA apropiada desde [pytorch.org](https://pytorch.org/).

### Variables de entorno relevantes

- `YOLOX_ONNX`: ruta al modelo `yolox_s.onnx`.
- `RAFT_DIR`: directorio local con el c\xC3\xB3digo de RAFT (alias `RAFT_REPO`).
- `RAFT_CHECKPOINT`: archivo `raft-sintel.pth` descargado por `download_models.sh` (alias `RAFT_CKPT`).
- `OPENFACE_BIN`: binario `FeatureExtraction` de OpenFace.
- `BACKEND`: `cpu`, `cuda` u `onnx` para seleccionar el modo de inferencia.
- `ONNX_MODEL`: ruta del modelo ONNX para el reconocedor.
- `LM_CKPT`: checkpoint del modelo de lenguaje opcional.
- `BEAM_SIZE` y `LM_WEIGHT`: par\xC3\xA1metros del decodificador beam-search.

### Uso de la API

Para exponer el servicio de transcripción ejecute:

```bash
uvicorn server.app:app
```

Endpoints disponibles:

- `POST /transcribe`: recibe archivos de video en el campo `files` y devuelve glosas transcritas.
- `WebSocket /ws`: envíe fragmentos de video y reciba transcripciones en tiempo real.

La especificación de OpenAPI se encuentra en `server/openapi.yaml`.

### Cliente web

Se incluye un cliente mínimo en `frontend/index.html` que captura la cámara o
permite cargar un archivo de video y envía fragmentos al backend mediante
WebSocket, mostrando la transcripción recibida en tiempo real.

1. Inicie el backend:

   ```bash
   uvicorn server.app:app
   ```

2. Sirva la carpeta `frontend` con un servidor estático, por ejemplo:

   ```bash
   cd frontend
   python -m http.server 3000
   ```

3. Abra `http://localhost:3000` en el navegador. Al seleccionar un archivo de
   video o permitir el acceso a la cámara, el cliente enviará datos al backend
   (`ws://localhost:8000/ws`) y mostrará las glosas transcritas.

### Servidor gRPC

El servicio también expone una interfaz gRPC definida en `server/protos/transcriber.proto`.
Para regenerar los módulos de Python a partir del `.proto` utilice:

```bash
python -m grpc_tools.protoc -I server/protos \
    --python_out=server/protos --grpc_python_out=server/protos \
    server/protos/transcriber.proto
```

Inicie el servidor gRPC con:

```bash
python run_grpc_server.py --port 50051
```

Ejemplos de clientes en Python:

```bash
python scripts/grpc_client.py demo.mp4
python scripts/grpc_client_async.py demo.mp4
```


## 1. Adquisición y Anotación de Datos

### 1.1 Corpora Paralelos

Se parte de **LSA‑T**, un corpus continuo de LSA con 14 880 vídeos de oraciones a 30 FPS, extraídos del canal CN Sordos y anotados con glosas y keypoints .
Se añade **LSA64**, compuesto por 3 200 clips de señas aisladas para pre‑entrenamiento controlado .
Para robustez cross‑lingual, se incorporan **PHOENIX‑Weather‑2014T** y **CoL‑SLTD** en fases iniciales de pre‑entrenamiento .

Para descargar y normalizar automáticamente estos corpora se incluye el script
`data/download.py`:

```bash
python data/download.py <corpus> DEST --username USER --password PASS
```

`<corpus>` puede ser `lsa_t`, `lsa64`, `phoenix` o `col-sltd`. El script
verifica la suma SHA‑256, muestra una barra de progreso durante la descarga,
extrae los archivos y coloca los vídeos en `DEST/videos` y las anotaciones en
`DEST/meta.csv`.

### 1.2 Esquema de Anotación Multitarea

Cada clip se enriquece con:

* **Glosas** sincronizadas (`start`, `end`) para entrenamiento con CTC .
* **Marcadores no‑manuales** (expresiones faciales y movimientos de cabeza) etiquetados como tareas auxiliares .
* **Referencias deícticas** y silencios pre/post (`prev_delta`, `post_delta`) parseados con `ast.literal_eval` para modelar transiciones discursivas .
* Los metadatos se almacenan en `meta.csv`, un archivo CSV separado por punto y coma (`;`).

---

## 2. Preprocesamiento y Extracción Multimodal

### 2.1 Chunking de Vídeo

Los vídeos se fragmentan en **chunks de 16–32 frames** conforme a la estrategia de STTN, reduciendo la carga de memoria y permitiendo batch‑training eficiente .

### 2.2 Detección de Sujetos (ROI)

Se aplica **YOLOX** exportado a ONNX para extraer bounding boxes de la persona firmante a ≥ 25 FPS, con padding dinámico para incluir manos, torso y rostro .
Para obtener el modelo ONNX puede utilizar el repositorio oficial de YOLOX:

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX
cd YOLOX
python tools/export_onnx.py -f exps/example/yolox_s.py -c yolox_s.pth \
    --output-name yolox_s.onnx --decode-in-inference --dynamic
```

Coloque `yolox_s.onnx` en `checkpoints/` o indique su ruta con la variable de entorno `YOLOX_ONNX`.

### 2.3 Extracción de Landmarks y Flujo Óptico

* **MediaPipe Holistic** extrae **543 landmarks** (33 pose, 2×21 manos, 468 rostro) normalizados al torso para homogeneizar escala y perspectiva .
* **RAFT** calcula flujo óptico denso focalizado en manos, capturando movimientos sutiles que complementan los landmarks .
  Para ejecutar sin conexión debe descargarse el peso `raft-sintel.pth` desde
  [RAFT](https://github.com/princeton-vl/RAFT/releases) y colocarse en
  `checkpoints/` o indicar su ruta mediante la variable de entorno `RAFT_CKPT`.

### 2.4 Construcción de Grafos Esqueléticos

Los landmarks se modelan como nodos de un grafo esquelético procesado por un **Spatial‑Temporal GCN**, reforzando la representación de la conectividad articular .

---

## 3. Arquitectura del Modelo

### 3.1 Pre‑Entrenamiento Self‑Supervised (SHuBERT)

Se implementa **SHuBERT**, que enmascara y predice flujos de landmarks de manos, pose y rostro para aprender representaciones robustas en \~1 000 h de vídeo .

### 3.2 Spatio‑Temporal Transformer Network (STTN)

El **STTN** alterna bloques de atención espacial y temporal sobre cada chunk, extrayendo características globales y locales con gran eficiencia en secuencias prolongadas .

### 3.3 Multi‑Scale Channel Transformer (MCST‑Transformer)

Se integran kernels de tamaño 3, 5 y 7 frames en bloques de atención multi‑escala, adaptando el campo receptivo temporal a la velocidad variable de las glosas .

### 3.4 CorrNet+ para Correlación Temporal

**CorrNet+** genera mapas de correlación entre frames adyacentes y aplica atención selectiva a trayectorias críticas de manos y rostro, reduciendo el ruido de jitter y la complejidad computacional .

### 3.5 Cabezas Multitarea

El encoder alimenta tres cabezas:

1. **CTC‑head** para reconocimiento continuo de glosas.
2. **No‑manual head** para clasificación de expresiones faciales y de cabeza.
3. **Classifier head** para sufijos numerales y clasificadores morfológicos simultáneos .

---

## 4. Estrategia de Entrenamiento

### 4.1 Funciones de Pérdida

* **CTC Loss** alinea vídeo y glosas sin segmentación manual .
* **Contrastive Loss** separa representaciones de clases cercanas en el espacio latente .
* **Cross‑Entropy** auxiliar para marcadores no‑manuales y clasificadores morfológicos .

### 4.2 Curriculum Learning y Augmentación

Se inicia el entrenamiento con clips cortos (< 10 frames) y gradualmente se incorporan secuencias largas (> 32 frames) para estabilizar la convergencia .
Se emplea MixUp temporal, speed perturbation y fondos sintéticos generados por GANs para robustez ante variaciones adversariales .

Cada técnica puede activarse con el parámetro ``augment`` de ``SignDataset``:

```python
from augmentations import speed_perturbation
ds = SignDataset("data.h5", "labels.csv", augment=lambda x: speed_perturbation(x, 0.9))
```

### 4.3 Adaptación Adversarial de Dominio

Se integra un **Gradient Reversal Layer** (DANN) para alinear las características entre LSA‑T y otros corpus (ASL, GSL), reduciendo el gap de dominio y mejorando la generalización.

El entrenamiento adversarial se activa con `--adversarial` y requiere un CSV que indique el dominio de cada muestra. El archivo debe contener dos columnas separadas por `;`: `id` (identificador del video) y `domain` (entero empezando en 0). Un ejemplo se provee en `data/domain_labels_example.csv`.

```csv
id;domain
0001;0
0002;1
0003;0
```

Para entrenar con DANN:

```bash
python train.py --h5_file datos.h5 --csv_file labels.csv \
  --domain_csv data/domain_labels_example.csv --adversarial \
  --adv_mix_rate 1.0 --adv_weight 0.1 --adv_steps 1
```

Los parámetros `--adv_mix_rate` controla la lambda del gradiente invertido, `--adv_weight` pondera la pérdida adversarial y `--adv_steps` define cuántas iteraciones de actualización se realizan sobre el discriminador por batch.

---

## 5. Decodificación e Inferencia

### 5.1 Beam‑Search con Transformer‑LM

Durante la inferencia, un **beam‑search** acompañado de un **Transformer‑based Language Model** entrenado en glosas LSA‑T refuerza la sintaxis SOV y las estructuras topic–comment .

### 5.2 Optimización y Quantización

El modelo se exporta a **ONNX Runtime** y se cuantiza dinámicamente a INT8, alcanzando > 25 FPS en CPU/GPU con una pérdida de WER inferior al 1 % .
Para ello se provee el script `export_onnx.py`:

```bash
python export_onnx.py --checkpoint ckpt.pt --arch stgcn --output model.onnx
```

Genera `model.onnx_int8.onnx` ya cuantizado y verifica una inferencia con `onnxruntime`.
Se genera además una versión “lite” mediante **Knowledge Distillation**, reduciendo parámetros en un 50 % con alta precisión para dispositivos móviles .
Para entrenarla ejecute:
```bash
python distill.py --h5_file datos.h5 --csv_file labels.csv \
  --teacher_ckpt checkpoints/epoch_10.pt --model stgcn
```
El script guarda el checkpoint reducido en `checkpoints/` y reporta WER y
precisión de NMM comparando alumno y profesor.

### 5.3 Configuración por defecto

Los scripts de entrenamiento, exportación y distilación pueden leer valores
por omisión desde `configs/config.yaml`. Si una opción de la línea de comandos
no se proporciona, se utilizará el valor definido en este archivo. Allí se
especifican las rutas de los datasets, la carpeta de `checkpoints`, la
arquitectura del modelo y los hiperparámetros más comunes.

---

## 6. Despliegue y Monitoreo

### 6.1 Backend y Servido

El backend se implementa con **FastAPI** y gRPC en contenedores Docker orquestados por Kubernetes, con auto‑escalado y batching de peticiones para alta disponibilidad y latencia < 100 ms .

### 6.2 Frontend en Tiempo Real

Un cliente React/WebSocket captura la webcam o stream, envía chunks al servidor y superpone landmarks y transcripción en tiempo real para feedback inmediato al usuario .

### 6.3 Mantenimiento y Aprendizaje Continuo

Se monitorizan métricas clave (WER, precisión de RNM, latencia, FPS) y se orquesta un pipeline de **active learning** semanal con ejemplos de baja confianza capturados en producción, permitiendo reentrenamientos periódicos .
Las métricas se almacenan automáticamente en `logs/metrics.db` para su visualización posterior.

### 6.4 Reanotación y regeneración de features

Para seleccionar ejemplos con menor confianza y exportarlos para su revisión
manual utilice el script `scripts/run_active_learning.py`:

```bash
python scripts/run_active_learning.py \
  --h5_file data/data.h5 \
  --csv_file data/labels.csv \
  --video_dir videos \
  --out_dir reannotation \
  --top_k 20
```

El script toma el checkpoint más reciente de `checkpoints/` y genera los
archivos `reannotation/selected.csv`, `reannotation/selected.h5` y una copia de
los vídeos en `reannotation/videos/` (o los sube al bucket indicado).

1. Corrija manualmente las glosas en `reannotation/selected.csv`.
2. Una las nuevas filas al corpus original:

   ```bash
   cat reannotation/selected.csv >> data/labels.csv
   ```
3. Extraiga de nuevo las características HDF5 para los vídeos revisados y
   fusione los datasets:

   ```bash
   python track_lsa_4.py --input_dir reannotation/videos \
     --output_file reannotation/new_feats.h5 --no_window

   python - <<'PY'
   import h5py
   with h5py.File('data/data.h5','a') as dst, h5py.File('reannotation/new_feats.h5') as src:
       for k in src.keys():
           src.copy(k, dst)
   PY
   ```

Tras este proceso, las muestras reanotadas quedan integradas y se pueden
regenerar los features para futuros entrenamientos.

---

## 7. Archivos Requeridos para el Servidor

El servicio `server/app.py` busca un modelo TorchScript en `checkpoints/model.ts` y un archivo de vocabulario `vocab.txt`.
Para realizar pruebas rápidas sin modelos entrenados, cree ficheros de relleno:

```
mkdir -p checkpoints
:>checkpoints/model.ts
printf "<blank>\n<sos>\n<eos>\n" > vocab.txt
```

Si dispone de los pesos reales y un vocabulario entrenado, reemplace estos archivos por los originales.

Si desea usar detección con YOLOX desde `server/app.py`, coloque el modelo
`yolox_s.onnx` en `checkpoints/` o establezca la variable de entorno
`YOLOX_ONNX` apuntando al archivo.

Para seleccionar el backend de inferencia del reconocedor se puede definir la
variable `BACKEND` con los valores `cpu`, `cuda` u `onnx`. Con `cpu` o `cuda`
se cargará `checkpoints/model.ts` con PyTorch en el dispositivo indicado. Si se
elige `onnx` se cargará `checkpoints/model.onnx` (o la ruta indicada en
`ONNX_MODEL`) usando `onnxruntime`.

El endpoint `/transcribe` ahora acepta varios archivos en la clave `files` y
devuelve una lista de transcripciones en el mismo orden.

## 8. Modelos preentrenados y pesos de referencia

Se publican checkpoints preentrenados de las arquitecturas principales en la
sección de *releases* del repositorio. Descargue los pesos y colóquelos en la
carpeta `checkpoints/` o ajuste las rutas en `configs/config.yaml`:

| Arquitectura | Archivo de ejemplo |
|--------------|--------------------|
| ST-GCN       | `https://github.com/OWNER/REPO/releases/download/models/stgcn.pt` |
| STTN         | `https://github.com/OWNER/REPO/releases/download/models/sttn.pt` |
| CorrNet+     | `https://github.com/OWNER/REPO/releases/download/models/corrnet_plus.pt` |
| MCST         | `https://github.com/OWNER/REPO/releases/download/models/mcst.pt` |

Ejemplo de descarga y uso del modelo ST-GCN:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/stgcn.pt \
  https://github.com/OWNER/REPO/releases/download/models/stgcn.pt

python infer.py --checkpoint checkpoints/stgcn.pt \
  --lm checkpoints/lm.pt --vocab vocab.txt \
  --video demo.mp4 --h5 data/features.h5
```

Sustituya `stgcn.pt` por `sttn.pt`, `corrnet_plus.pt` o `mcst.pt` para evaluar
las otras arquitecturas.

## 9. Benchmarks con Datos Públicos

Se incluye un subconjunto de **LibriSpeech** en `tests/data/librispeech_subset.csv` para validar la precisión real mediante WER y CER.
El script `scripts/wer_benchmark.py` calcula estas métricas para varias arquitecturas y guarda los resultados en un JSON opcional.

Ejemplo de uso:

```bash
python scripts/wer_benchmark.py tests/data/librispeech_subset.csv --models model_a model_b --output docs/benchmark_results.json
```

Los resultados de referencia se documentan en `docs/benchmark_results.md`.
