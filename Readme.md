Readme
---

## 1. Adquisición y Anotación de Datos

### 1.1 Corpora Paralelos

Se parte de **LSA‑T**, un corpus continuo de LSA con 14 880 vídeos de oraciones a 30 FPS, extraídos del canal CN Sordos y anotados con glosas y keypoints .
Se añade **LSA64**, compuesto por 3 200 clips de señas aisladas para pre‑entrenamiento controlado .
Para robustez cross‑lingual, se incorporan **PHOENIX‑Weather‑2014T** y **CoL‑SLTD** en fases iniciales de pre‑entrenamiento .

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

### 4.3 Adaptación Adversarial de Dominio

Se integra un **Gradient Reversal Layer** (DANN) para alinear las características entre LSA‑T y otros corpus (ASL, GSL), reduciendo el gap de dominio y mejorando la generalización .

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

---

## 6. Despliegue y Monitoreo

### 6.1 Backend y Servido

El backend se implementa con **FastAPI** y gRPC en contenedores Docker orquestados por Kubernetes, con auto‑escalado y batching de peticiones para alta disponibilidad y latencia < 100 ms .

### 6.2 Frontend en Tiempo Real

Un cliente React/WebSocket captura la webcam o stream, envía chunks al servidor y superpone landmarks y transcripción en tiempo real para feedback inmediato al usuario .

### 6.3 Mantenimiento y Aprendizaje Continuo

Se monitorizan métricas clave (WER, precisión de RNM, latencia, FPS) y se orquesta un pipeline de **active learning** semanal con ejemplos de baja confianza capturados en producción, permitiendo reentrenamientos periódicos .
Las métricas se almacenan automáticamente en `logs/metrics.db` para su visualización posterior.

---

## 7. Archivos Requeridos para el Servidor

El servicio `server/app.py` busca un modelo TorchScript en `checkpoints/model.ts` y un archivo de vocabulario `vocab.txt`.
Para realizar pruebas rápidas sin modelos entrenados, cree ficheros de relleno:

```
mkdir -p checkpoints
:>checkpoints/model.ts
printf "<unk>\n<pad>\n" > vocab.txt
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
