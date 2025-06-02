# 📡 vLLM + Fine-Tuning de Mistral-7B para Telecomunicaciones

Este repositorio contiene el flujo completo para descargar, afinar, fusionar y desplegar **Mistral-7B-Instruct-v0.3** con conocimientos específicos del dominio de telecomunicaciones.

## ✨ Funcionalidades

- **Descarga** automática del modelo base desde Hugging Face.
- **Fine-tuning** con LoRA (4-bit) empleando los conjuntos `qa_dataset.csv` y `glosario_dataset.csv`.
- **Fusión (merge)** de los adaptadores LoRA con el modelo original.
- **Despliegue** en `vLLM` para inferencia concurrente.
- **Conversión** a formato **GGUF** compatible con **LM Studio** o `llama.cpp`.

## 🖥️ Requisitos

| Recurso | Recomendación mínima |
|---------|----------------------|
| GPU     | NVIDIA RTX 3080 (≥16 GB VRAM) |
| CPU     | 8 núcleos |
| RAM     | 32 GB |
| SO      | Linux, WSL 2 o Windows 10+ |
| Software| Python 3.10+, Docker |

Instala las dependencias de Python:

```bash
pip install -r requirements.txt
```

## 📁 Estructura del proyecto

```text
.
├── dataset/
│   ├── qa_dataset.csv
│   └── glosario_dataset.csv
├── models/
│   ├── mistral/              # Modelo descargado de HF
│   ├── mistral-7b-finetuned/ # Modelo después del fine‑tuning
│   ├── mistral-7b-merged/    # Modelo con LoRA fusionado
│   └── mistral-7b-gguf/      # Modelo exportado a GGUF
├── scripts/
│   ├── download_model.py
│   ├── fine_tuning.py
│   ├── merge_models.py
│   ├── convert_hf_to_gguf.py
│   ├── gguf_writer.py
│   └── vllm_test.py
├── docker-compose.yml
└── README.md
```

## 🚀 Flujo de trabajo

### 1. Descargar el modelo base

```bash
python scripts/download_model.py
```

Descarga **`mistralai/Mistral-7B-Instruct-v0.3`** en `./models/mistral/`.

### 2. Fine‑tuning con datos de telecomunicaciones

```bash
python scripts/fine_tuning.py
```

Entrena el modelo con los archivos del directorio `dataset/`. El resultado se guarda en `./models/mistral-7b-finetuned/`.

### 3. Fusión del modelo base y LoRA

```bash
python scripts/merge_models.py
```

Genera `./models/mistral-7b-merged/` con los pesos consolidados.

### 4. Pruebas de carga concurrente

```bash
python scripts/vllm_test.py --threads 10
```

El script lanza 10 hilos simultáneos usando la API de `vLLM` y muestra *throughput* y *latency* antes y después del fine-tuning.

### 5. Despliegue con Docker + vLLM

Asegúrate de que la ruta configurada en `docker-compose.yml` apunte a `./models/mistral-7b-merged/`, luego:

```bash
docker compose up -d
```

El endpoint REST estará en <http://localhost:8000/v1>.

### 6. Conversión a GGUF

```bash
python scripts/convert_hf_to_gguf.py   --dir-model ./models/mistral-7b-merged   --outfile   ./models/mistral-7b-gguf/mistral-7b-telecom.gguf   --outtype   f16
```

El archivo `.gguf` resultante es compatible con **LM Studio** y `llama.cpp`.

## 🧪 Ejemplo de evaluación

```python
MESSAGES = [
    {
        "role": "system",
        "content": "Eres un ingeniero experto en telecomunicaciones..."
    },
    {
        "role": "user",
        "content": "Describe un problema típico en redes GPON o HFC."
    }
]
```

- **Sin fine‑tuning:** respuestas vagas o alucinaciones.  
- **Con fine‑tuning:** explicaciones técnicas, referencias correctas a GPON/HFC y mejores prácticas.

## 🧰 Descripción de scripts

| Script | Descripción |
|--------|-------------|
| `download_model.py` | Descarga el modelo base desde Hugging Face. |
| `fine_tuning.py` | Aplica LoRA + cuantización 4‑bit. |
| `merge_models.py` | Fusiona pesos LoRA y produce el modelo final. |
| `vllm_test.py` | Benchmark multihilo usando la API de `vLLM`. |
| `convert_hf_to_gguf.py` | Convierte el modelo a GGUF. |
| `gguf_writer.py` | Utilidad auxiliar para la conversión. |

## 📌 Notas

- El tokenizer de Mistral no trae `pad_token`; los scripts lo añaden automáticamente.
- Después de convertir a GGUF, carga el modelo en **LM Studio** sin pasos adicionales.
- Ajusta los hiperparámetros de `fine_tuning.py` (batch size, lr, epochs) según tu GPU.

## 🤝 Créditos

Proyecto basado en:

- [Hugging Face Transformers]  
- [vLLM Project]  
- [PEFT / LoRA]  
- [llama.cpp]

## 📜 Licencia

Este repositorio se proporciona únicamente con fines educativos y de experimentación. Verifica y respeta las licencias y términos de uso del modelo base y de cada dependencia.


---

# 📡 vLLM + Mistral‑7B Fine‑Tuning for Telecommunications

This repository provides an end‑to‑end workflow to download, fine‑tune, merge, and deploy **Mistral‑7B‑Instruct‑v0.3** with domain‑specific knowledge in telecommunications.

## ✨ Features

- **Automatic download** of the base model from Hugging Face.
- **Fine‑tuning** with LoRA (4‑bit) using `qa_dataset.csv` and `glosario_dataset.csv`.
- **Merge** of LoRA adapters with the original model.
- **Deployment** on `vLLM` for concurrent inference.
- **Conversion** to **GGUF** format for **LM Studio** or `llama.cpp`.

## 🖥️ Requirements

| Resource | Minimum recommendation |
|----------|------------------------|
| GPU      | NVIDIA RTX 3080 (≥16 GB VRAM) |
| CPU      | 8 cores |
| RAM      | 32 GB |
| OS       | Linux, WSL 2, or Windows 10+ |
| Software | Python 3.10+, Docker |

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## 📁 Project structure

```text
.
├── dataset/
│   ├── qa_dataset.csv
│   └── glosario_dataset.csv
├── models/
│   ├── mistral/              # Model downloaded from HF
│   ├── mistral-7b-finetuned/ # Model after fine‑tuning
│   ├── mistral-7b-merged/    # Model with LoRA merged
│   └── mistral-7b-gguf/      # Model exported to GGUF
├── scripts/
│   ├── download_model.py
│   ├── fine_tuning.py
│   ├── merge_models.py
│   ├── convert_hf_to_gguf.py
│   ├── gguf_writer.py
│   └── vllm_test.py
├── docker-compose.yml
└── README.md
```

## 🚀 Workflow

### 1. Download the base model

```bash
python scripts/download_model.py
```

Downloads **`mistralai/Mistral-7B-Instruct-v0.3`** into `./models/mistral/`.

### 2. Fine‑tune with telecom data

```bash
python scripts/fine_tuning.py
```

Trains the model using files in `dataset/`. The result is saved to `./models/mistral-7b-finetuned/`.

### 3. Merge base model and LoRA

```bash
python scripts/merge_models.py
```

Creates `./models/mistral-7b-merged/` with consolidated weights.

### 4. Concurrent load testing

```bash
python scripts/vllm_test.py --threads 10
```

The script launches 10 concurrent threads via the `vLLM` API, reporting *throughput* and *latency* before and after fine‑tuning.

### 5. Deploy with Docker + vLLM

Ensure the path in `docker-compose.yml` points to `./models/mistral-7b-merged/`, then run:

```bash
docker compose up -d
```

The REST endpoint will be available at <http://localhost:8000/v1>.

### 6. Convert to GGUF

```bash
python scripts/convert_hf_to_gguf.py   --dir-model ./models/mistral-7b-merged   --outfile   ./models/mistral-7b-gguf/mistral-7b-telecom.gguf   --outtype   f16
```

The resulting `.gguf` file is compatible with **LM Studio** and `llama.cpp`.

## 🧪 Evaluation example

```python
MESSAGES = [
    {
        "role": "system",
        "content": "You are a telecommunications engineering expert..."
    },
    {
        "role": "user",
        "content": "Describe a typical issue in GPON or HFC networks."
    }
]
```

- **Without fine‑tuning:** vague answers or hallucinations.  
- **With fine‑tuning:** technical explanations, correct references to GPON/HFC, and best practices.

## 🧰 Script overview

| Script | Description |
|--------|-------------|
| `download_model.py` | Downloads the base model from Hugging Face. |
| `fine_tuning.py` | Applies LoRA + 4‑bit quantization. |
| `merge_models.py` | Merges LoRA weights to produce the final model. |
| `vllm_test.py` | Multi‑thread benchmark using the `vLLM` API. |
| `convert_hf_to_gguf.py` | Converts the model to GGUF. |
| `gguf_writer.py` | Helper utility for conversion. |

## 📌 Notes

- Mistral’s tokenizer lacks a `pad_token`; the scripts add one automatically.
- After conversion to GGUF, load the model in **LM Studio** without extra steps.
- Adjust hyperparameters in `fine_tuning.py` (batch size, lr, epochs) according to your GPU.

## 🤝 Credits

Project built with:

- [Hugging Face Transformers]  
- [vLLM Project]  
- [PEFT / LoRA]  
- [llama.cpp]

## 📜 License

This repository is provided for educational and experimental purposes only. Check and respect the licenses and terms of use of the base model and each dependency.

