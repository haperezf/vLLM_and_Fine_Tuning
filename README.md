# 📡 vLLM + Fine-tuning de Mistral 7B para Telecomunicaciones

Este repositorio permite:

- Descargar y preparar el modelo **Mistral-7B-Instruct-v0.3** desde Hugging Face.
- Realizar **fine-tuning especializado en telecomunicaciones** utilizando LoRA con cuantización en 4 bits.
- Unir el modelo original con los pesos ajustados (merge).
- Desplegar el modelo en una instancia de `vLLM` para inferencia concurrente.
- Convertir el modelo final al formato **GGUF** para uso en **LM Studio**.

---

## 🖥️ Requisitos

- GPU NVIDIA RTX 3080 o superior (mínimo 16 GB VRAM)
- Python 3.10+
- Docker
- pip packages:
  ```bash
  pip install -r requirements.txt
````

---

## 📁 Estructura del Proyecto

```
.
├── dataset/
│   ├── qa_dataset.csv
│   └── glosario_dataset.csv
├── models/
│   ├── mistral/                 # Modelo descargado desde HF
│   ├── mistral-7b-finetuned/    # Modelo después del fine-tuning
│   ├── mistral-7b-merged/       # Modelo con LoRA fusionado
│   └── mistral-7b-gguf/         # Modelo exportado a GGUF
├── scripts/
│   ├── download_model.py
│   ├── fine_tuning.py
│   ├── merge_models.py
│   ├── convert_hf_to_gguf.py
│   ├── gguf_writer.py
│   └── vLLM_Test.py
├── docker-compose.yml
└── README.md
```

---

## 🚀 Flujo de Trabajo

### 1. 📥 Descargar el Modelo Base

```bash
python download_model.py
```

Esto descargará `mistralai/Mistral-7B-Instruct-v0.3` a `./models/mistral/`.

---

### 2. 🧠 Fine-tuning con Datos de Telecomunicaciones

```bash
python fine_tuning.py
```

Entrena el modelo con `qa_dataset.csv` y `glosario_dataset.csv`. El modelo ajustado se guarda en `./models/mistral-7b-finetuned/`.

---

### 3. 🔗 Fusión del Modelo Base + LoRA

```bash
python merge_models.py
```

Esto crea un modelo final en `./models/mistral-7b-merged/` con los pesos combinados.

---

### 4. 🧪 Pruebas de Sesiones Simultáneas

Lanza 10 hilos concurrentes con peticiones tipo Chat OpenAI para medir desempeño del modelo usando `vLLM`.

```bash
python vLLM_Test.py
```

📌 **Resultados esperados**:

* Antes del fine-tuning: El modelo “alucina” al hablar de telecomunicaciones.
* Después del fine-tuning: Da definiciones precisas y relevantes de redes como **HFC**, **GPON**, etc.

---

### 5. 🐳 Despliegue con Docker y vLLM

Asegúrate de que el path en `docker-compose.yml` apunte a `./models/mistral-7b-merged/`.

```bash
docker-compose up -d
```

El API estará disponible en: [http://localhost:8000/v1](http://localhost:8000/v1)

---

### 6. 🔄 Conversión a GGUF para LM Studio

```bash
python convert_hf_to_gguf.py \
  --dir-model ./models/mistral-7b-merged \
  --outfile ./models/mistral-7b-gguf/mistral-7b-telecom.gguf \
  --outtype f16
```

Esto genera un archivo `.gguf` compatible con LM Studio o llama.cpp.

---

## 🧪 Evaluación del Fine-tuning

El archivo `vLLM_Test.py` demuestra cómo evaluar la capacidad del modelo antes y después del ajuste.

```python
MESSAGES = [
    {"role": "system", "content": "Eres un ingeniero experto en telecomunicaciones..."},
    {"role": "user", "content": "Describe un problema típico en redes GPON o HFC."}
]
```

Después del entrenamiento, el modelo responde con precisión técnica en español, sin inventar información.

---

## 🧰 Scripts Incluidos

* `download_model.py`: descarga el modelo base desde Hugging Face.
* `fine_tuning.py`: realiza fine-tuning con LoRA + 4-bit quantization.
* `merge_models.py`: fusiona el modelo base con el adaptador LoRA.
* `vLLM_Test.py`: lanza pruebas multihilo vía API de vLLM.
* `convert_hf_to_gguf.py` y `gguf_writer.py`: convierten el modelo al formato GGUF.

---

## 📌 Notas

* El tokenizer de Mistral no incluye `pad_token` por defecto; se ajusta automáticamente en los scripts.
* El modelo final puede ser usado en **LM Studio** sin problemas luego de la conversión.

---

## 🤝 Créditos

Este proyecto combina herramientas de:

* [HuggingFace Transformers](https://huggingface.co)
* [vLLM](https://github.com/vllm-project/vllm)
* [PEFT / LoRA](https://github.com/huggingface/peft)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)

---

## 📜 Licencia

Este repositorio es solo para fines educativos y de experimentación. Respeta las licencias de cada modelo base utilizado.