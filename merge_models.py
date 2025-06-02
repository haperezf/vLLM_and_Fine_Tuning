import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM

# ---------------------------------------------------------------
# 1. RUTAS QUE DEBES AJUSTAR A TU ENTORNO
# ---------------------------------------------------------------

# Ruta al modelo base original de Mistral 7B (sin LoRA)
BASE_MODEL_PATH = r"C:\LLM\FTLLM\models\mistral"

# Carpeta donde tu Trainer guardó los adaptadores LoRA
# (debe contener adapter_config.json + pytorch_model*.bin de LoRA)
LORA_ADAPTER_PATH = r"C:\LLM\FTLLM\mistral-7b-finetuned"

# Carpeta donde queremos guardar el modelo base + LoRA fusionados
MERGED_OUTPUT_DIR = r"C:\LLM\FTLLM\mistral-7b-merged"

# Carpeta (vacía) para usar como offload_dir cuando Accelerate pone partes en disco
OFFLOAD_DIR = r"C:\LLM\FTLLM\offload_temp"

# ---------------------------------------------------------------
# 2. CREAR CARPETAS NECESARIAS
# ---------------------------------------------------------------
os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 3. CARGAR TOKENIZER (opcional pero recomendado)
# ---------------------------------------------------------------
print(f"[1/5] Cargando tokenizer desde {BASE_MODEL_PATH} …")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)

# ---------------------------------------------------------------
# 4. CARGAR EL MODELO BASE
# ---------------------------------------------------------------
print(f"[2/5] Cargando modelo base desde {BASE_MODEL_PATH} …")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",           # Auto asigna capas a GPU/CPU/disk
    trust_remote_code=True,
    torch_dtype=torch.float16    # Opcional: reduce uso de VRAM
)

# ---------------------------------------------------------------
# 5. APLICAR EL ADAPTADOR LoRA COMO PeftModelForCausalLM
# ---------------------------------------------------------------
print(f"[3/5] Cargando adaptador LoRA desde {LORA_ADAPTER_PATH} …")
model = PeftModelForCausalLM.from_pretrained(
    model, 
    LORA_ADAPTER_PATH,
    device_map="auto",     # “auto” para GPU/CPU/Disk según convenga
    offload_dir=OFFLOAD_DIR
)

# ---------------------------------------------------------------
# 6. FUSIONAR (merge) Y DESCARGAR EL ADAPTADOR LoRA EN EL MODELO BASE
# ---------------------------------------------------------------
print("[4/5] Fusionando adaptador LoRA en el modelo base … (puede tardar unos minutos) …")
model = model.merge_and_unload()

# ---------------------------------------------------------------
# 7. GUARDAR EL MODELO FUSIONADO Y EL TOKENIZER
# ---------------------------------------------------------------
print(f"[5/5] Guardando modelo final en {MERGED_OUTPUT_DIR} …")
model.save_pretrained(MERGED_OUTPUT_DIR)
tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

print("✅ ¡Modelo fusionado guardado correctamente!")
print("   Revisa que en esa carpeta aparezcan:")
print("   - config.json")
print("   - pytorch_model.bin         (pesos base + LoRA)")
print("   - tokenizer.json, tokenizer_config.json, etc.")
