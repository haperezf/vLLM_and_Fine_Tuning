import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

# ---------------------------------------------------------------
# 1) RUTA LOCAL CORRECTA A TU MISTRAL 7B
# ---------------------------------------------------------------
# Asegúrate de reemplazar esta ruta con la ubicación exacta en tu disco.
# Ejemplo: r"C:\LLM\FTLLM\models\mistral-7b"
model_path = "./models/mistral"

# ---------------------------------------------------------------
# 1.1) RUTA ABSOLUTA DONDE GUARDARÁS EL MODELO FINETUNED
# ---------------------------------------------------------------
# Debe existir o se creará automáticamente
output_dir = "./models/mistral-7b-finetuned"

# Asegurémonos de que la carpeta exista
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------
# 2) CONFIGURACIÓN DE CUANTIZACIÓN 4-bit usando BitsAndBytesConfig
# ---------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Si no quieres 4-bit, pon False
    bnb_4bit_compute_dtype="float16",      # Tipo de dato interno (float16 en RTX 3080)
    bnb_4bit_use_double_quant=True,        # Habilita doble cuantización
    bnb_4bit_quant_type="nf4"              # Tipo de cuantización (nf4 es habitual para 4-bit)
)

# ---------------------------------------------------------------
# 3) CARGA TOKENIZER y MODELO con quantization_config
# ---------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# ---------------------------------------------------------------
# 3.1) ASEGURARNOS DE QUE HAYA PAD_TOKEN
# ---------------------------------------------------------------
# El tokenizer de Mistral/LLaMA no trae pad_token por defecto. 
# Asignamos eos_token como pad_token para poder hacer padding="max_length".
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,   # usamos BitsAndBytesConfig en lugar de load_in_4bit
    device_map="auto",
    trust_remote_code=True
)

# ---------------------------------------------------------------
# 4) PREPARAR EL MODELO PARA ENTRENAR EN 4-bit + LoRA
# ---------------------------------------------------------------
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ---------------------------------------------------------------
# 5) CARGA LOS CSVs PARA FINE-TUNING
# ---------------------------------------------------------------
# Asegúrate de tener qa_dataset.csv y glosario_dataset.csv en el mismo directorio
qa_df = pd.read_csv("./dataset/qa_dataset.csv", encoding="utf-8")
glosario_df = pd.read_csv("./dataset/glosario_dataset.csv", encoding="utf-8")

combined_df = pd.concat([qa_df, glosario_df], ignore_index=True)
dataset = Dataset.from_pandas(combined_df)

# ---------------------------------------------------------------
# 6) TOKENIZACIÓN
# ---------------------------------------------------------------
def tokenize_fn(example):
    # Concatena prompt + completion y tokeniza a max_length=512
    return tokenizer(
        example["prompt"] + " " + example["completion"],
        truncation=True,
        padding="max_length",    # Ahora funciona porque definimos pad_token
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=False)

# ---------------------------------------------------------------
# 7) CONFIGURA EL DATA COLLATOR
# ---------------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---------------------------------------------------------------
# 8) ARGUMENTOS DE ENTRENAMIENTO (optimizado para RTX 3080 16GB)
# ---------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,               # RUTA ABSOLUTA para guardar checkpoints
    per_device_train_batch_size=2,       # 2 es razonable con 4-bit en 16 GB
    gradient_accumulation_steps=4,       # simula un batch de 8
    num_train_epochs=3,
    learning_rate=2e-4,
    save_strategy="epoch",
    save_total_limit=1,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=10,
    fp16=True,                           # usa float16
    report_to="none"
)

# ---------------------------------------------------------------
# 9) INICIALIZA EL TRAINER
# ---------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# ---------------------------------------------------------------
# 10) EJECUTA ENTRENAMIENTO
# ---------------------------------------------------------------
trainer.train()

# ---------------------------------------------------------------
# 11) GUARDA EL MODELO Y EL TOKENIZER FINE-TUNEADOS
# ---------------------------------------------------------------
# Usamos la misma ruta absoluta
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Modelo guardado en: {output_dir}")
