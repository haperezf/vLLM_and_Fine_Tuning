import os
import subprocess

# Paths – Update these to your actual paths
MODEL_PATH = r"C:\LLM\vllm-windows\models\mistral-7b-merged"  # e.g., "./Llama-2-7B"
OUTFILE_NAME = "llama-converted.gguf"

# Optional: Specify vocab type – 'sentencepiece' or 'tokenizer.json'
VOCAB_TYPE = "sentencepiece"

# Optional: Add any flags you need
extra_args = []

# Build the command
command = [
    "python",
    "scripts/convert-hf-to-gguf.py",  # Path relative to llama.cpp root
    MODEL_PATH,
    "--outfile", OUTFILE_NAME,
    "--vocab-type", VOCAB_TYPE,
] + extra_args

# Run the command
print("Running conversion command:\n", " ".join(command))
subprocess.run(command, check=True)

print(f"\n✅ Model converted successfully: {OUTFILE_NAME}")
