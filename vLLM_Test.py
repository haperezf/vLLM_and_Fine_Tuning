import threading
import time
from typing import List, Tuple
from openai import OpenAI, APIError

# Configuration
N_THREADS = 10
VLLM_API_URL = "http://localhost:8000/v1"
API_KEY = "not-needed"

# Usa el nombre correcto del modelo como lo ve vLLM (por ejemplo, "mistral")
MODEL_NAME = "/models/mistral"  # Este nombre debe coincidir con el nombre cargado por vLLM

# Mensajes de tipo chat, usando el formato OpenAI-compatible
MESSAGES = [
    {
        "role": "system",
        "content": "Eres un ingeniero experto en telecomunicaciones SIEMPRE EN ESPANOL describes y ayudas a solucionar problemas, si no sabes algo NO NUNCA LO INVENTAS."
    },
    {
        "role": "user",
        "content": (
            "Escribe un problema tipico de las redes HFC o de las redes GPON"
        )
    }
]

# Inicializar cliente OpenAI para vLLM
client = OpenAI(base_url=VLLM_API_URL, api_key=API_KEY)

# Resultados por hilo
results: List[Tuple[int, float, float, str]] = []

def generate_story(thread_id: int):
    """Genera una historia y mide tokens y tiempo."""
    print(f"[Thread {thread_id}] Started.")
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES,
            max_tokens=800,
            temperature=0.7,
            stream=False
        )

        end_time = time.time()
        elapsed = end_time - start_time

        content = response.choices[0].message.content.strip()

        # Uso de tokens
        tokens_used = response.usage.total_tokens if response.usage else len(content.split())
        tps = tokens_used / elapsed if elapsed > 0 else 0
        results.append((thread_id, tokens_used, tps, content))

        print(f"[Thread {thread_id}] Finished: {tokens_used} tokens in {elapsed:.2f}s (TPS: {tps:.2f})\n")

    except APIError as e:
        print(f"[Thread {thread_id}] API error: {e}")
    except Exception as e:
        print(f"[Thread {thread_id}] Unexpected error: {e}")

def main():
    print("🚀 Starting story generation using vLLM...\n")
    threads = []

    global_start = time.time()

    for i in range(N_THREADS):
        thread = threading.Thread(target=generate_story, args=(i + 1,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    global_end = time.time()
    total_time = global_end - global_start

    total_tokens = sum(tokens for _, tokens, _, _ in results)
    avg_tps = total_tokens / total_time if total_time > 0 else 0

    print("\n📊 Summary Report")
    print(f"Total threads completed: {len(results)} / {N_THREADS}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Average tokens per second (TPS): {avg_tps:.2f}\n")

    print("🔍 Per-thread results:")
    for thread_id, tokens, tps, _ in results:
        print(f" - Thread {thread_id}: {tokens} tokens, {tps:.2f} TPS")

    print("\n📖 Model Outputs:")
    for thread_id, _, _, content in results:
        print(f"\n🧵 [Thread {thread_id}] Output:\n{'-'*60}\n{content}\n{'='*60}")

if __name__ == "__main__":
    main()
