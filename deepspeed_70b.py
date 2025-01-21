import time
import torch
from colorama import Fore, Style, init
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
# Para esta demo no necesitamos hacer `import deepspeed`
# porque DeepSpeed se configurará desde la CLI y ds_config.json

# Inicializa colorama para colores en la terminal
init(autoreset=True)

model_id = "meta-llama/Llama-3.3-70B-Instruct"

# Config de bitsandbytes para 4 bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Cargando el modelo a 4 bits (bitsandbytes) + offload CPU...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,     # Cuantización en 4 bits
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"                  # Reparto auto en GPU/CPU
    # NO PASAMOS: deepspeed=...
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"    # El pipeline también se reparte
)

print("¡Chat interactivo con Llama-3.3-70B (4bits + ZeRO Stage 3)! Escribe 'exit' o 'quit' para salir.\n")

while True:
    user_input = input("Tú: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Saliendo del chat...")
        break

    start_time = time.time()
    output = pipe(user_input, max_new_tokens=128)
    end_time = time.time()

    response = output[0]["generated_text"]
    print(f"Llama: {response}")

    elapsed = end_time - start_time
    print(Fore.GREEN + f"Tiempo de respuesta: {elapsed:.2f} seg" + Style.RESET_ALL)
    print()
