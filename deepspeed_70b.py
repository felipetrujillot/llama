import time
import torch
from colorama import Fore, Style, init
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)

init(autoreset=True)

model_id = "meta-llama/Llama-3.3-70B-Instruct"

# Config de bitsandbytes para 4 bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Cargando el modelo a 4 bits (bitsandbytes) + device_map='auto'...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,  # 4-bit
    torch_dtype=torch.bfloat16,
    device_map="auto",              # Offload CPU / GPU automáticamente
    trust_remote_code=True
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
    device_map="auto"
)

print("¡Chat interactivo con Llama-3.3-70B en 4bits! Escribe 'exit' o 'quit' para salir.\n")

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
