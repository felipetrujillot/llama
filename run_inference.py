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

# Configuración para cuantización en 4 bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Cuantiza el modelo a 4 bits
    bnb_4bit_use_double_quant=True,       # Activar doble cuantización para ahorrar memoria
    bnb_4bit_quant_type="nf4",            # Tipo de cuantización (nf4 o fp4)
    bnb_4bit_compute_dtype=torch.bfloat16 # Para usar bfloat16 en el cómputo
)

# Control preciso del uso de VRAM y RAM con max_memory
# Ajusta los valores según tu GPU (48GB) y tu RAM (188GB)
max_memory = {
    0: "47GiB",   # GPU ID 0: le reservamos 47 GB para no llegar al límite
    "cpu": "170GiB"
}

print("Cargando el modelo en 4 bits con offload automático...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",       # Esto reparte capas GPU/CPU según la memoria disponible
    max_memory=max_memory,   # Límite de memoria para cada dispositivo
    torch_dtype=torch.bfloat16,
    trust_remote_code=True   # Por si el repositorio requiere código adicional
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("Cargando...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("¡Chat interactivo con Llama-3.3-70B en 4 bits! Escribe 'exit' o 'quit' para salir.\n")

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
