import time
import torch
from colorama import Fore, Style, init
from transformers import pipeline

# Inicializa colorama para imprimir con colores en la terminal
init(autoreset=True)

# Identificador del modelo en Hugging Face
model_id = "meta-llama/Llama-3.3-70B-Instruct"

# Crea un pipeline de text-generation
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    torch_dtype=torch.bfloat16,  # Usa torch.float16 si necesitas
    device_map="auto",           # Permite offload a CPU/GPU automáticamente
)

print("¡Chat interactivo con Llama-3.3-70B! Escribe 'exit' o 'quit' para salir.\n")

while True:
    # Solicita input del usuario
    user_input = input("Tú: ")

    # Condición de salida
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Saliendo del chat...")
        break

    # Mide el tiempo de generación
    start_time = time.time()
    # Genera la respuesta con el pipeline
    output = pipe(user_input, max_new_tokens=128)
    end_time = time.time()

    # Extrae la cadena generada
    response = output[0]["generated_text"]

    # Imprime la respuesta del modelo
    print(f"Llama: {response}")

    # Calcula y muestra el tiempo en color verde
    elapsed = end_time - start_time
    print(Fore.GREEN + f"Tiempo de respuesta: {elapsed:.2f} seg" + Style.RESET_ALL)
    print()  # Salto de línea para separar turnos de chat
