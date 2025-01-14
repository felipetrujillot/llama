import torch
from transformers import pipeline, AutoTokenizer
from time import time
from colorama import Fore, Style

# Configuración del modelo
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=128001
)

# Leer contenido del archivo .txt
file_path = "archivo.txt"  # Nombre del archivo en la raíz
with open(file_path, "r", encoding="utf-8") as file:
    file_content = file.read()

# Crear el mensaje de entrada
prompt = (
    "You are a chatbot that summarizes and answers questions about documents.\n"
    f"The document is as follows:\n{file_content}\n\nWhat is this document about?"
)

# Medir el tiempo de inicio de generación
start_time = time()
# Usamos `generate` para medir el tiempo inicial del token.
outputs = pipe(
    prompt,
    max_new_tokens=9999999,  # Límite razonable
    return_full_text=False,  # Solo la respuesta generada
)
generation_start_time = time()

# Extraer la respuesta generada
assistant_content = outputs[0]["generated_text"]

# Medir tiempo total al finalizar
end_time = time()

# Contar los tokens en la respuesta
token_count = len(tokenizer.encode(assistant_content))

# Imprimir solo la respuesta relevante
print(assistant_content)

# Imprimir tiempos y tokens en color verde
print(f"{Fore.GREEN}Tiempo para comenzar a generar: {generation_start_time - start_time:.3f} segundos{Style.RESET_ALL}")
print(f"{Fore.GREEN}Tiempo total de respuesta: {end_time - start_time:.3f} segundos{Style.RESET_ALL}")
print(f"{Fore.GREEN}Tokens generados: {token_count}{Style.RESET_ALL}")
