import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from time import time
from colorama import Fore, Style
from accelerate import infer_auto_device_map, dispatch_model

# Configuración del modelo
model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Llama-3.3-70B-Instruct"

def setup_model():
    print(f"{Fore.GREEN}Inicializando el modelo en múltiples GPUs...{Style.RESET_ALL}")

    # Cargar el tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Cargar el modelo completamente en memoria primero
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Inicialización automática para una sola GPU
        low_cpu_mem_usage=True  # Optimización de memoria al cargar
    )

    # Generar el mapa de dispositivos automáticamente
    device_map = infer_auto_device_map(model)

    # Redistribuir el modelo entre múltiples GPUs
    model = dispatch_model(model, device_map=device_map)

    # Crear el pipeline con el modelo distribuido
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        pad_token_id=128001,
    )

    return pipe, tokenizer

pipe, tokenizer = setup_model()

# Leer contenido del archivo .txt
file_path = "archivo.txt"  # Nombre del archivo en la raíz
with open(file_path, "r", encoding="utf-8") as file:
    file_content = file.read()

# Crear mensajes con el contenido del archivo
messages = [
    {"role": "system", "content": "You are a chatbot that summarizes and answers questions about documents."},
    {"role": "user", "content": f"The document is as follows:\n{file_content}\n\nWhat is this document about?"},
]

# Medir el tiempo de generación
start_time = time()
outputs = pipe(
    messages,
    max_new_tokens=1024,  # Ajusta según sea necesario
)
end_time = time()

# Extraer la respuesta del asistente
assistant_content = outputs[0]["generated_text"]

# Contar los tokens en la respuesta
token_count = len(tokenizer.encode(assistant_content))

# Imprimir solo la respuesta relevante
print(assistant_content)

# Imprimir tiempo y tokens en color verde
print(f"{Fore.GREEN}Tiempo de Respuesta: {end_time - start_time:.3f} segundos{Style.RESET_ALL}")
print(f"{Fore.GREEN}Tokens generados: {token_count}{Style.RESET_ALL}")
