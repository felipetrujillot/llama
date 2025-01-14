import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from time import time
from colorama import Fore, Style
import deepspeed

# Configuración del modelo
model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Ajusta según tu modelo
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configuración de DeepSpeed
ds_config = {
    "replace_with_kernel_inject": True,
    "tensor_parallel": {
        "tp_size": 4  # Número de GPUs disponibles
    }
}

# Cargar modelo con DeepSpeed
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

model = deepspeed.init_inference(
    model,
    config=ds_config
)

# Crear el pipeline con el modelo optimizado
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    pad_token_id=tokenizer.pad_token_id
)

# Leer contenido del archivo .txt
file_path = "archivo.txt"  # Nombre del archivo en la raíz
with open(file_path, "r", encoding="utf-8") as file:
    file_content = file.read()

# Crear mensajes con el contenido del archivo
messages = [
    {"role": "system", "content": "You are a chatbot that summarizes and answers questions about documents."},
    {"role": "user", "content": f"The document is as follows:\n{file_content}\n\nWhat is this document about?"}
]

# Medir el tiempo de generación
start_time = time()
outputs = pipe(
    messages,
    max_new_tokens=1024,  # Ajusta según tus necesidades
)
end_time = time()

# Extraer la respuesta generada
assistant_content = outputs[0]["generated_text"]

# Contar los tokens en la respuesta
token_count = len(tokenizer.encode(assistant_content))

# Imprimir solo la respuesta relevante
print(assistant_content)

# Imprimir tiempo y tokens en color verde
print(f"{Fore.GREEN}Tiempo de Respuesta: {end_time - start_time:.3f} segundos{Style.RESET_ALL}")
print(f"{Fore.GREEN}Tokens generados: {token_count}{Style.RESET_ALL}")
