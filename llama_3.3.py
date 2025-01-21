import transformers
import torch
from transformers import pipeline

# Establece el modelo y la configuración de DeepSpeed
model_id = "meta-llama/Llama-3.3-70B-Instruct"

# Configuración del modelo de DeepSpeed
deepspeed_config = "deepspeed_config.json"  # Ruta local del archivo de configuración

# Configuración de la pipeline de transformers con DeepSpeed
pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    deepspeed=deepspeed_config  # Utiliza el archivo local de configuración de DeepSpeed
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])
