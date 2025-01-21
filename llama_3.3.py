import transformers
import torch
from transformers import pipeline
from accelerate import Accelerator

# Configura Accelerator
accelerator = Accelerator()

# Establece el modelo y la configuración de DeepSpeed
model_id = "meta-llama/Llama-3.3-70B-Instruct"
deepspeed_config = "deepspeed_config.json"  # Ruta al archivo de configuración

# Configura la pipeline con DeepSpeed usando Accelerate
pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    deepspeed=deepspeed_config,  # Usar DeepSpeed con Accelerate
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
