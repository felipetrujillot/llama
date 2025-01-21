import transformers
import torch

# Establece el modelo y la configuración de DeepSpeed
model_id = "meta-llama/Llama-3.3-70B-Instruct"

# Configuración del modelo de DeepSpeed
deepspeed_config = "deepspeed_config.json"

# Configuración de la pipeline de transformers con DeepSpeed
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    config=deepspeed_config,  # Configuración de DeepSpeed
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])
