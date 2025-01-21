import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

# Establece el modelo y la configuración de DeepSpeed
model_id = "meta-llama/Llama-3.3-70B-Instruct"
deepspeed_config = "deepspeed_config.json"  # Ruta al archivo de configuración de DeepSpeed

# Inicializa Accelerator para gestionar múltiples GPUs si es necesario
accelerator = Accelerator()

# Carga el modelo y el tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                            torch_dtype=torch.bfloat16, 
                                            device_map="auto", 
                                            deepspeed=deepspeed_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configura el pipeline de transformers
pipe = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=accelerator.device,  # Asegura que se use el dispositivo adecuado
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# Usando el pipeline para generar texto
outputs = pipe(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])
