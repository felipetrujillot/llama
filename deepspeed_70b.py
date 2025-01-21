import os
import torch
import transformers
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)

model_id = "meta-llama/Llama-3.3-70B-Instruct"
dtype = torch.bfloat16

print("Cargando modelo en CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=None,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Aquí usamos la API para inferencia de DeepSpeed
ds_engine = deepspeed.init_inference(
    model=model,
    mp_size=1,  # tamaño de parallelismo tensor
    dtype=dtype,
    replace_method='auto',  # o 'auto_layer', a veces ayuda con modelos grandes
    replace_with_kernel_inject=True,
    config_fp='deepspeed_config.json'  # tu config Zero Stage 3
)
model = ds_engine.module  # El modelo envuelto por DeepSpeed para inferencia

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=0 si quieres forzar la GPU 0 para la salida
    device=0
)

print("Generando respuesta...")
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user",   "content": "Who are you?"}
]

outputs = pipe(messages, max_new_tokens=256)
print(outputs[0]["generated_text"][-1])
