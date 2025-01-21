import os
import torch
import transformers
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ========================================
# Variables de entorno para 1 solo proceso
# ========================================
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

model_id = "meta-llama/Llama-3.3-70B-Instruct"
deepspeed_config = "deepspeed_config.json"
dtype = torch.bfloat16

print("Cargando modelo y tokenizer desde Hugging Face...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Inicializando DeepSpeed con offload a CPU...")
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=deepspeed_config,
    dist_init_required=True,
    dist_backend='nccl'  # o 'gloo' si no quieres usar la GPU para comm
)

pipe = pipeline(
    task="text-generation",
    model=engine.module,  
    tokenizer=tokenizer,
    device=0
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipe(messages, max_new_tokens=256)
print(outputs[0]["generated_text"][-1])
