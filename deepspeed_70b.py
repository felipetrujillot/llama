import os
import torch
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

# Configuración de DeepSpeed Inference
ds_inference_config = {
    "replace_with_kernel_inject": False,  # Desactivar inyección de kernels
    "tensor_parallel": {
        "tp_size": 1
    },
    "dtype": dtype,
    "zero": {
        "stage": 3,
        "offload_param": {
            "device": "cpu"
        }
    }
}

print("Inicializando DeepSpeed en modo INFERENCE con Zero Offload Stage 3...")
ds_engine = deepspeed.init_inference(
    model=model,
    **ds_inference_config
)

pipe = pipeline(
    task="text-generation",
    model=ds_engine.module,
    tokenizer=tokenizer,
    device=0
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"}
]

print("Generando respuesta...")
outputs = pipe(messages, max_new_tokens=256)
print(outputs[0]["generated_text"][-1])
