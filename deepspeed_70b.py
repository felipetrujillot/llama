import os
import torch
import transformers
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

###############################################################################
# 1) CONFIGURAR ENTORNO DISTRIBUIDO DE 1 PROCESO
###############################################################################
# Variables de entorno para simular "1 proceso / 1 GPU"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

# Evita que DeepSpeed intente buscar MPI automáticamente:
deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)

###############################################################################
# 2) CONFIGURACIÓN DEL MODELO
###############################################################################
model_id = "meta-llama/Llama-3.3-70B-Instruct"
deepspeed_config = "deepspeed_config.json"
dtype = torch.bfloat16  # Cámbialo a float16 si tienes problemas con bfloat16

print("Cargando modelo y tokenizer desde Hugging Face...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype, 
    device_map=None,           # <-- Asegurarte de no asignarlo a la GPU
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

###############################################################################
# 3) INICIALIZAR DEEPSPEED
###############################################################################
print("Inicializando DeepSpeed con offload a CPU...")
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=deepspeed_config,
    dist_init_required=True  # Aquí sí lo dejamos True, 
                             # porque ya hicimos init_distributed()
)

###############################################################################
# 4) CREAR PIPELINE DE TRANSFORMERS
###############################################################################
pipe = pipeline(
    task="text-generation",
    model=engine.module,  
    tokenizer=tokenizer,
    device=0  # Usa la GPU 0
)

###############################################################################
# 5) EJEMPLO DE INFERENCIA
###############################################################################
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user",   "content": "Who are you?"}
]

print("Generando respuesta...")
outputs = pipe(messages, max_new_tokens=256)
print(outputs[0]["generated_text"][-1])
