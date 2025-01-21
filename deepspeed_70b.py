import os
import torch
import transformers
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

##############################################################################
# 1) CONFIGURAR "DISTRIBUCIÓN" LOCAL
##############################################################################
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

# Evita que DeepSpeed busque MPI
deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)

##############################################################################
# 2) CARGA DEL MODELO EN CPU
##############################################################################
model_id = "meta-llama/Llama-3.3-70B-Instruct"
dtype = torch.bfloat16  # o torch.float16 si hay OOM

print("Cargando modelo en CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=None,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

##############################################################################
# 3) CONFIGURAR DEEPSPEED INFERENCE
##############################################################################
# Creamos un diccionario que describa la configuración de ZeRO-Inference
# con stage=3 y offload de parámetros a CPU
ds_inference_config = {
    "replace_method": "auto",            # o "auto_layer"
    "replace_with_kernel_inject": True,  # inyección de kernels optimizados
    "mp_size": 1,                        # 1 proceso (sin tensor-parallel)
    "dtype": dtype,
    # Config específico de ZeRO-Inference
    "zero": {
        "stage": 3,
        "offload_param": {
            "device": "cpu"
            # "pin_memory": False  # si quieres especificarlo
        }
    }
}

print("Inicializando DeepSpeed en modo INFERENCE con offload CPU (ZeRO Stage 3)...")
ds_engine = deepspeed.init_inference(
    model=model,
    **ds_inference_config
)
# ds_engine.module es el modelo envuelto

##############################################################################
# 4) PIPELINE DE TRANSFORMERS
##############################################################################
pipe = pipeline(
    "text-generation",
    model=ds_engine.module,
    tokenizer=tokenizer,
    device=0  # GPU 0
)

##############################################################################
# 5) EJEMPLO DE INFERENCIA
##############################################################################
print("Generando respuesta...")
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user",   "content": "Who are you?"}
]

outputs = pipe(messages, max_new_tokens=256)
print(outputs[0]["generated_text"][-1])
