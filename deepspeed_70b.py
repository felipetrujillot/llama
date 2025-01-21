import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

##############################################################################
# 1) CONFIGURAR AMBIENTE DISTRIBUIDO “SIMPLE” (1 GPU)
##############################################################################
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

# Evita que DeepSpeed busque MPI
deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)

##############################################################################
# 2) CARGAR MODELO Y TOKENIZER DESDE HUGGING FACE (EN CPU)
##############################################################################
model_id = "meta-llama/Llama-3.3-70B-Instruct"
dtype = torch.bfloat16  # O torch.float16 si tu GPU no maneja bfloat16 o te da OOM

print("Cargando modelo en CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=None,        # Asegura que no lo asigne a GPU
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

##############################################################################
# 3) CONFIGURAR DEEPSPEED PARA INFERENCIA (Zero Offload Stage 3)
##############################################################################
ds_inference_config = {
    "replace_with_kernel_inject": False,  # Evita errores con LLaMA

    # 1 sola GPU (sin tensor parallel)
    "tensor_parallel": {
        "tp_size": 1
    },

    # bfloat16 o fp16
    "dtype": "bf16",

    "zero": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": False
        }
    },

    "moe": {},
    "quant": {
        "enabled": False
    }
}

##############################################################################
# 4) INICIALIZAR DEEPSPEED EN MODO INFERENCE
##############################################################################
print("Inicializando DeepSpeed en modo INFERENCE con Zero Offload Stage 3...")
ds_engine = deepspeed.init_inference(
    model=model,
    **ds_inference_config
)

##############################################################################
# 5) CREAR PIPELINE DE TRANSFORMERS
##############################################################################
pipe = pipeline(
    task="text-generation",
    model=ds_engine.module,
    tokenizer=tokenizer,
    device=0  # GPU 0
)

##############################################################################
# 6) EJEMPLO DE INFERENCIA
##############################################################################
print("Generando respuesta...")

prompt = "Explain the difference between a pirate and a ninja."
outputs = pipe(prompt, max_new_tokens=128)
print(outputs[0]["generated_text"])
