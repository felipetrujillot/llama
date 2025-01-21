import torch
import transformers
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

##################################################################
# 1) CONFIGURACIÓN
##################################################################
model_id = "meta-llama/Llama-3.3-70B-Instruct"
deepspeed_config = "deepspeed_config.json" 
dtype = torch.bfloat16  # Cambia a float16 si da problemas

##################################################################
# 2) CARGA DEL MODELO Y TOKENIZER
##################################################################
print("Cargando modelo y tokenizer desde Hugging Face...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype, 
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

##################################################################
# 3) INICIALIZAR DEEPSPEED
##################################################################
print("Inicializando DeepSpeed con offload a CPU (según deepspeed_config.json)...")
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=deepspeed_config,
    dist_init_required=False  # <--- Importante para no usar MPI
)

##################################################################
# 4) CONFIGURAR PIPELINE DE TRANSFORMERS
##################################################################
pipe = pipeline(
    task="text-generation",
    model=engine.module,  
    tokenizer=tokenizer,
    device=0  # GPU 0 si tienes una sola tarjeta
)

##################################################################
# 5) EJEMPLO DE INFERENCIA
##################################################################
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"}
]

print("Generando respuesta...")
outputs = pipe(messages, max_new_tokens=256)
print(outputs[0]["generated_text"][-1])
