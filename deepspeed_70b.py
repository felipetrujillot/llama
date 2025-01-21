import torch
import transformers
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

##################################################################
# 1) CONFIGURACIÓN
##################################################################
model_id = "meta-llama/Llama-3.3-70B-Instruct"
deepspeed_config = "deepspeed_config.json"  # Asegúrate de que este archivo exista en la misma carpeta

# Si quieres que sea float16 en vez de bfloat16, cámbialo abajo.
# bfloat16 requiere soporte particular en tu GPU, pero la RTX 3090 sí suele tenerlo.
dtype = torch.bfloat16

##################################################################
# 2) CARGA DEL MODELO Y TOKENIZER
##################################################################
print("Cargando modelo y tokenizer desde Hugging Face...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype, 
    # NO uses device_map="auto" aquí, porque DeepSpeed manejará la distribución/offload
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

##################################################################
# 3) INICIALIZAR DEEPSPEED
##################################################################
print("Inicializando DeepSpeed con offload a CPU (según deepspeed_config.json)...")
# model_parameters = list(model.parameters()) es necesario si usas optimizer interno de DS
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=deepspeed_config
)
# 'engine.module' es el modelo real envuelto por DeepSpeed

##################################################################
# 4) CONFIGURAR PIPELINE DE TRANSFORMERS
##################################################################
# Importante: como DeepSpeed ya controla el modelo, no se recomienda usar device_map="auto" aquí.
# En la práctica, si tienes 1 sola GPU, puedes forzar device=0. 
# DeepSpeed seguirá usando la CPU para offload de parámetros y optimizador gracias a tu config.
pipe = pipeline(
    task="text-generation",
    model=engine.module,  
    tokenizer=tokenizer,
    # Si es 1 sola GPU, puedes poner device=0. Así PyTorch sabe que la salida final va a la GPU 0.
    device=0
)

##################################################################
# 5) EJEMPLO DE INFERENCIA
##################################################################
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"}
]

print("Generando respuesta...")
outputs = pipe(
    messages,          # El pipeline de text-generation normalmente espera strings. 
                       # En versiones recientes, pipeline() puede aceptar mensajes estilo Chat. 
    max_new_tokens=256
)

# Si el pipeline te devuelve un dict con la clave 'generated_text', imprimimos el último carácter
# Ajusta según la estructura real de la salida.
print(outputs[0]["generated_text"][-1])
