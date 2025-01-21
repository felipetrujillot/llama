import time
import torch
from colorama import Fore, Style, init
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
import deepspeed

init(autoreset=True)

model_id = "meta-llama/Llama-3.3-70B-Instruct"

# Config de bitsandbytes para 4 bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Configuración DeepSpeed (ZeRO Stage 3 + offload a CPU)
# Ajusta según tus necesidades. Importante "inference=True".
ds_config = {
    "train_batch_size": 1,
    "inference": {
        "enabled": True,
        "use_cuda_graph": False
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "cpu_offload": True,
        "contiguous_gradients": True,
        "overlap_comm": True
    }
}

print("Cargando el modelo con DeepSpeed ZeRO Stage 3 + Cuantización 4bits...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    deepspeed=ds_config  # <- Parámetro para que transformers use DeepSpeed
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("¡Chat interactivo con Llama-3.3-70B (DeepSpeed ZeRO + 4bits)! Escribe 'exit' o 'quit' para salir.\n")

while True:
    user_input = input("Tú: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Saliendo del chat...")
        break

    start_time = time.time()
    output = pipe(user_input, max_new_tokens=128)
    end_time = time.time()

    response = output[0]["generated_text"]
    print(f"Llama: {response}")

    elapsed = end_time - start_time
    print(Fore.GREEN + f"Tiempo de respuesta: {elapsed:.2f} seg" + Style.RESET_ALL)
    print()
