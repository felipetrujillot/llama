import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_distributed():
    """
    Configura el entorno distribuido para PyTorch y DeepSpeed.
    """
    if not torch.distributed.is_initialized():
        # Determinar cuántas GPUs están disponibles
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        rank = int(os.getenv("RANK", 0))
        
        # Inicializar el proceso distribuido
        torch.distributed.init_process_group(
            backend="nccl",  # Utiliza NCCL para GPUs
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        print(f"Proceso inicializado: rank={rank}, local_rank={local_rank}, world_size={world_size}")

def setup_model():
    """
    Carga el modelo y lo inicializa con DeepSpeed para múltiples GPUs.
    """
    # Nombre del modelo (puedes usar cualquier modelo compatible con Transformers)
    model_name = "EleutherAI/gpt-neo-125M"

    # Cargar el modelo y el tokenizador
    print("Cargando el modelo y el tokenizador...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configuración de DeepSpeed
    ds_config = {
        "replace_method": "auto",
        "tensor_parallel": {
            "tp_size": torch.cuda.device_count()  # Número de GPUs disponibles
        },
        "quantize": {
            "enabled": True,
            "bits": 8
        }
    }

    # Inicializar el modelo con DeepSpeed
    print("Inicializando el modelo con DeepSpeed...")
    model = deepspeed.init_inference(
        model=model,
        config=ds_config,
        mp_size=torch.cuda.device_count()
    )

    return model, tokenizer

def main():
    # Configura el entorno distribuido
    setup_distributed()

    # Carga y configura el modelo con DeepSpeed
    model, tokenizer = setup_model()

    # Prueba básica para generar texto
    prompt = "DeepSpeed es increíble para"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print("Generando texto...")
    outputs = model.generate(inputs.input_ids, max_length=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
