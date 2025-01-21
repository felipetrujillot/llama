import torch
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main():
    accelerator = Accelerator()
    model_id = "meta-llama/Llama-3.3-70B-Instruct"

    print("Cargando el modelo en CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # O torch.float16 si prefieres
        device_map=None,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # -----------------------------------------------------------------
    # CREAR UN DATASET FICTICIO PARA QUE ACCELERATE NO LANCE ERROR
    # -----------------------------------------------------------------
    dummy_data = TensorDataset(torch.arange(1))  # 1 elemento
    dummy_loader = DataLoader(dummy_data, batch_size=1)

    # Prepara MODELO + DATALOADER
    model, dummy_loader = accelerator.prepare(model, dummy_loader)

    # Crea el pipeline con el modelo “acelerado” (DeepSpeed Offload)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=accelerator.device
    )

    print("Generando respuesta...")
    prompt = "Explain the difference between a pirate and a ninja."
    outputs = pipe(prompt, max_new_tokens=128)
    print(outputs[0]["generated_text"])

if __name__ == "__main__":
    main()
