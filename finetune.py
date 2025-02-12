import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Configuración inicial
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
DATASET_NAME = "preemware/pentesting-eval"  # Dataset de Hugging Face
OUTPUT_DIR = "./qwen-finetuned"

# 2. Cargar el tokenizer y el modelo preentrenado
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 3. Cargar y preparar el dataset
def preprocess_function(examples):
    return tokenizer(
        examples['text'],  # Asegúrate de que 'text' es la columna correcta en tu dataset
        truncation=True,
        padding='max_length',
        max_length=512
    )

dataset = load_dataset(DATASET_NAME)  # Carga el dataset desde Hugging Face
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 4. Dividir el dataset en entrenamiento y validación
# Para pruebas rápidas, seleccionamos un subconjunto pequeño del dataset
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))  # 100 ejemplos para entrenamiento
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(20))     # 20 ejemplos para validación

# 5. Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Tamaño de lote para entrenamiento
    per_device_eval_batch_size=4,   # Tamaño de lote para validación
    num_train_epochs=2,             # Reducimos a 2 épocas para pruebas iniciales
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),  # Usa FP16 si hay GPU disponible
    push_to_hub=False,               # Cambiar a True si quieres subir el modelo a Hugging Face Hub
)

# 6. Crear el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 7. Entrenar el modelo
print("Iniciando entrenamiento...")
trainer.train()

# 8. Guardar el modelo finetuneado
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modelo finetuneado guardado en {OUTPUT_DIR}")