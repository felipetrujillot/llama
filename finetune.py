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

# 3. Cargar el dataset
dataset = load_dataset(DATASET_NAME)

# Inspeccionar las columnas del dataset
print("Columnas del dataset:")
print(dataset["train"].column_names)

# 4. Preprocesar el dataset
def preprocess_function(examples):
    # Concatenar las columnas 'question', 'choices', y 'explanation' en un solo texto
    texts = [
        f"Question: {q}\nChoices: {c}\nExplanation: {e}"
        for q, c, e in zip(examples['question'], examples['choices'], examples['explanation'])
    ]
    return tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=512
    )

# Aplicar la función de preprocesamiento
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 5. Dividir el dataset en entrenamiento y validación
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))  # 100 ejemplos para entrenamiento
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(20))     # 20 ejemplos para validación

# 6. Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
)

# 7. Crear el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 8. Entrenar el modelo
print("Iniciando entrenamiento...")
trainer.train()

# 9. Guardar el modelo finetuneado
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modelo finetuneado guardado en {OUTPUT_DIR}")