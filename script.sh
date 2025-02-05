#!/bin/bash

set -e  # Detener ejecución si ocurre un error

# Descargar e instalar Anaconda
echo "Descargando Anaconda..."
wget -q https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O /tmp/anaconda.sh

echo "Instalando Anaconda..."
bash /tmp/anaconda.sh -b -p /root/anaconda3

# Agregar Anaconda al PATH
export PATH="/root/anaconda3/bin:$PATH"

# Activar Anaconda y crear el entorno
echo "Configurando entorno Conda..."
source /root/anaconda3/bin/activate
conda create -n ia python=3.12 -y
source activate ia  # Opción alternativa a `conda activate ia`

# Instalar paquetes
echo "Instalando dependencias..."
pip install accelerate
pip install huggingface_hub
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers colorama
pip install PyPDF2
pip install langchain chromadb sentence-transformers transformers
pip install -U langchain-community
pip install pypdf
pip install fastapi uvicorn

#instalando nvtop
apt update && apt install -y nvtop

echo "Configuración completada. Usa 'conda activate ia' para entrar al entorno."
