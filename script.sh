#!/bin/bash

# Configuración inicial
set -e  # Detener el script si hay errores
export DEBIAN_FRONTEND=noninteractive

# Actualizar el sistema y dependencias esenciales
apt-get update && apt-get install -y wget git

# Clonar el repositorio
git clone https://github.com/felipetrujillot/llama.git

# Descargar Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O /tmp/anaconda.sh

# Instalar Anaconda en modo silencioso
bash /tmp/anaconda.sh -b -p /root/anaconda3

# Agregar Anaconda al PATH
export PATH="/root/anaconda3/bin:$PATH"

# Crear el entorno Conda
source /root/anaconda3/bin/activate
conda create -n ia python=3.12 -y

# Activar el entorno
source activate ia

# Instalar dependencias
pip install accelerate
pip install huggingface_hub
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers colorama
pip install PyPDF2

echo "Instalación completada. Para activar el entorno, use: source /root/anaconda3/bin/activate && conda activate ia"
