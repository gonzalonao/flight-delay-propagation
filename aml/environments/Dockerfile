FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest

RUN pip install --upgrade pip

# PyTorch — must match CUDA 11.8 on the base image.
RUN pip install \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric + compiled CUDA extensions.
# Install from pre-built wheels to avoid source compilation failures.
RUN pip install torch_geometric==2.4.0
RUN pip install \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.2+cu118.html

# Data / ML stack
RUN pip install \
    pandas>=2.0 \
    pyarrow>=14.0 \
    scikit-learn>=1.3 \
    pyyaml>=6.0 \
    numpy>=1.24

# Azure SDK — used by the training script to log metrics and register the model.
RUN pip install azure-ai-ml>=1.12 azure-storage-file-datalake>=12.0 azure-identity>=1.15
