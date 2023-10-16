#!/usr/bin/env bash

export CONDA_ENV_NAME=py3.8_NVDIFFRAST
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.8

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install .
pip3 install -i https://pypi.mirrors.ustc.edu.cn/simple/ -r requirements.txt