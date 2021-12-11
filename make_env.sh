#!/usr/bin/env bash

# Modified from ruxi's make_conda_env.sh
# Link: https://gist.github.com/ruxi/949e3d326c5a8a24ecffa8a225b2be2a

CUDA=10.2
TORCH=1.8
PYTHON=3.7

declare -A TORCHVISION=([1.8]=0.9 [1.6]=0.7 [1.5]=0.6)

read -p "Create new conda env (y/n)?" CONT

if [ $CONT == "n" ]; then
  echo "exit";
else
# user chooses to create conda env
# prompt user for conda env name
  echo "Creating new conda environment, choose name"
  read input_variable
  echo "Name $input_variable was chosen";

  # Create environment.yml or not
  read -p "Create 'enviroment.yml', will overwrite if exist (y/n)?" CONT
    if [ "$CONT" == "y" ]; then
      # yes: create enviroment.yml
      echo "# BASH: conda env create

name: $input_variable
channels:
  - conda-forge
dependencies:
- python=${PYTHON}
- cudatoolkit=${CUDA}
- opencv
- pip
- pip:
  - http://dl.fbaipublicfiles.com/detectron2/wheels/cu${CUDA//.}/torch${TORCH}/detectron2-0.6%2Bcu${CUDA//.}-cp${PYTHON//.}-cp${PYTHON//.}m-linux_x86_64.whl
  - --find-links https://download.pytorch.org/whl/cu${CUDA//.}/torch_stable.html
  - torch==${TORCH}
  - torchvision==${TORCHVISION[$TORCH]}" >environment.yml

  #list name of packages
  conda env create
    else
        echo "installing base packages"
        conda_path="$(which conda)"
        conda create --name $input_variable python=${PYTHON}
        conda install -n $input_variable pytorch==${TORCH} torchvision==${TORCHVISION[$TORCH]} cudatoolkit=${CUDA} -c pytorch
        python_path=$"$(dirname $(dirname $conda_path))/envs/${input_variable}/bin/python"
        $python_path -m pip install opencv-contrib-python
        $python_path -m pip install "http://dl.fbaipublicfiles.com/detectron2/wheels/cu${CUDA//.}/torch${TORCH}/detectron2-0.6%2Bcu${CUDA//.}-cp${PYTHON//.}-cp${PYTHON//.}m-linux_x86_64.whl"
    fi
  echo "to exit: source deactivate"
fi
