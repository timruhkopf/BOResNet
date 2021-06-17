#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=BoResnet
#SBATCH --output=BoResnet.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=timruhkopf@googlemail.com

# requiries
module purge
#module --ignore-cache load cuda90/fft
#module --ignore-cache load cuda90/nsight
#module --ignore-cache load cuda90/profiler

# GWDG tutorial on tensorflow
# module load cuda90/toolkit/9.0.176
# module load cuda90/blas/9.0.176
# module load cudnn/90v7.3.1

# GWDG tutorial on pytorch:
module load cuda10.1/toolkit/10.1.105

# NOTICE Execution via bash shell/Jobs/Unittests.sh
# install packages:
module load python/3.9.0 # remember to load this before calling python3!
# python3 -m  pip install torch  # when it was not installed previously
# python3 -m pip install pyro-ppl
# python3 -m pip install numpy
# python3 -m pip install matplotlib

# to avoid matplotlib error :
# python3 -m pip install pyro-ppl
# python3 -m pip install --upgrade --force-reinstall  matplotlib

# to install third party implementations directly from github
# python3 -m pip install git+https://github.com/AdamCobb/hamiltorch # 3rd party

echo 'currently at dir: ' $PWD
echo 'be aware to change to /BOResNet/ and start script using "bash shell/Jobs/Unittests.sh"'
#cd /home/tim/PycharmProjects/BOResNet/

# make sure not to add .py ending when calling a module file
COMMIT_ID=$(git rev-parse --short HEAD)
echo 'current commit: ' $COMMIT_ID


# make sure not to add .py ending when calling a module file
python3 -m main # &>/usr/users/truhkop/BOResNet/consolelog_main.out

wait
