#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 00:10:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=hello_world
#SBATCH --output=hello_world.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=timruhkopf@googlemail.com

module load python/3.9.0

python3 -m hello_world.py
wait