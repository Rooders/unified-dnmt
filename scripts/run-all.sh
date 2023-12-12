#!/bin/bash
#SBATCH -J v7ldc80
#SBATCH -n 48
#SBATCH -x gpu112
#SBATCH --gres=gpu:2


USER=/public/home/jhli/xllv
PROJ=$USER/proj/unified-dnmt-v7
OUTPUT=$PROJ/logs
EXPT=exp_ldc80w
bash $PROJ/scripts/$EXPT.sh

