#!/bin/bash
# Author: xinglin
USER=/data/xllv
BASH_PATH=$USER/proj/distill-ctx-dnmt-v8/scripts
GPU_N=8
LOG_PATH=$USER/proj/distill-ctx-dnmt-v8/logs
J_NAME=exp_ldc80w

nohup bash $BASH_PATH/$J_NAME.sh &

