#!/bin/sh
root=/share03/xxlyu
proj_dir=$root/proj/effective-DNMT-v2
python=$root/anaconda3/bin/python
cuda_app=$root/toolkits/idle-gpus.pl
model_dir=$proj_dir/workspace/model/sent-200w-model/_step_120000.pt
data_dir=$root/data/LDC-zh2en-data 
sets=(nist nist06)
src=zh
tgt=en

for set in "${sets[@]}"; do
CUDA_VISIBLE_DEVICES=`$cuda_app -n 1` $python $proj_dir/translate.py \
                -model $model_dir \
                -src $data_dir/$set.bpe.$src \
                -batch_size 64 \
                -output $data_dir/$set.200tran.$tgt \
                -minimal_relative_prob 0.0 -gpu 0 2>&1 | tee $data_dir/$set.200tran.log
done