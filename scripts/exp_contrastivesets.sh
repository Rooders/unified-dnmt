#!/bin/sh
# copyright: xinglin lyu 2022.12.17

proj_dir=/data/xllv/proj/distill-ctx-dnmt-v8
model_dir=/data/xllv/proj/distill-ctx-dnmt-v8/workspace/model/opensubtitle/doc-enru-se0-uoc1-uat1-ct1-cb1-of0-gas1-drop0.1-lr0.2-acc2-ls0.1-mw0.7-cwg0-mb0.15-ddl0-dl1.0-seca0-sdca0-ics1-md0-sds100000-ng0-smde0-dt0.2-da0-model
python=/data/xllv/anaconda3/envs/nict/bin/python
step=100000
type=document
device=4
eval_tookit=/data/xllv/toolkits/contrastive-set-eval
data_dir=/data/xllv/toolkits/contrastive-set-eval/consistency_testsets/clean_data
all_sets=(lex_cohesion_test deixis_test ellipsis_infl ellipsis_vp)
src=en
tgt=ru

eval_out_dir=$model_dir/$step-contras-eval
mkdir -p $eval_out_dir

for set in "${all_sets[@]}"; do
    CUDA_VISIBLE_DEVICES=$device $python $proj_dir/translate.py \
                    -model $model_dir/_step_$step.pt \
                    -src $data_dir/$set.$type.bpe.$src \
                    -output $eval_out_dir/$set.scores \
                    -batch_size 1 \
                    -force_decoding \
                    -tgt_tran $data_dir/$set.$type.bpe.$tgt \
                    -tgt $data_dir/$set.$type.bpe.$tgt \
                    -minimal_relative_prob 0.0 -gpu 0 2>&1 | tee $eval_out_dir/$set.log
    awk '{if (NR%4 == 0) print $0; }' $eval_out_dir/$set.scores > $eval_out_dir/$set.final.scores
    $python $eval_tookit/scripts/evaluate_consistency.py --repo-dir $eval_tookit \
     --test $set --scores $eval_out_dir/$set.final.scores 2>&1 | tee $eval_out_dir/$set.evalmb.score
done
