#!/bin/sh
# toolkits path
root=/share03/xxlyu
proj_dir=$root/proj/joint-learning-dnmt
python=$root/anaconda3/bin/python
cuda_app=$root/toolkits/idle-gpus.pl
LTCR=$root/toolkits/alignment-LTCR/compute_lctr_awe.sh
blonde_m=$root/toolkits/blonde-metric/blonde_metric.py
multi_bleu=$root/toolkits/multi-bleu.perl
d_bleu_prepare=$root/toolkits/prepare4doc-bleu.py
split_script=$proj_dir/split_translation.py
py_file=/share03/xxlyu/toolkits/doc2sent.py
# path of files for LTCR evaluation 
en_stopwords_file=$root/data/LDC-zh2en-data/enstopwords.txt
zh_stopwords_file=$root/data/LDC-zh2en-data/topword.zh.1000
# workspace path
workspace=$proj_dir/workspace
binary_data=$workspace/data
model_dir=$workspace/model
mkdir -p $workspace $binary_data $model_dir

# path of traing, dev, test data
data_dir=$root/data/LDC-zh2en-data
# languages pair and type
src=zh
tgt=en
type=200w
# training hyperparameters
word_size=8
dropout=0.1
lr=0.1
accum_count=2
hyper=drop$dropout-lr$lr-acc$accum_count
# model hyperparameters for document-level
use_auto_trans=1
segment_embedding=1
use_ord_ctx=1
cross_before=1
cross_attn=1
decoder_cross_before=0
only_fixed=0
cross_out_encoder=0
gated_auto_src=1
pretrain_from_doc=0
doc_double_lr=1
doc_lr=100.0

info_use=se$segment_embedding-uoc$use_ord_ctx-uat$use_auto_trans-ct$cross_attn-cb$cross_before-dcb$decoder_cross_before-of$only_fixed-coe$cross_out_encoder-gas$gated_auto_src-$hyper-ddl$doc_double_lr-dl$doc_lr
((rk=$word_size-1))
ranks=($(seq 0 1 $rk))
ranks=`echo "${ranks[@]}"`


# The procedures of training
generate_model_path=$workspace/model/doc-se1-uoc1-uat1-ct0-cb0-dcb0-of0-coe0-gas1-drop0.3-lr0.1-acc4-ddl0-dl1.0-model/_step_72000.pt
fine_tune_model_path=$generate_model_path
generate_process=1

# sent_preprocessing=0
# sent_stage_training=0
# sent_stage_evaluation=01234567890-0 
doc_preprocessing=1
doc_stage_training=1
doc_stage_evaluation=1

dev_set=nist06
test_set=nist
all_sets=(nist06 nist)

best_sent_checkpoint=130000


if [ $generate_process -eq 1 ]; then
  for set in train nist06 nist; do
    # CUDA_VISIBLE_DEVICES=`$cuda_app -n 1` $python $proj_dir/translate.py \
    #                 -model $sent_model_dir/_step_$best_sent_checkpoint.pt \
    #                 -src $data_dir/$F.sent.bpe.$src \
    #                 -batch_size 64 \
    #                 -share_vocab \
    #                 -output $data_dir/$F.sent.tran.bpe.$tgt \
    #                 -minimal_relative_prob 0.0 -gpu 0 2>&1 | tee $data_dir/$F.tran.log
    touch $data_dir/$set.tran
    CUDA_VISIBLE_DEVICES=`$cuda_app -n 1` $python $proj_dir/translate.py \
                  -model $generate_model_path \
                  -src $data_dir/$set.document.bpe.$src \
                  -tgt_tran $data_dir/$set.document.200tran.bpe.$tgt \
                  -output $data_dir/$set.tran \
                  -minimal_relative_prob 0.0 -gpu 0 2>&1 | tee $data_dir/$set.log
    
    # rm $data_dir/$set.document.200tran.bpe.$tgt
    # mv $dat_dir/$set.tran $data_dir/$set.document.200tran.bpe.$tgt
    
    
    python $py_file --doc_file $data_dir/$set.document.bpe.$src \
                        --sent_file $data_dir/$set.tran \
                        --out_file $data_dir/$set.document.200tran1.bpe.$tgt \
                        --mode sent2doc
    # mv $dat_dir/$set.tran $data_dir/$set.document.200tran1.bpe.$tgt
  done
fi





# doc_binary_dir=$binary_data/doc-$type-data
# doc_model_dir=$model_dir/$type/doc-$info_use-model/
# mkdir -p $doc_binary_dir $doc_model_dir
# if [ $doc_preprocessing -eq 1 ]; then
# python $proj_dir/preprocess.py -train_src $data_dir/train.document.bpe.$src \
#                        -train_tgt $data_dir/train.document.bpe.$tgt \
#                        -train_auto_trans $data_dir/train.document.200tran.bpe.$tgt \
#                        -valid_src $data_dir/$dev_set.document.bpe.$src  \
#                        -valid_tgt $data_dir/$dev_set.document.bpe.en0 \
#                        -valid_auto_trans $data_dir/$dev_set.document.200tran.bpe.$tgt \
#                        -save_data $doc_binary_dir/gq \
#                        -use_auto_trans $use_auto_trans \
#                        -src_vocab_size 40000 \
#                        -tgt_vocab_size 40000 \
#                        -src_seq_length 10000 \
#                        -tgt_seq_length 10000 2>&1 | tee $doc_binary_dir/preprocess.log
# fi

# if [ $doc_stage_training -eq 1 ]; then
# pretrain_model=$sent_model_dir/_step_$best_sent_checkpoint.pt
# if [ $pretrain_from_doc -eq 1 ]; then
#   pretrain_model=$model_dir/doc-nopaired-model/_step_36000.pt
# fi

# CUDA_VISIBLE_DEVICES=`$cuda_app -n $word_size` \
#         $python -W ignore $proj_dir/train.py \
#             -data $doc_binary_dir/gq \
#             -save_model $doc_model_dir \
#             -world_size $word_size \
#             -gpu_ranks $ranks \
#             -master_port 62594 \
#             -save_checkpoint_steps 2000 \
#             -valid_steps 2000 \
#             -report_every 20 \
#             -keep_checkpoint 40 \
#             -seed 3435 \
#             -train_steps 80000 \
#             -warmup_steps 4000 \
#             --share_decoder_embeddings \
#             --position_encoding \
#             --optim adam \
#             -adam_beta1 0.9 \
#             -adam_beta2 0.998 \
#             -decay_method noam \
#             -learning_rate $lr \
#             -max_grad_norm 0.0 \
#             -batch_size 2048 \
#             -accum_count $accum_count \
#             -batch_type tokens \
#             -mixed_precision \
#             -normalization tokens \
#             -dropout $dropout \
#             -label_smoothing 0.1 \
#             -use_auto_trans $use_auto_trans \
#             -use_ord_ctx $use_ord_ctx \
#             -cross_attn $cross_attn \
#             -cross_before $cross_before \
#             -decoder_cross_before $decoder_cross_before \
#             -cross_out_encoder $cross_out_encoder \
#             -gated_auto_src $gated_auto_src \
#             -only_fixed $only_fixed \
#             -segment_embedding $segment_embedding \
#             -train_from $pretrain_model \
#             -doc_double_lr $doc_double_lr \
#             -doc_lr $doc_lr \
#             -reset_optim all \
#             -max_generator_batches 50 \
#             -tensorboard \
#             -tensorboard_log_dir $doc_model_dir \
#             -param_init 0.0 \
#             -param_init_glorot 2>&1 | tee $doc_model_dir/train.log
# fi


# if [ $doc_stage_evaluation -eq 1 ]; then
# for set in "${all_sets[@]}"; do
#     start=2000
#     end=80000
#     step=2000
#     best_bleu=0.01
#     best_d_bleu=0.01
#     best_ltcr=0.01
#     best_bleu_checkpoint=0
#     best_d_bleu_checkpoint=0
#     best_ltcr_checkpoint=0
#     while [ $start -le $end ]; do
#     if [ -f "$doc_model_dir/_step_$start.pt" ] && [ ! -f "$doc_model_dir/$start/$set.tran" ]; then
#         echo "Decoding using $doc_model_dir/_step_$start.pt"
#         mkdir -p $doc_model_dir/$start
#         touch $doc_model_dir/$start/$set.tran
#         CUDA_VISIBLE_DEVICES=`$cuda_app -n 1` $python $proj_dir/translate.py \
#                   -model $doc_model_dir/_step_$start.pt \
#                   -src $data_dir/$set.document.bpe.$src \
#                   -tgt_tran $data_dir/$set.document.200tran.bpe.$tgt \
#                   -output $doc_model_dir/$start/$set.tran \
#                   -minimal_relative_prob 0.0 -gpu 0 2>&1 | tee $doc_model_dir/$start/$set.log
        
#         # $python $split_script --path_trans $doc_model_dir/$start/$set.paired.tran --out_file $doc_model_dir/$start/$set.tran
#         sed -i 's/@@ //g' $doc_model_dir/$start/$set.tran
#         perl $multi_bleu $data_dir/$set.clean.$tgt < $doc_model_dir/$start/$set.tran > $doc_model_dir/$start/$set.tran.evalmb
#         # evaluation of ltcr
#         CUDA_VISIBLE_DEVICES=`$cuda_app -n 1` bash $LTCR $data_dir/$set.clean.$src \
#                 $data_dir/$set.document.bpe.$src \
#                 $doc_model_dir/$start/$set.tran \
#                 $doc_model_dir/$start \
#                 $doc_model_dir/$start/$set.LTCR.evalmb \
#                 $src $tgt $en_stopwords_file $zh_stopwords_file
#         # evaluation of blonde
#         $python $blonde_m --ref $data_dir/$set.clean.$tgt \
#                           --ref_num 4 \
#                           --doc_file $data_dir/$set.document.bpe.$src \
#                           --trans $doc_model_dir/$start/$set.tran > $doc_model_dir/$start/$set.blonde.evalmb \
        
#         $python $d_bleu_prepare --doc_file $data_dir/$set.document.bpe.$src \
#                                 --sent_file  $doc_model_dir/$start/$set.tran \
#                                 --out_file  $doc_model_dir/$start/$set.doc.tran \
#                                 --mode sent2doc
#         perl $multi_bleu $data_dir/$set.document.clean.$tgt <  $doc_model_dir/$start/$set.doc.tran >  $doc_model_dir/$start/$set.doc.tran.evalmb
    
#         BLEU_d=$(cat  $doc_model_dir/$start/$set.doc.tran.evalmb | sed 's/,/\n/g' | grep 'BLEU' | sed -e 's/BLEU = //g' | sed -e 's/\n//g')
#         LTCR_s=$(cat $doc_model_dir/$start/$set.LTCR.evalmb | grep 'LTCR = ' | sed 's/,/\n/g' | grep 'LTCR' | sed -e 's/LTCR = //g' | sed -e 's/\n//g')
#         BLEU_s=$(cat $doc_model_dir/$start/$set.tran.evalmb | sed 's/,/\n/g' | grep 'BLEU' | sed -e 's/BLEU = //g' | sed -e 's/\n//g')
#         if [ `echo "$LTCR_s > $best_ltcr" | bc` -eq 1 ]; then
#             best_ltcr=$LTCR_s
#             best_ltcr_checkpoint=$start
#         fi

#         if [ `echo "$BLEU_s > $best_bleu" | bc` -eq 1 ]; then
#             best_bleu=$BLEU_s
#             best_bleu_checkpoint=$start
#         fi
#         if [ `echo "$BLEU_d > $best_d_bleu" | bc` -eq 1 ]; then
#             best_d_bleu=$BLEU_d
#             best_d_bleu_checkpoint=$start
#         fi
#     fi
#     start=$((${start}+$step))
#     done
#     echo -e "$set:\n best_itcr checkpoint: $best_ltcr_checkpoint, $best_ltcr\n best_bleu checkpoint: $best_bleu_checkpoint, $best_bleu\n" >> $doc_model_dir/results.evalmb
# done
# fi