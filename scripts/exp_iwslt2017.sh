#!/bin/sh
# toolkits path
root=/public/home/jhli/xllv
proj_dir=$root/proj/unified-dnmt-v7
python=/public/home/jhli/anaconda3/envs/unified/bin/python
cuda_app=$root/toolkits/idle-gpus.pl
LTCR=$root/toolkits/alignment-LTCR/compute_lctr_awe.sh
blonde_m=$root/toolkits/blonde-metric/blonde_metric.py
multi_bleu=$root/toolkits/multi-bleu.perl
d_bleu_prepare=$root/toolkits/prepare4doc-bleu.py
split_script=$proj_dir/split_translation.py
DETOKENIZER=$root/toolkits/scripts/tokenizer/detokenizer.perl
py_file=$root/toolkits/doc2sent.py
doc_data_bulider=$proj_dir/doc_data_bulider.py
# path of files for LTCR evaluation 
en_stopwords_file=$root/data/LDC-zh2en-data/enstopwords.txt
zh_stopwords_file=$root/data/LDC-zh2en-data/topword.zh.1000
# workspace path
workspace=$proj_dir/workspace
binary_data=$workspace/data
model_dir=$workspace/model
mkdir -p $workspace $binary_data $model_dir

# path of traing, dev, test data
data_dir=$root/data/IWSLT2017
# languages pair and type
src=en
tgt=de
type=iwslt2017
# training hyperparameters
word_size=8
dropout=0.3
lr=0.02
label_smoothing=0.1
weight_trans_kl=0.3
accum_count=2

hyper=drop$dropout-lr$lr-acc$accum_count-ls$label_smoothing-wtk$weight_trans_kl

use_auto_trans=1
segment_embedding=0
use_ord_ctx=1
cross_before=1
cross_attn=1
only_fixed=0
gated_auto_src=1
auto_truth_trans_kl=1
src_mlm=1
multi_task_training=0
doc_double_lr=0
doc_lr=1.0
# option of use gold translation
use_z_contronl=0
use_affine=1
share_dec_cross_attn=0
share_enc_cross_attn=1
doc_context_layers=4

gold_use=uzc$use_z_contronl-sdca$share_dec_cross_attn-seca$share_enc_cross_attn-ua$use_affine


info_use=sm$src_mlm-se$segment_embedding-uoc$use_ord_ctx-uat$use_auto_trans-ct$cross_attn-cb$cross_before-of$only_fixed-gas$gated_auto_src-$hyper-ddl$doc_double_lr-dl$doc_lr-mtt$multi_task_training-attk$auto_truth_trans_kl-$gold_use

((rk=$word_size-1))
ranks=($(seq 0 1 $rk))
ranks=`echo "${ranks[@]}"`


# The procedures of training
sent_preprocessing=0
sent_stage_training=0
sent_stage_evaluation=0


gen_pseudo_data=0
doc_data_bulid=0
doc_preprocessing=0
doc_stage_training=1
doc_stage_evaluation=1

dev_set=valid
test_set=test
all_sets=(valid test)

best_sent_checkpoint=30000

sent_binary_dir=$binary_data/$type/sent-$src$tgt-data
sent_model_dir=$model_dir/$type/sent-$src$tgt-model/
mkdir -p $sent_binary_dir $sent_model_dir
if [ $sent_preprocessing -eq 1 ]; then
$python $proj_dir/preprocess.py -train_src $data_dir/train.sent.bpe.$src \
                    -train_tgt $data_dir/train.sent.bpe.$tgt \
                    -valid_src $data_dir/$dev_set.sent.bpe.$src  \
                    -valid_tgt $data_dir/$dev_set.sent.bpe.$tgt \
                    -save_data $sent_binary_dir/gq \
                    -sentence_level 1 \
                    -share_vocab \
                    -src_vocab_size 40000  \
                    -tgt_vocab_size 40000 \
                    -src_seq_length 128 \
                    -tgt_seq_length 128 \
                    --src_seq_length_trunc 128 \
                    --tgt_seq_length_trunc 128 2>&1 | tee $sent_binary_dir/preprocess.log
fi

if [ $sent_stage_training -eq 1 ]; then
CUDA_VISIBLE_DEVICES=`$cuda_app -n $word_size` \
        $python -W ignore $proj_dir/train.py \
            -data $sent_binary_dir/gq \
            -save_model $sent_model_dir \
            -world_size $word_size \
            -gpu_ranks $ranks \
            -master_port 62394 \
            -save_checkpoint_steps 5000 \
            -share_embeddings \
            -valid_steps 5000 \
            -report_every 20 \
            -keep_checkpoint 40 \
            -seed 3435 \
            -train_steps 150000 \
            -warmup_steps 4000 \
            --share_decoder_embeddings \
            --position_encoding \
            --optim adam \
            -adam_beta1 0.9 \
            -adam_beta2 0.998 \
            -decay_method noam \
            -learning_rate 1.0 \
            -max_grad_norm 0.0 \
            -batch_size 4096 \
            -accum_count $accum_count \
            -batch_type tokens \
            -mixed_precision \
            -normalization tokens \
            -dropout $dropout \
            -label_smoothing 0.1 \
            -sentence_level True \
            -max_generator_batches 50 \
            -param_init 0.0 \
            -param_init_glorot 2>&1 | tee $sent_model_dir/train.log
fi

if [ $sent_stage_evaluation -eq 1 ]; then
start=30000
end=200000
step=5000
dev_best_bleu=0.01
while [ $start -le $end ]; do
if [ -f "$sent_model_dir/_step_$start.pt" ] && [ ! -f "$sent_model_dir/$start/$dev_set.tran" ]; then
    echo "Decoding using $sent_model_dir/_step_$start.pt"
    mkdir -p $sent_model_dir/$start
    touch $sent_model_dir/$start/$dev_set.tran
    CUDA_VISIBLE_DEVICES=`$cuda_app -n 1` $python $proj_dir/translate.py \
                -model $sent_model_dir/_step_$start.pt \
                -src $data_dir/$dev_set.sent.bpe.$src \
                -output $sent_model_dir/$start/$dev_set.tran \
                -batch_size 64 \
                -share_vocab \
                -minimal_relative_prob 0.0 -gpu 0 2>&1 | tee $sent_model_dir/$start/$dev_set.log
    sed -i 's/@@ //g' $sent_model_dir/$start/$dev_set.tran
    perl $DETOKENIZER -l $tgt < $sent_model_dir/$start/$dev_set.tran > $sent_model_dir/$start/$dev_set.tran.detok
    perl $multi_bleu $data_dir/$dev_set.sent.detok.$tgt < $sent_model_dir/$start/$dev_set.tran.detok > $sent_model_dir/$start/$dev_set.tran.detok.evalmb
    perl $multi_bleu $data_dir/$dev_set.sent.clean.$tgt < $sent_model_dir/$start/$dev_set.tran > $sent_model_dir/$start/$dev_set.tran.evalmb
    # BLEU_s=$(cat $sent_model_dir/$start/$dev_set.tran.detok.evalmb | sed 's/,/\n/g' | grep 'BLEU' | sed -e 's/BLEU = //g' | sed -e 's/\n//g')
    $python -m sacrebleu.sacrebleu $data_dir/$dev_set.sent.detok.$tgt -i $sent_model_dir/$start/$dev_set.tran.detok -w 2 --score-only > $sent_model_dir/$start/$dev_set.sent.tran.detok.sacrebleu
    BLEU_s=$(cat $sent_model_dir/$start/$dev_set.sent.tran.detok.sacrebleu)
    if [ `echo "$BLEU_s > $dev_best_bleu" | bc` -eq 1 ]; then
        dev_best_bleu=$BLEU_s
        best_sent_checkpoint=$start
    fi
fi
start=$((${start}+$step))
done

CUDA_VISIBLE_DEVICES=`$cuda_app -n 1` $python $proj_dir/translate.py \
                -model $sent_model_dir/_step_$best_sent_checkpoint.pt \
                -src $data_dir/$test_set.sent.bpe.$src \
                -batch_size 64 \
                -share_vocab \
                -output $sent_model_dir/$best_sent_checkpoint/$test_set.tran \
                -minimal_relative_prob 0.0 -gpu 0 2>&1 | tee $sent_model_dir/$best_sent_checkpoint/$test_set.log
sed -i 's/@@ //g' $sent_model_dir/$best_sent_checkpoint/$test_set.tran
perl $DETOKENIZER -l $tgt < $sent_model_dir/$best_sent_checkpoint/$test_set.tran > $sent_model_dir/$best_sent_checkpoint/$test_set.tran.detok
perl $multi_bleu $data_dir/$test_set.sent.detok.$tgt < $sent_model_dir/$best_sent_checkpoint/$test_set.tran.detok > $sent_model_dir/$best_sent_checkpoint/$test_set.tran.detok.evalmb
perl $multi_bleu $data_dir/$test_set.sent.clean.$tgt < $sent_model_dir/$best_sent_checkpoint/$test_set.tran > $sent_model_dir/$best_sent_checkpoint/$test_set.tran.evalmb
# $python -m sacrebleu.sacrebleu $data_dir/$test_set.sent.detok.$tgt -i $best_sent_checkpoint/$test_set.sent.tran.detok > $sent_model_dir/$best_sent_checkpoint/$test_set.sent.tran.detok.sacrebleu
$python -m sacrebleu.sacrebleu $data_dir/$test_set.sent.detok.$tgt -i $sent_model_dir/$best_sent_checkpoint/$test_set.tran.detok --score-only -w 2 > $sent_model_dir/$best_sent_checkpoint/$test_set.sent.tran.detok.sacrebleu
test_best_bleu=$(cat $sent_model_dir/$best_sent_checkpoint/$test_set.sent.tran.detok.sacrebleu)

$python $blonde_m --ref $data_dir/$test_set.sent.clean.$tgt \
                  --ref_num 1 \
                  --doc_file $data_dir/$test_set.document.bpe.$src \
                  --trans $sent_model_dir/$best_sent_checkpoint/$test_set.tran > $sent_model_dir/$best_sent_checkpoint/$test_set.blonde.evalmb 
        

$python $d_bleu_prepare --doc_file $data_dir/$test_set.document.bpe.$src \
                            --sent_file  $sent_model_dir/$best_sent_checkpoint/$test_set.tran.detok \
                            --out_file  $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran.detok \
                            --mode sent2doc

$python $d_bleu_prepare --doc_file $data_dir/$test_set.document.bpe.$src \
                            --sent_file  $sent_model_dir/$best_sent_checkpoint/$test_set.tran \
                            --out_file  $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran \
                            --mode sent2doc


# Need to delete later
# $python $d_bleu_prepare --doc_file $data_dir/$test_set.document.bpe.$src \
#                             --sent_file  $data_dir/$test_set.sent.detok.$tgt \
#                             --out_file  $data_dir/$test_set.document.detok.$tgt \
#                             --mode sent2doc
# $python $d_bleu_prepare --doc_file $data_dir/$test_set.document.bpe.$src \
#                             --sent_file  $data_dir/$test_set.sent.detok.$src \
#                             --out_file  $data_dir/$test_set.document.detok.$src \
#                             --mode sent2doc

# $python $d_bleu_prepare --doc_file $data_dir/$dev_set.document.bpe.$src \
#                             --sent_file  $data_dir/$dev_set.sent.detok.$tgt \
#                             --out_file  $data_dir/$dev_set.document.detok.$tgt \
#                             --mode sent2doc

# $python $d_bleu_prepare --doc_file $data_dir/$dev_set.document.bpe.$src \
#                             --sent_file  $data_dir/$dev_set.sent.detok.$src \
#                             --out_file  $data_dir/$dev_set.document.detok.$src \
#                             --mode sent2doc



perl $multi_bleu $data_dir/$test_set.document.detok.$tgt <  $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran.detok >  $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran.detok.evalmb
perl $multi_bleu $data_dir/$test_set.document.clean.$tgt <  $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran >  $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran.evalmb
$python -m sacrebleu.sacrebleu $data_dir/$test_set.document.detok.$tgt -i $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran.detok -w 2 --score-only > $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran.detok.sacrebleu

test_best_dbleu=$(cat $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran.detok.sacrebleu)
echo -e "best_checkpoint: $best_sent_checkpoint\n $dev_set: s-bleu=$dev_best_bleu\n $test_set: s-bleu=$test_best_bleu, d-bleu=$test_best_dbleu, " >> $sent_model_dir/results.evalmb


fi


if [ $gen_pseudo_data -eq 1 ]; then
  for F in train valid test; do
    CUDA_VISIBLE_DEVICES=`$cuda_app -n 1` $python $proj_dir/translate.py \
                    -model $sent_model_dir/_step_$best_sent_checkpoint.pt \
                    -src $data_dir/$F.sent.bpe.$src \
                    -batch_size 256 \
                    -share_vocab \
                    -output $data_dir/$F.sent.tran.bpe.$tgt \
                    -minimal_relative_prob 0.0 -gpu 0 2>&1 | tee $data_dir/$F.tran.log

    $python $py_file --doc_file $data_dir/$F.document.bpe.$src \
                        --sent_file $data_dir/$F.sent.tran.bpe.$tgt \
                        --out_file $data_dir/$F.document.tran.bpe.$tgt \
                        --mode sent2doc
  done
fi



if [ $doc_data_bulid -eq 1 ]; then
  for F in train valid test; do
  $python $doc_data_bulider --src_doc_path $data_dir/$F.document.bpe.$src \
                            --tgt_doc_path $data_dir/$F.document.bpe.$tgt \
                            --tran_doc_path $data_dir/$F.document.tran.bpe.$tgt
  done

fi



doc_binary_dir=$binary_data/$type/doc-$src$tgt-data
doc_model_dir=$model_dir/$type/doc-$src$tgt-$info_use-model/
mkdir -p $doc_binary_dir $doc_model_dir
if [ $doc_preprocessing -eq 1 ]; then
$python $proj_dir/preprocess.py -train_src $data_dir/train.document.bpe.$src.mini \
                       -train_tgt $data_dir/train.document.bpe.$tgt.mini \
                       -train_auto_trans $data_dir/train.document.tran.bpe.$tgt.mini \
                       -valid_src $data_dir/$dev_set.document.bpe.$src.mini  \
                       -valid_tgt $data_dir/$dev_set.document.bpe.$tgt.mini \
                       -valid_auto_trans $data_dir/$dev_set.document.tran.bpe.$tgt.mini \
                       -save_data $doc_binary_dir/gq \
                       -use_auto_trans $use_auto_trans \
                       -src_vocab_size 40000 \
                       -tgt_vocab_size 40000 \
                       --shard_size 40000 \
                       -share_vocab \
                       -src_seq_length 128 \
                       -tgt_seq_length 128 \
                       --src_seq_length_trunc 128 \
                       --tgt_seq_length_trunc 128 2>&1 | tee $doc_binary_dir/preprocess.log
fi

if [ $doc_stage_training -eq 1 ]; then
pretrain_model=$sent_model_dir/_step_$best_sent_checkpoint.pt

CUDA_VISIBLE_DEVICES=`$cuda_app -n $word_size` \
        $python -W ignore $proj_dir/train.py \
            -data $doc_binary_dir/gq \
            -save_model $doc_model_dir \
            -world_size $word_size \
            -gpu_ranks $ranks \
            -master_port 62384 \
            -save_checkpoint_steps 2500 \
            -valid_steps 2500 \
            -report_every 20 \
            -keep_checkpoint 40 \
            -seed 3435 \
            -train_steps 100000 \
            -warmup_steps 4000 \
            --share_decoder_embeddings \
            --position_encoding \
            --optim adam \
            -adam_beta1 0.9 \
            -adam_beta2 0.998 \
            -decay_method noam \
            -learning_rate $lr \
            -max_grad_norm 0.0 \
            -batch_size 4096 \
            -accum_count $accum_count \
            -batch_type tokens \
            -mixed_precision \
            -normalization tokens \
            -dropout $dropout \
            -label_smoothing $label_smoothing \
            -use_auto_trans $use_auto_trans \
            -use_ord_ctx $use_ord_ctx \
            -cross_attn $cross_attn \
            -cross_before $cross_before \
            -auto_truth_trans_kl $auto_truth_trans_kl \
            -gated_auto_src $gated_auto_src \
            -only_fixed $only_fixed \
            -segment_embedding $segment_embedding \
            -multi_task_training $multi_task_training \
            -train_from $pretrain_model \
            -weight_trans_kl $weight_trans_kl \
            -use_z_contronl $use_z_contronl \
            -use_affine $use_affine \
            -share_dec_cross_attn $share_dec_cross_attn \
            -share_enc_cross_attn $share_enc_cross_attn \
            -doc_context_layers $doc_context_layers \
            -src_mlm $src_mlm \
            -train_from $pretrain_model \
            -doc_double_lr $doc_double_lr \
            -doc_lr $doc_lr \
            -reset_optim all \
            -max_generator_batches 50 \
            -tensorboard \
            -tensorboard_log_dir $doc_model_dir \
            -param_init 0.0 \
            -param_init_glorot 2>&1 | tee $doc_model_dir/train.log
fi


if [ $doc_stage_evaluation -eq 1 ]; then
for set in "${all_sets[@]}"; do
    start=2500
    end=100000
    step=2500
    best_bleu=0.01
    best_d_bleu=0.01
    best_ltcr=0.01
    best_bleu_checkpoint=0
    best_d_bleu_checkpoint=0
    best_ltcr_checkpoint=0
    while [ $start -le $end ]; do
    if [ -f "$doc_model_dir/_step_$start.pt" ] && [ ! -f "$doc_model_dir/$start/$set.tran" ]; then
        echo "Decoding using $doc_model_dir/_step_$start.pt"
        mkdir -p $doc_model_dir/$start
        touch $doc_model_dir/$start/$set.tran
        CUDA_VISIBLE_DEVICES=`$cuda_app -n 1` $python $proj_dir/translate.py \
                  -model $doc_model_dir/_step_$start.pt \
                  -src $data_dir/$set.document.bpe.$src.mini \
                  -tgt_tran $data_dir/$set.document.tran.bpe.$tgt.mini \
                  -share_vocab \
                  -output $doc_model_dir/$start/$set.tran \
                  -minimal_relative_prob 0.0 -gpu 0 2>&1 | tee $doc_model_dir/$start/$set.log
        
        # $python $split_script --path_trans $doc_model_dir/$start/$set.paired.tran --out_file $doc_model_dir/$start/$set.tran
        sed -i 's/@@ //g' $doc_model_dir/$start/$set.tran
        
        perl $multi_bleu $data_dir/$set.sent.clean.$tgt < $doc_model_dir/$start/$set.tran > $doc_model_dir/$start/$set.tran.evalmb
        perl $DETOKENIZER -l $tgt < $doc_model_dir/$start/$set.tran > $doc_model_dir/$start/$set.tran.detok
        perl $multi_bleu $data_dir/$set.sent.detok.$tgt < $doc_model_dir/$start/$set.tran.detok > $doc_model_dir/$start/$set.sent.tran.detok.evalmb
        $python -m sacrebleu.sacrebleu $data_dir/$set.sent.detok.$tgt -i $doc_model_dir/$start/$set.tran.detok -w 2 --score-only > $doc_model_dir/$start/$set.sent.tran.sacrebleu
        
        $python $blonde_m --ref $data_dir/$set.sent.clean.$tgt \
                          --ref_num 1 \
                          --doc_file $data_dir/$set.document.bpe.$src \
                          --trans $doc_model_dir/$start/$set.tran > $doc_model_dir/$start/$set.blonde.evalmb \
        


        $python $d_bleu_prepare --doc_file $data_dir/$set.document.bpe.$src \
                                --sent_file  $doc_model_dir/$start/$set.tran \
                                --out_file  $doc_model_dir/$start/$set.doc.tran \
                                --mode sent2doc
        
        $python $d_bleu_prepare --doc_file $data_dir/$set.document.bpe.$src \
                                --sent_file  $doc_model_dir/$start/$set.tran.detok \
                                --out_file  $doc_model_dir/$start/$set.doc.tran.detok \
                                --mode sent2doc
        
        perl $multi_bleu $data_dir/$set.document.clean.$tgt <  $doc_model_dir/$start/$set.doc.tran >  $doc_model_dir/$start/$set.doc.tran.evalmb
        perl $multi_bleu $data_dir/$set.document.detok.$tgt < $doc_model_dir/$start/$set.doc.tran.detok > $doc_model_dir/$start/$set.doc.tran.detok.evalmb
        $python -m sacrebleu.sacrebleu $data_dir/$set.document.detok.$tgt -i $doc_model_dir/$start/$set.doc.tran.detok -w 2 --score-only > $doc_model_dir/$start/$set.doc.tran.sacrebleu
        BLEU_d=$(cat $doc_model_dir/$start/$set.doc.tran.sacrebleu)
        BLEU_s=$(cat $doc_model_dir/$start/$set.sent.tran.sacrebleu)
        

        if [ `echo "$BLEU_s > $best_bleu" | bc` -eq 1 ]; then
            best_bleu=$BLEU_s
            best_bleu_checkpoint=$start
        fi
        if [ `echo "$BLEU_d > $best_d_bleu" | bc` -eq 1 ]; then
            best_d_bleu=$BLEU_d
            best_d_bleu_checkpoint=$start
        fi
    fi
    start=$((${start}+$step))
    done
    echo -e "$set:\n best_bleu checkpoint: $best_bleu_checkpoint, $best_bleu\n" >> $doc_model_dir/results.evalmb
done
fi