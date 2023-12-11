#!/bin/sh
# toolkits path
root=/data/xllv
proj_dir=$root/proj/distill-ctx-dnmt-v8
python=$root/anaconda3/envs/nict/bin/python
python2=$root/anaconda3/envs/python2.7/bin/python
cuda_app=$root/toolkits/idle-gpus.pl
multi_bleu=$root/toolkits/multi-bleu.perl
d_bleu_prepare=$root/toolkits/prepare4doc-bleu.py
split_script=$proj_dir/split_translation.py
DETOKENIZER=$root/toolkits/scripts/tokenizer/detokenizer.perl
py_file=$root/toolkits/doc2sent.py
doc_data_bulider=$proj_dir/doc_data_bulider.py
discoures_terms_eval=$root/toolkits/discourse-metric
en_stopwords_file=$root/data/LDC-zh2en-data/enstopwords.txt
zh_stopwords_file=$root/data/LDC-zh2en-data/topword.zh.1000
alignment_tool=$root/toolkits/awesome-align
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
type=ldc80w
# training hyperparameters
device=2
test_device=3
world_size=2
dropout=0.3
lr=0.2
lr_decay=noam
label_smoothing=0.1
mlm_weight=0.1
mlm_prob=0.4
accum_count=4
connect_with_gate=0

hyper=drop$dropout-lr$lr-acc$accum_count-ls$label_smoothing-mw$mlm_weight-cwg$connect_with_gate-mb$mlm_prob
# model hyperparameters for document-level
use_auto_trans=1
segment_embedding=1
use_ord_ctx=1
cross_before=1
cross_attn=1
only_fixed=0
gated_auto_src=1
doc_double_lr=1
doc_lr=10.0
# option of use gold translation
share_dec_cross_attn=0
share_enc_cross_attn=1
init_cross_sent=1
new_gen=1
share_mlm_decoder_embeddings=1
distill_threshold=0.5
distill_annealing=1
mlm_distill=1
start_distill_step=50000



gold_use=seca$share_enc_cross_attn-sdca$share_dec_cross_attn-ics$init_cross_sent-md$mlm_distill-sds$start_distill_step-ng$new_gen-smde$share_mlm_decoder_embeddings-dt$distill_threshold-da$distill_annealing
info_use=se$segment_embedding-uoc$use_ord_ctx-uat$use_auto_trans-ct$cross_attn-cb$cross_before-of$only_fixed-gas$gated_auto_src-$hyper-ddl$doc_double_lr-dl$doc_lr-$gold_use

((rk=$world_size-1))
ranks=($(seq 0 1 $rk))
ranks=`echo "${ranks[@]}"`

# The procedures of training
sent_preprocessing=0
sent_stage_training=0
sent_stage_evaluation=0


gen_pseudo_data=0
doc_data_bulid=1
doc_preprocessing=1
doc_stage_training=1
doc_stage_evaluation=1

dev_set=nist06
test_set=nist
all_sets=(nist nist06)

best_sent_checkpoint=140000

sent_binary_dir=$binary_data/$type/sent-$src$tgt-data
sent_model_dir=$model_dir/$type/sent-$src$tgt-model/
mkdir -p $sent_binary_dir $sent_model_dir
if [ $sent_preprocessing -eq 1 ]; then
$python $proj_dir/preprocess.py -train_src $data_dir/train.80wsent.bpe.$src \
                    -train_tgt $data_dir/train.80wsent.bpe.$tgt \
                    -valid_src $data_dir/$dev_set.bpe.$src  \
                    -valid_tgt $data_dir/$dev_set.bpe.en0 \
                    -save_data $sent_binary_dir/gq \
                    -sentence_level 1 \
                    -src_vocab_size 40000 \
                    -tgt_vocab_size 40000 \
                    -src_seq_length 128 \
                    -tgt_seq_length 128 \
                    --src_seq_length_trunc 128 \
                    --tgt_seq_length_trunc 128 2>&1 | tee $sent_binary_dir/preprocess.log
fi


if [ $sent_stage_training -eq 1 ]; then
CUDA_VISIBLE_DEVICES=`$cuda_app -n $world_size` \
        $python -W ignore $proj_dir/train.py \
            -data $sent_binary_dir/gq \
            -save_model $sent_model_dir \
            -world_size $world_size \
            -gpu_ranks $ranks \
            -master_port 62594 \
            -save_checkpoint_steps 5000 \
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
            -decay_method $lr_decay \
            -learning_rate $lr \
            -max_grad_norm 0.0 \
            -batch_size 4096 \
            -accum_count 1 \
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
start=5000
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
                -src $data_dir/$dev_set.bpe.$src \
                -output $sent_model_dir/$start/$dev_set.tran \
                -batch_size 64 \
                -minimal_relative_prob 0.0 -gpu 0 2>&1 | tee $sent_model_dir/$start/$dev_set.log
    sed -i 's/@@ //g' $sent_model_dir/$start/$dev_set.tran
    perl $multi_bleu $data_dir/$dev_set.clean.$tgt < $sent_model_dir/$start/$dev_set.tran > $sent_model_dir/$start/$dev_set.tran.evalmb
    BLEU_s=$(cat $sent_model_dir/$start/$dev_set.tran.evalmb | sed 's/,/\n/g' | grep 'BLEU' | sed -e 's/BLEU = //g' | sed -e 's/\n//g')
    
    if [ `echo "$BLEU_s > $dev_best_bleu" | bc` -eq 1 ]; then
        dev_best_bleu=$BLEU_s
        best_sent_checkpoint=$start
    fi
fi
start=$((${start}+$step))
done

CUDA_VISIBLE_DEVICES=`$cuda_app -n 1` $python $proj_dir/translate.py \
                -model $sent_model_dir/_step_$best_sent_checkpoint.pt \
                -src $data_dir/$test_set.bpe.$src \
                -batch_size 64 \
                -output $sent_model_dir/$best_sent_checkpoint/$test_set.tran \
                -minimal_relative_prob 0.0 -gpu 0 2>&1 | tee $sent_model_dir/$best_sent_checkpoint/$test_set.log
sed -i 's/@@ //g' $sent_model_dir/$best_sent_checkpoint/$test_set.tran
perl $multi_bleu $data_dir/$test_set.clean.$tgt < $sent_model_dir/$best_sent_checkpoint/$test_set.tran > $sent_model_dir/$best_sent_checkpoint/$test_set.tran.evalmb
test_best_bleu=$(cat $sent_model_dir/$best_sent_checkpoint/$test_set.tran.evalmb | sed 's/,/\n/g' | grep 'BLEU' | sed -e 's/BLEU = //g' | sed -e 's/\n//g')
$python $d_bleu_prepare --doc_file $data_dir/$test_set.document.bpe.$src \
                            --sent_file  $sent_model_dir/$best_sent_checkpoint/$test_set.tran \
                            --out_file  $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran \
                            --mode sent2doc
perl $multi_bleu $data_dir/$test_set.document.clean.$tgt <  $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran >  $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran.evalmb

CUDA_VISIBLE_DEVICES=`$cuda_app -n 1` bash $LTCR $data_dir/$test_set.clean.$src \
                $data_dir/$test_set.document.bpe.$src \
                $sent_model_dir/$best_sent_checkpoint/$test_set.tran \
                $sent_model_dir/$best_sent_checkpoint \
                $sent_model_dir/$best_sent_checkpoint/$test_set.LTCR.evalmb \
                $src $tgt $en_stopwords_file $zh_stopwords_file
$python $blonde_m --ref $data_dir/$test_set.clean.$tgt \
                  --ref_num 4 \
                  --doc_file $data_dir/$test_set.document.bpe.$src \
                  --trans $sent_model_dir/$best_sent_checkpoint/$test_set.tran > $sent_model_dir/$best_sent_checkpoint/$test_set.blonde.evalmb \

test_best_dbleu=$(cat  $sent_model_dir/$best_sent_checkpoint/$test_set.doc.tran.evalmb | sed 's/,/\n/g' | grep 'BLEU' | sed -e 's/BLEU = //g' | sed -e 's/\n//g')
test_best_LTCR=$(cat $sent_model_dir/$best_sent_checkpoint/$test_set.LTCR.evalmb | grep 'LTCR = ' | sed 's/,/\n/g' | grep 'LTCR' | sed -e 's/LTCR = //g' | sed -e 's/\n//g')
echo -e "best_checkpoint: $best_sent_checkpoint\n $dev_set: s-bleu=$dev_best_bleu\n $test_set: s-bleu=$test_best_bleu, d-bleu=$test_best_dbleu, LTCR=$test_best_LTCR\n" >> $sent_model_dir/results.evalmb
fi


if [ $doc_data_bulid -eq 1 ]; then

  $python $doc_data_bulider --src_doc_path $data_dir/train.document.bpe.$src \
                            --tgt_doc_path $data_dir/train.document.bpe.$tgt \
                            --tran_doc_path $data_dir/train.document.tran.bpe.$tgt

fi



doc_binary_dir=$binary_data/$type/doc-$src$tgt-data
doc_model_dir=$model_dir/$type/doc-$src$tgt-$info_use-model/
# doc_model_dir=$model_dir/$type/doc-joint-model/
mkdir -p $doc_binary_dir $doc_model_dir
if [ $doc_preprocessing -eq 1 ]; then
$python $proj_dir/preprocess.py -train_src $data_dir/train.document.bpe.$src.mini \
                       -train_tgt $data_dir/train.document.bpe.$tgt.mini \
                       -train_auto_trans $data_dir/train.document.tran.bpe.$tgt.mini \
                       -valid_src $data_dir/$dev_set.document.bpe.$src  \
                       -valid_tgt $data_dir/$dev_set.document.bpe.en0 \
                       -valid_auto_trans $data_dir/$dev_set.document.tran.bpe.$tgt \
                       -save_data $doc_binary_dir/gq \
                       -use_auto_trans $use_auto_trans \
                       -src_vocab_size 40000 \
                       -tgt_vocab_size 40000 \
                       -src_seq_length 128 \
                       -tgt_seq_length 128 \
                       --src_seq_length_trunc 128 \
                       --tgt_seq_length_trunc 128 2>&1 | tee $doc_binary_dir/preprocess.log
fi

if [ $doc_stage_training -eq 1 ]; then
pretrain_model=$sent_model_dir/_step_$best_sent_checkpoint.pt
CUDA_VISIBLE_DEVICES=`$cuda_app -n $world_size` \
        $python -W ignore $proj_dir/train.py \
            -data $doc_binary_dir/gq \
            -save_model $doc_model_dir \
            -world_size $world_size \
            -gpu_ranks $ranks \
            -master_port 62571 \
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
            -decay_method $lr_decay \
            -learning_rate $lr \
            -max_grad_norm 0.0 \
            -batch_size 4096 \
            -accum_count $accum_count \
            -batch_type tokens \
            -mixed_precision \
            -normalization tokens \
            -dropout $dropout \
            -label_smoothing 0.1 \
            -use_auto_trans $use_auto_trans \
            -use_ord_ctx $use_ord_ctx \
            -cross_attn $cross_attn \
            -cross_before $cross_before \
            -gated_auto_src $gated_auto_src \
            -only_fixed $only_fixed \
            -segment_embedding $segment_embedding \
            -mlm_distill $mlm_distill \
            -start_distill_step $start_distill_step \
            -new_gen $new_gen \
            -share_mlm_decoder_embeddings $share_mlm_decoder_embeddings \
            -mlm_weight $mlm_weight \
            -mlm_prob $mlm_prob \
            -distill_threshold $distill_threshold \
            -distill_annealing $distill_annealing \
            -doc_double_lr $doc_double_lr \
            -doc_lr $doc_lr \
            -share_dec_cross_attn $share_dec_cross_attn \
            -share_enc_cross_attn $share_enc_cross_attn \
            -init_cross_sent $init_cross_sent \
            -train_from $pretrain_model \
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
    end=200000
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
        CUDA_VISIBLE_DEVICES=$test_device $python $proj_dir/translate.py \
                  -model $doc_model_dir/_step_$start.pt \
                  -src $data_dir/$set.document.bpe.$src \
                  -tgt_tran $data_dir/$set.document.tran.bpe.$tgt \
                  -output $doc_model_dir/$start/$set.tran \
                  -minimal_relative_prob 0.0 -gpu 0 2>&1 | tee $doc_model_dir/$start/$set.log
        
        # $python $split_script --path_trans $doc_model_dir/$start/$set.paired.tran --out_file $doc_model_dir/$start/$set.tran
        sed -i 's/@@ //g' $doc_model_dir/$start/$set.tran
        perl $multi_bleu $data_dir/$set.clean.$tgt < $doc_model_dir/$start/$set.tran > $doc_model_dir/$start/$set.tran.evalmb
        # evaluation of ltcr
        CUDA_VISIBLE_DEVICES=`$cuda_app -n 1` bash $LTCR $data_dir/$set.clean.$src \
                $data_dir/$set.document.bpe.$src \
                $doc_model_dir/$start/$set.tran \
                $doc_model_dir/$start \
                $doc_model_dir/$start/$set.LTCR.evalmb \
                $src $tgt $en_stopwords_file $zh_stopwords_file
        # evaluation of blonde
        $python $blonde_m --ref $data_dir/$set.clean.$tgt \
                          --ref_num 4 \
                          --doc_file $data_dir/$set.document.bpe.$src \
                          --trans $doc_model_dir/$start/$set.tran > $doc_model_dir/$start/$set.blonde.evalmb \
        
        $python $d_bleu_prepare --doc_file $data_dir/$set.document.bpe.$src \
                                --sent_file  $doc_model_dir/$start/$set.tran \
                                --out_file  $doc_model_dir/$start/$set.doc.tran \
                                --mode sent2doc
        perl $multi_bleu $data_dir/$set.document.clean.$tgt <  $doc_model_dir/$start/$set.doc.tran >  $doc_model_dir/$start/$set.doc.tran.evalmb
        

        # Compute LTCR score
        bash $discoures_terms_eval/LTCR/compute_ltcr_awe.sh $discoures_terms_eval/LTCR \
                                            $data_dir/$set.clean.$src \
                                            $data_dir/$set.document.bpe.$src \
                                            $doc_model_dir/$start/$set.tran \
                                            $doc_model_dir/$start \
                                            $doc_model_dir/$start/$set.LTCR.evalmb \
                                            $set.src tran $en_stopwords_file $zh_stopwords_file $test_device $python


        # Compute APT score
        bash $discoures_terms_eval/APT/APT.sh  $doc_model_dir/$start \
                                            $data_dir/$set.clean.$src \
                                            $doc_model_dir/$start/$set.tran \
                                            $data_dir/$set.clean.en0 \
                                            $set.$src $test_device $python2 $python

        # # Compute cohension
        # $python $discoures_terms_eval/Cohesion/compute_cohesion.py --sent_file $doc_model_dir/$start/$set.tran --device $test_device \
        #                             --doc_file $data_dir/$set.document.bpe.$src 2>&1 | tee $doc_model_dir/$start/$set.CHS.evalmb

        # # Compute coherence
        # $python $discoures_terms_eval/Coherence/compute_coherence.py --sent_file $doc_model_dir/$start/$set.tran  \
        #                             --doc_file $data_dir/$set.document.bpe.$src 2>&1 | tee $doc_model_dir/$start/$set.CHR.evalmb              
                        
                
        
        $python $d_bleu_prepare --doc_file $data_dir/$set.document.bpe.$src \
                                --sent_file  $doc_model_dir/$start/$set.tran \
                                --out_file  $doc_model_dir/$start/$set.doc.tran \
                                --mode sent2doc
        perl $multi_bleu $data_dir/$set.document.clean.$tgt <  $doc_model_dir/$start/$set.doc.tran >  $doc_model_dir/$start/$set.doc.tran.evalmb
    
        BLEU_d=$(cat  $doc_model_dir/$start/$set.doc.tran.evalmb | sed 's/,/\n/g' | grep 'BLEU' | sed -e 's/BLEU = //g' | sed -e 's/\n//g')
        LTCR_s=$(cat $doc_model_dir/$start/$set.LTCR.evalmb | grep 'LTCR = ' | sed 's/,/\n/g' | grep 'LTCR' | sed -e 's/LTCR = //g' | sed -e 's/\n//g')
        BLEU_s=$(cat $doc_model_dir/$start/$set.tran.evalmb | sed 's/,/\n/g' | grep 'BLEU' | sed -e 's/BLEU = //g' | sed -e 's/\n//g')
        if [ `echo "$LTCR_s > $best_ltcr" | bc` -eq 1 ]; then
            best_ltcr=$LTCR_s
            best_ltcr_checkpoint=$start
        fi

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
    echo -e "$set:\n best_itcr checkpoint: $best_ltcr_checkpoint, $best_ltcr\n best_bleu checkpoint: $best_bleu_checkpoint, $best_bleu\n" >> $doc_model_dir/results.evalmb
done
fi
