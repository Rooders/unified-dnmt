#!/bin/bash

root=/share03/xxlyu
data_path=$root/data/LDC-zh2en-v2-data
MOSES=$root/toolkits/scripts
bpe_dir=$root/toolkits/subword_nmt

clean_data=1
LEARN_BPE=1
APPLY_BPE=1

test_sets=(nist02 nist03 nist04 nist05 nist06 nist08 nist)
train_sets=(train train.80wsent)

# cleaning data
if [ $clean_data -eq 1 ] ; then
   # clean train data
   for item in "${train_sets[@]}"; do
      raw_data_path=$data_path/$item
      echo 'Cleaning' $raw_data_path'.zh'
      cat $raw_data_path.zh \
        | perl $MOSES/tokenizer/lowercase.perl \
        | perl $MOSES/tokenizer/remove-non-printing-char.perl \
        | perl $MOSES/tokenizer/normalize-punctuation.perl -l zh \
        | perl $MOSES/tokenizer/tokenizer.perl -l zh -threads 40 \
        > $raw_data_path.clean.zh
      echo 'Cleaning' $raw_data_path'.en'
        cat $raw_data_path.en \
         | perl $MOSES/tokenizer/lowercase.perl \
         | perl $MOSES/tokenizer/remove-non-printing-char.perl \
         | perl $MOSES/tokenizer/normalize-punctuation.perl -l en \
         | perl $MOSES/tokenizer/tokenizer.perl -threads 40 \
         > $raw_data_path.clean.en
    done
    # clean dev, test data
    for item in "${test_sets[@]}"; do
      raw_data_path=$data_path/$item
      echo 'Cleaning' $raw_data_path'.zh'
      cat $raw_data_path.zh \
        | perl $MOSES/tokenizer/lowercase.perl \
        | perl $MOSES/tokenizer/tokenizer.perl -l zh -threads 40 \
        > $raw_data_path.clean.zh 
      for num in {0..3}; do
        echo 'Cleaning' $raw_data_path'.en'$num
        cat $raw_data_path.en$num \
         | perl $MOSES/tokenizer/lowercase.perl \
         | perl $MOSES/tokenizer/tokenizer.perl -threads 40 \
         > $raw_data_path.clean.en$num
      done
    done
fi

# learn bpe
if [ $LEARN_BPE -eq 1 ] ; then
    echo 'info: learn bpe from train data ...' 
    bpe_size=32000
    cat $data_path/train.clean.en $data_path/train.clean.zh \
        | python $bpe_dir/learn_bpe.py -s $bpe_size > $data_path/train.32k.bpe

fi

# apply bpe
if [ $APPLY_BPE -eq 1 ] ; then
    # echo 'info: apply bpe to test data ...' 
    lm_set=(en zh)
    # BPE train
    for pref in ${train_sets[@]}; do
      for lm in ${lm_set[@]}; do
        echo 'info: apply bpe to '$data_path/$pref.clean.$lm' data ...' 
        python $bpe_dir/apply_bpe.py -c $data_path/train.32k.bpe \
            < $data_path/$pref.clean.$lm \
            > $data_path/$pref.bpe.$lm
      done
    done
    # BPE test dev
    for pref in ${test_sets[@]}; do
      for lm in ${lm_set[@]}; do
        if [[ "${lm}" == "zh" ]]; then
          echo 'info: apply bpe to '$data_path/$pref.clean.$lm' data ...' 
          python $bpe_dir/apply_bpe.py -c $data_path/train.32k.bpe \
            < $data_path/$pref.clean.$lm \
            > $data_path/$pref.bpe.$lm
        else
          for num in {0..3}; do
            echo 'info: apply bpe to '$data_path/$pref.clean.$lm$num' data ...' 
            python $bpe_dir/apply_bpe.py -c $data_path/train.32k.bpe \
              < $data_path/$pref.clean.$lm$num \
              > $data_path/$pref.bpe.$lm$num
          done
        fi
      done
    done
fi


