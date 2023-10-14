#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

if [ -e mosesdecoder ]; then
    echo "mosesdecoder already exists, skipping download"
else
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git
fi

if [ -e subword-nmt ]; then
    echo "subword-nmt already exists, skipping download"
else
    echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
    git clone https://github.com/rsennrich/subword-nmt.git
fi

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

CORPORA=(
    "test_classify_data/wmt14_en_fr_test"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=fr
lang=en-fr
dest=data-bin/wmt14_en_fr_2023/wmt14_en_fr_raw_2023_50k
prep=test_classify_data/wmt14_en_fr_test_processed
tmp=$prep/tmp
orig=test_classify_data/orig

FILES=(
    $orig/src.en
    $orig/target.fr
    $orig/ht_mt_target.fr
)

echo  "checking if existing files exist"
if [ -e test_classify_data/orig ]; then
    echo "deleting test_classify_data/orig."
    rm -r test_classify_data/orig
fi

if [ -e test_classify_data/wmt14_en_fr_test_processed ]; then
    echo "deleting test_classify_data/wmt14_en_fr_test_processed"
    rm -r test_classify_data/wmt14_en_fr_test_processed
fi

if [ -e test_classify_data/wmt14_en_fr_test_processed/tmp ]; then
    echo "deleting test_classify_data/wmt14_en_fr_test_processed/tmp"
    rm -r test_classify_data/wmt14_en_fr_test_processed/tmp
fi

mkdir -p $orig $prep $tmp

# test_classify_data/
#   -orig/
#   -prep(wmt14_fr_en_test_processed)/
#       --tmp (wmt14_fr_en_test_processed/tmp)/
#   -wmt14_fr_en_test
#       --src.fr
#       --target.en
#       --ht_mt_target.en

# cd $orig

cp test_classify_data/wmt14_en_fr_test/src.en $orig/src.en
cp test_classify_data/wmt14_en_fr_test/target.fr $orig/target.fr
cp test_classify_data/wmt14_en_fr_test/ht_mt_target.fr $orig/ht_mt_target.fr

# echo "pre-processing src data..."
# grep '<seg id' $orig/src.fr | \
#     sed -e 's/<seg id="[0-9]*">\s*//g' | \
#     sed -e 's/\s*<\/seg>\s*//g' | \
#     sed -e "s/\’/\'/g" | \
# perl $TOKENIZER -threads 8 -a -l fr > $tmp/src.fr
# echo ""

echo "pre-processing src data..."
cat $orig/src.en | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l en >> $tmp/src.en
echo ""

# cat $orig/$f.$l | \
# perl $NORM_PUNC $l | \
# perl $REM_NON_PRINT_CHAR | \
# perl $TOKENIZER -threads 8 -a -l $l >> $tmp/src.fr

echo "pre-processing target data..."
cat $orig/target.fr | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
perl $TOKENIZER -threads 8 -a -l fr >> $tmp/target.fr
echo ""

echo "pre-processing ht_mt_target data..."
cat $orig/ht_mt_target.fr | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
perl $TOKENIZER -threads 8 -a -l fr >> $tmp/ht_mt_target.fr
echo ""

BPE_CODE=test_classify_data/wmt14_en_fr_test/code
# echo "learn_bpe.py on ${TRAIN}..."
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
# HERE SHOULD BE USING THE BPE CODE GENERATED DURING THE TRAINING STAGE, PLEASE VERIFY AGAIN

echo "apply_bpe.py to src.en ..."
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/src.en > $tmp/src.bpe.en

echo "apply_bpe.py to target.fr ..."
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/target.fr > $tmp/target.bpe.fr

echo "apply_bpe.py to ht_mt_target.fr ..."
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/ht_mt_target.fr > $tmp/ht_mt_target.bpe.fr

# perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
# perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

cp $tmp/src.bpe.en $prep/src.en
cp $tmp/target.bpe.fr $prep/target.fr
cp $tmp/ht_mt_target.bpe.fr $prep/ht_mt_target.fr

cp $prep/src.en $dest/src.en
cp $prep/target.fr $dest/target.fr
cp $prep/ht_mt_target.fr $dest/ht_mt_target.fr
cp test_classify_data/wmt14_en_fr_test/ht_mt_label $dest/ht_mt_label