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
    "test_classify_data/wmt14_fr_en_test"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=fr
tgt=en
lang=fr-en
dest=data-bin/wmt14_en_fr_raw_sm
prep=test_classify_data/wmt14_fr_en_test_processed
tmp=$prep/tmp
orig=test_classify_data/orig

FILES=(
    $orig/src.fr
    $orig/target.en
    $orig/ht_mt_target.en
)

echo  "checking if existing files exist"
if [ -e test_classify_data/orig ]; then
    echo "deleting test_classify_data/orig."
    rm -r test_classify_data/orig
fi

if [ -e test_classify_data/wmt14_fr_en_test_processed ]; then
    echo "deleting test_classify_data/wmt14_fr_en_test_processed"
    rm -r test_classify_data/wmt14_fr_en_test_processed
fi

if [ -e test_classify_data/wmt14_fr_en_test_processed/tmp ]; then
    echo "deleting test_classify_data/wmt14_fr_en_test_processed/tmp"
    rm -r test_classify_data/wmt14_fr_en_test_processed/tmp
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

cp test_classify_data/wmt14_fr_en_test/src.fr $orig/src.fr
cp test_classify_data/wmt14_fr_en_test/target.en $orig/target.en
cp test_classify_data/wmt14_fr_en_test/ht_mt_target.en $orig/ht_mt_target.en

# echo "pre-processing src data..."
# grep '<seg id' $orig/src.fr | \
#     sed -e 's/<seg id="[0-9]*">\s*//g' | \
#     sed -e 's/\s*<\/seg>\s*//g' | \
#     sed -e "s/\’/\'/g" | \
# perl $TOKENIZER -threads 8 -a -l fr > $tmp/src.fr
# echo ""

echo "pre-processing src data..."
cat $orig/src.fr | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l $l >> $tmp/src.fr
echo ""

# cat $orig/$f.$l | \
# perl $NORM_PUNC $l | \
# perl $REM_NON_PRINT_CHAR | \
# perl $TOKENIZER -threads 8 -a -l $l >> $tmp/src.fr

echo "pre-processing target data..."
cat $orig/target.en | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
perl $TOKENIZER -threads 8 -a -l en >> $tmp/target.en
echo ""

echo "pre-processing ht_mt_target data..."
cat $orig/ht_mt_target.en | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
perl $TOKENIZER -threads 8 -a -l en >> $tmp/ht_mt_target.en
echo ""

BPE_CODE=test_classify_data/wmt14_fr_en_test/code
# echo "learn_bpe.py on ${TRAIN}..."
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
# HERE SHOULD BE USING THE BPE CODE GENERATED DURING THE TRAINING STAGE, PLEASE VERIFY AGAIN

echo "apply_bpe.py to src.fr ..."
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/src.fr > $tmp/src.bpe.fr

echo "apply_bpe.py to target.en ..."
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/target.en > $tmp/target.bpe.en

echo "apply_bpe.py to ht_mt_target.en ..."
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/ht_mt_target.en > $tmp/ht_mt_target.bpe.en

# perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
# perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

cp $tmp/src.bpe.fr $prep/src.fr
cp $tmp/target.bpe.en $prep/target.en
cp $tmp/ht_mt_target.bpe.en $prep/ht_mt_target.en

cp $prep/src.fr $dest/src.fr
cp $prep/target.en $dest/target.en
cp $prep/ht_mt_target.en $dest/ht_mt_target.en
cp test_classify_data/wmt14_fr_en_test/ht_mt_label $dest/ht_mt_label