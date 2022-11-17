#!/bin/bash

#generate distilled data

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

module load cuda
module load cudnn

moses_scripts=/fs/clip-scratch/sweagraw/software/mosesdecoder/scripts
bpe_scripts_path=/fs/clip-scratch/sweagraw/software/subword-nmt/subword_nmt

exp_dir=experiments/exp-2
new_exp_dir=experiments/exp-6
out_dir=experiments/exp-6/data
mkdir -p ${out_dir}

tc_model=${exp_dir}/data/tc
bpe_model=${exp_dir}/data/bpe

for data in train dev; do
	if [ ! -f $out_dir/$data.txt ]; then
		fairseq-generate $exp_dir/data-bin \
			--gen-subset $data \
		    --path $exp_dir/checkpoints/checkpoint_best.pt \
		    --batch-size 128 --beam 5 --remove-bpe > $out_dir/$data.txt
    fi;

	grep ^S $out_dir/$data.txt | cut -f2- \
					   | $moses_scripts/recaser/detruecase.perl 2>/dev/null \
				       | $moses_scripts/tokenizer/detokenizer.perl -q -l en 2>/dev/null > $out_dir/$data.src
	grep ^H $out_dir/$data.txt | cut -f3- \
					   | $moses_scripts/recaser/detruecase.perl 2>/dev/null \
				       | $moses_scripts/tokenizer/detokenizer.perl -q -l en 2>/dev/null > $out_dir/$data.tgt

	# check if this source is same as original source
	for type in src tgt; do
		if [ ! -f ${out_dir}/${data}.tc.$type  ]; then
			$moses_scripts/recaser/truecase.perl \
		            -model $tc_model                       \
		            < $out_dir/${data}.$type                       \
		            > ${out_dir}/${data}.tc.$type 
		fi;
		if [ ! -f ${out_dir}/${data}.tc.bpe.$type  ]; then
		    python $bpe_scripts_path/apply_bpe.py \
	            --codes $bpe_model                  \
	            < ${out_dir}/${data}.tc.$type    \
	            > ${out_dir}/${data}.tc.bpe.$type 
        fi;
	done;
done;

if [ ! -f $out_dir/train.tc.bpe.clean.src ] && [ ! -f $out_dir/train.tc.bpe.clean.tgt ]; then
    echo "Cleaning data..."
    $moses_scripts/training/clean-corpus-n.perl -ratio 3 $out_dir/train.tc.bpe src tgt $out_dir/train.tc.bpe.clean 1 1000
fi;


if [ ! -d ${new_exp_dir}/data-bin ]; then
    fairseq-preprocess --source-lang src --target-lang tgt \
        --trainpref $out_dir/train.tc.bpe.clean --validpref $out_dir/dev.tc.bpe --testpref $exp_dir/data/test_en.tc.bpe \
        --joined-dictionary \
        --destdir ${new_exp_dir}/data-bin \
        --workers 20
fi;