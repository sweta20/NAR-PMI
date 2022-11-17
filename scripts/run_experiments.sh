#!/bin/bash

# Source Information 

# AR
bash scripts/train.sh -i 1 -j 1 -m seqseq -u ours-fil 

# AR with PMI loss -> the weights can be generated from get_pmi_weights.py
bash scripts/train.sh -i 2 -j 1 -m seqseq -u ours-fil -a " --weighted data/weight_matrix.npy --grade-map data/grade_map.pkl "

# EDITOR
bash scripts/train.sh -i 3 -j 1 -m nat -u ours-fil 

# EDITOR Refine finetune
bash scripts/train.sh -i 4 -j 1 -m nat -u ours-fil -r experiments/exp-2/checkpoints1/checkpoint_best.pt -a " --use-source 1 --skip-tokens-refine 2 --lr 0.0001 --max-epoch 20 " 
