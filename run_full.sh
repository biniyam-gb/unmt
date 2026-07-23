#!/bin/bash
set -e

export UNMT_WORK_DIR=/kaggle/working/unmt-en-fi
D="$UNMT_WORK_DIR/data"
C="$UNMT_WORK_DIR/checkpoints"
SPM="$D/spm_joint.model"

python3 data_prepare.py --max_sentences_per_lang 500000
python3 train_tokenizer.py --vocab_size 8000
python3 binarize.py --spm_model "$SPM"
python3 profile_throughput.py --spm_model "$SPM"
python3 run_stage_a.py --spm_model "$SPM"
torchrun --nproc_per_node=2 train_dae.py --spm_model "$SPM" --max_steps 80000
torchrun --nproc_per_node=2 train_bt.py --spm_model "$SPM" --max_steps 14500
python3 evaluate.py --spm_model "$SPM" --split devtest
