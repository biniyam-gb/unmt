#!/bin/bash
# Run with: bash run_all.sh   (NOT `sh run_all.sh` -- dash/sh can choke on
# things bash handles fine; this script assumes bash).
set -e  # stop on the first real error instead of silently continuing

export UNMT_WORK_DIR=/kaggle/working/unmt-en-fi
D="$UNMT_WORK_DIR/data"
C="$UNMT_WORK_DIR/checkpoints"
SPM="$D/spm_joint.model"

# Stage 0: data (takes a while -- Wikipedia streaming + LID filtering)
python3 data_prepare.py --max_sentences_per_lang 3000000
python3 train_tokenizer.py --vocab_size 32000
python3 binarize.py --spm_model "$SPM"

# Measure YOUR actual throughput before committing to a step count --
# see profile_throughput.py's output for realistic DAE/BT step budgets
# given your actual 12h-session / 30h-week Kaggle limits.
python3 profile_throughput.py --spm_model "$SPM"

# Stage A: embedding alignment (CPU-bound, no GPU needed)
python3 run_stage_a.py --spm_model "$SPM"

# Stage B: DAE pretraining -- replace --max_steps with the number
# profile_throughput.py suggested for your session/quota budget
torchrun --nproc_per_node=2 train_dae.py --spm_model "$SPM" --max_steps 60000

# Stage C: online back-translation -- same: use YOUR measured budget, not this default
torchrun --nproc_per_node=2 train_bt.py --spm_model "$SPM" --max_steps 150000

# Evaluation (only place FLORES+ ground truth is used)
python3 evaluate.py --spm_model "$SPM" --split devtest
