export UNMT_WORK_DIR=/kaggle/working/unmt-en-fi

# Stage 0: data (takes a while -- Wikipedia streaming + LID filtering)
python3 data_prepare.py --max_sentences_per_lang 3000000
python3 train_tokenizer.py
python3 binarize.py

# Measure YOUR actual throughput before committing to a step count
python3 profile_throughput.py

# Stage A: embedding alignment (CPU-bound, no GPU needed, a few hours for
# the skip-gram training depending on corpus size)
python3 run_stage_a.py

# Stage B: DAE pretraining (use the step count profile_throughput.py suggested
# for your session/quota budget, not the config.py default -- see below)
torchrun --nproc_per_node=2 train_dae.py --max_steps <YOUR_NUMBER>

# Stage C: online back-translation (bootstraps from the DAE checkpoint
# automatically; also re-run with a higher --max_steps to resume)
torchrun --nproc_per_node=2 train_bt.py --max_steps <YOUR_NUMBER>

# Evaluation (only place FLORES+ ground truth is used)
python3 evaluate.py --split devtest

