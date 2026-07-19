export UNMT_WORK_DIR=/kaggle/working/unmt-en-fi

# Stage 0: data (takes a while -- Wikipedia streaming + LID filtering)
python3 data_prepare.py --max_sentences_per_lang 3000
python3 train_tokenizer.py
python3 binarize.py

# Measure YOUR actual throughput before committing to a step count
python3 profile_throughput.py

# Stage A: embedding alignment (CPU-bound, no GPU needed, a few hours for
# the skip-gram training depending on corpus size)
python3 run_stage_a.py

# Stage B: DAE pretraining 
torchrun --nproc_per_node=2 train_dae.py --max_steps 1000

# Stage C: online back-translation
torchrun --nproc_per_node=2 train_bt.py --max_steps 2000

# Evaluation (only place FLORES+ ground truth is used)
python3 evaluate.py --split devtest
