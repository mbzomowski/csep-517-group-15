# train a character-level model for Flores 200 dataset
# optimized for better accuracy and runtime

out_dir = 'work'
eval_interval = 1000  # Less frequent evaluation for faster training
eval_iters = 100      # Fewer evaluation iterations
log_interval = 50     # Less frequent logging

# Save checkpoints when validation improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'flores200'
wandb_run_name = 'flores-gpt'

dataset = ''
gradient_accumulation_steps = 4  # Simulate larger batch sizes
batch_size = 16                  # Optimized batch size
block_size = 512                 # Larger context window for better understanding

# Larger model for complex multilingual data
n_layer = 12      # More layers for better representation
n_head = 12       # More attention heads
n_embd = 768      # Larger embedding dimension
dropout = 0.05    # Lower dropout for better learning on complex data

# Optimized learning rate schedule for better accuracy
learning_rate = 3e-4    # Lower for more stable training
max_iters = 15000       # Balanced training duration for M1
lr_decay_iters = 15000  # Match max_iters
min_lr = 3e-5          # Lower minimum learning rate
beta2 = 0.95           # Standard value for larger models

warmup_iters = 1500    # More warmup for stability

# Performance optimizations
compile = True         # Enable PyTorch 2.0 compilation for speedup
dtype = 'bfloat16'     # Faster and more stable than float16
