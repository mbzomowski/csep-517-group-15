# train a character-level model for Flores 200 dataset
# optimized for better accuracy and runtime

out_dir = 'work'
eval_interval = 500   # More frequent checkpoints to catch peak performance
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
max_iters = 3500        # Stop near peak performance (based on iter 3000 results)
lr_decay_iters = 3500   # Match max_iters
min_lr = 3e-5          # Lower minimum learning rate
beta2 = 0.95           # Standard value for larger models

warmup_iters = 500     # Shorter warmup for shorter training

# Performance optimizations
compile = True         # Enable PyTorch 2.0 compilation for speedup
dtype = 'bfloat16'     # Faster and more stable than float16

# Auto-fallback device selection with GPU-specific optimizations
import torch
if torch.cuda.is_available():
    device = 'cuda'
    # Check GPU capability for T4 optimization
    gpu_name = torch.cuda.get_device_name(0)
    print(f"CUDA detected - using GPU: {gpu_name}")
    
    if 'T4' in gpu_name or 'Tesla' in gpu_name:
        # T4-specific optimizations (Turing architecture)
        dtype = 'float16'  # T4 supports float16 better than bfloat16
        compile = False    # Disable compilation for T4 stability
        print("Optimizing for T4 GPU: using float16, disabling compilation")
    elif any(x in gpu_name for x in ['A100', 'A10', 'V100', 'RTX', 'RTX 30', 'RTX 40']):
        # Modern GPU optimizations
        dtype = 'bfloat16'
        compile = True
        print("Modern GPU detected: using bfloat16 with compilation")
    else:
        # Conservative settings for unknown GPUs
        dtype = 'float16'
        compile = False
        print("Unknown GPU: using conservative float16 settings")
        
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'  
    compile = False  # Disable compilation on MPS
    dtype = 'float32'  # Use float32 on MPS
    print("MPS detected - using Apple Silicon acceleration")
else:
    device = 'cpu'
    compile = False  # Disable compilation on CPU
    dtype = 'float32'  # Use float32 on CPU
    batch_size = 4  # Smaller batch for CPU
    gradient_accumulation_steps = 16  # Compensate with more accumulation
    print("No GPU detected - falling back to CPU (will be slower)")
