# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'work'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = ''
wandb_run_name = 'mini-gpt'

dataset = ''
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

learning_rate = 3e-4 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
