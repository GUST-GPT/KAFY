# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = "out-model"
eval_interval = 50  # keep frequent because we'll overfit
eval_iters = 10
log_interval = 5  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = "trajectoryGenerationDummy"
wandb_run_name = "mini-gpt"
init_from = "scratch"
dataset = ""
gradient_accumulation_steps = 1
batch_size = 12
block_size = 64  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 2
n_head = 4
n_embd = 64
dropout = 0.2

learning_rate = 1e-1  # with baby networks can afford to go a bit higher
max_iters = 100
lr_decay_iters = 100  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 10  # not super necessary potentially

# on macbook also add
device = "cpu"  # run on cpu only
compile = False  # do not torch compile the model
