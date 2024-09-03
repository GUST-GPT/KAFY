import os
import time
import math
import pickle
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext
import matplotlib.pyplot as plt  # Added for plotting
from model import GPTConfig, GPT
import json
from torch.distributed import init_process_group, destroy_process_group

# self.initialize_params()
# self.setup_ddp()
# self.setup_model()


class PretrainGPT:
    def __init__(self, config_path):
        """
        Start the pretraining process
        """
        # Load the configuration from JSON
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.initialize_params()
        self.setup_ddp()
        self.setup_model()
        self.train_gpt(self.config)

    def initialize_params(self):
        # Assign configuration values to instance variables
        self.out_dir = self.config.get("out_dir", "out")
        self.eval_interval = self.config.get("eval_interval", 2000)
        self.log_interval = self.config.get("log_interval", 1)
        self.eval_iters = self.config.get("eval_iters", 200)
        self.eval_only = self.config.get("eval_only", False)
        self.always_save_checkpoint = self.config.get("always_save_checkpoint", True)
        self.init_from = self.config.get("init_from", "scratch")
        self.wandb_log = self.config.get("wandb_log", False)
        self.wandb_project = self.config.get("wandb_project", "owt")
        self.wandb_run_name = self.config.get("wandb_run_name", "gpt2")
        self.dataset = self.config.get("dataset", "openwebtext")
        self.gradient_accumulation_steps = self.config.get(
            "gradient_accumulation_steps", 40
        )
        self.batch_size = self.config.get("batch_size", 12)
        self.block_size = self.config.get("block_size", 1024)
        self.n_layer = self.config.get("n_layer", 12)
        self.n_head = self.config.get("n_head", 12)
        self.n_embd = self.config.get("n_embd", 768)
        self.dropout = self.config.get("dropout", 0.0)
        self.bias = self.config.get("bias", False)
        self.learning_rate = self.config.get("learning_rate", 6e-4)
        self.max_iters = self.config.get("max_iters", 600000)
        self.weight_decay = self.config.get("weight_decay", 1e-1)
        self.beta1 = self.config.get("beta1", 0.9)
        self.beta2 = self.config.get("beta2", 0.95)
        self.grad_clip = self.config.get("grad_clip", 1.0)
        self.decay_lr = self.config.get("decay_lr", True)
        self.warmup_iters = self.config.get("warmup_iters", 2000)
        self.lr_decay_iters = self.config.get("lr_decay_iters", 600000)
        self.min_lr = self.config.get("min_lr", 6e-5)
        self.backend = self.config.get("backend", "nccl")
        self.device = self.config.get("device", "cuda")
        self.dtype = self.config.get(
            "dtype",
            (
                "bfloat16"
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else "float16"
            ),
        )
        self.compile = self.config.get("compile", True)
        self.seed = self.config.get("seed", 1337)

        self.iter_num = 0
        self.best_val_loss = 1e9

    def setup_ddp(self):
        self.ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        if self.ddp:
            init_process_group(backend=self.backend)
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
            assert self.gradient_accumulation_steps % self.ddp_world_size == 0
            self.gradient_accumulation_steps //= self.ddp_world_size
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1

        self.tokens_per_iter = (
            self.gradient_accumulation_steps
            * self.ddp_world_size
            * self.batch_size
            * self.block_size
        )
        if self.master_process:
            os.makedirs(self.out_dir, exist_ok=True)
        torch.manual_seed(self.seed + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        )

    def setup_model(self):
        # Initialize the model
        model_args = dict(
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            block_size=self.block_size,
            bias=self.bias,
            vocab_size=None,
            dropout=self.dropout,
        )
        if self.init_from == "scratch":
            print("Initializing a new model from scratch")
            self.setup_scratch_model(model_args)
        elif self.init_from == "resume":
            print(f"Resuming training from {self.out_dir}")
            self.setup_resume_model(model_args)
        elif self.init_from.startswith("trajGPT"):
            print(f"Initializing from trajGPT: {self.init_from}")
            self.setup_trajGPT_model(model_args)
        elif self.init_from.startswith("gpt2"):
            print(f"Initializing from OpenAI GPT-2 weights: {self.init_from}")
            self.setup_gpt2_model(model_args)

        if self.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.block_size)
            model_args["block_size"] = self.block_size

        self.model.to(self.device)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == "float16"))
        self.optimizer = self.model.configure_optimizers(
            self.weight_decay,
            self.learning_rate,
            (self.beta1, self.beta2),
            self.device_type,
        )
        if self.init_from == "resume":
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])
        self.checkpoint = None  # free up memory

        if self.compile:
            print("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

    def setup_scratch_model(self, model_args):
        meta_vocab_size = self.get_meta_vocab_size()
        model_args["vocab_size"] = (
            meta_vocab_size if meta_vocab_size is not None else 50304
        )
        gptconf = GPTConfig(**model_args)
        self.model = GPT(gptconf)

    def setup_resume_model(self, model_args):
        ckpt_path = os.path.join(self.out_dir, "ckpt.pt")
        self.checkpoint = torch.load(ckpt_path, map_location=self.device)
        checkpoint_model_args = self.checkpoint["model_args"]
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        self.model = GPT(gptconf)
        self.model.load_state_dict(self.checkpoint["model"])
        self.iter_num = self.checkpoint["iter_num"]
        self.best_val_loss = self.checkpoint["best_val_loss"]

    def setup_trajGPT_model(self, model_args):
        override_args = dict(dropout=self.dropout)
        self.model = GPT.from_pretrained_TrajGPT(
            self.init_from, override_args, model_args
        )
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(self.model.config, k)

    def setup_gpt2_model(self, model_args):
        override_args = dict(dropout=self.dropout)
        self.model = GPT.from_pretrained(self.init_from, override_args)
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(self.model.config, k)

    def get_meta_vocab_size(self):
        data_dir = os.path.join("data", self.dataset)
        meta_path = os.path.join(data_dir, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            print(f"found vocab_size = {meta['vocab_size']} (inside {meta_path})")
            return meta["vocab_size"]
        return None

    def train_gpt(self, config: dict):
        # Initial setup
        model = self.model
        block_size = config.get("block_size", self.model.config.block_size)
        device = config.get("device", "cuda")

        # Crop block size if necessary
        if block_size < self.model.config.block_size:
            self.model.crop_block_size(block_size)
            config["block_size"] = block_size

        self.model.to(device)

        # Initialize GradScaler for mixed precision
        dtype = config.get("dtype", "float32")
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

        # Configure optimizer
        optimizer = self.model.configure_optimizers(
            config.get("weight_decay", 0.01),
            config.get("learning_rate", 3e-5),
            (config.get("beta1", 0.9), config.get("beta2", 0.999)),
            config.get("device_type", "cuda"),
        )

        # Resume from checkpoint if required
        init_from = config.get("init_from", "scratch")
        if init_from == "resume":
            checkpoint = torch.load(config["checkpoint_path"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            self.model.load_state_dict(checkpoint["model"])
            config.update(checkpoint["config"])
            iter_num = checkpoint["iter_num"]
            best_val_loss = checkpoint["best_val_loss"]
        else:
            iter_num = 0
            best_val_loss = float("inf")

        # Compile model if required
        if config.get("compile", False):
            print("Compiling the model... (this may take a minute)")
            self.model = torch.compile(self.model)  # Requires PyTorch 2.0

        # Wrap self.model in DDP if necessary
        ddp = config.get("ddp", False)
        if ddp:
            self.model = DDP(self.model, device_ids=[config.get("ddp_local_rank", 0)])

        # Training loop setup
        eval_interval = config.get("eval_interval", 100)
        log_interval = config.get("log_interval", 10)
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        max_iters = config.get("max_iters", 10000)
        warmup_iters = config.get("warmup_iters", 100)
        lr_decay_iters = config.get("lr_decay_iters", 10000)
        min_lr = config.get("min_lr", 0)
        decay_lr = config.get("decay_lr", True)
        grad_clip = config.get("grad_clip", 1.0)
        wandb_log = config.get("wandb_log", False)
        wandb_project = config.get("wandb_project", "")
        wandb_run_name = config.get("wandb_run_name", "")
        eval_only = config.get("eval_only", False)
        always_save_checkpoint = config.get("always_save_checkpoint", False)
        out_dir = config.get("out_dir", "./")

        # Initialize loss tracking
        training_losses = []
        validation_losses = []

        # Initialize logging if using wandb
        if wandb_log:
            import wandb

            wandb.init(project=wandb_project, name=wandb_run_name, config=config)

        # Function to estimate loss
        @torch.no_grad()
        def estimate_loss():
            out = {}
            model.eval()
            for split in ["train", "val"]:
                losses = torch.zeros(config["eval_iters"])
                for k in range(config["eval_iters"]):
                    X, Y = self.get_batch(split)
                    with torch.cuda.amp.autocast(enabled=(dtype == "float16")):
                        logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            model.train()
            return out

        # Learning rate decay function
        def get_lr(it):
            if it < warmup_iters:
                return config["learning_rate"] * it / warmup_iters
            if it > lr_decay_iters:
                return min_lr
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (config["learning_rate"] - min_lr)

        # Start the training loop
        t0 = time.time()
        raw_model = self.model.module if ddp else self.model
        running_mfu = -1.0
        while iter_num <= max_iters:

            # Set learning rate for the current iteration
            lr = get_lr(iter_num) if decay_lr else config["learning_rate"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Evaluate loss and write checkpoints
            if iter_num % eval_interval == 0:
                losses = estimate_loss()
                print(
                    f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if wandb_log:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,
                        }
                    )
                if losses["val"] < best_val_loss or always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_args": config,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "config": config,
                        }
                        torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                        print(f"Checkpoint saved to {os.path.join(out_dir, 'ckpt.pt')}")

            if iter_num == 0 and eval_only:
                break

            # Forward-backward-update
            X, Y = self.get_batch("train")
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    self.model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1
                    )
                with torch.cuda.amp.autocast(enabled=(dtype == "float16")):
                    logits, loss = self.model(X, Y)
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()

            # Clip gradients if required
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0:
                lossf = loss.item() * gradient_accumulation_steps
                if iter_num >= 5:
                    mfu = raw_model.estimate_mfu(
                        config["batch_size"] * gradient_accumulation_steps, dt
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                print(
                    f"Iter {iter_num}: train loss {lossf:.4f}, val loss {losses['val']:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
                )

            iter_num += 1

            # Termination condition
            if iter_num > max_iters:
                break

        if ddp:
            destroy_process_group()

    def initialize_model(self, config):
        # Your model initialization logic here
        pass

    def initialize_optimizer(self, config, params):
        # Your optimizer initialization logic here
        pass

    # """
    def get_batch(self, split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        # ix = torch.randint(0, (len(data) - block_size) // (block_size * 2) + 1, (batch_size,))
        # ix = ix * (block_size * 2)
        x = torch.stack(
            [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
                for i in ix
            ]
        )
        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y


if __name__ == "__main__":
    with open("test_config.json", "r") as file:
        configs = json.load(file)
    PretrainGPT(configs)
