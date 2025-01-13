# Python version of the notebook in this folder

# import libs
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from IPython.display import display, Markdown
import tiktoken
import wandb
import math
import os
from .model import NanoGPT
from .utils import load_config, call_with_matching_args, get_batch, compute_loss

# ==== CONFIG ====
CONFIG = load_config(config_path="config.yml")

# Helper functions

def get_data(data_path, tokenizer_model="gpt-2", split_ratio=0.8):
    # get the raw tiny shakespeare dataset and split into train and val sets
    with open(data_path) as file:
        data = file.read()

    tokenizer = tiktoken.encoding_for_model(tokenizer_model)
    # Set the vocab size in config
    CONFIG["vocab_size"] = tokenizer.n_vocab
    tokens = tokenizer.encode(data)

    split_idx = int(split_ratio * len(tokens))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    return train_tokens, val_tokens

def train(train_tokens, val_tokens, model, optimizer, scheduler, device, block_size, batch_size, n_iters, eval_interval, out_dir, always_save_checkpoint):
    # train_lossi, val_lossi = [], []
    best_val_loss = float("inf")

    for i in range(n_iters):
        model.train()
        Xb, yb = get_batch(train_tokens, block_size, batch_size)
        Xb, yb = Xb.to(device), yb.to(device)

        # forward
        _, loss = model(Xb, yb)

        # set grads to zero
        optimizer.zero_grad(set_to_none=True)

        # do backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # scheduler step
        if scheduler:
            scheduler.step()

        wandb.log({"learning_rate": scheduler.get_last_lr()[0]})

        if (i % eval_interval == 0) or (i == n_iters - 1):
            model.eval()
            train_loss = compute_loss(train_tokens, block_size, batch_size, model, device)
            val_loss = compute_loss(val_tokens, block_size, batch_size, model, device)

            print(f"Step {i:4d} | Learning Rate: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                if i > 0:
                    checkpoint = {
                        'step': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'loss': val_loss,
                        'config': CONFIG  # Save configuration as well
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
             # log metrics to wandb
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        # break


# Model classes
# lr scheduler with cosine decay copied from here: https://github.com/karpathy/nanoGPT/blob/93a43d9a5c22450bbf06e78da2cb6eeef084b717/train.py#L230
def get_lr_multiplier(it):
    # 1) linear warmup for warmup_iters steps
    if it < CONFIG["warmup_iters"]:
        return (it + 1) / (CONFIG["warmup_iters"] + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > CONFIG["lr_decay_iters"]:
        return CONFIG["min_lr"] / CONFIG["learning_rate"]
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - CONFIG["warmup_iters"]) / (CONFIG["lr_decay_iters"] - CONFIG["warmup_iters"])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return (CONFIG["min_lr"] + coeff * (CONFIG["learning_rate"] - CONFIG["min_lr"])) / CONFIG["learning_rate"]


# Wandb init
wandb.init(
    # set the wandb project where this run will be logged
    project="nano-gpt-token-tiny-shakespeare-large",
    # track hyperparameters and run metadata
    config=CONFIG
)

def main():
    torch.random.manual_seed(1337)

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    sorted_config = sorted(CONFIG.items())
    out_dir = "-".join(str(key) + "_" + str(value) for key, value in sorted_config if isinstance(value, (int, float)))
    out_dir = os.path.join(CONFIG["checkpoint_dir"], out_dir)
    CONFIG["out_dir"] = out_dir
    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    # data
    CONFIG["data_path"] = "../data/tiny-shakespeare/input.txt"
    train_tokens, val_tokens = call_with_matching_args(get_data, CONFIG)
    CONFIG["train_tokens"] = train_tokens
    CONFIG["val_tokens"] = val_tokens

    print(f"Train size: {len(train_tokens)}, Val Size: {len(val_tokens)}")

    # model specific params
    model = call_with_matching_args(NanoGPT, CONFIG)
    model = model.to(CONFIG["device"])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    if not CONFIG["fixed_lr"]:
        scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    else:
        scheduler = None

    CONFIG["model"] = model
    CONFIG["optimizer"] = optimizer
    CONFIG["scheduler"] = scheduler
    CONFIG["eval_interval"] = CONFIG["n_iters"] // 10

    # train
    call_with_matching_args(train, CONFIG)

if __name__=="__main__":
    main()



        


    


    