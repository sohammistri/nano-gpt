# Python version of the notebook in this folder

# import libs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from IPython.display import display, Markdown
import tiktoken
import wandb
import math
import inspect
from typing import Any, Callable
import os

# ==== CONFIG ====
CONFIG = {
    "batch_size": 64,
    "block_size": 8,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'),
    "emb_dim": 32,
    "n_heads": 4,
    "head_dim": 8,
    "n_layers": 1,
    "dropout":0.0,
    "n_iters": 5000,
    "warmup_iters": 500,
    "lr_decay_iters": 5000,
    "learning_rate": 3e-4, # max lr
    "min_lr": 3e-5, # min lr,
    "tokenizer_model": "gpt-2",
    "split_ratio": 0.8,
    "checkpoint_dir": "./checkpoint/",
    "always_save_checkpoint": False
}

# Helper functions

def call_with_matching_args(func: Callable, data_dict: dict) -> Any:
    """
    Call a function or method with only the arguments it accepts from a dictionary.
    Works with both regular functions and class methods.
    """
    params = inspect.signature(func).parameters
    # Remove 'self' parameter if it's a method
    if 'self' in params:
        params = dict(list(params.items())[1:])
    filtered_args = {k: data_dict[k] for k in params if k in data_dict}
    return func(**filtered_args)

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

def get_batch(tokens, block_size, batch_size):
    batch = torch.randint(0, len(tokens)-block_size, (batch_size,)) # B dimension array of random indices
    Xb = torch.stack([torch.LongTensor(tokens[i:i+block_size]) for i in batch], dim=0) # Create (B, T) dimension array
    yb = torch.stack([torch.LongTensor(tokens[i+1:i+block_size+1]) for i in batch], dim=0) # Create (B, T) dimension array
    return Xb, yb

@torch.no_grad()
def compute_loss(tokens, block_size, batch_size, model, device):
    loss_values = []
    for _ in range(100):
        Xb, yb = get_batch(tokens, block_size, batch_size)
        Xb, yb = Xb.to(device), yb.to(device)

        _, loss = model(Xb, yb)
        loss_values.append(loss.item())

    mean_loss = torch.FloatTensor(loss_values).mean().item()
    return mean_loss

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
class MHA(nn.Module):
    def __init__(self, emb_dim, block_size, n_heads, head_dim, dropout):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim

        # 1st LayerNorm
        self.ln1 = nn.LayerNorm(emb_dim)

        # first Linear to get from emb_dim --> 3 * n_heads*head_dim, to get k,q,v, then proj back to emb_dim
        self.c_proj = nn.Linear(emb_dim, 3 * n_heads * head_dim, bias=False)
        self.proj = nn.Linear(n_heads * head_dim, emb_dim)

        # 2nd LayerNorm
        self.ln2 = nn.LayerNorm(emb_dim)

        # finally thinking layer
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.think_dropout = nn.Dropout(dropout)

        # finally register the tril matrix
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        # get the shape
        B, T, C = x.shape

        # Layer norm
        ln_x = self.ln1(x)

        # Project and extract k,q,v
        c = self.c_proj(ln_x) # (B,T,C)  --> (B,T,3*nh*H)
        c = c.view(B, T, self.n_heads, 3 * self.head_dim) # (B,T,nh,3*H)
        k, q, v = torch.split(c, self.head_dim, dim=-1) # each of shape B,T,nh,H
        k, q, v = k.transpose(-3, -2), q.transpose(-3, -2), v.transpose(-3, -2) # B, nh, T, H

        # Get the attention weights
        wei = q @ k.transpose(-2, -1) * (self.head_dim**-0.50) # (B,nh,T,H) @ (B,nh,H,T) -> (B,nh,T,T)
        wei = wei.masked_fill(self.mask[:, :, :T, :T] == 0, -float("inf"))
        wei = torch.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        # Apply to v
        act = wei @ v # (B,nh,T,T) @ (B,nh,T,H) -> (B,nh,T,H)
        act = act.transpose(-3, -2) # B,T,nh,H
        act = act.contiguous().view(B, T, self.n_heads * self.head_dim)

        # Transform to emb_dim and skip connection
        act = self.proj(act) # (B, T,C)
        act = self.proj_dropout(act)
        act = x + act

        # Think and skip connections
        ln_act = self.ln2(act)
        out = self.ffn(ln_act) # (B,T,C)
        out = self.think_dropout(out)
        out = x + out # x shape (B,T,C)

        return out


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, block_size, emb_dim, n_layers, n_heads, head_dim, dropout, device):
        super().__init__()

        # helper variables
        self.block_size = block_size
        self.device = device

        # Embedding lookup table
        self.token_embbeding_table = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding_table = nn.Embedding(block_size, emb_dim)
        self.drop = nn.Dropout(dropout)

        # MHA head
        self.MHA = nn.Sequential(*[MHA(emb_dim, block_size, n_heads, head_dim, dropout) for _ in range(n_layers)])

        # Layernorm
        self.ln = nn.LayerNorm(emb_dim)

        # final linear layer
        self.lm_layer = nn.Linear(emb_dim, vocab_size)

        # init weights
        self.apply(self._init_weights)

        print(f"Number of parameters: {sum([p.numel() for p in self.parameters()])}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, targets=None):
        # x shape (B, T)
        B, T = x.shape

        token_emb = self.token_embbeding_table(x)
        pos_emb = self.position_embedding_table(torch.arange(0, T).to(self.device))
        emb = self.drop(token_emb + pos_emb)

        emb = self.MHA(emb)
        emb = self.ln(emb)
        logits = self.lm_layer(emb) # (B, T, V)

        loss = None

        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))

        return logits, loss

    def generate(self, tokenizer_model="gpt-2", max_tokens=1000):
        with torch.no_grad():
            cur_window, idx_list = torch.LongTensor([[0]]).to(self.device), [0] # (1, 1)

            for i in range(max_tokens):
                cur_window = cur_window[:, -self.block_size:] # (1, B)
                logits, _ = self.forward(cur_window) # (1,B,V)
                probs = torch.softmax(logits, dim=-1).squeeze(dim=0) # (B,V)
                idx = torch.multinomial(probs, num_samples=1, replacement=True)[-1].item()
                cur_window = torch.concat([cur_window, torch.LongTensor([[idx]]).view(1, 1).to(self.device)], dim=-1)
                idx_list.append(idx)

            tokenizer = tiktoken.encoding_for_model(tokenizer_model)
            generated_text = tokenizer.decode(idx_list)

            return generated_text

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
    project="nano-gpt-token-small",
    # track hyperparameters and run metadata
    config=CONFIG
)

def main():
    torch.random.manual_seed(1337)

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
    scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

    CONFIG["model"] = model
    CONFIG["optimizer"] = optimizer
    CONFIG["scheduler"] = scheduler
    CONFIG["eval_interval"] = CONFIG["n_iters"] // 10

    # train
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    out_dir = "-".join(str(key) + "_" + str(value) for key, value in CONFIG.items() if isinstance(value, (int, float)))
    out_dir = os.path.join(CONFIG["checkpoint_dir"], out_dir)
    CONFIG["out_dir"] = out_dir
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    call_with_matching_args(train, CONFIG)

if __name__=="__main__":
    main()



        


    


    