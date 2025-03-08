import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
import tiktoken
import os
import time
import math 
import inspect



@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layers: int = 12 # number of layers
    n_heads: int = 12 # number of heads
    n_embed: int = 768 # embedding dimension

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embed % config.n_heads == 0

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_RESIDUAL_SCALE = True
        # NOT NEEDED after FlashAttention
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).\
        #                      view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x) # (B, T, 3*C)
        q, k, v = torch.split(qkv, C, dim=2) # (B, T, C)

        q = q.view(B, T, self.config.n_heads, C//self.config.n_heads).transpose(1, 2) # (B, nh, T, ch)
        k = k.view(B, T, self.config.n_heads, C//self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, C//self.config.n_heads).transpose(1, 2)

        # Flash attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # attn_scores = q @ k.transpose(-2, -1) * ((k.shape[-1])**-0.5) # (B, nh, T, T)
        # attn_scores = attn_scores.masked_fill(self.bias[:, :, :T, :T] == 0, -float("inf")) # (B, nh, T, T)
        # attn_scores = torch.softmax(attn_scores, dim=-1)
        # out = attn_scores @ v # (B, nh, T, ch)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)

        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_RESIDUAL_SCALE = True

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight tying scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self.init_weights)

    def forward(self, idx, targets=None):
        # idx: (B, T)
        B, T = idx.shape
        token_emb = self.transformer.wte(idx) # (B, T, C)
        pos_emb = self.transformer.wpe(torch.arange(0, T).long().to(device)) # (T, C)
        emb = token_emb + pos_emb

        for layer in self.transformer.h:
            emb = layer(emb)
        emb = self.transformer.ln_f(emb)

        logits = self.lm_head(emb)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, self.config.vocab_size), targets.view(B * T))

        return logits, loss
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_RESIDUAL_SCALE"):
                std *= (self.config.n_layers * 2) ** (-0.50)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(self, tokens ,max_length, num_return_sequences, k=50):
        # make it (B, T)
        tokens = tokens.unsqueeze(0).expand(num_return_sequences, -1).to(device)

        for _ in range(max_length):
            x = tokens[:, -self.config.block_size:] # (B, T)
            logits, _ = self(x) # (B, T, V)

            logits = logits[:, -1, :] # (B, V)

            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, k, dim=-1)

            ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
            sampled_indices = torch.gather(topk_indices, -1, ix) # (B, 1)

            tokens = torch.cat((tokens, sampled_indices), dim=1)

        return tokens
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layers, n_heads and n_embed are determined from model_type
        config_args = {
            'gpt2':         dict(n_layers=12, n_heads=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layers=24, n_heads=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layers=36, n_heads=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layers=48, n_heads=25, n_embed=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model, config
    
    # set weight decay for only 2d params
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
class DataloaderLite:
    def __init__(self, B, T, rank, world_size, dataset="tiny_shakespeare", dir=None):
        self.B = B
        self.T = T
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

        if self.dataset == "tiny_shakespeare":
            assert dir is not None
            with open(os.path.join(dir, "input.txt"), "r") as file:
                self.txt = file.read()
            self.tokens = tokenizer.encode(self.txt)
            self.tokens = torch.tensor(self.tokens).long()

        self.cur_idx = self.B * self.T * self.rank

    def sample(self):
        sampled_tokens = self.tokens[self.cur_idx:self.cur_idx + self.B * self.T + 1]
        x = sampled_tokens[:-1].view(self.B, self.T)
        y = sampled_tokens[1:].view(self.B, self.T)

        self.cur_idx += self.B * self.T * self.world_size
        # reset if overflow
        if self.cur_idx + self.B * self.T * self.world_size + 1 > self.tokens.shape[0]:
            self.cur_idx = self.B * self.T * self.rank

        return x, y

def cosine_scheduler(it, min_lr, max_lr, warmup_steps, max_steps, base_lr):
    if it < warmup_steps:
        lr = max_lr * ((it + 1) / warmup_steps)
    elif it > max_steps:
        lr = min_lr
    else:
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(decay_ratio * math.pi)) # starts with 1, ends at 0
        lr = min_lr + coeff * (max_lr - min_lr)

    return lr / base_lr

def train(model, dataloader, n_iters, grad_accum_steps, optimizer, scheduler):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) # prevent underflow of grads in mixed precision training

    for iter in range(n_iters):
        t0 = time.time()

        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = dataloader.sample()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                logits, loss = model(x, y)
                # print(logits.dtype, loss.dtype) logits: bfloat16, loss: float32
                loss /= grad_accum_steps
                loss_accum += loss.detach()
                scaler.scale(loss).backward()
                if ddp:
                    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # Collect all losses from all GPUs and make them avg

        scaler.unscale_(optimizer)
        # Grad scaling
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()

        tokens_processed = x.numel() * grad_accum_steps * ddp_world_size
        tokens_processed_per_sec = tokens_processed / (t1 - t0)

        if master_process:
            print(f"Iteration: {iter:3d} | Train Loss: {loss_accum.item():.4f} | lr: {scheduler.get_last_lr()[0]:.4e} | Grad norm: {norm:.4f} | Time per iter: {(t1 - t0)*1000:.2f} ms | Tokens Processed per sec: {int(tokens_processed_per_sec)} toks/sec")

def main():
    config = GPTConfig(vocab_size=50304)

    total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
    B = 16
    T = config.block_size
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # FP32 -> TF32 matrix muls
    torch.set_float32_matmul_precision('high')

    dataloader = DataloaderLite(B=B, T=T, dir="../data/tiny-shakespeare/", rank=ddp_rank, world_size=ddp_world_size)

    model = GPT(config)
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    n_iters = 50
    base_lr = 6e-4
    max_lr = 6e-4
    min_lr = max_lr * 0.10
    warmup_steps = 10
    max_steps = 50
    
    optimizer = raw_model.configure_optimizers(weight_decay=0.10, learning_rate=base_lr, device=device)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: cosine_scheduler(epoch, min_lr, max_lr, warmup_steps, max_steps, base_lr))

    if master_process:
        print(f"Training on device {device}")
    train(model, dataloader, n_iters=n_iters, grad_accum_steps=grad_accum_steps, optimizer=optimizer, scheduler=scheduler)

    if ddp:
        dist.destroy_process_group()
    # crush a single batch
    # x, y = dataloader.sample()
    # x, y = x.to(device), y.to(device)


    # for i in range(n_iters):
    #     optimizer.zero_grad(set_to_none=True)
    #     logits, loss = model(x, y)
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Iteration: {i}|Train Loss: {loss.item()}")

    
    # print("Starting loading gpt2....")
    # print(f"Device: {device}")
    # model, config = GPT.from_pretrained(model_type="gpt2")
    # model.to(device)
    # model.eval()
    # print("Didn't crash yay!!")

    # print("-" * 50)

    # print("Lets sample some tokens:")
    # txt = "Hello, I'm a language model,"
    # tokens = tokenizer.encode(txt)
    # tokens = torch.tensor(tokens).long()
    # print(f"Input seq: {txt}")
    # out_tokens = model.generate(tokens, max_length=30, num_return_sequences=5)
    # out_tokens = out_tokens.cpu().detach().numpy()
    # out_sentences = tokenizer.decode_batch(out_tokens)
    # for sentence in out_sentences:
    #     print(sentence)

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    tokenizer = tiktoken.encoding_for_model("gpt2")

    use_amp = True

    # Distributed check
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        torch.cuda.manual_seed(42)
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        torch.manual_seed(42)
        if device == "mps":
            torch.mps.manual_seed(42)
        elif device == "cuda":
            torch.cuda.manual_seed(42)


    main()