import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

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
            cur_window, idx_list = torch.LongTensor([[198]]).to(self.device), [198] # "\n" gpt-2 key=198  (1, 1)

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

