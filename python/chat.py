import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------
# Hyperparameters (Matched to training setup)
# ---------------------------------------------------------
block_size = 64
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------
# Vocabulary Mapping
# ---------------------------------------------------------
dataset_file = 'tiny_shakespeare.txt'
if not os.path.exists(dataset_file):
    print(f"Error: '{dataset_file}' not found.")
    sys.exit(1)

with open(dataset_file, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# ---------------------------------------------------------
# Model Definition
# ---------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=0.0, 
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def stream_generate(self, idx, max_new_tokens=250, temperature=0.7, top_k=20):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Print token immediately to the terminal
            char = itos[idx_next.item()]
            sys.stdout.write(char)
            sys.stdout.flush()
        print()


# ---------------------------------------------------------
# Interactive Prompt Execution
# ---------------------------------------------------------
if __name__ == '__main__':
    weights_path = 'mini_llm.pt'
    if not os.path.exists(weights_path):
        print(f"Error: Could not find '{weights_path}'.")
        sys.exit(1)

    model = MiniLLM().to(device)
    
    # Load checkpoint ignoring unused buffer keys
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights from {weights_path} onto {device}.\n")
    print("Enter a starting prompt (or 'quit' to exit):")

    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
            if not prompt.strip():
                prompt = "\n"

            tokens = encode(prompt)
            if not tokens:
                print("Prompt contains characters not present in training vocabulary.")
                continue

            x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

            print("\nModel:\n" + "-" * 30)
            model.stream_generate(x, max_new_tokens=250, temperature=0.7, top_k=20)
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nExiting.")
            break