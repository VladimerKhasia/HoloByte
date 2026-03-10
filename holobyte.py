#@title HoloByte: Continuous Hyperspherical Distillation for Tokenizer-Free Modeling
import os
import gc
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tiktoken
from datasets import load_dataset

# =========================================================================
# 0. STRICT REPRODUCIBILITY
# =========================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# =========================================================================
# EXPERIMENT CONFIGURATION
# =========================================================================
BATCH_SIZE = 4          
SEQ_LEN = 1024          
D_MODEL = 768           
W_CAPACITY = 8          
MAX_STEPS = 20_000     
EVAL_INTERVAL = 500
LEARNING_RATE = 6e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Initializing V9 Hyperspherical Distillation on {DEVICE}")

# =========================================================================
# 1. STREAMING DATASET & DATALOADER
# =========================================================================
def get_fineweb_text(num_chars=5_000_000):
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    text = ""
    for item in ds:
        text += item['text'] + "<|endoftext|>"
        if len(text) > num_chars: break
    return text[:num_chars]

text_data = get_fineweb_text()
split_idx = int(len(text_data) * 0.9)
train_text, val_text = text_data[:split_idx], text_data[split_idx:]

class DataLoaderLite:
    def __init__(self, text_subset, is_hss):
        self.is_hss = is_hss
        if is_hss:
            self.data = torch.tensor(list(text_subset.encode('utf-8')), dtype=torch.long, device=DEVICE)
        else:
            enc = tiktoken.get_encoding("gpt2")
            tokens = enc.encode(text_subset, allowed_special={'<|endoftext|>'})
            self.data = torch.tensor(tokens, dtype=torch.long, device=DEVICE)

    def get_batch(self):
        if self.is_hss:
            bytes_needed = (SEQ_LEN + 1) * W_CAPACITY
            ix = torch.randint(len(self.data) - bytes_needed, (BATCH_SIZE,))
            chunked = torch.stack([self.data[i : i + bytes_needed] for i in ix]).view(BATCH_SIZE, SEQ_LEN + 1, W_CAPACITY)
            return chunked[:, :-1, :], chunked[:, 1:, :]  
        else:
            ix = torch.randint(len(self.data) - SEQ_LEN - 1, (BATCH_SIZE,))
            return torch.stack([self.data[i : i + SEQ_LEN] for i in ix]), \
                   torch.stack([self.data[i + 1 : i + SEQ_LEN + 1] for i in ix])

# =========================================================================
# 2. UNITARY MATH HELPER
# =========================================================================
def apply_unitary_rotation(x, cos, sin, reverse=False):
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    if reverse: sin = -sin 
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# =========================================================================
# 3. MODDED CONTINUOUS GPT ARCHITECTURE
# =========================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.c_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.n_head = 12

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(D_MODEL, dim=2)
        k, q, v =[t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (k, q, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class ModdedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL, bias=False), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL, bias=False)
        )
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

class ContinuousGPT(nn.Module):
    def __init__(self, is_hss, n_layer):
        super().__init__()
        self.is_hss = is_hss
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([ModdedBlock() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(D_MODEL)

        if self.is_hss:
            # Standard init, but strictly normalized in forward pass
            self.byte_manifold = nn.Parameter(torch.randn(256, D_MODEL))
            self.hss_head = nn.Linear(D_MODEL, D_MODEL, bias=False)
            
            # Learnable Temperature for Cosine Similarity Unbinding (Starts at ~14.3)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            
            # Autoregressive sequence start vector
            self.chunk_start = nn.Parameter(torch.randn(1, 1, D_MODEL))

            # Extremely lightweight causal attention for intra-chunk autoregression (W=8 is almost free)
            self.micro_decoder = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=4, dim_feedforward=2*D_MODEL, 
                batch_first=True, norm_first=True
            )
            # Causal mask for the 8-byte micro-sequence
            self.register_buffer(
                'causal_mask', 
                nn.Transformer.generate_square_subsequent_mask(W_CAPACITY)
            )
            
            inv_freq = 1.0 / (10000 ** (torch.arange(0, D_MODEL, 2, device=DEVICE).float() / D_MODEL))
            positions = torch.arange(W_CAPACITY, device=DEVICE).float()
            angles = torch.einsum('i,j->ij', positions, inv_freq)
            self.register_buffer('mux_cos', torch.cos(angles))
            self.register_buffer('mux_sin', torch.sin(angles))
        else:
            self.tok_emb = nn.Embedding(50257, D_MODEL)
            self.lm_head = nn.Linear(D_MODEL, 50257, bias=False)
            self.tok_emb.weight = self.lm_head.weight 

    def hss_encode(self, bytes_tensor):
        """Strict Unit-Hypersphere Encoding"""
        # 1. Map to unit sphere to prevent weight explosion
        manifold_norm = F.normalize(self.byte_manifold, p=2, dim=-1)
        vectors = manifold_norm[bytes_tensor] 
        rotated = apply_unitary_rotation(vectors, self.mux_cos, self.mux_sin, reverse=False)
        return rotated.sum(dim=2) / math.sqrt(W_CAPACITY)

    @torch.no_grad()
    def decode_vector(self, predicted_vector, temperature=0.8):
        expanded = predicted_vector.unsqueeze(0).expand(W_CAPACITY, -1)
        unbound = apply_unitary_rotation(expanded, self.mux_cos, self.mux_sin, reverse=True)
        unbound = unbound.unsqueeze(0) # [1, W, D]
        
        generated_chunk =[]
        manifold_norm = F.normalize(self.byte_manifold, p=2, dim=-1)
        
        # Current prefix embedding starts with the chunk_start token
        current_embs = self.chunk_start.clone()
        
        for i in range(W_CAPACITY):
            # Pad the current_embs to length W to match unbound spatial dimensions (causal mask ignores future)
            pad_len = W_CAPACITY - current_embs.size(1)
            padded_embs = F.pad(current_embs, (0, 0, 0, pad_len))
            
            combined_signal = unbound + padded_embs
            
            # Pass through 1-layer micro decoder
            cleaned = self.micro_decoder(combined_signal, src_mask=self.causal_mask, is_causal=True)
            
            # Take the prediction for the current step `i`
            step_vector = F.normalize(cleaned[:, i, :].float(), p=2, dim=-1)
            manifold_norm_fp32 = F.normalize(self.byte_manifold.float(), p=2, dim=-1)
            
            byte_logits = torch.matmul(step_vector, manifold_norm_fp32.transpose(0, 1)) * self.logit_scale.float().exp()
            probs = F.softmax(byte_logits / temperature, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1).item()
            
            generated_chunk.append(next_byte)
            
            # Append newly generated byte's embedding to prefix
            next_emb = manifold_norm[next_byte].view(1, 1, -1)
            current_embs = torch.cat([current_embs, next_emb], dim=1)

        return bytes(generated_chunk)

    def forward(self, idx, targets=None):
        if self.is_hss:
            x_sketches = self.hss_encode(idx) 
            B, T = x_sketches.size()[:2]
            x = x_sketches + self.pos_emb(torch.arange(0, T, dtype=torch.long, device=DEVICE))
            for block in self.blocks: x = block(x)
            logits_continuous = self.hss_head(self.ln_f(x)) 
            
            loss = None
            if targets is not None:
                # Holographic Latent Distillation
                # We mathematically know what the target continuous vector MUST be.
                with torch.no_grad():
                    perfect_target_sketches = self.hss_encode(targets)
                
                # Hyperspherical Unbinding (Cosine Geometry)
                expanded_logits = logits_continuous.unsqueeze(2).expand(-1, -1, W_CAPACITY, -1)
                unbound_spatial = apply_unitary_rotation(expanded_logits, self.mux_cos, self.mux_sin, reverse=True)
                
                # Causal Micro-Decoder with Teacher Forcing 
                B, T, W, D = unbound_spatial.size()
                unbound_flat = unbound_spatial.view(B * T, W, D)
                
                # Embed the true targets (shifted right by 1 for causality)
                manifold_norm = F.normalize(self.byte_manifold, p=2, dim=-1)
                target_embs = manifold_norm[targets].view(B * T, W, D)
                
                shifted_embs = torch.cat([
                    self.chunk_start.expand(B * T, 1, -1), 
                    target_embs[:, :-1, :]
                ], dim=1)
                
                # Combine Holographic signal with AR prefix and apply 1-layer Causal Attention
                combined_signal = unbound_flat + shifted_embs
                cleaned_flat = self.micro_decoder(combined_signal, src_mask=self.causal_mask, is_causal=True)
                
                # --- FP32: Force geometric math to Float32 ---
                with torch.amp.autocast(DEVICE, enabled=False):
                    cleaned_norm = F.normalize(cleaned_flat.float(), p=2, dim=-1)
                    manifold_norm_fp32 = F.normalize(self.byte_manifold.float(), p=2, dim=-1)
                    byte_logits = torch.matmul(cleaned_norm, manifold_norm_fp32.transpose(0, 1)) * self.logit_scale.float().exp()
                    
                    # Also compute Latent Loss in FP32
                    loss_latent = F.mse_loss(logits_continuous.float(), perfect_target_sketches.float())
                
                loss_ce = F.cross_entropy(byte_logits.view(-1, 256), targets.reshape(-1))
                
                # Rebalanced weights
                loss = loss_ce + 0.5 * loss_latent
                
            return logits_continuous, loss
        else:
            # Standard discrete baseline
            B, T = idx.size()[:2]
            x = self.tok_emb(idx) + self.pos_emb(torch.arange(0, T, dtype=torch.long, device=DEVICE))
            for block in self.blocks: x = block(x)
            logits = self.lm_head(self.ln_f(x))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
            return logits, loss

# =========================================================================
# 5. EXPERIMENT MASTER LOOP 
# =========================================================================
results = {}
saved_hss_model = None

# Create a directory to store the saved models
os.makedirs("saved_models", exist_ok=True)

# Run BOTH Baseline and HSS so we can plot them against each other
for mode in ["HSS", "Baseline"]: 
    is_hss = (mode == "HSS")
    n_layer = 11 if is_hss else 6 
    print(f"\n{'='*50}\nSTARTING {mode.upper()} RUN\n{'='*50}")
    
    train_loader = DataLoaderLite(train_text, is_hss=is_hss)
    val_loader = DataLoaderLite(val_text, is_hss=is_hss)
    
    model = ContinuousGPT(is_hss=is_hss, n_layer=n_layer).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    scaler = torch.amp.GradScaler(DEVICE, enabled=(DEVICE=='cuda')) 

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    val_losses, steps = [],[]

    for step in range(MAX_STEPS + 1):
        if step % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = val_loader.get_batch()
                with torch.amp.autocast(DEVICE, dtype=torch.float16 if DEVICE=='cuda' else torch.bfloat16):
                    _, val_loss = model(x_val, y_val)
            model.train()
            print(f"Step {step:04d} | Val Loss: {val_loss.item():.4f}")
            steps.append(step)
            val_losses.append(val_loss.item())

        x, y = train_loader.get_batch()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(DEVICE, dtype=torch.float16 if DEVICE=='cuda' else torch.bfloat16):
            _, loss = model(x, y)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    # Store results for our new plotting function
    results[mode] = {'steps': steps, 'val_losses': val_losses}
    
    # --- SAVE THE MODEL ---
    save_path = f"saved_models/ContinuousGPT_V9_{mode}.pth"
    torch.save({
        'step': MAX_STEPS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_losses': val_losses
    }, save_path)
    print(f"[*] Successfully saved {mode} model to {save_path}")

    if is_hss:
        saved_hss_model = model

print(f"\n{'='*50}\nPLOTTING VALIDATION LOSSES\n{'='*50}")


def plot_definitive_comparisons(results):
    # Force a clean, solid background 
    plt.style.use('default')
    
    # ---------------------------------------------------------
    # PLOT 1: THE "FAIR START" (Normalized to % of Random Chance)
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    fig1.patch.set_facecolor('white')
    
    baseline_random_chance = math.log(50257) # ~10.82
    hss_random_chance = math.log(256)        # ~5.54 
    
    for mode, data in results.items():
        # Choose the right mathematical baseline
        norm_factor = baseline_random_chance if mode == "Baseline" else hss_random_chance
        
        # ACTUALLY APPLY THE MATH: Divide raw loss by random chance
        normalized_losses = [l / norm_factor for l in data['val_losses']]
        
        color = '#1f77b4' if mode == "Baseline" else '#d62728'
        ax1.plot(data['steps'], normalized_losses, label=f'{mode} Architecture', 
                 linewidth=2.5, marker='o', color=color)
        
    ax1.set_title('Plot 1: Learning Rate (Both start at ~1.0 = Random Guess)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Normalized Loss Ratio')
    ax1.set_xlabel('Training Steps')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    plt.show()

    # ---------------------------------------------------------
    # PLOT 2: THE "BRUTAL REALITY" (Average Loss Per Byte)
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=150)
    fig2.patch.set_facecolor('white')
    
    # GPT-2 tokenizer naturally compresses ~4 chars/bytes into 1 token
    BYTES_PER_TOKEN = 4.0 
    
    for mode, data in results.items():
        if mode == "Baseline":
            # Divide token loss by 4 to get the per-byte equivalent
            per_byte_losses =[l / BYTES_PER_TOKEN for l in data['val_losses']]
            label = 'Baseline (BPE converted to per-byte)'
            color = '#1f77b4'
        else:
            # HSS is already predicting byte-by-byte
            per_byte_losses = data['val_losses'] 
            label = 'Continuous HSS (Native per-byte)'
            color = '#d62728'
            
        ax2.plot(data['steps'], per_byte_losses, label=label, 
                 linewidth=2.5, marker='o', color=color)
        
    ax2.set_title('Plot 2: Absolute Information Compression (Loss per Byte)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Nats per Byte (Lower is Better)')
    ax2.set_xlabel('Training Steps')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    plt.show()

# Execute the plotter using the dictionary from our training loop
plot_definitive_comparisons(results)

# =========================================================================
# 6. INFERENCE GENERATION TEST
# =========================================================================
if saved_hss_model is not None:
    print(f"\n{'='*50}\nTESTING HSS INFERENCE GENERATION\n{'='*50}")
    saved_hss_model.eval()
    
    prompt_str = "The future of AI is " 
    pad_len = (W_CAPACITY - (len(prompt_str) % W_CAPACITY)) % W_CAPACITY
    prompt_str += " " * pad_len
    prompt_bytes = prompt_str.encode('utf-8')
    
    generated_bytes = bytearray(prompt_bytes)
    
    with torch.no_grad():
        for _ in range(10): # 10 iterations * 8 bytes = 80 generated bytes
            x_bytes = torch.tensor(list(generated_bytes), dtype=torch.long, device=DEVICE)
            L = len(x_bytes) // W_CAPACITY
            x_bytes = x_bytes[:L*W_CAPACITY].view(1, L, W_CAPACITY)
            
            if x_bytes.size(1) > SEQ_LEN:
                x_bytes = x_bytes[:, -SEQ_LEN:, :]
                
            with torch.amp.autocast(DEVICE, dtype=torch.float16 if DEVICE=='cuda' else torch.bfloat16):
                sketches = saved_hss_model.hss_encode(x_bytes) 
                x_transformer = sketches + saved_hss_model.pos_emb(torch.arange(0, sketches.size(1), dtype=torch.long, device=DEVICE))
                for block in saved_hss_model.blocks: x_transformer = block(x_transformer)
                logits = saved_hss_model.hss_head(saved_hss_model.ln_f(x_transformer))
            
            # Parallel Holographic Decoding via Cosine Sampling
            new_bytes = saved_hss_model.decode_vector(logits[:, -1, :][0], temperature=0.8)
            generated_bytes.extend(new_bytes)
    
    # Print raw representation to ensure we see exactly what is generated (even if unprintable)
    print(f"\nRaw Byte Output:\n{repr(bytes(generated_bytes))}")
    print(f"\nDecoded Text Output:\n{generated_bytes.decode('utf-8', errors='replace')}\n")