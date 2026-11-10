# =============================================================================
# OGI EXECUTIVE GATER + SAD IMPARTIAL OBSERVER
# Version 2 — seed fix + GRU spectral norm penalty (Lemma 4.2)
#
# Implements directly from:
#   Singleton (2024) "Addressing Mathematical Rigor in the OGI Framework"
#   Equations 1-11, Theorems 1-4, Lemmas 4.1-4.2
#
# Changes from v1:
#   - Removed torch.manual_seed(0) from StreamGenerator and SADProbe
#     so results actually vary across seeds (was deterministic before)
#   - Added GRU spectral norm penalty to enforce Lemma 4.2 condition (C1)
#
# SAD (Nelson's navi-SAD) is the IMPARTIAL OBSERVER.
# It watches the gater's routing decisions from outside.
# It never feeds into the gater's decision loop.
#
# Requirements:
#   pip install scipy matplotlib torch
#   Copy ordinal.py from navi-SAD src/navi_sad/signal/ into same directory
# =============================================================================

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

try:
    from ordinal import permutation_entropy, recommended_min_pe_length
    USING_NELSON_PE = True
except ImportError:
    USING_NELSON_PE = False
    print("WARNING: ordinal.py not found — using fallback PE (no tie exclusion)")

# =============================================================================
# CONFIG
# =============================================================================

# Architecture — from paper Section II, Eq. 1-2
N_MODULES     = 32      # n: total modules
K_ACTIVE      = 4       # k: Top-K active per step (Theorem 3)
D_CONTEXT     = 128     # dc: context vector dimension
D_TASK        = 64      # de: task embedding dimension
D_ATTN        = 64      # da: attention hidden dimension
D_MODULE      = 64      # dm: module output dimension

# Training
STEPS_WARMUP  = 500     # clean phase — both streams reliable
STEPS_CORRUPT = 1000    # degradation phase — one stream gets noisy
BATCH_SIZE    = 64
LR_ENCODER    = 1e-3
LR_GATER      = 5e-5
LAMBDA_COH    = 0.5     # λ: coherence weight (paper Section VI)
MU_SPARSE     = 0.01    # μ: sparsity weight (Eq. 3)
LAMBDA_SPEC   = 50,0    # spectral norm penalty weight (Lemma 4.2)
SPEC_TARGET   = 0.99    # GRU spectral norm target (must stay < 1)

# Degradation
DEGRADE_STREAM = "B"    # which stream degrades after warmup
SNR_CLEAN      = 10.0   # signal-to-noise during warmup
SNR_CORRUPT    = 0.5    # signal-to-noise after degradation

# SAD probe
PE_D          = 3
PE_TAU        = 1
PE_WINDOW     = 64

# Lipschitz bound from paper Theorem 1: L = La/2
LIPSCHITZ_BOUND = 0.456

SEEDS  = [42, 123, 777]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device:       {DEVICE}")
print(f"PE engine:    {'Nelson ordinal.py' if USING_NELSON_PE else 'fallback'}")
print(f"N={N_MODULES}, K={K_ACTIVE}, streams: A+B, degrade={DEGRADE_STREAM} after step {STEPS_WARMUP}")
print(f"Spectral norm penalty: weight={LAMBDA_SPEC}, target={SPEC_TARGET}")

# =============================================================================
# SYNTHETIC STREAMS — controlled SNR, equal architecture
# No dominant modality baked in. Both streams carry real signal.
# After warmup, one stream's noise increases to SNR_CORRUPT.
# =============================================================================

class StreamGenerator:
    """
    Two synthetic Gaussian streams with controlled SNR.
    Both encode the same underlying signal (class label).
    Reliability is controlled by noise level, not architecture.

    NOTE: No fixed seed here — prototypes vary with the run seed
    set in run() via torch.manual_seed(seed). This ensures results
    actually differ across seeds.
    """
    def __init__(self, n_classes=10, signal_dim=D_CONTEXT, device=DEVICE):
        self.n_classes  = n_classes
        self.signal_dim = signal_dim
        self.device     = device

        # Prototypes use whatever seed was set by run() — no override
        self.prototypes = F.normalize(
            torch.randn(n_classes, signal_dim, device=device), dim=1
        )

    def sample(self, batch_size, snr_a=SNR_CLEAN, snr_b=SNR_CLEAN):
        """
        Returns (stream_a, stream_b, labels).
        Both streams encode the same label with controlled SNR.
        SNR = signal_power / noise_power.
        """
        labels   = torch.randint(0, self.n_classes, (batch_size,), device=self.device)
        signal   = self.prototypes[labels]   # (B, signal_dim)
        noise_a  = torch.randn_like(signal) / math.sqrt(snr_a)
        noise_b  = torch.randn_like(signal) / math.sqrt(snr_b)
        return signal + noise_a, signal + noise_b, labels

# =============================================================================
# EXECUTIVE GATER — from paper Equations 1-2, Theorem 3
#
# Φ(c, e^t) = softmax(A(c, e^t))                      [Eq. 1]
# A(c, e^t) = W_A^T tanh(W_C c + W_E e^t + b_A)       [Eq. 2]
#
# Top-K mask enforces sparsity: O(k²dm + n(dc·da))     [Theorem 3]
# =============================================================================

class ExecutiveGater(nn.Module):
    """
    Implements OGI Executive Attention Gating (paper Section IV.A).
    Selects Top-K modules from N candidates based on context + task.
    """
    def __init__(self, n_modules=N_MODULES, k=K_ACTIVE,
                 d_context=D_CONTEXT, d_task=D_TASK, d_attn=D_ATTN):
        super().__init__()
        self.n = n_modules
        self.k = k

        # Eq. 2: W_C, W_E, b_A, W_A
        self.Wc = nn.Linear(d_context, d_attn,    bias=False)
        self.We = nn.Linear(d_task,    d_attn,    bias=False)
        self.ba = nn.Parameter(torch.zeros(d_attn))
        self.Wa = nn.Linear(d_attn,    n_modules, bias=False)

        nn.init.xavier_uniform_(self.Wc.weight, gain=0.1)
        nn.init.xavier_uniform_(self.We.weight, gain=0.1)
        nn.init.xavier_uniform_(self.Wa.weight, gain=0.1)

    def forward(self, c, e):
        """
        Args:
            c: (B, d_context) — context vector
            e: (B, d_task)    — task embedding

        Returns:
            phi:      (B, N) softmax attention weights [Eq. 1]
            phi_k:    (B, N) Top-K masked weights (sparse)
            topk_idx: (B, K) indices of selected modules
        """
        h     = torch.tanh(self.Wc(c) + self.We(e) + self.ba)   # (B, da)
        A     = self.Wa(h)                                        # (B, N)
        phi   = torch.softmax(A, dim=1)                          # (B, N)

        # Top-K masking — Theorem 3
        topk_vals, topk_idx = torch.topk(phi, self.k, dim=1)
        phi_k = torch.zeros_like(phi)
        phi_k.scatter_(1, topk_idx, topk_vals)
        phi_k = phi_k / (phi_k.sum(dim=1, keepdim=True) + 1e-8)

        return phi, phi_k, topk_idx

    def lipschitz_bound(self):
        """
        Analytical Lipschitz bound from Theorem 1: L = La/2
        La = ||W_A||_2 * ||W_C||_2
        """
        sv_Wa = torch.linalg.svdvals(self.Wa.weight.float())
        sv_Wc = torch.linalg.svdvals(self.Wc.weight.float())
        La    = sv_Wa.max().item() * sv_Wc.max().item()
        return La / 2.0, La

# =============================================================================
# FUSION — GRU-based semantic binding, Eq. 9-10, Lemma 4.2
# =============================================================================

class OGIFusion(nn.Module):
    """
    Semantic-preserving fusion via GRU (paper Section V, Eq. 9-10).
    h_fusion = GRU(h_prev, Σ_i Φ_i ⊙ o_i)    [Eq. 10]

    Lemma 4.2: GRU is contractive when spectral norms < 1.
    Enforced via spectral norm penalty in training loss.
    """
    def __init__(self, d_in=D_CONTEXT, d_module=D_MODULE,
                 d_hidden=D_CONTEXT, n_classes=10):
        super().__init__()
        self.gru      = nn.GRUCell(d_module, d_hidden)
        self.classify = nn.Linear(d_hidden, n_classes)
        # MINE critic T(c, O^t) for coherence objective [Eq. 11, Lemma 4.1]
        self.critic   = nn.Sequential(
            nn.Linear(d_hidden * 2, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.proj_a = nn.Linear(d_in, d_module)
        self.proj_b = nn.Linear(d_in, d_module)

    def fuse_streams(self, stream_a, stream_b, phi_k):
        """Weighted combination of streams via Top-K attention weights."""
        w_a = phi_k[:, :N_MODULES//2].sum(dim=1, keepdim=True)
        w_b = phi_k[:, N_MODULES//2:].sum(dim=1, keepdim=True)
        total = w_a + w_b + 1e-8
        w_a, w_b = w_a / total, w_b / total
        fused = w_a * self.proj_a(stream_a) + w_b * self.proj_b(stream_b)
        return fused, w_a.squeeze(1), w_b.squeeze(1)

    def forward(self, stream_a, stream_b, phi_k, topk_idx, h):
        fused, w_a, w_b = self.fuse_streams(stream_a, stream_b, phi_k)
        h_new  = self.gru(fused, h)
        logits = self.classify(h_new)
        return h_new, logits, w_a, w_b

    def coherence_loss(self, h, context):
        """MINE lower bound on I(c; O^t) [Eq. 11, Lemma 4.1]."""
        joint    = torch.cat([h, context], dim=1)
        perm     = torch.randperm(context.size(0), device=context.device)
        marginal = torch.cat([h, context[perm]], dim=1)
        return -(torch.mean(self.critic(joint)) -
                 torch.log(torch.mean(torch.exp(self.critic(marginal))) + 1e-8))

    def sparsity_loss(self, phi):
        """L_sparsity: entropy penalty on Φ [Eq. 3]."""
        return -(phi * torch.log(phi + 1e-8)).sum(dim=1).mean()

    def spectral_norm_penalty(self, target=SPEC_TARGET):
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                # Frobenius norm as differentiable proxy for spectral norm
                # svdvals() is not differentiable — this one is
                penalty = penalty + F.relu(torch.norm(param, p='fro') - target)
        return penalty

    def gru_spectral_norm(self):
        """Monitor GRU spectral norms — Lemma 4.2 condition (C1)."""
        norms = {}
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                sv = torch.linalg.svdvals(param.float())
                norms[name] = sv.max().item()
        return norms

# =============================================================================
# SAD PROBE — impartial observer
# Watches gater's attention geometry from outside.
# Never feeds into the gater's decision.
# =============================================================================

def compute_pe(series, D=PE_D, tau=PE_TAU):
    if USING_NELSON_PE:
        if len(series) < recommended_min_pe_length(D, tau):
            return None
        pe_val, _, _ = permutation_entropy(series, D=D, tau=tau)
        return pe_val
    else:
        if len(series) < D:
            return None
        patterns = {}
        for i in range(len(series) - D + 1):
            key = tuple(np.argsort(series[i:i+D]))
            patterns[key] = patterns.get(key, 0) + 1
        probs = np.array(list(patterns.values()), dtype=float)
        probs /= probs.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-12)) /
                     np.log2(math.factorial(D)))


def _softmax_attn(Q, K, V):
    scale = Q.shape[-1] ** -0.5
    return torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1) @ V


def _linear_attn(Q, K, V, eps=1e-6):
    phi_Q = F.elu(Q) + 1.0
    phi_K = F.elu(K) + 1.0
    KV    = phi_K.transpose(-2, -1) @ V
    Z     = phi_Q @ phi_K.sum(dim=-2, keepdim=True).transpose(-2, -1)
    return (phi_Q @ KV) / (Z + eps)


class SADProbe:
    """
    Impartial SAD observer. Frozen random projections.
    Watches stream_a and stream_b — never the gater internals.

    NOTE: No fixed seed — projections vary with run seed set in run().
    This ensures PE measurements actually differ across seeds.
    SAD remains impartial — random projections don't bias the measurement,
    they just determine which attractor geometry gets observed.
    """
    def __init__(self, n_heads=4, window=PE_WINDOW):
        self.n_heads = n_heads
        self.window  = window

        # No torch.manual_seed(0) here — uses run() seed
        self.Wq = nn.Linear(D_CONTEXT, D_CONTEXT, bias=False).to(DEVICE)
        self.Wk = nn.Linear(D_CONTEXT, D_CONTEXT, bias=False).to(DEVICE)
        self.Wv = nn.Linear(D_CONTEXT, D_CONTEXT, bias=False).to(DEVICE)
        for p in [*self.Wq.parameters(),
                  *self.Wk.parameters(),
                  *self.Wv.parameters()]:
            p.requires_grad_(False)

        self.sad:       list[float] = []
        self.w_a:       list[float] = []
        self.w_b:       list[float] = []
        self.lip:       list[float] = []
        self.corrupt:   list[bool]  = []
        self.gru_norms: list[dict]  = []

    @torch.no_grad()
    def step(self, stream_a, stream_b, w_a, w_b, corrupt,
             gater=None, fusion=None,
             prev_context=None, curr_context=None):
        B  = stream_a.shape[0]
        H  = self.n_heads
        dh = D_CONTEXT // H

        x = torch.stack([stream_a, stream_b], dim=1).float()  # (B, 2, D)

        Q = self.Wq(x).view(B, 2, H, dh).transpose(1, 2)
        K = self.Wk(x).view(B, 2, H, dh).transpose(1, 2)
        V = self.Wv(x).view(B, 2, H, dh).transpose(1, 2)

        out_s = _softmax_attn(Q, K, V)
        out_l = _linear_attn(Q, K, V)

        cos     = F.cosine_similarity(
            out_s.reshape(B, H, -1),
            out_l.reshape(B, H, -1),
            dim=2,
        )
        sad_val = (1.0 - cos).mean().item()

        # Lipschitz ratio — observed from gater, not influenced
        lip_val = float("nan")
        if (gater is not None and
                prev_context is not None and
                curr_context is not None):
            task_e = torch.zeros(B, D_TASK, device=DEVICE)
            phi1, _, _ = gater(prev_context, task_e)
            phi2, _, _ = gater(curr_context, task_e)
            in_d  = torch.norm(prev_context - curr_context, dim=1)
            out_d = torch.norm(phi1 - phi2, dim=1)
            valid = in_d > 1e-8
            if valid.any():
                lip_val = (out_d[valid] / in_d[valid]).mean().item()

        gru_norm = fusion.gru_spectral_norm() if fusion is not None else {}

        self.sad.append(sad_val)
        self.w_a.append(w_a.mean().item() if torch.is_tensor(w_a) else w_a)
        self.w_b.append(w_b.mean().item() if torch.is_tensor(w_b) else w_b)
        self.lip.append(lip_val)
        self.corrupt.append(corrupt)
        self.gru_norms.append(gru_norm)

    def rolling_pe(self):
        n  = len(self.sad)
        pe = np.full(n, np.nan)
        for i in range(n - self.window + 1):
            val = compute_pe(self.sad[i:i+self.window])
            pe[i + self.window//2] = val if val is not None else np.nan
        return pe, ~np.isnan(pe)

    def analyze(self):
        sad     = np.array(self.sad)
        w_a     = np.array(self.w_a)
        w_b     = np.array(self.w_b)
        lip     = np.array(self.lip)
        corrupt = np.array(self.corrupt, dtype=bool)
        pe, valid = self.rolling_pe()

        wa_clean   = w_a[~corrupt].mean() if (~corrupt).any() else float("nan")
        wa_corrupt = w_a[corrupt].mean()  if corrupt.any()   else float("nan")
        routing_shift = wa_corrupt - wa_clean

        sad_clean   = sad[~corrupt].mean() if (~corrupt).any() else float("nan")
        sad_corrupt = sad[corrupt].mean()  if corrupt.any()   else float("nan")

        def pearson_safe(a, b):
            if len(a) < 4 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return None, None
            r, p = stats.pearsonr(a, b)
            return float(r), float(p)

        mc          = valid & corrupt
        r_wa, p_wa  = pearson_safe(pe[mc], w_a[mc])
        r_wb, p_wb  = pearson_safe(pe[mc], w_b[mc])

        lip_valid   = ~np.isnan(lip)
        ml          = valid & corrupt & lip_valid
        r_lip, p_lip = pearson_safe(pe[ml], lip[ml])

        mc_null      = valid & ~corrupt
        r_null, p_null = pearson_safe(pe[mc_null], w_a[mc_null])

        gru_max_norms = [
            max(d.values()) for d in self.gru_norms if d
        ]

        return {
            "sad_clean":      round(float(sad_clean),      6),
            "sad_corrupt":    round(float(sad_corrupt),     6),
            "sad_delta":      round(float(sad_corrupt - sad_clean), 6),
            "wa_clean":       round(float(wa_clean),        4),
            "wa_corrupt":     round(float(wa_corrupt),      4),
            "routing_shift":  round(float(routing_shift),   4),
            "pe_wa_corrupt":  (
                {"r": round(r_wa, 4), "p": round(p_wa, 6), "n": int(mc.sum()),
                 "significant": p_wa < 0.05}
                if r_wa is not None else {"error": "insufficient data"}
            ),
            "pe_wb_corrupt":  (
                {"r": round(r_wb, 4), "p": round(p_wb, 6),
                 "significant": p_wb < 0.05}
                if r_wb is not None else {"error": "insufficient data"}
            ),
            "pe_lip_corrupt": (
                {"r": round(r_lip, 4), "p": round(p_lip, 6),
                 "significant": p_lip < 0.05}
                if r_lip is not None else {"error": "insufficient data"}
            ),
            "pe_null":        (
                {"r": round(r_null, 4), "p": round(p_null, 6),
                 "significant": p_null < 0.05}
                if r_null is not None else {"error": "insufficient data"}
            ),
            "gru_max_norm":   round(max(gru_max_norms), 6)
                              if gru_max_norms else float("nan"),
            "gru_stable":     max(gru_max_norms) < 1.0
                              if gru_max_norms else None,
            "_arrays": {
                "sad": sad, "w_a": w_a, "w_b": w_b,
                "pe": pe, "valid": valid, "lip": lip,
                "corrupt": corrupt, "lip_valid": lip_valid,
            },
        }

# =============================================================================
# TRAINING LOOP
# =============================================================================

def run(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen    = StreamGenerator()   # uses seed set above — no internal override
    gater  = ExecutiveGater().to(DEVICE)
    fusion = OGIFusion().to(DEVICE)
    probe  = SADProbe()          # uses seed set above — no internal override

    task_e = torch.randn(1, D_TASK, device=DEVICE)
    task_e = F.normalize(task_e, dim=1).expand(BATCH_SIZE, -1)

    opt = optim.Adam([
        {"params": list(fusion.parameters()), "lr": LR_ENCODER},
        {"params": list(gater.parameters()),  "lr": LR_GATER},
    ], weight_decay=1e-4)

    crit        = nn.CrossEntropyLoss()
    total_steps = STEPS_WARMUP + STEPS_CORRUPT
    prev_context = None

    print(f"\n[Seed {seed}] Training {total_steps} steps "
          f"(warmup={STEPS_WARMUP}, corrupt={STEPS_CORRUPT})")

    for step in range(total_steps):
        is_corrupt = step >= STEPS_WARMUP

        if is_corrupt and DEGRADE_STREAM == "B":
            snr_a, snr_b = SNR_CLEAN, SNR_CORRUPT
        elif is_corrupt and DEGRADE_STREAM == "A":
            snr_a, snr_b = SNR_CORRUPT, SNR_CLEAN
        else:
            snr_a, snr_b = SNR_CLEAN, SNR_CLEAN

        stream_a, stream_b, labels = gen.sample(BATCH_SIZE, snr_a, snr_b)
        context = (stream_a + stream_b) / 2.0
        h       = torch.zeros(BATCH_SIZE, D_CONTEXT, device=DEVICE)

        # Executive Gater: Eq. 1-2, Theorem 3
        phi, phi_k, topk_idx = gater(context, task_e[:BATCH_SIZE])

        # Fusion: Eq. 9-10
        h_new, logits, w_a, w_b = fusion(
            stream_a, stream_b, phi_k, topk_idx, h
        )

        # Global Coherence Objective: Eq. 3
        # L^t = L_task + λ·L_coherence + μ·L_sparsity + λ_spec·L_spectral
        L_task = crit(logits, labels)
        L_coh  = fusion.coherence_loss(h_new, context)
        L_sp   = fusion.sparsity_loss(phi)
        L_spec = fusion.spectral_norm_penalty()
        if step == 0:
            print(f"  L_task={L_task.item():.4f} L_coh={L_coh.item():.4f} "
                  f"L_sp={L_sp.item():.4f} L_spec={L_spec.item():.4f} "
                  f"weighted_spec={LAMBDA_SPEC*L_spec.item():.4f}")

        L_coh_clamped = torch.clamp(L_coh, min=-2.0, max=0.0)  # ← ADD
        loss   = (L_task
                  + LAMBDA_COH  * L_coh_clamped                 # ← changed
                  + MU_SPARSE   * L_sp
                  + LAMBDA_SPEC * L_spec)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # SAD probe — impartial observer, never influences loss or gater
        probe.step(
            stream_a.detach(), stream_b.detach(),
            w_a.detach(), w_b.detach(),
            corrupt=is_corrupt,
            gater=gater,
            fusion=fusion,
            prev_context=prev_context,
            curr_context=context.detach(),
        )
        prev_context = context.detach()

        if step % 200 == 0 or step == STEPS_WARMUP:
            phase = "CORRUPT" if is_corrupt else "warmup"
            gru_n = max(fusion.gru_spectral_norm().values())
            print(f"  step={step:4d} [{phase}] "
                  f"loss={loss.item():.4f} "
                  f"w_a={w_a.mean().item():.3f} "
                  f"w_b={w_b.mean().item():.3f} "
                  f"sad={probe.sad[-1]:.4f} "
                  f"gru_norm={gru_n:.3f}")

    # Theorem 1: Lipschitz bound on trained weights
    lip_bound, La = gater.lipschitz_bound()
    print(f"\n  Theorem 1: La/2 = {lip_bound:.6f} "
          f"(La={La:.4f}, paper bound={LIPSCHITZ_BOUND}) "
          f"{'✓' if lip_bound <= LIPSCHITZ_BOUND else '⚠'}")

    results = probe.analyze()
    print(f"  SAD: clean={results['sad_clean']:.4f} "
          f"corrupt={results['sad_corrupt']:.4f} "
          f"delta={results['sad_delta']:+.4f}")
    print(f"  Routing: w_a {results['wa_clean']:.3f} → {results['wa_corrupt']:.3f} "
          f"(shift={results['routing_shift']:+.3f})")
    print(f"  GRU norm: {results['gru_max_norm']:.4f} "
          f"({'✓ < 1' if results['gru_stable'] else '⚠ >= 1 Lemma 4.2 violated'})")

    return probe, results, {"lip_bound": lip_bound, "La": La, "seed": seed}

# =============================================================================
# UNIFIED RESULTS TABLE
# =============================================================================

def unified_results_table(all_results):
    print("\n" + "="*72)
    print("OGI EXECUTIVE GATER + SAD IMPARTIAL OBSERVER — UNIFIED RESULTS")
    print("="*72)

    print(f"\nTHEOREM 1: Lipschitz Stability  L = La/2  (paper bound = {LIPSCHITZ_BOUND})")
    print(f"  {'Seed':<8} {'La/2':<12} {'La':<12} {'≤ bound?'}")
    print(f"  {'-'*40}")
    for _, _, meta in all_results:
        sat = "YES ✓" if meta["lip_bound"] <= LIPSCHITZ_BOUND else "NO ⚠"
        print(f"  {meta['seed']:<8} {meta['lip_bound']:<12.6f} "
              f"{meta['La']:<12.4f} {sat}")

    print(f"\nTHEOREM 3: Top-K Routing  (N={N_MODULES}, K={K_ACTIVE})")
    print(f"  Complexity: O(k²dm + n(dc·da)) = "
          f"O({K_ACTIVE**2}·{D_MODULE} + {N_MODULES}·{D_CONTEXT*D_ATTN})")
    print(f"  vs full-mesh O(n²dm) = O({N_MODULES**2}·{D_MODULE}) "
          f"— {N_MODULES**2//K_ACTIVE**2}x reduction")

    print(f"\nLEMMA 4.2: GRU Contractive Mapping  (spectral norm < 1 required)")
    print(f"  Spectral norm penalty applied during training "
          f"(weight={LAMBDA_SPEC}, target={SPEC_TARGET})")
    print(f"  {'Seed':<8} {'GRU max norm':<16} {'Stable?'}")
    print(f"  {'-'*40}")
    for _, res, meta in all_results:
        stable = "YES ✓" if res["gru_stable"] else "NO ⚠ — semantic drift risk"
        print(f"  {meta['seed']:<8} {res['gru_max_norm']:<16.6f} {stable}")

    print(f"\nLEMMA 4.1: MI Routing + SAD Attractor Geometry")
    print(f"  Stream {DEGRADE_STREAM} degrades at step {STEPS_WARMUP} "
          f"(SNR: {SNR_CLEAN} → {SNR_CORRUPT})")
    print(f"  SAD is impartial observer — never feeds into gater decision")
    print()
    print(f"  {'Seed':<8} {'SAD Δ':>10} {'w_a Δ':>10} "
          f"{'PE↔w_a r':>12} {'p':>10} {'Sig':>6} {'Null r':>10}")
    print(f"  {'-'*70}")
    for _, res, meta in all_results:
        pac  = res.get("pe_wa_corrupt", {})
        null = res.get("pe_null", {})
        ra   = f"{pac['r']:+.4f}"  if "r" in pac  else "n/a"
        pa   = f"{pac['p']:.4f}"   if "p" in pac  else "n/a"
        sig  = "***" if pac.get("significant") else ("   " if "r" in pac else "n/a")
        rn   = f"{null['r']:+.4f}" if "r" in null else "n/a"
        print(f"  {meta['seed']:<8} {res['sad_delta']:>+10.4f} "
              f"{res['routing_shift']:>+10.4f} "
              f"{ra:>12} {pa:>10} {sig:>6} {rn:>10}")

    print(f"\nINTERPRETATION")
    for _, res, meta in all_results:
        pac   = res.get("pe_wa_corrupt", {})
        shift = res["routing_shift"]

        if abs(shift) > 0.05:
            routing_str = (
                f"Gater correctly rerouted "
                f"({'away from' if shift < 0 else 'toward'} stream A "
                f"by {abs(shift):.3f}) when stream {DEGRADE_STREAM} degraded."
            )
        else:
            routing_str = ("Routing shift small — "
                           "consider stronger SNR degradation.")

        if pac.get("significant"):
            sad_str = (
                "SAD PE tracked routing independently. "
                "Attractor geometry co-varies with MI routing. "
                "→ Lemma 4.1 ↔ SAD link: OBSERVED by impartial instrument."
            )
        else:
            sad_str = (
                f"SAD PE not significant (p={pac.get('p', '?')}). "
                "Grand-mean SAD consistent with Nelson Gate 3 pilot finding — "
                "signal may be in per-head PE, not mean."
            )

        print(f"\n  Seed {meta['seed']}:")
        print(f"    {routing_str}")
        print(f"    {sad_str}")

    print("\n" + "="*72)

# =============================================================================
# PLOT
# =============================================================================

def plot_results(probe, results, meta):
    arr     = results["_arrays"]
    sad     = arr["sad"]
    w_a     = arr["w_a"]
    w_b     = arr["w_b"]
    pe      = arr["pe"]
    valid   = arr["valid"]
    lip     = arr["lip"]
    corrupt = arr["corrupt"]
    lip_v   = arr["lip_valid"]
    steps   = np.arange(len(sad))

    # GRU norm trajectory
    gru_norms = [
        max(d.values()) if d else float("nan")
        for d in probe.gru_norms
    ]
    gru_arr = np.array(gru_norms)

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig)
    fig.suptitle(
        f"OGI Executive Gater + SAD Impartial Observer — Seed {meta['seed']}\n"
        f"N={N_MODULES} modules, K={K_ACTIVE} active  |  "
        f"Stream {DEGRADE_STREAM} degrades at step {STEPS_WARMUP}  |  "
        f"Spectral norm penalty: {LAMBDA_SPEC}",
        fontsize=11
    )

    def vline(ax):
        ax.axvline(STEPS_WARMUP, color="red", ls="--", lw=1.5,
                   label=f"Stream {DEGRADE_STREAM} degrades")

    # Row 0
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, sad, lw=0.7, color="steelblue")
    vline(ax); ax.legend(fontsize=8)
    ax.set_title("SAD Trajectory (Impartial Observer)")
    ax.set_xlabel("Step"); ax.set_ylabel("Softmax-Linear divergence")

    ax = fig.add_subplot(gs[0, 1])
    if valid.any():
        ax.plot(steps[valid], pe[valid], lw=0.7, color="purple")
    vline(ax)
    ax.set_title(f"Attractor PE (Nelson ordinal.py, D={PE_D}, w={PE_WINDOW})")
    ax.set_xlabel("Step"); ax.set_ylabel("Normalized PE")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(steps, w_a, lw=0.7, color="forestgreen", label="w_a (stream A)")
    ax.plot(steps, w_b, lw=0.7, color="coral",       label="w_b (stream B)")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    vline(ax); ax.legend(fontsize=8)
    ax.set_title("Executive Gater Routing Weights  [Lemma 4.1]")
    ax.set_xlabel("Step"); ax.set_ylabel("Weight")

    # Row 1
    ax = fig.add_subplot(gs[1, 0])
    if lip_v.any():
        ax.plot(steps[lip_v], lip[lip_v], lw=0.5, color="coral", alpha=0.7)
        ax.axhline(LIPSCHITZ_BOUND, color="red", ls="--", lw=1,
                   label=f"Theorem 1 bound ({LIPSCHITZ_BOUND})")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Lipschitz ratio\nnot available",
                ha="center", va="center", transform=ax.transAxes)
    vline(ax)
    ax.set_title("Lipschitz Ratio  [Theorem 1]")
    ax.set_xlabel("Step"); ax.set_ylabel("|Δφ|/|Δc|")

    ax = fig.add_subplot(gs[1, 1])
    valid_gru = ~np.isnan(gru_arr)
    if valid_gru.any():
        ax.plot(steps[valid_gru], gru_arr[valid_gru],
                lw=0.7, color="darkorange")
        ax.axhline(1.0, color="red", ls="--", lw=1,
                   label="Lemma 4.2 limit (1.0)")
        ax.axhline(SPEC_TARGET, color="green", ls=":", lw=1,
                   label=f"Penalty target ({SPEC_TARGET})")
        ax.legend(fontsize=8)
    vline(ax)
    ax.set_title("GRU Spectral Norm  [Lemma 4.2]")
    ax.set_xlabel("Step"); ax.set_ylabel("Max spectral norm")

    ax = fig.add_subplot(gs[1, 2])
    mc  = valid & corrupt
    pac = results.get("pe_wa_corrupt", {})
    if mc.sum() > 3 and "r" in pac:
        ax.scatter(pe[mc], w_a[mc], alpha=0.3, s=8,
                   color="forestgreen", label="w_a")
        ax.scatter(pe[mc], w_b[mc], alpha=0.3, s=8,
                   color="coral",       label="w_b")
        if np.std(pe[mc]) > 1e-12 and np.std(w_a[mc]) > 1e-12:
            m_, b_ = np.polyfit(pe[mc], w_a[mc], 1)
            xs = np.linspace(pe[mc].min(), pe[mc].max(), 50)
            ax.plot(xs, m_*xs + b_, color="darkgreen", lw=1.5)
        ax.set_title(f"PE vs Routing (corrupt)\n"
                     f"r(w_a)={pac['r']:.4f}, p={pac['p']:.4f} "
                     f"{'***' if pac.get('significant') else ''}")
        ax.legend(fontsize=8)
    else:
        ax.set_title("PE vs Routing (corrupt)\ninsufficient variation")
    ax.set_xlabel("Permutation Entropy")
    ax.set_ylabel("Routing weight")

    # Row 2
    ax = fig.add_subplot(gs[2, 0:2])
    ax.plot(steps, sad, lw=0.5, color="steelblue", alpha=0.6, label="SAD")
    ax2 = ax.twinx()
    ax2.plot(steps, w_a, lw=0.8, color="forestgreen", alpha=0.8,
             label="w_a")
    ax.axvline(STEPS_WARMUP, color="red", ls="--", lw=1.5)
    ax.set_title("SAD + Routing — Full Training Run")
    ax.set_xlabel("Step")
    ax.set_ylabel("SAD divergence", color="steelblue")
    ax2.set_ylabel("w_a routing weight", color="forestgreen")
    ax.legend(loc="upper left",  fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(gs[2, 2])
    phases  = ["Warmup\n(clean)", "Corrupt\n(degraded)"]
    wa_vals = [results["wa_clean"],       results["wa_corrupt"]]
    wb_vals = [1 - results["wa_clean"],   1 - results["wa_corrupt"]]
    x = np.arange(2)
    ax.bar(x - 0.2, wa_vals, 0.35, label="Stream A",
           color="forestgreen", alpha=0.8)
    ax.bar(x + 0.2, wb_vals, 0.35, label="Stream B",
           color="coral", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(phases)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_title(f"Routing Shift Summary\n"
                 f"Δw_a = {results['routing_shift']:+.3f}")
    ax.set_ylabel("Mean routing weight")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fname = f"ogi_executive_gater_seed{meta['seed']}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*55}\nSeed {seed}\n{'='*55}")
        probe, results, meta = run(seed)
        all_results.append((probe, results, meta))
        plot_results(probe, results, meta)
    unified_results_table(all_results)

if __name__ == "__main__":
    main()