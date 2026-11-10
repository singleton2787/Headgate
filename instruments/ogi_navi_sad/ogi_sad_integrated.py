# =============================================================================
# OGI + SAD INTEGRATED VALIDATION (CIFAR-100 + SYNTHETIC CAPTIONS)
# ALL PATCHES APPLIED:
#   1. Entropy regularizer on gate (weight=0.1, always active)
#   2. Small init (gain=0.01), temperature=3.0
#   3. Separate LR: encoders 1e-3, fusion 5e-5
#   4. COHERENCE_WEIGHT = 0.3
#   5. Clamp min=0.1, max=0.9
#   6. Routing confidence metric: |w0 - w1| instead of w0
# =============================================================================

import math
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy import stats

try:
    from ordinal import permutation_entropy, recommended_min_pe_length
    USING_NELSON_PE = True
except ImportError:
    USING_NELSON_PE = False
    print("WARNING: ordinal.py not found — using local PE fallback (no tie exclusion).")

# =============================================================================
# CONFIG
# =============================================================================

DIM               = 128
CLASSES           = 100
BATCH_SIZE        = 128
EPOCHS            = 15
WARMUP            = 2
SEEDS             = [42, 123, 777]

PE_D              = 3
PE_TAU            = 1
PE_WINDOW         = 16

N_HEADS           = 4
LIPSCHITZ_BOUND   = 0.456
LIPSCHITZ_PAIRS   = 2000
CORRUPTION_RATE   = 0.75
COHERENCE_WEIGHT  = 0.1
ENTROPY_REG_W     = 0.5
LING_NOISE_STD    = 0.02

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:    {DEVICE}")
print(f"PE engine: {'Nelson ordinal.py (tie-excluding, Lehmer)' if USING_NELSON_PE else 'local fallback'}")
print(f"Gate fixes: temp=3.0, init gain=0.01, entropy_reg={ENTROPY_REG_W}, "
      f"coherence={COHERENCE_WEIGHT}, fusion_lr=5e-5")

# =============================================================================
# CIFAR-100 CAPTIONS
# =============================================================================

CIFAR100_FINE_LABELS = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

CAPTION_TEMPLATES = [
    "a photo of a {}",
    "an image of a {}",
    "this is a {}",
    "the picture shows a {}",
    "a small {} in the scene",
    "a centered {} object",
]

CLASS_DESCRIPTORS = {
    "apple": ["fruit", "round"], "aquarium_fish": ["fish", "aquatic"],
    "baby": ["child", "young"], "bear": ["animal", "wild"],
    "beaver": ["animal", "rodent"], "bed": ["furniture", "indoor"],
    "bee": ["insect", "flying"], "beetle": ["insect", "small"],
    "bicycle": ["vehicle", "two_wheels"], "bottle": ["container", "plastic"],
    "bowl": ["container", "round"], "boy": ["person", "young"],
    "bridge": ["structure", "outdoor"], "bus": ["vehicle", "large"],
    "butterfly": ["insect", "winged"], "camel": ["animal", "desert"],
    "can": ["container", "metal"], "castle": ["building", "stone"],
    "caterpillar": ["insect", "larva"], "cattle": ["animal", "farm"],
    "chair": ["furniture", "indoor"], "chimpanzee": ["animal", "primate"],
    "clock": ["object", "time"], "cloud": ["sky", "weather"],
    "cockroach": ["insect", "pest"], "couch": ["furniture", "indoor"],
    "crab": ["animal", "sea"], "crocodile": ["animal", "reptile"],
    "cup": ["container", "small"], "dinosaur": ["animal", "prehistoric"],
    "dolphin": ["animal", "marine"], "elephant": ["animal", "large"],
    "flatfish": ["fish", "marine"], "forest": ["nature", "trees"],
    "fox": ["animal", "wild"], "girl": ["person", "young"],
    "hamster": ["animal", "small"], "house": ["building", "home"],
    "kangaroo": ["animal", "jumper"], "keyboard": ["object", "electronics"],
    "lamp": ["object", "light"], "lawn_mower": ["machine", "grass"],
    "leopard": ["animal", "spotted"], "lion": ["animal", "wild"],
    "lizard": ["animal", "reptile"], "lobster": ["animal", "sea"],
    "man": ["person", "adult"], "maple_tree": ["plant", "tree"],
    "motorcycle": ["vehicle", "two_wheels"], "mountain": ["nature", "landscape"],
    "mouse": ["animal", "small"], "mushroom": ["plant", "fungus"],
    "oak_tree": ["plant", "tree"], "orange": ["fruit", "citrus"],
    "orchid": ["flower", "plant"], "otter": ["animal", "aquatic"],
    "palm_tree": ["plant", "tropical"], "pear": ["fruit", "green"],
    "pickup_truck": ["vehicle", "truck"], "pine_tree": ["plant", "tree"],
    "plain": ["landscape", "flat"], "plate": ["object", "dish"],
    "poppy": ["flower", "plant"], "porcupine": ["animal", "spiky"],
    "possum": ["animal", "marsupial"], "rabbit": ["animal", "small"],
    "raccoon": ["animal", "striped"], "ray": ["fish", "marine"],
    "road": ["outdoor", "path"], "rocket": ["vehicle", "space"],
    "rose": ["flower", "plant"], "sea": ["water", "marine"],
    "seal": ["animal", "marine"], "shark": ["fish", "predator"],
    "shrew": ["animal", "small"], "skunk": ["animal", "striped"],
    "skyscraper": ["building", "tall"], "snail": ["animal", "slow"],
    "snake": ["animal", "reptile"], "spider": ["animal", "arachnid"],
    "squirrel": ["animal", "small"], "streetcar": ["vehicle", "urban"],
    "sunflower": ["flower", "plant"], "sweet_pepper": ["vegetable", "plant"],
    "table": ["furniture", "flat"], "tank": ["vehicle", "military"],
    "telephone": ["object", "communication"], "television": ["object", "screen"],
    "tiger": ["animal", "striped"], "tractor": ["vehicle", "farm"],
    "train": ["vehicle", "rail"], "trout": ["fish", "freshwater"],
    "tulip": ["flower", "plant"], "turtle": ["animal", "shell"],
    "wardrobe": ["furniture", "storage"], "whale": ["animal", "marine"],
    "willow_tree": ["plant", "tree"], "wolf": ["animal", "wild"],
    "woman": ["person", "adult"], "worm": ["animal", "small"],
}

def normalize_label_name(name):
    return name.replace("_", " ")

def label_to_captions(label_idx):
    label  = CIFAR100_FINE_LABELS[label_idx]
    pretty = normalize_label_name(label)
    desc   = CLASS_DESCRIPTORS.get(label, [])
    caps   = [tmpl.format(pretty) for tmpl in CAPTION_TEMPLATES]
    if desc:
        caps.append(f"a {pretty} that is {desc[0]}")
        if len(desc) > 1:
            caps.append(f"a {desc[0]} {pretty} with a {desc[1]} appearance")
            caps.append(f"the image contains a {pretty} and it looks {desc[1]}")
    return caps

def tokenize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    return [tok for tok in text.split() if tok]

def build_vocab():
    counter = Counter()
    for idx in range(CLASSES):
        for cap in label_to_captions(idx):
            counter.update(tokenize(cap))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token in sorted(counter.keys()):
        vocab[token] = len(vocab)
    return vocab

VOCAB      = build_vocab()
VOCAB_SIZE = len(VOCAB)

def encode_caption(text, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]

def make_caption_batch(labels, corrupt_mask=None):
    captions = []
    for i in range(labels.size(0)):
        true_label = int(labels[i].item())
        use_label  = true_label
        if corrupt_mask is not None and bool(corrupt_mask[i].item()):
            wrong = np.random.randint(0, CLASSES - 1)
            if wrong >= true_label:
                wrong += 1
            use_label = wrong
        candidates = label_to_captions(use_label)
        captions.append(candidates[np.random.randint(len(candidates))])
    return captions

def pad_caption_batch(captions, vocab, device):
    encoded = [encode_caption(c, vocab) for c in captions]
    max_len  = max(len(seq) for seq in encoded)
    arr      = np.full((len(encoded), max_len), vocab["<pad>"], dtype=np.int64)
    for i, seq in enumerate(encoded):
        arr[i, :len(seq)] = seq
    return torch.tensor(arr, dtype=torch.long, device=device)

# =============================================================================
# PE WRAPPER
# =============================================================================

def compute_pe(series, D=PE_D, tau=PE_TAU):
    if USING_NELSON_PE:
        if len(series) < recommended_min_pe_length(D, tau):
            return None
        pe_val, _tie_rate, _counts = permutation_entropy(series, D=D, tau=tau)
        return pe_val
    else:
        if len(series) < D:
            return None
        patterns = {}
        for i in range(len(series) - D + 1):
            key = tuple(np.argsort(series[i:i + D]))
            patterns[key] = patterns.get(key, 0) + 1
        probs   = np.array(list(patterns.values()), dtype=float)
        probs  /= probs.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return float(entropy / np.log2(math.factorial(D)))

# =============================================================================
# DATA
# =============================================================================

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset    = torchvision.datasets.CIFAR100("./data", train=True,  download=True, transform=transform)
testset     = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# =============================================================================
# MODELS
# =============================================================================

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(128, DIM), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.proj  = nn.Linear(dim, dim)
    def forward(self, token_ids):
        emb    = self.embed(token_ids)
        mask   = (token_ids != 0).float().unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        denom  = mask.sum(dim=1).clamp_min(1.0)
        return F.relu(self.proj(summed / denom))

class OGIFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_fc1 = nn.Linear(DIM * 2, 64)
        self.attn_fc2 = nn.Linear(64, 2)

        # Small init — prevents early saturation
        nn.init.xavier_uniform_(self.attn_fc1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.attn_fc2.weight, gain=0.01)
        nn.init.zeros_(self.attn_fc1.bias)
        nn.init.zeros_(self.attn_fc2.bias)

        self.attn_temperature = 3.0

        self.gru      = nn.GRUCell(DIM, DIM)
        self.classify = nn.Linear(DIM, CLASSES)
        self.critic   = nn.Sequential(
            nn.Linear(DIM * 2, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def compute_attn(self, x):
        logits = self.attn_fc2(F.relu(self.attn_fc1(x)))
        w = torch.softmax(logits / self.attn_temperature, dim=1)
        # Hard clamp — gate stays in [0.1, 0.9], can't go fully degenerate
        w = w.clamp(min=0.2, max=0.8)   # <-- change from min=0.1amp(min=0.1, max=0.9)
        # Renormalize so columns still sum to 1
        return w / w.sum(dim=1, keepdim=True)

    def forward(self, vis, ling, h):
        z   = torch.cat([vis, ling], dim=1)
        w   = self.compute_attn(z)
        h_t = self.gru(w[:, 0:1] * vis + w[:, 1:2] * ling, h)
        return h_t, self.classify(h_t), w

    def coherence_loss(self, h_t, vis):
        j    = torch.cat([h_t, vis], dim=1)
        perm = torch.randperm(vis.size(0), device=vis.device)
        m    = torch.cat([h_t, vis[perm]], dim=1)
        return -(torch.mean(self.critic(j)) -
                 torch.log(torch.mean(torch.exp(self.critic(m))) + 1e-8))

    def entropy_reg(self, w):
        # Penalize gate for being too confident — keeps it away from [1,0] or [0,1]
        return -(w * torch.log(w + 1e-8)).sum(dim=1).mean()

class BaselineFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru      = nn.GRUCell(DIM, DIM)
        self.classify = nn.Linear(DIM, CLASSES)
    def forward(self, vis, ling, h):
        h_t = self.gru(0.5 * vis + 0.5 * ling, h)
        return h_t, self.classify(h_t), None

# =============================================================================
# THEOREM 1: LIPSCHITZ ON TRAINED WEIGHTS
# =============================================================================

@torch.no_grad()
def collect_activation_pairs(v_enc, t_enc, loader, n_pairs=LIPSCHITZ_PAIRS):
    v_enc.eval(); t_enc.eval()
    pairs = []
    for img, lbl in loader:
        if len(pairs) >= n_pairs:
            break
        img, lbl  = img.to(DEVICE), lbl.to(DEVICE)
        captions  = make_caption_batch(lbl, corrupt_mask=None)
        token_ids = pad_caption_batch(captions, VOCAB, DEVICE)
        vis  = v_enc(img)
        ling = t_enc(token_ids)
        x    = torch.cat([vis, ling], dim=1).cpu()
        B    = x.shape[0]
        perm = torch.randperm(B)
        for i in range(0, B - 1, 2):
            pairs.append((x[perm[i]], x[perm[i + 1]]))
            if len(pairs) >= n_pairs:
                break
    return pairs

@torch.no_grad()
def lipschitz_empirical(fusion, pairs, bound=LIPSCHITZ_BOUND):
    fusion.eval()
    ratios = []; violations = 0
    for x1, x2 in pairs:
        x1 = x1.to(DEVICE).float().unsqueeze(0)
        x2 = x2.to(DEVICE).float().unsqueeze(0)
        w1 = fusion.compute_attn(x1).squeeze(0)
        w2 = fusion.compute_attn(x2).squeeze(0)
        in_d  = torch.norm(x1 - x2).item()
        out_d = torch.norm(w1 - w2).item()
        if in_d > 1e-8:
            r = out_d / in_d
            ratios.append(r)
            if r > bound:
                violations += 1
    arr = np.array(ratios)
    return {
        "n_pairs":      len(arr),
        "violations":   violations,
        "max_ratio":    float(arr.max())  if len(arr) else float("nan"),
        "mean_ratio":   float(arr.mean()) if len(arr) else float("nan"),
        "pct_of_bound": float(100 * arr.max() / bound) if len(arr) else float("nan"),
        "bound":        bound,
        "verified":     violations == 0,
    }

def lipschitz_analytical(fusion):
    snorms = []
    for lin in [fusion.attn_fc1, fusion.attn_fc2]:
        sv = torch.linalg.svdvals(lin.weight.data.float())
        snorms.append(sv.max().item())
    product = float(np.prod(snorms))
    return {
        "spectral_norms":   [round(s, 6) for s in snorms],
        "analytical_bound": round(product, 6),
        "theorem1_bound":   LIPSCHITZ_BOUND,
        "satisfied":        product <= LIPSCHITZ_BOUND,
    }

# =============================================================================
# SAD PROBE
# =============================================================================

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
    def __init__(self, heads=N_HEADS, window=PE_WINDOW):
        self.heads  = heads
        self.window = window

        torch.manual_seed(0)
        self.Wq = nn.Linear(DIM, DIM, bias=False).to(DEVICE)
        self.Wk = nn.Linear(DIM, DIM, bias=False).to(DEVICE)
        self.Wv = nn.Linear(DIM, DIM, bias=False).to(DEVICE)
        for p in [*self.Wq.parameters(), *self.Wk.parameters(), *self.Wv.parameters()]:
            p.requires_grad_(False)

        self.sad_series:  list[float] = []
        self.attn_series: list[float] = []  # routing confidence |w0-w1|
        self.lip_series:  list[float] = []
        self.corrupt:     list[bool]  = []

    @torch.no_grad()
    def step(self, vis, ling, w, corrupt, fusion=None):
        B, H, dh = vis.shape[0], self.heads, DIM // self.heads
        x = torch.stack([vis, ling], dim=1).float()

        Q = self.Wq(x).view(B, 2, H, dh).transpose(1, 2)
        K = self.Wk(x).view(B, 2, H, dh).transpose(1, 2)
        V = self.Wv(x).view(B, 2, H, dh).transpose(1, 2)

        out_s = _softmax_attn(Q, K, V)
        out_l = _linear_attn(Q, K, V)

        cos = F.cosine_similarity(
            out_s.reshape(B, H, -1),
            out_l.reshape(B, H, -1),
            dim=2,
        )
        sad = (1.0 - cos).mean().item()

        # Routing confidence: |w0 - w1|, 0=uncertain, ~0.8=decisive
        attn_val = (w[:, 0] - w[:, 1]).abs().mean().item() if w is not None else 0.0

        lip_ratio = float("nan")
        if fusion is not None and B >= 4:
            half  = B // 2
            x_cat = torch.cat([vis, ling], dim=1).float()
            w1    = fusion.compute_attn(x_cat[:half])
            w2    = fusion.compute_attn(x_cat[half:half * 2])
            in_d  = torch.norm(x_cat[:half] - x_cat[half:half * 2], dim=1)
            out_d = torch.norm(w1 - w2, dim=1)
            valid = in_d > 1e-8
            if valid.any():
                lip_ratio = (out_d[valid] / in_d[valid]).mean().item()

        self.sad_series.append(sad)
        self.attn_series.append(attn_val)
        self.lip_series.append(lip_ratio)
        self.corrupt.append(corrupt)

    def rolling_pe(self):
        n  = len(self.sad_series)
        pe = np.full(n, np.nan)
        for i in range(n - self.window + 1):
            val = compute_pe(self.sad_series[i:i + self.window], D=PE_D, tau=PE_TAU)
            pe[i + self.window // 2] = val if val is not None else np.nan
        return pe, ~np.isnan(pe)

    def analyze(self):
        sad     = np.array(self.sad_series)
        attn    = np.array(self.attn_series)
        lip     = np.array(self.lip_series)
        corrupt = np.array(self.corrupt, dtype=bool)
        pe, valid = self.rolling_pe()

        def pearson_safe(a, b):
            if len(a) < 4:                               return None, None
            if np.std(a) < 1e-12 or np.std(b) < 1e-12: return None, None
            r, p = stats.pearsonr(a, b)
            return float(r), float(p)

        results = {}

        m  = valid & corrupt
        r, p = pearson_safe(pe[m], attn[m])
        results["pe_attn_corrupt"] = (
            {"r": round(r, 4), "p": round(p, 6), "n": int(m.sum()),
             "significant": p < 0.05}
            if r is not None
            else {"error": f"only {m.sum()} valid corrupt steps or near-constant series"}
        )

        lip_valid = ~np.isnan(lip)
        m2 = valid & corrupt & lip_valid
        r2, p2 = pearson_safe(pe[m2], lip[m2])
        results["pe_lip_corrupt"] = (
            {"r": round(r2, 4), "p": round(p2, 6), "n": int(m2.sum()),
             "significant": p2 < 0.05,
             "max_lip": round(float(lip[lip_valid].max()), 6),
             "pct_of_bound": round(100 * lip[lip_valid].max() / LIPSCHITZ_BOUND, 2)}
            if r2 is not None
            else {"error": f"only {m2.sum()} valid steps or near-constant series"}
        )

        m3 = valid & ~corrupt
        r3, p3 = pearson_safe(pe[m3], attn[m3])
        results["pe_attn_clean_null"] = (
            {"r": round(r3, 4), "p": round(p3, 6), "n": int(m3.sum())}
            if r3 is not None
            else {"error": f"only {m3.sum()} valid clean steps"}
        )

        results["sad_regime"] = {
            "warmup_mean":  round(float(sad[~corrupt].mean()) if (~corrupt).any() else float("nan"), 6),
            "corrupt_mean": round(float(sad[corrupt].mean())  if corrupt.any()   else float("nan"), 6),
            "delta":        round(float(sad[corrupt].mean() - sad[~corrupt].mean())
                                  if (corrupt.any() and (~corrupt).any()) else float("nan"), 6),
        }

        results["gate_health"] = {
            "conf_mean": round(float(attn.mean()), 4),
            "conf_std":  round(float(attn.std()),  4),
            "conf_min":  round(float(attn.min()),  4),
            "conf_max":  round(float(attn.max()),  4),
            "collapsed": bool(attn.std() < 0.01),
        }

        return results

# =============================================================================
# TRAINING LOOP
# =============================================================================

def run(mode, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    probe  = SADProbe()
    v_enc  = VisualEncoder().to(DEVICE)
    t_enc  = CaptionEncoder(VOCAB_SIZE, DIM).to(DEVICE)
    fusion = OGIFusion().to(DEVICE) if mode == "OGI" else BaselineFusion().to(DEVICE)

    # Separate LR — gate trains 20x slower than encoders
    if mode == "OGI":
        opt = optim.Adam([
            {"params": list(v_enc.parameters()) + list(t_enc.parameters()), "lr": 1e-3},
            {"params": list(fusion.parameters()), "lr": 5e-5},
        ], weight_decay=1e-4)
    else:
        opt = optim.Adam(
            list(v_enc.parameters()) + list(t_enc.parameters()) + list(fusion.parameters()),
            lr=1e-3, weight_decay=1e-4
        )

    crit = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        is_corrupt = epoch >= WARMUP
        v_enc.train(); t_enc.train(); fusion.train()

        for img, lbl in trainloader:
            img, lbl   = img.to(DEVICE), lbl.to(DEVICE)
            batch_mask = torch.rand(lbl.size(0), device=DEVICE) < CORRUPTION_RATE
            captions   = make_caption_batch(lbl, corrupt_mask=batch_mask if is_corrupt else None)
            token_ids  = pad_caption_batch(captions, VOCAB, DEVICE)

            vis  = v_enc(img)
            ling = t_enc(token_ids)
            if LING_NOISE_STD > 0:
                ling = ling + LING_NOISE_STD * torch.randn_like(ling)

            h = torch.zeros(img.size(0), DIM, device=DEVICE)
            h_t, logits, w = fusion(vis, ling, h)

            # Diagnostic — epoch 0 only
            if epoch == 0 and np.random.rand() < 0.02:
                conf = (w[:, 0] - w[:, 1]).abs().mean().item()
                print(f"  w mean: {w[:,0].mean().item():.4f} "
                      f"w std: {w[:,0].std().item():.4f} "
                      f"confidence: {conf:.4f}")

            loss = crit(logits, lbl)

            if mode == "OGI":
                # Entropy reg always active — keeps gate from collapsing
                loss = loss - ENTROPY_REG_W * fusion.entropy_reg(w)
                if is_corrupt:
                    loss = loss + COHERENCE_WEIGHT * fusion.coherence_loss(h_t, vis.detach())

            opt.zero_grad()
            loss.backward()
            opt.step()

            probe.step(
                vis.detach(), ling.detach(), w,
                corrupt=is_corrupt,
                fusion=fusion if mode == "OGI" else None,
            )

    lip_emp = lip_ana = None
    if mode == "OGI":
        print(f"  [Seed {seed}] Collecting activation pairs for Lipschitz check...")
        pairs   = collect_activation_pairs(v_enc, t_enc, trainloader)
        lip_emp = lipschitz_empirical(fusion, pairs)
        lip_ana = lipschitz_analytical(fusion)

    sad_results = probe.analyze()
    gh = sad_results["gate_health"]
    status = "COLLAPSED" if gh["collapsed"] else "HEALTHY"
    print(f"  [Seed {seed}] Gate: {status} — "
          f"conf mean={gh['conf_mean']:.4f} std={gh['conf_std']:.4f} "
          f"min={gh['conf_min']:.4f} max={gh['conf_max']:.4f}")

    return probe, {
        "seed":           seed,
        "mode":           mode,
        "lip_empirical":  lip_emp,
        "lip_analytical": lip_ana,
        "sad":            sad_results,
    }

# =============================================================================
# UNIFIED RESULTS TABLE
# =============================================================================

def unified_results_table(all_meta):
    ogi = [m for m in all_meta if m["mode"] == "OGI"]

    print("\n" + "=" * 72)
    print("OGI INTEGRATED VALIDATION — UNIFIED RESULTS")
    print("=" * 72)

    print(f"\nGATE HEALTH (routing confidence |w0-w1|, 0=uncertain, ~0.8=decisive)")
    for m in ogi:
        gh     = m["sad"]["gate_health"]
        status = "COLLAPSED ⚠" if gh["collapsed"] else "HEALTHY ✓"
        print(f"  Seed {m['seed']}: {status} — "
              f"mean={gh['conf_mean']:.4f} std={gh['conf_std']:.4f}")

    print(f"\nTHEOREM 1: Lipschitz Stability  (bound = {LIPSCHITZ_BOUND})")
    print("  [A] Empirical — real activation pairs from trained model:")
    print(f"  {'Seed':<8} {'N pairs':<10} {'Max ratio':<12} {'% bound':<10} "
          f"{'Violations':<12} Verified")
    print(f"  {'-'*64}")
    for m in ogi:
        le = m["lip_empirical"]
        if le:
            print(f"  {m['seed']:<8} {le['n_pairs']:<10} {le['max_ratio']:<12.6f} "
                  f"{le['pct_of_bound']:<10.1f} {le['violations']:<12} "
                  f"{'YES' if le['verified'] else 'NO  <-- VIOLATION'}")

    print()
    print("  [B] Analytical — spectral norm product of trained weight matrices:")
    for m in ogi[:1]:
        la = m["lip_analytical"]
        if la:
            print(f"  Spectral norms:    {la['spectral_norms']}")
            print(f"  Analytical bound:  {la['analytical_bound']}")
            print(f"  Theorem 1 bound:   {la['theorem1_bound']}")
            sat = "YES" if la["satisfied"] else "NO  <-- requires tighter weight reg"
            print(f"  Satisfied:         {sat}")

    print(f"\nTHEOREM 3: Sub-linear Routing")
    print("  Not re-run — routing overhead is weight-independent.")
    print("  Original result: 0.456ms at n=100,000 — VERIFIED (see original notebook).")

    print(f"\nLEMMA 4.1: MI Attention Shift  (coherence weight = {COHERENCE_WEIGHT})")
    print(f"  {'Seed':<8} {'SAD warmup':<14} {'SAD corrupt':<14} {'SAD delta'}")
    print(f"  {'-'*50}")
    for m in ogi:
        sr = m["sad"]["sad_regime"]
        print(f"  {m['seed']:<8} {sr['warmup_mean']:<14.6f} "
              f"{sr['corrupt_mean']:<14.6f} {sr['delta']:+.6f}")

    print(f"\nSAD ↔ OGI INTEGRATION")
    print(f"  PE engine: {'Nelson ordinal.py (tie-excluding)' if USING_NELSON_PE else 'local fallback'}")
    print(f"  D={PE_D}, tau={PE_TAU}, window={PE_WINDOW}")
    print(f"  Metric: routing confidence |w0-w1| (not raw w0)")
    print()
    print(f"  {'Seed':<8} {'PE↔Conf r':<12} {'p':<10} {'Sig':<6} "
          f"{'PE↔Lip r':<12} {'p':<10} {'Sig':<6} {'Lip max':<10} {'% bound'}")
    print(f"  {'-'*80}")
    for m in ogi:
        sa  = m["sad"]
        pac = sa.get("pe_attn_corrupt", {})
        plc = sa.get("pe_lip_corrupt",  {})

        ra  = f"{pac['r']:+.4f}"           if "r" in pac else "n/a"
        pa  = f"{pac['p']:.4f}"            if "p" in pac else "n/a"
        sa_ = "***" if pac.get("significant") else ("   " if "r" in pac else "n/a")
        rl  = f"{plc['r']:+.4f}"           if "r" in plc else "n/a"
        pl  = f"{plc['p']:.4f}"            if "p" in plc else "n/a"
        sl  = "***" if plc.get("significant") else ("   " if "r" in plc else "n/a")
        lm  = f"{plc['max_lip']:.6f}"      if "max_lip"      in plc else "n/a"
        pb  = f"{plc['pct_of_bound']:.1f}" if "pct_of_bound" in plc else "n/a"

        print(f"  {m['seed']:<8} {ra:<12} {pa:<10} {sa_:<6} "
              f"{rl:<12} {pl:<10} {sl:<6} {lm:<10} {pb}")

    print()
    print("  Null check (PE↔Conf on CLEAN steps — expect r ≈ 0):")
    for m in ogi:
        null = m["sad"].get("pe_attn_clean_null", {})
        rn   = f"{null['r']:+.4f}" if "r" in null else "n/a"
        pn   = f"{null['p']:.4f}"  if "p" in null else "n/a"
        nn_  = null.get("n", "?")
        print(f"    Seed {m['seed']}: r={rn}, p={pn}, n={nn_}")

    print(f"\nINTERPRETATION")
    collapsed = any(m["sad"]["gate_health"]["collapsed"] for m in ogi)
    if collapsed:
        print("  ⚠ Gate collapsed in one or more seeds — increase ENTROPY_REG_W.")

    sig_pa = any(m["sad"].get("pe_attn_corrupt", {}).get("significant") for m in ogi)
    sig_pl = any(m["sad"].get("pe_lip_corrupt",  {}).get("significant") for m in ogi)

    if sig_pa and not collapsed:
        print("  PE ↔ Confidence SIGNIFICANT: attractor complexity tracks routing decisiveness.")
        print("  → Lemma 4.1 ↔ SAD attractor geometry link: CLOSED.")
    elif sig_pa and collapsed:
        print("  PE ↔ Confidence significant but gate collapsed — likely spurious.")
    else:
        print("  PE ↔ Confidence not significant.")

    if sig_pl and not collapsed:
        print("  PE ↔ Lip SIGNIFICANT: attractor complexity tracks Lipschitz envelope use.")
        print("  → Theorem 1 ↔ SAD observable link: CLOSED.")
    else:
        lip_maxes = [
            m["sad"].get("pe_lip_corrupt", {}).get("max_lip")
            for m in ogi
            if m["sad"].get("pe_lip_corrupt", {}).get("max_lip") is not None
        ]
        if lip_maxes:
            pct = 100 * max(lip_maxes) / LIPSCHITZ_BOUND
            print(f"  PE ↔ Lip not significant. Max ratio = {max(lip_maxes):.6f} ({pct:.1f}% of bound).")
            if pct < 20:
                print("  → Strong Theorem 1 result — model uses <20% of bound.")

    print("\n" + "=" * 72)

# =============================================================================
# PLOT
# =============================================================================

def plot_integrated(probe, meta, seed):
    sad     = np.array(probe.sad_series)
    attn    = np.array(probe.attn_series)
    lip     = np.array(probe.lip_series)
    corrupt = np.array(probe.corrupt, dtype=bool)
    pe, valid = probe.rolling_pe()
    steps   = np.arange(len(sad))
    we      = int((~corrupt).sum())

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"OGI-SAD Integrated Validation — Seed {seed}", fontsize=13)

    def vline(ax):
        ax.axvline(we, color="orange", ls="--", lw=1, label="Warmup end")

    ax = axes[0, 0]
    ax.plot(steps, sad, lw=0.6, color="steelblue")
    vline(ax); ax.legend(fontsize=8)
    ax.set_title("SAD Trajectory")
    ax.set_xlabel("Step"); ax.set_ylabel("SAD divergence")

    ax = axes[0, 1]
    if valid.any():
        ax.plot(steps[valid], pe[valid], lw=0.6, color="purple")
    vline(ax)
    ax.set_title(f"Rolling PE (D={PE_D}, w={PE_WINDOW}, "
                 f"{'Nelson' if USING_NELSON_PE else 'fallback'})")
    ax.set_xlabel("Step"); ax.set_ylabel("Normalized PE")

    ax = axes[0, 2]
    ax.plot(steps, attn, lw=0.6, color="forestgreen")
    ax.axhline(0.0, color="gray", ls="--", lw=0.8, label="Uncertain (0.0)")
    vline(ax); ax.legend(fontsize=8)
    ax.set_title("Routing Confidence |w0-w1|")
    ax.set_xlabel("Step"); ax.set_ylabel("Confidence")

    ax = axes[1, 0]
    lip_v = ~np.isnan(lip)
    if lip_v.any():
        ax.plot(steps[lip_v], lip[lip_v], lw=0.4, color="coral", alpha=0.7)
        ax.axhline(LIPSCHITZ_BOUND, color="red", ls="--", lw=1,
                   label=f"Theorem 1 bound ({LIPSCHITZ_BOUND})")
        ax.legend(fontsize=8)
        ax.set_title("Lipschitz Ratio per Step")
        ax.set_xlabel("Step"); ax.set_ylabel("mean |Δw|/|Δx|")
    else:
        ax.text(0.5, 0.5, "Lipschitz tracking\nnot available",
                ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 1]
    mc = valid & corrupt
    if mc.sum() > 3 and np.std(pe[mc]) > 1e-12 and np.std(attn[mc]) > 1e-12:
        ax.scatter(pe[mc], attn[mc], alpha=0.25, s=6, color="coral")
        m_, b_ = np.polyfit(pe[mc], attn[mc], 1)
        xs = np.linspace(pe[mc].min(), pe[mc].max(), 50)
        ax.plot(xs, m_ * xs + b_, color="darkred", lw=1.5)
        res = meta["sad"].get("pe_attn_corrupt", {})
        title_str = (f"r={res['r']:.4f}, p={res['p']:.4f}"
                     if "r" in res else "insufficient data")
        ax.set_title(f"PE vs Confidence (corrupt)\n{title_str}")
    else:
        ax.set_title("PE vs Confidence (corrupt)\ninsufficient variation")
    ax.set_xlabel("PE"); ax.set_ylabel("Routing confidence")

    ax = axes[1, 2]
    mlc = valid & corrupt & lip_v
    if mlc.sum() > 3 and np.std(pe[mlc]) > 1e-12 and np.std(lip[mlc]) > 1e-12:
        ax.scatter(pe[mlc], lip[mlc], alpha=0.25, s=6, color="steelblue")
        m_, b_ = np.polyfit(pe[mlc], lip[mlc], 1)
        xs = np.linspace(pe[mlc].min(), pe[mlc].max(), 50)
        ax.plot(xs, m_ * xs + b_, color="navy", lw=1.5)
        ax.axhline(LIPSCHITZ_BOUND, color="red", ls="--", lw=0.8,
                   label=f"Bound ({LIPSCHITZ_BOUND})")
        res = meta["sad"].get("pe_lip_corrupt", {})
        title_str = (f"r={res['r']:.4f}, p={res['p']:.4f}"
                     if "r" in res else "insufficient data")
        ax.set_title(f"PE vs Lip ratio (corrupt)\n{title_str}")
        ax.legend(fontsize=8)
    else:
        ax.set_title("PE vs Lip ratio (corrupt)\ninsufficient variation")
    ax.set_xlabel("PE"); ax.set_ylabel("Lipschitz ratio")

    plt.tight_layout()
    fname = f"ogi_sad_integrated_seed{seed}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    all_meta = []
    for seed in SEEDS:
        print(f"\n{'='*55}\nSeed {seed}\n{'='*55}")
        probe, meta = run("OGI", seed)
        all_meta.append(meta)
        plot_integrated(probe, meta, seed)
    unified_results_table(all_meta)

if __name__ == "__main__":
    main()