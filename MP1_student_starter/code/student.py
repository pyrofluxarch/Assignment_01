"""Standalone GPT architectures with causal window copy and train-only n-grams.

Rare names and repeated phrases can benefit from an exact local memory: if
the current token occurred earlier, its observed successor is a candidate
next token. No data or state is carried between independent windows.

Set ``copy_weight`` to 0 to ablate window copy, or ``use_train_memory`` to
False to ablate the n-gram memory. The trainable checkpoint tensors are
identical to the baseline. If use_train_memory is true, a companion
student_stats.npz speeds startup; otherwise the tables are rebuilt from the
supplied training split (never validation or test text).

The rotary architecture is selected by config['architecture'] = 'rotary_swiglu'.
All model code is in this file; the classroom model.py remains untouched.
"""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pathlib import Path


def _training_statistics(data_dir):
    """Reconstruct the optional statistics asset from the supplied train split."""
    from tokenizers import Tokenizer

    tokenizer_path = data_dir / 'tokenizer.json'
    train_path = data_dir / 'wikitext_train.txt'
    if not tokenizer_path.is_file() or not train_path.is_file():
        raise FileNotFoundError(
            'student_stats.npz is missing; to rebuild it, place the supplied '
            'data/tokenizer.json and data/wikitext_train.txt beside student.py, '
            'or copy student_stats.npz into the code directory.'
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = np.asarray(tokenizer.encode(train_path.read_text(encoding='utf-8')).ids, dtype=np.int64)
    tables = {}
    for order in (2, 3):
        size = order + 1
        packed = np.zeros(len(ids) - size + 1, dtype=np.int64)
        for index in range(size):
            packed = (packed << 11) | ids[index:len(ids) - size + index + 1]
        keys, counts = np.unique(packed, return_counts=True)
        tables[order] = (keys, counts)
    return tables


class Block(nn.Module):
    """Classroom GPT block, retained to load earlier student checkpoints."""
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([Block(width, config['heads']) for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))


class CopyGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.copy_weight = float(config.get('copy_weight', 0.20))
        self.copy_decay = float(config.get('copy_decay', 64.0))
        self.ngram_smoothing = tuple(float(x) for x in config.get('ngram_smoothing', (150., 80.)))
        adaptive2 = config.get('adaptive_ngram2')
        adaptive3 = config.get('adaptive_ngram3')
        self.adaptive_ngram2 = tuple(float(x) for x in adaptive2) if adaptive2 is not None else None
        self.adaptive_ngram3 = tuple(float(x) for x in adaptive3) if adaptive3 is not None else None
        if not 0 <= self.copy_weight < 1:
            raise ValueError('copy_weight must lie in [0, 1).')
        if self.copy_decay <= 0:
            raise ValueError('copy_decay must be positive.')
        if len(self.ngram_smoothing) != 2 or any(x <= 0 for x in self.ngram_smoothing):
            raise ValueError('ngram_smoothing needs two positive values.')
        for name, adaptive in (('adaptive_ngram2', self.adaptive_ngram2), ('adaptive_ngram3', self.adaptive_ngram3)):
            if adaptive is not None and (len(adaptive) != 2 or adaptive[0] <= 0 or adaptive[1] < 0):
                raise ValueError(f'{name} needs a positive base and nonnegative count scale.')
        self.use_train_memory = bool(config.get('use_train_memory', False))
        if self.use_train_memory:
            root = Path(__file__).resolve().parent
            stats_path = root / 'student_stats.npz'
            if stats_path.is_file():
                with np.load(stats_path) as stats:
                    tables = {order: (stats[f'keys{order}'], stats[f'counts{order}']) for order in (2, 3)}
            else:
                tables = _training_statistics(root / 'data')
            for order, (keys, counts) in tables.items():
                self.register_buffer(f'keys{order}', torch.from_numpy(keys.astype(np.int64)), persistent=False)
                self.register_buffer(f'counts{order}', torch.from_numpy(counts.astype(np.int32)), persistent=False)

    def predict_log_probs(self, ids):
        logits = self(ids).float()
        neural_log = F.log_softmax(logits, dim=-1)
        if (self.copy_weight == 0 or ids.shape[1] < 2) and not self.use_train_memory:
            return neural_log
        if self.copy_weight == 0 or ids.shape[1] < 2:
            log_probs = neural_log
        else:
            log_probs = self._window_copy(ids, logits, neural_log)
        if self.use_train_memory:
            probabilities = log_probs.exp()
            for order, smoothing in zip((2, 3), self.ngram_smoothing):
                probabilities = self._add_ngram(probabilities, ids, order, smoothing)
            log_probs = probabilities.log()
        return log_probs

    def _window_copy(self, ids, logits, neural_log):
        batch, length = ids.shape
        # At t, a match at j<t contributes the already observed token j+1.
        # The last input position has no observed successor.
        positions = torch.arange(length, device=ids.device)
        previous = positions[:-1]
        allowed = previous[None, :] < positions[:, None]
        matches = (ids[:, :, None] == ids[:, None, :-1]) & allowed
        recency = torch.exp(-(positions[:, None] - previous[None, :]).clamp_min(0) / self.copy_decay)
        weights = matches.to(logits.dtype) * recency[None, :, :]
        totals = weights.sum(dim=-1, keepdim=True)

        copy = torch.zeros_like(logits)
        successors = ids[:, None, 1:].expand(batch, length, length - 1)
        copy.scatter_add_(-1, successors, weights)
        copy = copy / totals.clamp_min(1e-12)
        gate = self.copy_weight * (totals > 0).to(logits.dtype)
        # logaddexp retains finite neural mass even for extremely small
        # probabilities that would underflow in a direct softmax mixture.
        return torch.logaddexp(
            torch.log1p(-gate) + neural_log,
            gate.log() + copy.log(),
        )

    def _add_ngram(self, probabilities, ids, order, smoothing):
        """Dirichlet backoff to the current normalized lower-order predictor."""
        batch, length = ids.shape
        keys = getattr(self, f'keys{order}')
        counts = getattr(self, f'counts{order}')
        context = torch.zeros((batch, length), dtype=torch.long, device=ids.device)
        for offset in range(order - 1, -1, -1):
            context[:, order-1:] = (context[:, order-1:] << 11) | ids[:, order-1-offset:length-offset]
        query = (context.reshape(-1) << 11)
        starts = torch.searchsorted(keys, query, right=False)
        ends = torch.searchsorted(keys, query + 2048, right=False)
        valid = (torch.arange(length, device=ids.device) >= order-1).expand(batch, -1).reshape(-1)
        ends = torch.where(valid, ends, starts)
        lengths = ends - starts
        rows = torch.repeat_interleave(torch.arange(batch*length, device=ids.device), lengths)
        offsets = torch.cumsum(lengths, 0) - lengths
        source = starts[rows] + torch.arange(rows.numel(), device=ids.device) - torch.repeat_interleave(offsets, lengths)
        value = counts[source].to(probabilities.dtype)
        total = torch.zeros(batch*length, device=ids.device, dtype=probabilities.dtype)
        total.index_add_(0, rows, value)
        adaptive = self.adaptive_ngram2 if order == 2 else self.adaptive_ngram3
        if adaptive is not None:
            base, scale = adaptive
            smoothing = base + scale * total
        denominator = total + smoothing
        result = probabilities.reshape(-1, probabilities.shape[-1]) * (smoothing / denominator)[:, None]
        flat_indexes = rows * probabilities.shape[-1] + (keys[source] & 2047)
        result.reshape(-1).index_add_(0, flat_indexes, value / denominator[rows])
        return result.reshape_as(probabilities)


class RotaryBlock(nn.Module):
    def __init__(self, width, heads, context):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError('Rotary attention requires an even head dimension.')
        self.heads = heads
        self.norm1 = nn.RMSNorm(width)
        self.norm2 = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        hidden = ((8 * width // 3 + 15) // 16) * 16
        self.up = nn.Linear(width, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        dim = width // heads
        pos = torch.arange(context).float()[:, None]
        freq = 10000 ** (-torch.arange(0, dim, 2).float() / dim)
        angles = pos * freq
        self.register_buffer('cos', torch.cos(angles)[None, None, :, :], persistent=False)
        self.register_buffer('sin', torch.sin(angles)[None, None, :, :], persistent=False)

    def rotate(self, x):
        length = x.shape[-2]
        a, b = x[..., ::2], x[..., 1::2]
        c, s = self.cos[:, :, :length], self.sin[:, :, :length]
        return torch.stack((a*c-b*s, a*s+b*c), dim=-1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.rotate(q), self.rotate(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        a, b = self.up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(a) * b)


class RotaryCopyGPT(CopyGPT):
    def __init__(self, config):
        super().__init__(config)
        width = config['width']
        self.pos = None
        self.blocks = nn.ModuleList([RotaryBlock(width, config['heads'], self.context) for _ in range(config['depth'])])
        self.norm = nn.RMSNorm(width)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def build_model(config):
    architecture = config.get('architecture', 'baseline')
    if architecture == 'rotary_swiglu':
        return RotaryCopyGPT(config)
    if architecture == 'baseline':
        return CopyGPT(config)
    raise ValueError(f'Unknown student architecture: {architecture}')
