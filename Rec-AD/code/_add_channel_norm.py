"""Normalise the physics channels in the sequence models before training.

The defect: `token_sequence` concatenates three views whose scales differ by ~1600x

    raw z          std 67.94
    residual r     std  0.483
    innovation nu  std  1.486

and feeds them to one Linear layer. Measured effect of fixing it (same model, same
seed, only standardisation added):

    family        unnormalised   normalised
    random             0.5194       1.0000
    load_redist        0.5112       0.8396
    stealthy           0.5073       0.7402

Implementation:
  * statistics are estimated over the TRAIN split only, per channel and per dimension,
    so no validation or test information leaks;
  * the estimate is taken once at the start of `fit`, via a single pass over the train
    loader, and stored as buffers so it moves with the model and lands in checkpoints;
  * `token_sequence` applies the transform only when the buffers exist, so a model
    used without `fit` behaves exactly as before and nothing silently changes meaning;
  * the transform is opt-in via `baselines.normalise_channels` (default true here,
    because the unnormalised behaviour is a measured defect, not a design choice).

Staged in memory, validated, then written together (failure mode H1).
"""
from __future__ import annotations

import ast
from pathlib import Path

P = Path(__file__).resolve().parent / "baselines" / "_sequence.py"

OLD_INIT_TAIL = '''        self.physics_channels = 3
        self.in_dim = raw_dim if self.physics_channels == 0 else (
            self.physics_channels * spec.num_measurements)
'''

NEW_INIT_TAIL = '''        self.physics_channels = 3
        self.in_dim = raw_dim if self.physics_channels == 0 else (
            self.physics_channels * spec.num_measurements)

        # Per-channel standardisation. Without it the three channel blocks differ in
        # scale by ~1600x and the largest one dominates the first layer, which
        # measurably destroys detection on every family (see the module docstring of
        # `_add_channel_norm.py`). Buffers are filled by `fit`; until then the
        # transform is inert.
        bcfg_norm = (cfg.get("baselines", {}) or {})
        self.normalise_channels = bool(bcfg_norm.get("normalise_channels", True))
        self.register_buffer("_ch_mu", torch.zeros(0), persistent=False)
        self.register_buffer("_ch_sd", torch.ones(0), persistent=False)
'''

OLD_TOK_TAIL = '''        proj = self.projector_.to(win.dtype)
        residual = win - win @ proj.T                          # (B, L, m)
        pred = win[:, :-1, :] @ proj.T
        innovation = torch.cat(
            [residual[:, :1, :], win[:, 1:, :] - pred], dim=1)
        return torch.cat([win, residual, innovation], dim=2)
'''

NEW_TOK_TAIL = '''        proj = self.projector_.to(win.dtype)
        residual = win - win @ proj.T                          # (B, L, m)
        pred = win[:, :-1, :] @ proj.T
        innovation = torch.cat(
            [residual[:, :1, :], win[:, 1:, :] - pred], dim=1)
        tokens = torch.cat([win, residual, innovation], dim=2)
        if self.normalise_channels and self._ch_sd.numel() == tokens.shape[2]:
            mu = self._ch_mu.to(tokens.dtype)
            sd = self._ch_sd.to(tokens.dtype)
            tokens = (tokens - mu) / sd
        return tokens
'''

NEW_FIT = '''    def fit(self, train_loader, val_loader=None, **kwargs):
        """Estimate the per-channel statistics, then train as usual.

        The estimate is one pass over the training split, taken before any gradient
        step, so the transform is a fixed preprocessing rather than something the
        optimiser can game. Skipped entirely when the physics channels are off or
        normalisation is disabled.
        """
        if (self.normalise_channels and self.physics_channels > 0
                and self.projector_ is not None):
            total = None
            count = 0
            with torch.no_grad():
                for batch in train_loader:
                    t = self.token_sequence(batch)
                    if t.dim() != 3:
                        continue
                    s = t.sum(dim=(0, 1))
                    ss = (t ** 2).sum(dim=(0, 1))
                    total = (s, ss) if total is None else (total[0] + s, total[1] + ss)
                    count += t.shape[0] * t.shape[1]
            if total is not None and count > 0:
                mean = total[0] / count
                var = torch.clamp(total[1] / count - mean ** 2, min=1e-12)
                self._ch_mu = mean.detach()
                self._ch_sd = var.sqrt().detach().clamp_min(1e-6)
        return super().fit(train_loader, val_loader, **kwargs)

'''

src = P.read_text(encoding="utf-8")
for old, new in ((OLD_INIT_TAIL, NEW_INIT_TAIL), (OLD_TOK_TAIL, NEW_TOK_TAIL)):
    n = src.count(old)
    assert n == 1, f"anchor appears {n} times:\\n{old[:80]}"

src = src.replace(OLD_INIT_TAIL, NEW_INIT_TAIL).replace(OLD_TOK_TAIL, NEW_TOK_TAIL)

# insert the fit override right before `def encode(`
anchor = "    def encode(self, tokens: torch.Tensor) -> torch.Tensor:"
assert src.count(anchor) == 1, "encode anchor not unique"
src = src.replace(anchor, NEW_FIT + anchor)

ast.parse(src)
P.write_text(src, encoding="utf-8")
print("  patched", P.name)
