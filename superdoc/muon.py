"""
Muon optimizer integration for LayoutLMv3.

Uses native PyTorch Muon (torch.optim.Muon) when available (PyTorch 2.9+),
otherwise falls back to a custom implementation.

https://kellerjordan.github.io/posts/muon/
https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html

Muon should only be used for hidden weight layers. The input embedding, final output layer,
and any internal gains or biases should be optimized using a standard method such as AdamW.
"""

import torch

# Check if native Muon is available (PyTorch 2.9+)
_NATIVE_MUON_AVAILABLE = hasattr(torch.optim, "Muon")


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Fallback implementation for PyTorch < 2.9.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _muon_update(grad, momentum_buffer, beta=0.95, ns_steps=5, nesterov=True):
    """Compute Muon update with momentum and Newton-Schulz orthogonalization."""
    momentum_buffer.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buffer, beta) if nesterov else momentum_buffer
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def _adam_update(grad, exp_avg, exp_avg_sq, step, betas, eps):
    """Standard Adam update computation."""
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    exp_avg_corrected = exp_avg / (1 - betas[0] ** step)
    exp_avg_sq_corrected = exp_avg_sq / (1 - betas[1] ** step)
    return exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + eps)


class _FallbackMuon(torch.optim.Optimizer):
    """Fallback Muon implementation for PyTorch < 2.9."""

    def __init__(self, params, lr=0.02, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                update = _muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_steps"],
                    nesterov=group["nesterov"],
                )

                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


# Use native Muon if available, otherwise use fallback
if _NATIVE_MUON_AVAILABLE:
    Muon = torch.optim.Muon
else:
    Muon = _FallbackMuon


class MuonAdamW(torch.optim.Optimizer):
    """
    Combined Muon + AdamW optimizer for training models with mixed parameter types.

    Muon is used for 2D hidden weight matrices (better for transformer layers).
    AdamW is used for embeddings, output heads, biases, and layer norms.

    Each param group must have 'use_muon' set to True or False.

    Muon groups use: lr, weight_decay, momentum, nesterov, ns_steps
    AdamW groups use: lr, weight_decay, betas, eps

    Example:
        >>> muon_params = [p for n, p in model.named_parameters()
        ...               if p.ndim >= 2 and 'embed' not in n and 'head' not in n]
        >>> adam_params = [p for n, p in model.named_parameters()
        ...               if p.ndim < 2 or 'embed' in n or 'head' in n]
        >>> optimizer = MuonAdamW([
        ...     {'params': muon_params, 'use_muon': True, 'lr': 0.02},
        ...     {'params': adam_params, 'use_muon': False, 'lr': 1e-4},
        ... ])
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group, "Each param group must specify 'use_muon'"

            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("weight_decay", 0.1)
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
            else:
                group.setdefault("lr", 1e-4)
                group.setdefault("weight_decay", 0.01)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)

        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = _muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )

                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])

                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1

                    update = _adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )

                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])

                    p.add_(update, alpha=-group["lr"])

        return loss


def create_muon_optimizer(
    model,
    lr=1e-4,
    muon_lr=0.02,
    weight_decay=0.01,
    muon_weight_decay=0.1,
    betas=(0.9, 0.95),
    eps=1e-8,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
):
    """
    Create a Muon+AdamW optimizer with automatic parameter grouping.

    Muon is used for 2D hidden weight matrices (encoder/decoder layers).
    AdamW is used for embeddings, output heads, biases, and layer norms.

    Args:
        model: The model to optimize
        lr: Learning rate for AdamW parameters (default: 1e-4)
        muon_lr: Learning rate for Muon parameters (default: 0.02)
        weight_decay: Weight decay for AdamW parameters (default: 0.01)
        muon_weight_decay: Weight decay for Muon parameters (default: 0.1)
        betas: Adam betas (default: (0.9, 0.95))
        eps: Adam epsilon (default: 1e-8)
        momentum: Muon momentum (default: 0.95)
        nesterov: Use Nesterov momentum for Muon (default: True)
        ns_steps: Newton-Schulz iterations (default: 5)

    Returns:
        MuonAdamW optimizer with properly grouped parameters
    """
    muon_params = []
    adamw_params = []
    adamw_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()

        # Determine parameter category
        is_2d = param.ndim == 2  # Muon is specifically for 2D weight matrices
        is_embedding = "embed" in name_lower
        is_head = "head" in name_lower or "classifier" in name_lower
        is_bias = name_lower.endswith(".bias")
        is_norm = "norm" in name_lower or "ln" in name_lower
        is_special_token = "cls_token" in name_lower or "pos_embed" in name_lower
        is_proj = "patch_embed" in name_lower  # Conv projection layer

        # Muon only for 2D hidden weight matrices (not embeddings, heads, special tokens, etc.)
        if is_2d and not is_embedding and not is_head and not is_bias and not is_norm and not is_special_token and not is_proj:
            # 2D hidden weights -> Muon
            muon_params.append(param)
        elif is_bias or is_norm:
            # Biases and norms -> AdamW without weight decay
            adamw_no_decay_params.append(param)
        else:
            # Embeddings, heads, special tokens, etc -> AdamW with weight decay
            adamw_params.append(param)

    param_groups = []

    if muon_params:
        param_groups.append({
            "params": muon_params,
            "lr": muon_lr,
            "weight_decay": muon_weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "use_muon": True,
        })

    if adamw_params:
        param_groups.append({
            "params": adamw_params,
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_muon": False,
        })

    if adamw_no_decay_params:
        param_groups.append({
            "params": adamw_no_decay_params,
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": 0.0,
            "use_muon": False,
        })

    # Log parameter counts
    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    n_adamw_no_decay = sum(p.numel() for p in adamw_no_decay_params)
    total = n_muon + n_adamw + n_adamw_no_decay

    print(f"Optimizer parameter groups:")
    print(f"  Muon (2D hidden weights): {len(muon_params)} params, {n_muon:,} elements ({100*n_muon/total:.1f}%)")
    print(f"  AdamW (embeddings/heads): {len(adamw_params)} params, {n_adamw:,} elements ({100*n_adamw/total:.1f}%)")
    print(f"  AdamW (bias/norm, no decay): {len(adamw_no_decay_params)} params, {n_adamw_no_decay:,} elements ({100*n_adamw_no_decay/total:.1f}%)")

    return MuonAdamW(param_groups)


__all__ = ["Muon", "MuonAdamW", "create_muon_optimizer"]
