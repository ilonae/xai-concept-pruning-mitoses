"""
LAMB optimizer — Layer-wise Adaptive Moments Based on Batch size.

Reference: You et al., "Large Batch Optimization for Deep Learning:
           Training BERT in 76 minutes", ICLR 2020.
           https://arxiv.org/abs/1904.00962
"""

import torch
from torch.optim import Optimizer


class Lamb(Optimizer):
    """LAMB optimizer.

    Args:
        params:       Iterable of parameters or parameter groups.
        lr:           Learning rate (default: 1e-3).
        betas:        Coefficients for computing running averages of gradient
                      and its square (default: (0.9, 0.999)).
        eps:          Term added to denominator for numerical stability
                      (default: 1e-6).
        weight_decay: L2 weight decay (default: 0).
        clamp_value:  Clamp weight norm to this value before computing the
                      trust ratio (default: 10).
        adam:         Always apply the Adam update step, ignoring the trust
                      ratio (useful for debugging, default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        clamp_value: float = 10,
        adam: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value
        self.adam = adam
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

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")

                state = self.state[p]
                beta1, beta2 = group["betas"]

                # Initialise state on first step.
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # Decay running averages.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected Adam update direction.
                step = state["step"]
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2

                adam_update = exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + group["eps"])

                if group["weight_decay"] != 0:
                    adam_update.add_(p, alpha=group["weight_decay"])

                if self.adam:
                    # Plain Adam step — skip trust ratio.
                    p.add_(adam_update, alpha=-group["lr"])
                    continue

                # Compute trust ratio.
                weight_norm = p.norm(2.0).clamp(0, self.clamp_value)
                adam_norm = adam_update.norm(2.0)

                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = weight_norm / adam_norm

                p.add_(adam_update, alpha=-group["lr"] * trust_ratio)

        return loss
