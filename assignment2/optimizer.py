from typing import Callable, Iterable, Tuple

import torch
import math
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                #raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]

                if len(state.keys()) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(grad)
                    state["v"] = torch.zeros_like(grad)
                
                m, v = state["m"], state["v"]
                
                state["step"] += 1

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                lr = group["lr"]
                b1,b2 = group["betas"]
                c_b = group["correct_bias"]
                eps = group["eps"]
                w_d = group["weight_decay"]

                # Update first and second moments of the gradients
                m = torch.add(torch.mul(m, b1), grad, alpha = 1 - b1)
                v = torch.addcmul(torch.mul(v, b2), grad, grad, value = 1 - b2)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if c_b:
                    t = state["step"]
                    alpha = alpha * math.sqrt(1 - pow(b2,t)) / (1-pow(b1,t))

                # Update parameters
                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                p.data = torch.addcdiv(p.data, m, torch.add(v.sqrt(),eps), value = -alpha)
                p.data = torch.add(p.data, p.data, alpha = - w_d * lr)

                state["m"] = m
                state["v"] = v

        return loss
