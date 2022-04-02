from torch import optim
import torch
from torch.optim.optimizer import Optimizer, required

class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
        max_epoch: maximum training epoch to determine polynomial LR decay.
    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888
    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=.9,
                 weight_decay=.0005, eta=0.001, max_epoch=200):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        self.epoch = 0
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        eta=eta, max_epoch=max_epoch)
        super(LARS, self).__init__(params, defaults)

    def step(self, epoch=None, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            max_epoch = group['max_epoch']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)

                # Global LR computed on polynomial decay schedule
                decay = (1 - float(epoch) / max_epoch) ** 2
                global_lr = lr * decay

                # Compute local learning rate for this layer
                local_lr = eta * weight_norm / \
                    (grad_norm + weight_decay * weight_norm)

                # Update the momentum term
                actual_lr = local_lr * global_lr

                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = \
                            torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(actual_lr, d_p + weight_decay * p.data)
                p.data.add_(-buf)

        return loss
        
# class LARS(optim.Optimizer):
#     def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
#                  weight_decay_filter=False, lars_adaptation_filter=False):
#         defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
#                         eta=eta, weight_decay_filter=weight_decay_filter,
#                         lars_adaptation_filter=lars_adaptation_filter)
#         super().__init__(params, defaults)


#     def exclude_bias_and_norm(self, p):
#         return p.ndim == 1

#     @torch.no_grad()
#     def step(self):
#         for g in self.param_groups:
#             for p in g['params']:
#                 dp = p.grad

#                 if dp is None:
#                     continue

#                 if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
#                     dp = dp.add(p, alpha=g['weight_decay'])

#                 if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
#                     param_norm = torch.norm(p)
#                     update_norm = torch.norm(dp)
#                     one = torch.ones_like(param_norm)
#                     q = torch.where(param_norm > 0.,
#                                     torch.where(update_norm > 0,
#                                                 (g['eta'] * param_norm / update_norm), one), one)
#                     dp = dp.mul(q)

#                 param_state = self.state[p]
#                 if 'mu' not in param_state:
#                     param_state['mu'] = torch.zeros_like(p)
#                 mu = param_state['mu']
#                 mu.mul_(g['momentum']).add_(dp)

#                 p.add_(mu, alpha=-g['lr'])