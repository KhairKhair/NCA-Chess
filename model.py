import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_sobel_kernel(channels):
    """Fixed depthwise kernel: identity + Sobel-x + Sobel-y per channel.
    Returns weight of shape [3*channels, 1, 3, 3] for a grouped conv
    with groups=channels (each input channel -> its own 3 filters)."""
    identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
    sobel_x  = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
    sobel_y  = sobel_x.T
    filters  = torch.stack([identity, sobel_x, sobel_y])        
    weight   = filters.unsqueeze(0).repeat(channels, 1, 1, 1)   
    return weight.reshape(channels * 3, 1, 3, 3)                


class PolicyChessCA(nn.Module):
    def __init__(self, channels=80, steps=15, hidden=128, hidden_act="relu",
                 zero_init=True, n_input=12, fire_rate=1.0):
        super().__init__()
        assert channels >= n_input
        self.channels, self.steps = channels, steps
        self.n_input = n_input
        self.fire_rate = fire_rate
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}

        # perceive: depthwise Sobel filters, identity + grad_x + grad_y.
        self.register_buffer("perceive_kernel", make_sobel_kernel(channels))

        self.update = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1), acts[hidden_act](),
            nn.Conv2d(hidden, channels, 1),
        )
        if zero_init:
            nn.init.zeros_(self.update[-1].weight)
            nn.init.zeros_(self.update[-1].bias)

        # readout: full state -> 2 logit planes (from, to), each 8x8 -> flatten to 64
        self.head = nn.Conv2d(channels, 2, 1)

    def perceive(self, x):
        # depthwise: groups=channels, each channel convolved with its 3 fixed kernels
        return F.conv2d(x, self.perceive_kernel, padding=1, groups=self.channels)

 
    def _init_state(self, x):
        B, _, H, W = x.shape
        s = torch.zeros(B, self.channels, H, W, device=x.device)
        # copy input to state, leaving rest of channels as 0
        s[:, :self.n_input] = x
        return s
 
    def _step(self, s, x):
        # compute state update
        du = self.update(torch.relu(self.perceive(s)))
        # stochastic per-cell update
        if self.fire_rate < 1.0:
            mask = (torch.rand_like(s[:, :1]) < self.fire_rate).float()
            du = du * mask
        
        s = s + du
        return s
 
    def forward(self, x):
        s = self._init_state(x)
        for _ in range(self.steps):
            s = self._step(s, x)
        logits = self.head(s)                          # (B,2,8,8)
        B = x.shape[0]
        return logits[:, 0].reshape(B, 64), logits[:, 1].reshape(B, 64)
 
def policy_loss(from_logits, to_logits, fl, tl):
    # cross-entropy loss for from/to planes, averaged over batch
    return F.cross_entropy(from_logits, fl) + F.cross_entropy(to_logits, tl)