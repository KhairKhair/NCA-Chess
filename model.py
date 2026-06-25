import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataset import NUM_RETURN_BUCKETS


VISIBLE = 13   # 12 piece planes + 1 empty plane (must match helper.py)


class CA(torch.nn.Module):
    def __init__(self, chn=25, hidden_n=96):
        super().__init__()
        self.chn = chn
        self.out_chn = NUM_RETURN_BUCKETS
        total = chn + NUM_RETURN_BUCKETS
        self.perc = torch.nn.Conv2d(
            total, 8 * total, 3, padding=1, padding_mode="zeros", bias=False
        )
        self.dropout = torch.nn.Dropout2d(p=0.1)
        self.dropout2 = torch.nn.Dropout2d(p=0.1)
        self.w1 = torch.nn.Conv2d(9 * total, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, total, 1, bias=False)

    def forward(self, x, update_rate=0.5):
        y = self.perc(x)
        y = torch.cat((y, x), dim=1)
        y = self.dropout(y)
        y = self.w1(y)
        y = torch.relu(y)
        y = self.dropout2(y)
        y = self.w2(y)

        b, c, h, w = y.shape
        # mask must live on the same device as y, else CUDA mismatch
        update_mask = (
            torch.rand(b, 1, h, w, device=y.device) + update_rate
        ).floor()
        x = x + y * update_mask
        return x

