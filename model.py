import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

    
class CA(torch.nn.Module):
  def __init__(self, chn=12, hidden_n=96, mask_n = 0):
    super().__init__()
    self.chn = chn
    self.perc = torch.nn.Conv2d(chn, 8 * chn, 3, padding=1, padding_mode="zeros", bias=False)
    self.dropout = torch.nn.Dropout2d(p=0.4)
    self.dropout2 = torch.nn.Dropout2d(p=0.4)
    self.w1 = torch.nn.Conv2d(9*chn, hidden_n, 1)
    self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
    self.mask_n = mask_n

  def forward(self, x, update_rate=0.5):
      y = self.perc(x)

      y = torch.cat((y, x), dim=1)
      y = self.dropout(y)
      y = self.w1(y)
      y = torch.relu(y)
      y = self.dropout2(y)
      y = self.w2(y)
      b, c, h, w = y.shape
      update_mask = (torch.rand(b, 1, h, w) + update_rate).floor()
      #px = torch.nn.functional.pad(x, [1,1,1,1])
      #pre_life_mask = torch.nn.functional.max_pool2d(px[:, None, 3, ...], 3, 1, ) > 0.1
      # Perform update
      x = x + y * update_mask #* pre_life_mask
      return x
