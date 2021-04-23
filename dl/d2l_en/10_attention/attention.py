#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

import sys
sys.path.append("../..")
print(sys.path)
import torch
import math
from torch import nn
from d2l import d2l_en as dl

def masked_softmax(X,valid_len):
    if valid_len is None:
        return nn.functional.softmax(X,dim=1)
    else:
        shape = X.shape
        if valid_len.dim() == 1:
            valid_len = torch.repeat_interleave(valid_len,repeats=shape[1],dim=0)
        else:
            pass
