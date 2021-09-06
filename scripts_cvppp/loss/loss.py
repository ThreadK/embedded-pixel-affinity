from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################################
# 0. Main loss functions
#######################################################

class JaccardLoss(n