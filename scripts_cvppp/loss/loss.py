from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################################
# 0. Main loss functions
#######################################################

class JaccardLoss(nn.Module):
    """Jaccard loss.
    """
    # binary case

    def __init__(self, size_average=True, reduce=True, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce

    def jaccard_loss(self, pred, target):
        loss = 0.
        # for each sample in the batch
        for index in range(pred.size()[0]):
            iflat = pred[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()
            loss += 1 - ((intersection + self.smooth) / 
                    ( iflat.sum() + tflat.sum() - intersection + self.smooth))
            #print('loss:',intersection, iflat.sum(), tflat.sum())

        # size_average=True for the jaccard loss
        return loss / float(pred.size()[0])

    def jaccard_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        loss = 1 - ((intersection + self.smooth) / 
               ( iflat.sum() + tflat.sum() - intersection + self.smooth))
        #print('loss:',intersection, iflat.sum(), tflat.sum())
        return loss

    def forward(self, pred, target):
        #_assert_no_grad(target)
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})".format(target.size(), pred.size()))
        if self.re