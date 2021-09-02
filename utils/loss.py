import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import numpy as np

import pdb


def compute_cls_loss(pred, labels, use_cosface):

    if use_cosface:
        # CosFace Loss

        s = 30.0
        m = 0.4

        cos_value = torch.diagonal(pred.transpose(0, 1)[labels])
        numerator = s * (cos_value - m)
        excl = torch.cat([torch.cat((pred[i, :y], pred[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(s * excl), dim=1)
        L = numerator - torch.log(denominator)
        loss = -torch.mean(L)
    else:
        # Softmax Loss

        criterion = CrossEntropyLoss().cuda()
        loss = criterion(pred, labels)

    # pdb.set_trace()
    return loss


def compute_seq_loss(seq1, seq2):

    if seq1 == None or seq2 == None:
        return 0


    seq1 = F.normalize(seq1, 2, dim=2)
    seq2 = F.normalize(seq2, 2, dim=2)

    corr = torch.bmm(seq1, seq2.transpose(1, 2))

    corr1 = nn.Softmax(dim=1)(corr)  # Softmax across column
    corr2 = nn.Softmax(dim=2)(corr)

    loss = torch.sum(1 - torch.diagonal((corr1 + corr2) / 2, dim1=1, dim2=2)) / seq1.size(0)

    return loss

def compute_norm_loss(embed_feature):
    assert len(embed_feature.size()) == 2, 'Expect feature to compute norm loss have size of [bs, dim], but got %d dim of size' % len(embed_feature.size())

    return np.sum([x.norm() for x in embed_feature])

def compute_triplet_loss(triplet, margin=1.0, p=2):
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=p)
    return triplet_loss(triplet[0], triplet[1], triplet[2])