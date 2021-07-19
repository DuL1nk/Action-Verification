from torch import nn
import torch

from models.resnet.resnet import *

import sys
sys.path.append('/p300/code/ActionVerification/models/tsn')

from basic_ops import ConsensusModule

import pdb

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', partial_bn=True, pretrain=None):
        super(TSN, self).__init__()
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.consensus_type = consensus_type
        self.pretrain = pretrain


        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        self._prepare_base_model(base_model)

        # feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)


        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)


    def _prepare_base_model(self, base_model):

        # pdb.set_trace()

        if 'resnet' in base_model:
            self.base_model = resnet50(pretrain=self.pretrain, truncate=False)
            # modify the last fc layer
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, self.num_class)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def forward(self, input):

        # pdb.set_trace()

        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])


        output = self.consensus(base_out)
        return output.squeeze(1)