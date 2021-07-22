# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from models.resnet.resnet import *
from torch import nn
import numpy as np
import pdb

# import sys
# sys.path.append('/p300/code/ActionVerification/models/tsm')
# pdb.set_trace()


from ..tsn.basic_ops import ConsensusModule
# from ..ops.transforms import *
from torch.nn.init import normal_, constant_


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', img_feature_dim=256,
                 pretrain=None,
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False):
        super(TSN, self).__init__()
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True

        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local


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




    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        # pdb.set_trace()

        if 'resnet' in base_model:
            # self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            self.base_model = resnet50(pretrain=self.pretrain, truncate=False)
            # modify the last fc layer
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, self.num_class)

            if self.is_shift:
                print('Adding temporal shift...')
                from models.tsm.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments, n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if self.non_local:
                print('Adding non-local module...')
                from models.tsm.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)


        else:
            raise ValueError('Unknown base model: {}'.format(base_model))




    def forward(self, input):
        # pdb.set_trace()

        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))



        if self.is_shift and self.temporal_pool:
            base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
        else:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        output = self.consensus(base_out)
        return output.squeeze(1)

