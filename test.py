import torch
import torchvision
import torch.nn as nn
import pdb

from models.resnet.resnet import *

pretrain_path = 'pretrained_models/tsn_r50_1x1x3_100e_kinetics400_rgb.pth'

model = resnet50(pretrain=None, truncate=True)



state_dict = torch.load(pretrain_path)['state_dict']
new_state_dict = {}

for key in state_dict:
    segments = key[9:].split('.')

    if len(segments) == 3:
        new_key = segments[-2] + segments[-3][-1] + '.' + segments[-1]
        # new_state_dict[new_key] = state_dict[key]

    elif len(segments) == 5:
        if 'downsample' in segments:
            tmp = ['conv', 'bn']
            new_key = segments[0] + '.' + segments[1] + '.' + segments[2] + '.' + str(tmp.index(segments[3])) + '.' + segments[4]
        else:
            new_key = segments[0] + '.' + segments[1] + '.' + segments[-2] + segments[-3][-1] + '.' + segments[-1]

    else:
        new_key = 'fc.' + segments[-1]
    print(key, new_key)
    new_state_dict[new_key] = state_dict[key]

    # if new_key == 'layer1.0.conve.weight':
    #     pdb.set_trace()

pdb.set_trace()
torch.save(new_state_dict, 'pretrained_models/resnet50-kinetics400.pth')
model.load_state_dict({k[9:]: v for k, v in new_state_dict.items() if k[9:] in model.state_dict() and k[9:][:2] != 'fc'},strict=False)

