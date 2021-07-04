import torch
from torch import nn
from .layer_factory import get_basic_layer, parse_expr
import torch.utils.model_zoo as model_zoo
import yaml

import pdb


import logging
logger = logging.getLogger('ActionVerification')

class BNInception(nn.Module):
    def __init__(self, pretrain='imagenet',
                       model_path='/p300/code/ActionVerification/models/tsn/tf_model_zoo/bninception/bn_inception.yaml', num_classes=101,
                       weight_url='http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth'):
        super(BNInception, self).__init__()

        manifest = yaml.load(open(model_path))
        # pdb.set_trace()

        layers = manifest['layers']

        self._channel_dict = dict()

        self._op_list = list()
        for l in layers:
            out_var, op, in_var = parse_expr(l['expr'])
            if op != 'Concat':
                id, out_name, module, out_channel, in_name = get_basic_layer(l,
                                                                3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]],
                                                                             conv_bias=True)

                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            else:
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel


        # Solve the mismatch between model and checkpoint, eg:
        # size mismatch for inception_5b_pool_proj_bn.running_var:
        # copying a param with shape torch.Size([1, 128]) from checkpoint, the shape in current model is torch.Size([128]).
        if pretrain:
            if pretrain == 'imagenet':
                state_dict = torch.utils.model_zoo.load_url(weight_url)
                self.load_state_dict(state_dict, strict=False)
                # self.load_state_dict({k: v.flatten() for k, v in state_dict.items() if k.split('.')[0][-2:] == 'bn' }, strict=False)
                # self.load_state_dict({k: v for k, v in state_dict.items() if k.split('.')[0][-2:] != 'bn'}, strict=False)
                logger.info('Loading backbone state_dict from %s pretrained model' % pretrain)
            elif pretrain[-4:] == '.tar':
                checkpoint = torch.load(pretrain)
                self.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info('Loading backbone state_dict from %s' % pretrain)
        logger.info('Initializing backbone weights')
        # self.load_state_dict(torch.utils.model_zoo.load_url(weight_url))

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(1024, 1000)



    def forward(self, input):

        # pdb.set_trace()


        data_dict = dict()
        data_dict[self._op_list[0][-1]] = input

        def get_hook(name):

            def hook(m, grad_in, grad_out):
                print(name, grad_out[0].data.abs().mean())

            return hook
        for op in self._op_list:

            # if op[0] == 'global_pool':
            #     pdb.set_trace()

            if op[1] != 'Concat' and op[1] != 'InnerProduct':
                data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
                # getattr(self, op[0]).register_backward_hook(get_hook(op[0]))
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:

                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[x] for x in op[-1]), 1)
                except:
                    pdb.set_trace()

                    for x in op[-1]:
                        print(x, data_dict[x].size())
                    raise


        return data_dict[self._op_list[-1][2]]

        # x = self.avgpool(data_dict[self._op_list[-1][2]]).flatten(1)
        # x = self.fc(x)
        #
        # return x


class InceptionV3(BNInception):
    def __init__(self, model_path='model_zoo/bninception/inceptionv3.yaml', num_classes=101,
                 weight_url='https://yjxiong.blob.core.windows.net/models/inceptionv3-cuhk-0e09b300b493bc74c.pth'):
        super(InceptionV3, self).__init__(model_path=model_path, weight_url=weight_url, num_classes=num_classes)
