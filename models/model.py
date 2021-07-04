import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import logging


from utils.builder import builder, TRUNCATE_DIM
from models.tsn.TRNmodule import return_TRN

logger = logging.getLogger('ActionVerification')



class CAT(nn.Module):
    def __init__(self,
                 num_class=20,
                 num_clip=8,
                 len_clip=1,
                 dim_embedding=128,
                 backbone_model=None,
                 backbone_dim='2D',
                 base_model=None,
                 pretrain=None,
                 dropout=0,
                 use_ViT=False,
                 use_SeqAlign=False,
                 use_CosFace=False,
                 fix_ViT_projection=False):
        super(CAT, self).__init__()
        assert backbone_model, logger.info('CAT must have a backbone model, but get None')

        # Initialization
        self.num_cls = num_class
        self.num_clip = num_clip
        self.dim_embedding = dim_embedding
        self.backbone_model = backbone_model
        self.backbone_dim = backbone_dim
        self.use_ViT = use_ViT
        self.use_SeqAlign = use_SeqAlign
        self.use_CosFace = use_CosFace

        if base_model == None:
            self.base_model = backbone_model
        else:
            self.base_model = base_model

        # Construct the model
        model_builder = builder(backbone_model,
                                base_model,
                                num_class,
                                num_clip,
                                pretrain,
                                dropout,
                                dim_embedding,
                                True,
                                use_ViT,
                                use_SeqAlign,
                                fix_ViT_projection)


        self.backbone = model_builder.build_backbone()
        if use_ViT:
            self.vit = model_builder.build_vit()
            self.vit_fc = nn.Linear(1024, dim_embedding)
        if use_SeqAlign:
            self.seq_features_extractor = model_builder.build_seq_features_extractor()

        if backbone_model == 'trn' and not use_ViT:
            trn_type = 'TRN'
            # trn_type = 'TRNmultiscale'
            self.trn_mix = return_TRN(trn_type, TRUNCATE_DIM[self.base_model], num_clip, dim_embedding)

        if backbone_dim != '3D':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            if backbone_model == 'trn':
                self.embed_fc = nn.Linear(1024, dim_embedding)
            else:
                self.embed_fc = nn.Linear(TRUNCATE_DIM[self.base_model], dim_embedding)
            self.temporal_mix_conv = nn.Conv1d(num_clip, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_embedding, num_class, bias=False) if use_CosFace else nn.Linear(dim_embedding, num_class)


 

    def forward(self, x, labels=None, embed=False):

        # pdb.set_trace()
        x = self.backbone(x)    # [bs * num_clip, 512, 6, 10]
        # return x

        seq_features = None

        if self.backbone_dim != '3D':

            if self.use_SeqAlign:
                seq_features = self.seq_features_extractor(x)

            if self.use_ViT:
                _, c, h, w = x.size()
                x = x.reshape(-1, self.num_clip, c, h, w).permute(0,2,3,1,4).reshape(-1, c, h, w * self.num_clip)     # [bs, 512, 6, t*10]
                x = self.vit(x)         # [bs, 1024]
                x = self.vit_fc(x)      # [bs, 128]
            else:
                x = self.avgpool(x)     # [bs * num_clip, 512, 1, 1]
                x = x.flatten(1)        # [bs * num_clip, 512]
                x = x.reshape(-1, self.num_clip, TRUNCATE_DIM[self.base_model])    # [bs, num_clip, dim_feature]

                # Different aggregate method in temporal dim
                if self.backbone_model == 'tsn' or self.backbone_model == 'tsm':
                    x = torch.mean(x, dim=1)
                elif self.backbone_model == 'trn':
                    # pdb.set_trace()
                    x = self.trn_mix(x)
                else:
                    x = self.temporal_mix_conv(x)
                    x = x.squeeze(1)

                x = self.embed_fc(x)  # [bs, dim_embedding]

        # [bs, dim_embedding]
        if embed:
            # x = F.normalize(x, 2, dim=1)
            return x



        if self.use_CosFace:
            # Cosface loss
            self.fc.weight.data = F.normalize(self.fc.weight, p=2, dim=1)
            x = F.normalize(x, p=2, dim=1)

        x = self.dropout(x)
        x = self.fc(x)

        # if not self.use_CosFace:
        #     # Softmax loss
        #     x = F.softmax(x, dim=1)






        return x, seq_features




