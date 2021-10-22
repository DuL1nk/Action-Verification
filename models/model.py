import torch.nn as nn
import torch.nn.functional as F
import pdb
import logging


from utils.builder import builder, TRUNCATE_DIM
from models.trn.TRNmodule import return_TRN

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
                 fix_ViT_projection=False,
                 partial_bn=False,
                 freeze_backbone=False):
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
        self.enable_pbn = partial_bn
        self.freeze_backbone = freeze_backbone

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
                                use_ViT,
                                use_SeqAlign,
                                fix_ViT_projection)


        self.backbone = model_builder.build_backbone()

        if self.backbone_model == 'swin':
            self.swin_head = self.backbone.cls_head
            self.backbone = self.backbone.backbone

        if use_SeqAlign:
            self.seq_features_extractor = model_builder.build_seq_features_extractor()

        if self.backbone_model == 'cat':
            if use_ViT:
                # self.vit1, self.vit2 = model_builder.build_2vit()
                self.vit = model_builder.build_vit()
                self.embed_fc = nn.Linear(1024, dim_embedding)
            else:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.embed_fc = nn.Linear(TRUNCATE_DIM[self.base_model], dim_embedding)
            self.temporal_mix_conv = nn.Conv1d(num_clip, 1, 1)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(dim_embedding, num_class, bias=False) if use_CosFace else nn.Linear(dim_embedding, num_class)


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(CAT, self).train(mode)
        count = 0
        if self.enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self.enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

        if self.freeze_backbone:
            print('Freezeing backbone.')
            for param in self.backbone.parameters():
                param.requires_grad = False





    def forward(self, x, labels=None, embed=False):

        # pdb.set_trace()

        seq_features = None
        # if self.use_SeqAlign:
        #     seq_features = self.seq_features_extractor(x)

        x = self.backbone(x)    # [bs * num_clip, 512, 6, 10]
        # return x
        if self.use_SeqAlign:
            # print('*** sa-vit ***')
            seq_features = self.seq_features_extractor(x)
        if self.backbone_model == 'swin':
            x = self.swin_head(x)




        if self.backbone_model == 'cat':

            if self.use_ViT:

                # Single ViT
                if self.base_model == 'vit':
                    _, c = x.size()
                    x = x.reshape(-1, self.num_clip, c)
                    x = self.vit(x, embedded=True)  # [bs, 1024]
                else:
                    _, c, h, w = x.size()
                    # input for raw-vit
                    x = x.reshape(-1, self.num_clip, c, h, w).permute(0,2,3,1,4).reshape(-1, c, h, w * self.num_clip)     # [bs, dim, 6, t*10]
                    # input for dense-vit
                    # x = x.reshape(-1, self.num_clip, c, h, w).permute(0, 2, 1, 3, 4).reshape(-1, c, h * self.num_clip, w)   # [bs, dim, 6*t, 10]
                    # print('*** vit ***')
                    x = self.vit(x)  # [bs, 16, 1024]


                # Series ViT
                # print('*** vit1 ***')
                # x = self.vit1(x)    # [bs * self.num_clip, 1024]
                # x = x.reshape(-1, self.num_clip, 1024)
                # # print('*** vit2 ***')
                # x = self.vit2(x, embedded=True)

            else:
                if 'resnet' in self.base_model:
                    x = self.avgpool(x)     # [bs * num_clip, 512, 1, 1]
                    x = x.flatten(1)        # [bs * num_clip, 512]
                x = x.reshape(-1, self.num_clip, TRUNCATE_DIM[self.base_model])    # [bs, num_clip, dim_feature]


            # x: [bs, num_clip, dim_feature]
            x = self.temporal_mix_conv(x)
            x = x.squeeze(1)
            x = self.embed_fc(x)  # [bs, dim_embedding]


        # [bs, dim_embedding]
        embed_feature = x
        if embed:
            # x = F.normalize(x, 2, dim=1)
            return x



        if self.use_CosFace:
            # Cosface loss
            self.fc.weight.data = F.normalize(self.fc.weight, p=2, dim=1)
            x = F.normalize(x, p=2, dim=1)

        x = self.dropout(x)
        x = self.fc(x)

        return x, seq_features, embed_feature







