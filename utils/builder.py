import torch.nn as nn
import logging

from models.tsn.models import TSN
from models.tsm.models import TSN as TSM
from models.trn.models import TSN as TRN
from models.resnet.resnet import *
from models.vit.vit_pytorch import ViT
from models.c3d.c3d import C3D
from models.i3d.i3d import InceptionI3d as I3D
from models.vgg.vgg import vgg16
import pdb

# Dims for different backbone model when truncated
TRUNCATE_DIM = {'resnet18': 512,
                'resnet34': 512,
                'resnet50': 2048,
                'resnet101': 2048,
                'bninception': 1024,
                'BNInception': 1024,
                'vit': 1024,
                'vgg': 512,
                'tsn': 1024,
                'tsm': 1024,
                'trn': 1024}


logger = logging.getLogger('ActionVerification')



class builder:

    def __init__(self,
                 backbone_model,
                 base_model,
                 num_class,
                 num_clip,
                 pretrain,
                 dropout,
                 dim_embedding,
                 use_ViT,
                 use_SeqAlign,
                 fix_ViT_projection):

        self.backbone_model = backbone_model
        self.base_model = base_model
        self.num_class = num_class
        self.num_clip = num_clip
        self.pretrain = pretrain
        self.dropout = dropout
        self.dim_embedding = dim_embedding
        self.use_ViT = use_ViT
        self.use_SeqAlign = use_SeqAlign
        self.fix_ViT_projection = fix_ViT_projection

    def build_backbone(self):
        backbone = None

        if self.backbone_model == 'cat':

            if self.base_model in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
                depth = int(self.base_model[6:])

                if depth == 18:
                    backbone = resnet18(pretrain=self.pretrain, truncate=True)
                elif depth == 34:
                    backbone = resnet34(pretrain=self.pretrain, truncate=True)
                elif depth == 50:
                    backbone = resnet50(pretrain=self.pretrain, truncate=True)
                elif depth == 101:
                    backbone = resnet101(pretrain=self.pretrain, truncate=True)

                # backbone.fc = nn.Linear(backbone.fc.in_features, self.dim_embedding)
            elif self.base_model == 'vgg':
                backbone = vgg16(self.pretrain)
            elif self.base_model == 'bninception':
                pass
            elif self.base_model == 'vit':
                backbone = ViT(
                    image_size=(180, 320),
                    patch_size=(20, 20),
                    num_classes=self.num_class,
                    dim=1024,
                    depth=6,
                    heads=8,
                    mlp_dim=2048,
                    pool='all',
                    channels=3,
                    dropout=self.dropout,
                    emb_dropout=self.dropout,
                    fix_embedding=self.fix_ViT_projection
                )

        elif self.backbone_model == 'c3d':
            backbone = C3D(pretrain=self.pretrain,
                           dim_embedding=self.dim_embedding,
                           dropout=self.dropout)
        elif self.backbone_model == 'i3d':
            backbone = I3D(num_classes=self.dim_embedding,
                           dropout=self.dropout,
                           pretrain=self.pretrain)
        elif self.backbone_model == 'tsn':
            backbone = TSN(num_class=self.dim_embedding,
                           num_segments=self.num_clip,
                           modality='RGB',
                           base_model=self.base_model,
                           consensus_type='avg',
                           pretrain=self.pretrain)
        elif self.backbone_model == 'tsm':
            backbone = TSM(num_class=self.dim_embedding,
                           num_segments=self.num_clip,
                           modality='RGB',
                           base_model=self.base_model,
                           consensus_type='avg',
                           pretrain=self.pretrain,
                           is_shift=True, shift_div=2, shift_place='blockres',
                           fc_lr5=True,
                           temporal_pool=False,
                           non_local=False)
        elif self.backbone_model == 'trn':
            backbone = TRN(num_class=self.dim_embedding,
                           num_segments=self.num_clip,
                           modality='RGB',
                           base_model=self.base_model,
                           consensus_type='TRN',
                           pretrain=self.pretrain)
        elif self.backbone_model == 'tea':
            pass

        else:
            logger.info('Not support models %s' % self.backbone_model)
            raise NotImplementedError

        return backbone



    def build_vit(self):

        return ViT(
            image_size=(6, 10 * self.num_clip),
            patch_size=(6, 10),   # raw-vit
            # patch_size=(1, 1),      # dense-vit
            num_classes=self.num_class,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            pool='all',
            channels=TRUNCATE_DIM[self.base_model],
            dropout=self.dropout,
            emb_dropout=self.dropout,
            fix_embedding=self.fix_ViT_projection
        )

        # vit + vit
        return ViT(
            image_size=(1, 16),
            patch_size=(1, 1),
            num_classes=self.num_class,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            pool='all',
            channels=512,
            dropout=self.dropout,
            emb_dropout=self.dropout,
            fix_embedding=self.fix_ViT_projection
        )


    def build_2vit(self):

        # backbone + 2vit
        vit1 = ViT(
            image_size=(6, 10),
            patch_size=(1, 1),
            num_classes=self.num_class,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            pool='cls',
            channels=TRUNCATE_DIM[self.base_model],
            dropout=self.dropout,
            emb_dropout=self.dropout,
            fix_embedding=self.fix_ViT_projection
        )

        vit2 = ViT(
            image_size=(1, 16),
            patch_size=(1, 1),
            num_classes=self.num_class,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            pool='all',
            channels=TRUNCATE_DIM[self.base_model],
            dropout=self.dropout,
            emb_dropout=self.dropout,
            fix_embedding=self.fix_ViT_projection
        )

        return vit1, vit2

    def build_seq_features_extractor(self):

        # return nn.Sequential(
        #     ViT(
        #         image_size=(180, 320),
        #         patch_size=(20, 20),
        #         num_classes=self.num_class,
        #         dim=1024,
        #         depth=4,
        #         heads=4,
        #         mlp_dim=2048,
        #         channels=3,
        #         dropout=self.dropout,
        #         emb_dropout=self.dropout,
        #         fix_embedding=self.fix_ViT_projection
        #     ),
        #     Reshape(-1, self.num_clip, 1024)
        # )

        return nn.Sequential(
            ViT(
            image_size=(6, 10),
            patch_size=(1, 1),
            num_classes=self.num_class,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            channels=TRUNCATE_DIM[self.base_model],
            dropout=self.dropout,
            emb_dropout=self.dropout,
            fix_embedding=self.fix_ViT_projection
        ),
            Reshape(-1, self.num_clip, 1024)
        )

        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            Reshape(-1, self.num_clip, TRUNCATE_DIM[self.base_model])
        )



class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.reshape(self.shape)
        # return x.view(self.shape)
        # return x.view((x.size(0), ) + self.shape)










