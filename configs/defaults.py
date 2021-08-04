from yacs.config import CfgNode as CN

# 创建一个配置节点_C
_C = CN()


# default configuration

# Train configuration
_C.TRAIN = CN()
_C.TRAIN.SEED = 1234
_C.TRAIN.USE_CUDA = True
_C.TRAIN.MAX_EPOCH = 120
_C.TRAIN.DECAY_EPOCHS = 40
_C.TRAIN.DECAY_RATE = 0.1
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.LR = 0.0001
_C.TRAIN.DROPOUT = 0
_C.TRAIN.SAVE_PATH = None
_C.TRAIN.GRAD_MAX_NORM = None
_C.TRAIN.USE_ADAMW = False
_C.TRAIN.FIX_VIT_PROJECTION = False
_C.TRAIN.PARTIAL_BN = False     # Freeze BatchNorm2D except the first layer in backbone

# Model configuration
_C.MODEL = CN()
_C.MODEL.NAME = 'CAT'
_C.MODEL.BACKBONE_DIM = '2D'
_C.MODEL.BACKBONE = 'resnet18'  # 2D: resnet18/50/101, TSN, TSM, STM, TEA, ...
                                # 3D: C3D, I3D, ...
_C.MODEL.BASE_MODEL = None      # BNInception or resnet50...
_C.MODEL.PRETRAIN = None
_C.MODEL.DIM_EMBEDDING = 128
_C.MODEL.TRANSFORMER = False    # Whether to use ViT module
_C.MODEL.ALIGNMENT = False      # Whether to use sequence alignment module
_C.MODEL.COSFACE = False
_C.MODEL.CHECKPOINT_PATH = None
_C.MODEL.SEQ_LOSS_COEF = 1
_C.MODEL.SAVE_EPOCHS = 5        # Save model per 5 epochs


# Dataset configuration
_C.DATASET = CN()
_C.DATASET.TXT_PATH = '/p300/dataset/ActionVerification/train.txt'
_C.DATASET.NUM_WORKERS = 4
_C.DATASET.MODALITY = 'RGB'
_C.DATASET.NUM_CLASS = 20       # 20 classes in total
_C.DATASET.MODE = 'train'       # train, train_pairs, test
_C.DATASET.NUM_SAMPLE = 600     # num of samples while training
_C.DATASET.AUGMENT = True       # whether to apply data augmentation
_C.DATASET.NUM_CLIP = 8         # num of clips
_C.DATASET.LEN_CLIP = 1         # length of each clip (1 for 2D backbone; 8 for 3D backbone)
_C.DATASET.SHUFFLE = True




# # LOAD configuration
# _C.LOAD = CN()
# _C.LOAD.IF_LOAD = False
# _C.ROOT_PATH = '/p300/dataset/ActionVerification/defaults'
# _C.LOAD.CHECKPOINT_PATH = ''
#
#
# # Save configuration
# _C.SAVE = CN()
# _C.SAVE.IF_SAVE = False
# _C.ROOT_PATH = '/p300/dataset/ActionVerification/defaults'
# _C.SAVE.LOG_PATH = 'logs/defaults'
# _C.SAVE.CHECKPOINT_PATH = 'logs/defaults/checkpoint'



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # 克隆一份配置节点_C的信息返回，_C的信息不会改变
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`

