import argparse
import os

import torch
import torch.nn as nn

import torchvision.transforms as tf
from torch.utils import data
from data.dataset import load_dataset
import pdb
import logging

from utils.logger import setup_logger
from utils.input import frames_preprocess
from utils.loss import compute_cls_loss, compute_seq_loss
from configs.defaults import get_cfg_defaults
from models.model import CAT

import numpy as np
import time
import random



def train():

    train_loader = load_dataset(cfg)

    model = CAT(num_class=cfg.DATASET.NUM_CLASS,
                num_clip=cfg.DATASET.NUM_CLIP,
                dim_embedding=cfg.MODEL.DIM_EMBEDDING,
                backbone_model=cfg.MODEL.BACKBONE,
                backbone_dim=cfg.MODEL.BACKBONE_DIM,
                base_model=cfg.MODEL.BASE_MODEL,
                pretrain=cfg.MODEL.PRETRAIN,
                dropout=cfg.TRAIN.DROPOUT,
                use_ViT=cfg.MODEL.TRANSFORMER,
                use_SeqAlign=cfg.MODEL.ALIGNMENT,
                use_CosFace=cfg.MODEL.COSFACE,
                fix_ViT_projection=cfg.TRAIN.FIX_VIT_PROJECTION,
                partial_bn=cfg.TRAIN.PARTIAL_BN,
                freeze_backbone=cfg.TRAIN.FREEZE_BACKBONE).to(device)


    # pdb.set_trace()
    # for name, param in model.named_parameters():
    #     print(name, param.nelement())


    logger.info("Model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    if cfg.TRAIN.USE_ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=0.01)

    start_epoch = 0


    # Load checkpoint
    if args.load_path and os.path.isfile(args.load_path):
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info("-> Loaded checkpoint %s (epoch: %d)" % (args.load_path, start_epoch))

    # Mulitple gpu
    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        logger.info("Let's use %d GPUs" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    model.train()
    # pdb.set_trace()
    start_time = time.time()
    flag = False
    # Start training
    for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):

        if epoch >= 40 and cfg.TRAIN.FREEZE_BACKBONE and flag:
            print('Unfreezeing backbone from epoch %d.' % (epoch + 1))
            flag = False
            for param in model.backbone.parameters():
                param.requires_grad = True

        loss = 0
        loss_per_epoch = 0
        num_true_pred = 0
        indices = []

        # for name, param in model.named_parameters():
        #     print('层:', name, param.size())
        #     print('权值梯度', param.grad)
        #     print('权值', param)

        time_spot0 = time.time()
        for iter, sample in enumerate(train_loader):

            # pdb.set_trace()

            time_spot1 = time.time()
            # logger.info("Iter %d: Loading data costs %d" % (iter, time_spot1 - time_spot0))

            frames1 = frames_preprocess(sample['frames_list1'], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE).to(device, non_blocking=True)
            frames2 = frames_preprocess(sample['frames_list2'], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE).to(device, non_blocking=True)
            labels1 = sample['label1'].to(device, non_blocking=True)
            labels2 = sample['label2'].to(device, non_blocking=True)

            pred1, seq_features1 = model(frames1)
            pred2, seq_features2 = model(frames2)

            pred_labels1 = torch.argmax(pred1, dim=-1)
            pred_labels2 = torch.argmax(pred2, dim=-1)
            num_true_pred += torch.sum(pred_labels1 == labels1) + torch.sum(pred_labels2 == labels1)



            loss_cls1 = compute_cls_loss(pred1, labels1, cfg.MODEL.COSFACE)
            loss_cls2 = compute_cls_loss(pred2, labels2, cfg.MODEL.COSFACE)


            loss_seq = compute_seq_loss(seq_features1, seq_features2)
            loss = loss_cls1 + loss_cls2 + cfg.MODEL.SEQ_LOSS_COEF * loss_seq
            loss_per_epoch += loss.item()

            # pdb.set_trace()

            optimizer.zero_grad()
            loss.backward()
            if cfg.TRAIN.GRAD_MAX_NORM:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.TRAIN.GRAD_MAX_NORM, norm_type=2)
            optimizer.step()

            if (iter + 1) % 10 == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, cfg.TRAIN.MAX_EPOCH, iter + 1, len(train_loader), loss.item()))

            time_spot0 = time.time()
            # logger.info("Iter %d: Training costs %d" % (iter, time_spot0 - time_spot1))

        # Statistics per epoch
        loss_per_epoch /= (iter + 1)
        accuracy = num_true_pred / (cfg.DATASET.NUM_SAMPLE * 2)
        logger.info('Epoch [{}/{}], Accuracy: {:.4f}, Loss: {:.4f}'.format(epoch + 1, cfg.TRAIN.MAX_EPOCH, accuracy, loss_per_epoch))


        # Learning rate decay
        if ((epoch + 1) % cfg.TRAIN.DECAY_EPOCHS == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * cfg.TRAIN.DECAY_RATE
                logger.info('Epoch: %d, change lr to %.6f' % (epoch + 1, param_group['lr']))



        # Save checkpoint
        if cfg.TRAIN.SAVE_PATH:
            checkpoint_dir = os.path.join(cfg.TRAIN.SAVE_PATH, 'save_models')
            save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss': loss,
                         }
            try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = model.module.state_dict()
            except:
                save_dict['model_state_dict'] = model.state_dict()


            # Save model every 10 epochs
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if (epoch + 1) % cfg.MODEL.SAVE_EPOCHS == 0:
                save_name = 'epoch_' + str(epoch+1) + '.tar'
                torch.save(save_dict, os.path.join(checkpoint_dir, save_name))
                logger.info('Save ' + os.path.join(checkpoint_dir, save_name) + ' done!')

    end_time = time.time()
    duration = end_time - start_time

    hour = duration // 3600
    min = (duration % 3600) // 60
    sec = duration % 60

    logger.info('Training cost %dh%dm%ds' % (hour, min, sec))



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='configs/train_resnet_config.yml', help='config file path [default: configs/train_resnet_config.yml]')
    parser.add_argument('--save_path', default=None, help='path to save models and log [default: None]')
    parser.add_argument('--load_path', default=None, help='path to load the model [default: None]')


    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)





    setup_seed(cfg.TRAIN.SEED)
    # torch.manual_seed(cfg.TRAIN.SEED)
    # torch.cuda.manual_seed_all(cfg.TRAIN.SEED)
    use_cuda = cfg.TRAIN.USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")




    if cfg.TRAIN.SAVE_PATH:
        logger_path = os.path.join(cfg.TRAIN.SAVE_PATH, 'logs')
    else:
        logger_path = 'temp_log'

    logger = setup_logger("ActionVerification", logger_path, 'train_log.txt', 0)
    logger.info("Running with config:\n{}\n".format(cfg))





    train()