import torch
import torch.nn as nn

import os
import argparse
from utils.logger import setup_logger
from utils.input import frames_preprocess
from visualizer import get_local
get_local.activate() # 激活装饰器
from models.model import CAT
from train import setup_seed
from data.dataset import load_dataset
from configs.defaults import get_cfg_defaults
import pdb
from tqdm import tqdm
import cv2
import numpy as np



def load_model(model_path):

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
                use_CosFace=cfg.MODEL.COSFACE).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    print('Load state dict form %s' % model_path)

    return model

def process_one_epoch(model_path, vis_func, **kwargs):

    model = load_model(model_path)

    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        # logger.info("Let's use %d GPUs" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)


    # model = model.to(device)
    auc_value = 0
    indices = []

    # auc metric
    model.eval()
    with torch.no_grad():

        for iter, sample in enumerate(tqdm(test_loader)):

            # print(sample['index'])
            # indices += sample['index']
            # continue

            frames_list1 = sample["frames_list1"]
            frames_list2 = sample["frames_list2"]
            assert len(frames_list1) == len(frames_list2)

            labels1 = sample["label1"].to(device)
            labels2 = sample["label2"].to(device)
            label = labels1 == labels2

            pred1 = 0
            pred2 = 0
            # pdb.set_trace()
            for i in range(len(frames_list1)):

                frames1 = frames_preprocess(frames_list1[i], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE).to(device, non_blocking=True)
                frames2 = frames_preprocess(frames_list2[i], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE).to(device, non_blocking=True)

                # pdb.set_trace()
                pred1 += model(frames1, embed=True)
                attn1 = get_local.cache
                vis_func(frames_preprocess(sample["raw_frames_list1"][i], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE), attn1['Attention.forward'])

                pred2 += model(frames2, embed=True)

                # pdb.set_trace()


            # pdb.set_trace()

            pred1 /= len(frames_list1)
            pred2 /= len(frames_list1)

            # L1 distance
            # pred = torch.sum(torch.abs(pred1 - pred2), dim=1)

            # L2 distance
            pred = torch.sum((pred1 - pred2) ** 2, dim=1)

            if iter == 0:
                preds = pred
                labels = label
            else:
                preds = torch.cat([preds, pred])
                labels = torch.cat([labels, label])



def vis_attn(frames, attns):

    pdb.set_trace()

    bs, _, h, w = frames.size()
    frames = frames.reshape(cfg.TRAIN.BATCH_SIZE, cfg.DATASET.NUM_CLIP, 3, h, w)
    attns = [torch.diagonal(torch.from_numpy(attn),dim1=2,dim2=3).mean(dim=1, keepdim=True)[:,:,1:] for attn in attns]      # [[bs, 1, H*W],...]

    def attn_seq2map(attn_seq, intermediate_shape, output_shape):

        # pdb.set_trace()

        bs, c, _ = attn_seq.size()
        h, w = intermediate_shape
        length = h * w

        upsampler = nn.Upsample(size=output_shape, mode='bilinear')

        attn_maps = []
        for i in range(int(attn_seq.size(-1) / length)):
            seq = attn_seq[:,:,i*length:(i+1)*length]
            scale = 1 / seq.max()
            seq = seq * scale
            seq = seq.reshape(bs, c, h, w)
            attn_map =upsampler(seq)
            attn_maps.append(attn_map.unsqueeze(1))

        return torch.cat(attn_maps, dim=1)
        # return torch.cat([upsampler((]*scale).reshape(bs, c, h, w)).unsqueeze(1) for i in range(int(attn_seq.size(-1)/length))], dim=1)

    def save_heatmap_img(img, mask, path):

        # pdb.set_trace()

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cam = 0.4 * heatmap + 0.6 * np.float32(img)
        cam = cam / np.max(cam)

        cv2.imwrite(path, np.uint8(255 * cam))

        # cam = cam[:, :, ::-1]  # BGR > RGB
        # plt.figure(figsize=(10, 10))
        # plt.imshow(np.uint8(255 * cam))

    for i in range(cfg.TRAIN.BATCH_SIZE):
        for k in range(len(attns)):
            attn_maps = attn_seq2map(attns[k], (6, 10), (180, 320))  # [bs, 16, 1, 180, 320]

            for j in range(cfg.DATASET.NUM_CLIP):
                # pdb.set_trace()
                split_list = args.model_path.split('/')
                save_path = os.path.join('figs', split_list[1], split_list[3][:-4])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path = os.path.join(save_path, 'attn_bs%d_layers%d_idx%d.jpg' % (i+1, k+1, j+1))
                save_heatmap_img(frames[i, j].permute(1, 2, 0), attn_maps[i, j].permute(1, 2, 0), save_path)





    pdb.set_trace()





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/eval_resnet_config.yml', help='config file path [default: configs/test_config.yml]')
    parser.add_argument('--model_path', default=None, help='path to load one model [default: None]')
    # parser.add_argument('--log_name', default='eval_log', help='log name')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)

    setup_seed(cfg.TRAIN.SEED)
    use_cuda = cfg.TRAIN.USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    test_loader = load_dataset(cfg)


    process_one_epoch(args.model_path, vis_attn)

