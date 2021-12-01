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
import torch.nn.functional as F



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


def vis_embedding(model_path):

    id_bank = ['1.1', '1.2', '1.3', '1.4', '1.5']
    # id_bank = ['5.1', '5.2', '5.3', '5.4', '5.5']
    # id_bank = ['18.1', '18.3', '18.4', '18.5', '18.6', '18.7', '18.13', '18.15', '18.16', '18.18']
    model = load_model(model_path)

    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        # logger.info("Let's use %d GPUs" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    labels = []
    embeds = []
    with torch.no_grad():

        for iter, sample in enumerate(tqdm(test_loader)):

            # print(sample['index'])
            # indices += sample['index']
            # continue

            frames_list1 = sample['frames_list1']
            # frames_list2 = sample['frames_list2']
            # assert len(frames_list1) == len(frames_list2)


            # pdb.set_trace()
            labels1 = [id_bank.index(tmp) for tmp in sample['label1']]
            # labels2 = [id_bank.index(tmp) for tmp in sample['label2']]
            # label = labels1+labels2
            labels += labels1

            embed1s = []
            # embed2s = []
            # num_true_pred = 0
            # pdb.set_trace()
            for i in range(len(frames_list1)):

                frames1 = frames_preprocess(frames_list1[i], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE).to(device, non_blocking=True)
                # frames2 = frames_preprocess(frames_list2[i], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE).to(device, non_blocking=True)

                # pdb.set_trace()
                pred1, seq_features1, embed1 = model(frames1)
                # pred2, seq_features2, embed2 = model(frames2)

                # true_labels1 = torch.tensor([LABELS['COIN']['train'].index(label_str) for label_str in labels1]).to(device)
                # true_labels2 = torch.tensor([LABELS['COIN']['train'].index(label_str) for label_str in labels2]).to(device)
                #
                # pred_labels1 = torch.argmax(tmp1, dim=-1)
                # pred_labels2 = torch.argmax(tmp2, dim=-1)
                # num_true_pred = torch.sum(pred_labels1 == true_labels1) + torch.sum(pred_labels2 == true_labels2)
                # print('Accuracy: %4f' % (num_true_pred / (len(true_labels1) + len(true_labels2))))
                # # pdb.set_trace()
                #
                # pred1 = model(frames1, embed=True)
                # pred2 = model(frames2, embed=True)
                #
                # distance = torch.sum((pred1 - pred2) ** 2, dim=1)

                embed1s.append(embed1)
                # embed2s.append(embed2)

                # pdb.set_trace()

            # pdb.set_trace()

            embed1s = np.sum(embed1s) / len(embed1s)
            # embed2s = np.sum(embed2s) / len(embed2s)
            embeds.append(embed1s)
            # embeds.append(embed2s)

        # pdb.set_trace()
        embeds = torch.cat(embeds)
        embeds1, embeds2, embeds3 = 0, 0, 0
        for i in range(len(labels)):
            if labels[i] == 2:
                embeds1 += embeds[i]
            if labels[i] == 3:
                embeds2 += embeds[i]
            if labels[i] == 4:
                embeds3 += embeds[i]


        labels = torch.tensor(labels)

        # embeds = torch.cat([embeds, (embeds1/41).unsqueeze(0), (embeds2/41).unsqueeze(0), (embeds3/41).unsqueeze(0)], dim=0)
        # labels = torch.tensor(labels + [7,8,9])




        # PCA
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        embeds = F.normalize(embeds, p=2, dim=1)
        pca_sk = PCA(n_components=2)
        coords = pca_sk.fit_transform(np.array(embeds.cpu()))
        coords0 = coords[labels == 0]
        coords1 = coords[labels == 1]
        coords2 = coords[labels == 2]
        colors0 = ['c' for i in coords0]
        colors1 = ['y' for i in coords1]
        colors2 = ['g' for i in coords2]
        center0 = [coords0[:, 0].mean(), coords0[:, 1].mean()]
        center1 = [coords1[:, 0].mean(), coords1[:, 1].mean()]
        center2 = [coords2[:, 0].mean(), coords2[:, 1].mean()]

        coords = np.concatenate((coords0, coords1, coords2), axis=0)
        centers = np.concatenate(([center0], [center1], [center2]), axis=0)
        print('Variance: 1.1 - %.4f; 1.2 - %.4f; 1.3 - %.4f; centers - %.4f' %
              ((np.var(coords0[:,0])+np.var(coords0[:,1])) / 2,
               (np.var(coords1[:,0])+np.var(coords1[:,1])) / 2,
               (np.var(coords2[:,0])+np.var(coords2[:,1])) / 2,
               (np.var(centers[:,0])+np.var(centers[:,1])) / 2))
        plt.scatter(coords[:, 0], coords[:, 1], c=colors0+colors1+colors2, marker='o')
        # plt.scatter(centers[:, 0], centers[:, 1], c=['b', 'y', 'r'], marker='x')
        plt.savefig('vis1.png')

        pdb.set_trace()

        # Tensorboard visualization

        img = cv2.resize(cv2.imread('square.png'), (30, 30), interpolation=cv2.INTER_AREA)
        img = torch.tensor(img).unsqueeze(0).repeat(embeds.size(0), 1, 1, 1).permute(0, 3, 1, 2)

        pdb.set_trace()
        from tensorboardX import SummaryWriter
        # from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('runs/exp_123_new')
        # writer.add_embedding(embeds, labels, global_step=0)
        writer.add_embedding(embeds, labels, img, global_step=0)






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

            frames_list1 = sample['frames_list1']
            frames_list2 = sample['frames_list2']
            assert len(frames_list1) == len(frames_list2)

            labels1 = sample['label1']
            labels2 = sample['label2']
            label = torch.tensor(np.array(labels1) == np.array(labels2)).to(device)

            pred1 = 0
            pred2 = 0
            # pdb.set_trace()
            for i in range(len(frames_list1)):

                frames1 = frames_preprocess(frames_list1[i], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE).to(device, non_blocking=True)
                frames2 = frames_preprocess(frames_list2[i], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE).to(device, non_blocking=True)

                # pdb.set_trace()
                pred1 += model(frames1, embed=True)
                pred2 += model(frames2, embed=True)
                attn1 = get_local.cache
                vis_func(frames_preprocess(sample["raw_frames_list1"][i], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE), attn1['Attention.forward'])



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

    vis_embedding(args.model_path)
    # process_one_epoch(args.model_path, vis_attn)

