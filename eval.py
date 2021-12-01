import torch
import torch.nn.functional as F

import os
import argparse
from configs.defaults import get_cfg_defaults

from utils.logger import setup_logger
from utils.input import frames_preprocess
# from utils.visualization import vis_embedding

from models.model import CAT
from train import setup_seed

from data.dataset import load_dataset

import pdb
import time

from sklearn.metrics import auc, roc_curve

from tqdm import tqdm
import numpy as np
from data.label import LABELS
import xlwt



def compute_auc(model, dist='NormL2'):

    # pdb.set_trace()

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

            pred1s = []
            pred2s = []
            # num_true_pred = 0
            # pdb.set_trace()
            for i in range(len(frames_list1)):

                frames1 = frames_preprocess(frames_list1[i], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE).to(device, non_blocking=True)
                frames2 = frames_preprocess(frames_list2[i], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE).to(device, non_blocking=True)

                # pdb.set_trace()
                tmp1, seq_features1, pred1 = model(frames1)
                tmp2, seq_features2, pred2 = model(frames2)

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

                pred1s.append(pred1)
                pred2s.append(pred2)

                # pdb.set_trace()

            # pdb.set_trace()

            pred1s = np.sum(pred1s) / len(pred1s)
            pred2s = np.sum(pred2s) / len(pred2s)

            # pdb.set_trace()
            if dist == 'L1':
                # L1 distance
                pred = torch.sum(torch.abs(pred1s - pred2s), dim=1)
            elif dist == 'L2':
                # L2 distance
                pred = torch.sum((pred1s - pred2s) ** 2, dim=1)
            elif dist == 'NormL2':
                pred = torch.sum((F.normalize(pred1s, p=2, dim=1) - F.normalize(pred2s, p=2, dim=1)) ** 2, dim=1)
            elif dist == 'cos':
                pred = torch.cosine_similarity(pred1s, pred2s, dim=1)



            if iter == 0:
                preds = pred
                NormL2dist = torch.sum((F.normalize(pred1s, p=2, dim=1) - F.normalize(pred2s, p=2, dim=1)) ** 2, dim=1)
                cos_sim = torch.cosine_similarity(pred1s, pred2s, dim=1)
                labels = label
                label1_all = labels1
                label2_all = labels2
                data_path = sample['data']
            else:
                preds = torch.cat([preds, pred])
                NormL2dist = torch.cat([NormL2dist, torch.sum((F.normalize(pred1s, p=2, dim=1) - F.normalize(pred2s, p=2, dim=1)) ** 2, dim=1)])
                cos_sim = torch.cat([cos_sim, torch.cosine_similarity(pred1s, pred2s, dim=1)])
                labels = torch.cat([labels, label])
                label1_all += labels1
                label2_all += labels2
                data_path += sample['data']
            # pdb.set_trace()

    # print('min is', torch.min(preds))
    # print('mean is', torch.mean(preds))

    # m_scores = []
    # um_scores = []
    #
    # for i in range(len(data_path)):
    #     path1, label1, path2, label2 = data_path[i].split(' ')
    #     print(data_path[i], label1==label2, cos_sim[i].item())
    #
    #     if label1 == label2:
    #         m_scores.append(cos_sim[i].item())
    #     else:
    #         um_scores.append(cos_sim[i].item())
    #
    pdb.set_trace()

    # pairs = []
    # for i in range(len(data_path)):
    #     path1, label1, path2, label2 = data_path[i].split(' ')
    #     if label1 == label2 and cos_sim[i] > 0.8:
    #
    #         for j in range(len(data_path)):
    #             path11, label11, path21, label21 = data_path[j].split(' ')
    #             if path11 == path1 and label21 == '43' and cos_sim[j] < 0.8:
    #
    #                 for p in range(len(data_path)):
    #                     path111, label111, path211, label211 = data_path[j].split(' ')
    #                     if path111 == path1 and label211 == '47' and cos_sim[p] < cos_sim[j]:
    #                         pairs.append([data_path[i], data_path[j], data_path[p]])





    # 42-42, 0.8542
    # '/ssd0/qyc/dataset/Diving48/frames/42/xbQCwTHcGN8_00146 42 /ssd0/qyc/dataset/Diving48/frames/42/_tigfCJFLZg_00512 42 '

    # 42-43 0.6232,
    # '/ssd0/qyc/dataset/Diving48/frames/42/xbQCwTHcGN8_00146 42 /ssd0/qyc/dataset/Diving48/frames/43/6wVdnLa3Tes_00186 43'

    # 42-47 0.2516
    # '/ssd0/qyc/dataset/Diving48/frames/42/xbQCwTHcGN8_00146 42 /ssd0/qyc/dataset/Diving48/frames/47/xbQCwTHcGN8_00309 47'



    fpr, tpr, thresholds = roc_curve(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), pos_label=0)
    auc_value = auc(fpr, tpr)
    wdr = compute_WDR(NormL2dist, label1_all, label2_all)


    # pdb.set_trace()

    best_threshold = 0
    best_accuracy = 0
    for threshold in sorted(preds):
        accuracy = torch.sum((preds < threshold) == labels) / labels.size(0)
        # logger.info('Threshold is %.4f, accuracy is %.4f' % (threshold, accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    # logger.info('Best threshold is %.4f, best accuracy is %.4f, wdr is %.4f' % (best_threshold, best_accuracy, wdr))
    print('Best threshold is ', best_threshold)

    # pdb.set_trace()

    return auc_value, wdr


def save_WDR(data, save_path):

    # pdb.set_trace()


    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet('test')

    # 写入excel
    # 参数对应 行, 列, 值
    worksheet.write(0, 0, 'NormL2 dist')
    worksheet.write(0, 1, 'Edit dist')

    for i in range(len(data)):
        n_dist, e_dist = data[i]
        worksheet.write(i + 1, 0, n_dist)
        worksheet.write(i + 1, 1, e_dist)

    workbook.save(save_path)

    pdb.set_trace()

def compute_WDR(preds, label1, label2):
    # compute weighted dist ratio
    #        weighted dist / # unmatched pairs
    # WDR = ---------------------------------
    #             dist / # matched pairs

    import json
    def read_json(file_path):
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
        return data

    # pdb.set_trace()
    label_bank = read_json('/p300/dataset/COIN/splits/label_bank.json')
    # label_bank = read_json('/p300/dataset/Diving48/splits/label_bank.json')
    # label_bank = read_json('/p300/dataset/ActionVerification/splits/label_bank.json')


    data = []

    # Calcualte wdr
    labels = torch.tensor(np.array(label1) == np.array(label2))
    m_dists = preds[labels]
    um_dists = []
    for i in range(len(labels)):
        label = labels[i]
        if not label:
            # unmatched pair
            # NormL2 dist / edit distance
            um_dists.append(preds[i] / compute_edit_dist(label_bank[label1[i]], label_bank[label2[i]]))
            data.append([preds[i], compute_edit_dist(label_bank[label1[i]], label_bank[label2[i]])])


    # Calcluate averaged NormL2 dist over edit distances
    # pdb.set_trace()
    # edit_dists = list(set([i[1] for i in data]))
    # new_data = [[] for i in edit_dists]
    # for i in range(len(edit_dists)):
    #     new_data[i] = [j[0].item() for j in data if j[1] == edit_dists[i]]
    #     print(len(new_data[i]))
    #     new_data[i] = [np.mean(new_data[i]), edit_dists[i]]
    #
    # save_WDR(new_data, 'edit_dist0.xls')


    return torch.tensor(um_dists).mean() / m_dists.mean()






def compute_edit_dist(seq1, seq2):
    """
    计算字符串 seq1 和 seq1 的编辑距离
    :param seq1
    :param seq2
    :return:
    """


    matrix = [[i + j for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if (seq1[i - 1] == seq2[j - 1]):
                d = 0
            else:
                d = 2

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)



    return matrix[len(seq1)][len(seq2)]




def eval():

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

    # assert args.root_path, logger.info('Please appoint the root path')

    if args.model_path == None:
        model_path = os.path.join(args.root_path, 'save_models')
        # model_path = os.path.join(args.root_path, 'test_models')
    else:
        model_path = args.model_path

    start_time = time.time()
    # pdb.set_trace()
    if os.path.isdir(model_path):
        logger.info('To evaluate %d models in %s' % (len(os.listdir(model_path)) - args.start_epoch + 1, model_path))

        best_auc = 0
        best_model_path = ''

        # pdb.set_trace()
        model_paths = os.listdir(model_path)
        try:
            model_paths.remove('.DS_Store')
            model_paths.remove('._.DS_Store')
        except:
            pass

        # pdb.set_trace()
        model_paths.sort(key=lambda x: int(x[6:-4]))

        for path in model_paths:

            if int(path[6:-4]) < args.start_epoch:
                continue

            # pdb.set_trace()
            checkpoint = torch.load(os.path.join(model_path, path))
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            auc_value, wdr = compute_auc(model, args.dist)
            logger.info("Model is %s, AUC is %.4f, wdr is %.4f" % (os.path.join(model_path, path), auc_value, wdr))

            if auc_value > best_auc:
                best_auc = auc_value
                best_wdr = wdr
                best_model_path = os.path.join(model_path, path)

        logger.info("*** Best models is %s, Best AUC is %.4f, Best wdr is %.4f ***" % (best_model_path, best_auc, best_wdr))
        logger.info('----------------------------------------------------------------')
        # Run again
        for path in model_paths:

            if int(path[6:-4]) < args.start_epoch:
                continue

            # pdb.set_trace()
            checkpoint = torch.load(os.path.join(model_path, path))
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            auc_value, wdr = compute_auc(model, args.dist)
            auc_value, wdr = compute_auc(model, args.dist)
            auc_value, wdr = compute_auc(model, args.dist)
            auc_value, wdr = compute_auc(model, args.dist)
            auc_value, wdr = compute_auc(model, args.dist)
            auc_value, wdr = compute_auc(model, args.dist)
            auc_value, wdr = compute_auc(model, args.dist)
            auc_value, wdr = compute_auc(model, args.dist)
            auc_value, wdr = compute_auc(model, args.dist)
            print("Model is %s, AUC is %.4f" % (os.path.join(model_path, path), auc_value))

    elif os.path.isfile(model_path):
        logger.info('To evaluate 1 models in %s' % (model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        auc_value, wdr = compute_auc(model, args.dist)
        logger.info("Model is %s, AUC is %.4f" % (model_path, auc_value))

        pdb.set_trace()


        # vis
        # pdb.set_trace()
        # txt_path = '/p300/dataset/ActionVerification/vis_1_123.txt'
        # vis_embedding(model, txt_path, cfg, device)


    else:
        logger.info('Wrong model path: %s' % model_path)
        exit(-1)

    end_time = time.time()
    duration = end_time - start_time

    hour = duration // 3600
    min = (duration % 3600) // 60
    sec = duration % 60

    logger.info('Evaluate cost %dh%dm%ds' % (hour, min, sec))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='configs/test_config.yml', help='config file path [default: configs/test_config.yml]')
    parser.add_argument('--root_path', default=None, help='path to load models and save log [default: None]')
    parser.add_argument('--model_path', default=None, help='path to load one model [default: None]')
    parser.add_argument('--log_name', default='eval_log', help='log name')
    parser.add_argument('--start_epoch', default=1, type=int, help='index of the first evaluated epoch while evaluating epochs [default: 1]')
    parser.add_argument('--dist', default='NormL2')


    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # import json
    #
    #
    # def read_json(file_path):
    #     with open(file_path, 'r') as f:
    #         data = json.loads(f.read())
    #     return data
    #
    #
    # # pdb.set_trace()
    # # label_bank = read_json('/p300/dataset/COIN/splits/label_bank.json')
    # label_bank = read_json('/p300/dataset/Diving48/splits/label_bank.json')
    # # label_bank = read_json('/p300/dataset/ActionVerification/splits/label_bank.json')
    #
    #
    # pdb.set_trace()

    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)


    setup_seed(cfg.TRAIN.SEED)
    use_cuda = cfg.TRAIN.USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    if args.root_path:
        logger_path = os.path.join(args.root_path, 'logs')
    else:
        logger_path = 'temp_log'
    logger = setup_logger("ActionVerification", logger_path, args.log_name, 0)
    logger.info("Running with config:\n{}\n".format(cfg))


    test_loader = load_dataset(cfg)


    eval()