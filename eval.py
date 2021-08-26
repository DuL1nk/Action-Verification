import torch

import os
import argparse
from configs.defaults import get_cfg_defaults

from utils.logger import setup_logger
from utils.input import frames_preprocess
from utils.visualization import vis_embedding

from models.model import CAT
from train import setup_seed

from data.dataset import load_dataset

import pdb
import time

from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from tqdm import tqdm

def compute_auc(model, dist='L2'):

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

                pred1 += model(frames1, embed=True)

                pred2 += model(frames2, embed=True)

            # pdb.set_trace()

            pred1 /= len(frames_list1)
            pred2 /= len(frames_list1)

            # L1 distance
            if dist == 'L1':
                pred = torch.sum(torch.abs(pred1 - pred2), dim=1)

            # L2 distance
            if dist == 'L2':
                pred = torch.sum((pred1 - pred2) ** 2, dim=1)

            if iter == 0:
                preds = pred
                labels = label
            else:
                preds = torch.cat([preds, pred])
                labels = torch.cat([labels, label])

    # print('min is', torch.min(preds))
    # print('mean is', torch.mean(preds))
    # pdb.set_trace()

    fpr, tpr, thresholds = roc_curve(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), pos_label=0)
    auc_value = auc(fpr, tpr)

    best_threshold = 0
    best_accuracy = 0
    for threshold in sorted(preds):
        accuracy = torch.sum((preds < threshold) == labels) / labels.size(0)
        # logger.info('Threshold is %.4f, accuracy is %.4f' % (threshold, accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    logger.info('Best threshold is %.4f, best accuracy is %.4f' % (best_threshold, best_accuracy))


    # pdb.set_trace()

    return auc_value







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
            auc_value = compute_auc(model)
            logger.info("Model is %s, AUC is %.4f" % (os.path.join(model_path, path), auc_value))

            if auc_value > best_auc:
                best_auc = auc_value
                best_model_path = os.path.join(model_path, path)

        logger.info("*** Best models is %s, Best AUC is %.4f ***" % (best_model_path, best_auc))

    elif os.path.isfile(model_path):
        logger.info('To evaluate 1 models in %s' % (model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        auc_value = compute_auc(model)
        logger.info("Model is %s, AUC is %.4f" % (model_path, auc_value))


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


    if args.root_path:
        logger_path = os.path.join(args.root_path, 'logs')
    else:
        logger_path = 'temp_log'
    logger = setup_logger("ActionVerification", logger_path, args.log_name, 0)
    logger.info("Running with config:\n{}\n".format(cfg))


    test_loader = load_dataset(cfg)

    eval()