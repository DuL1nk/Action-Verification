import torch
import torchvision.transforms as tf
import os
import numpy as np

import pdb
from PIL import Image

from utils.input import frames_preprocess
from data.dataset import action_ids_bank
from tensorboardX import SummaryWriter
from tqdm import tqdm

import cv2



def sample_frames(data_path, mode=None, num_clip=16, len_clip=1, augment=False):

    all_frames = os.listdir(data_path)
    segments = np.linspace(0, len(all_frames) - len_clip - 1, num_clip + 1, dtype=int)

    eval_per_segment = 3  # num of samples per segment while evaluating
    sampled_clips_list = []
    for j in range(eval_per_segment):
        sampled_clips = []
        for i in range(num_clip):

            start_index = segments[i] + int((segments[i + 1] - segments[i]) / 4) * (j + 1)

            sampled_frames = []
            for i in range(len_clip):
                frame_index = start_index + i
                frame_path = os.path.join(data_path, str(frame_index + 1) + '.jpg')
                # print('Loading from:', frame_path)
                try:
                    frame = Image.open(frame_path)
                except:
                    # logger.info('Wrong image path %s' % frame_path)
                    pdb.set_trace()
                sampled_frames.append(frame)

            # Data augmentation on the clip
            transforms = []
            if augment:
                # if self.seed != None:
                #     np.random.seed(self.seed)

                # Flip
                if np.random.random() > 0.5:
                    transforms.append(tf.RandomHorizontalFlip(1))

                # Random crop
                if np.random.random() > 0.5:
                    transforms.append(tf.RandomResizedCrop((180, 320), (0.7, 1.0)))

                # Color augmentation
                if np.random.random() > 0.5:
                    transforms.append(tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5))

                # # Rotation
                # if np.random.random() > 0.5 and self.aug_rot:
                #     transforms.append(tf.RandomRotation(30))

            # PIL image to tensor
            transforms.append(tf.ToTensor())

            # Normalization
            transforms.append(tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

            transforms = tf.Compose(transforms)

            for i in range(len(sampled_frames)):
                sampled_frames[i] = transforms(sampled_frames[i])

            # Concat all sampled frames
            for i in range(len(sampled_frames)):
                sampled_frames[i] = sampled_frames[i].unsqueeze(-1)

            sampled_frames = torch.cat(sampled_frames, dim=-1)






            # sampled_clips.append(sampled_frames.unsqueeze(-1))
            sampled_clips.append(sampled_frames.unsqueeze(-1))

        sampled_clips = torch.cat(sampled_clips, dim=-1)
        sampled_clips_list.append(sampled_clips.unsqueeze(0))
    # sampled_clips_list = torch.cat(sampled_clips_list, dim=-1)
    return sampled_clips_list


def vis_embedding(model, txt_path, cfg, device):

    # pdb.set_trace()

    id_bank = ['', '', '', '1.1', '1.2', '1.3']

    model = model.to(device)
    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        # logger.info("Let's use %d GPUs" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    with torch.no_grad():

        data_list = [line.strip() for line in open(txt_path, 'r'). readlines()]
        embeds = []
        labels = []

        for i in tqdm(range(len(data_list))):

            data_path = data_list[i]

            embed = 0
            frames_list = sample_frames(data_path)

            for i in range(len(frames_list)):
                frames = frames_preprocess(frames_list[i], cfg.MODEL.BACKBONE_DIM, cfg.MODEL.BACKBONE).to(device)
                embed += model(frames, embed=True)

            embed /= len(frames_list)
            segments = data_path.strip().split('/')
            label = id_bank.index(segments[-2])

            embeds.append(embed)
            labels.append(label)


        embeds = torch.cat(embeds, dim=0)
        labels = torch.tensor(labels)

        size = (30, 30)
        # img1 = cv2.resize(cv2.imread('5.1.png'), size, interpolation=cv2.INTER_AREA)
        # img2 = cv2.resize(cv2.imread('5.2.png'), size, interpolation=cv2.INTER_AREA)
        # img3 = cv2.resize(cv2.imread('5.3.png'), size, interpolation=cv2.INTER_AREA)
        #
        # img1 = torch.tensor(img1).unsqueeze(0).repeat(42, 1, 1, 1).permute(0, 3, 1, 2)
        # img2 = torch.tensor(img2).unsqueeze(0).repeat(42, 1, 1, 1).permute(0, 3, 1, 2)
        # img3 = torch.tensor(img3).unsqueeze(0).repeat(42, 1, 1, 1).permute(0, 3, 1, 2)
        #
        # img = torch.cat([img1, img2, img3], dim=0)



        img = cv2.resize(cv2.imread('square.png'), size, interpolation=cv2.INTER_AREA)
        img = torch.tensor(img).unsqueeze(0).repeat(126, 1, 1, 1).permute(0, 3, 1, 2)



        pdb.set_trace()
        writer = SummaryWriter('runs/exp')
        writer.add_embedding(embeds, labels, img, global_step=0)

    return 0