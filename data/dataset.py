import os
import torch
from torch.utils import data
from PIL import Image
import cv2
import pdb
import time
import numpy as np
import logging

from torchvision import transforms as tf

logger = logging.getLogger('ActionVerification')
# All actions are divided to 3 batches, corresponding to action_ids[0], action_ids[1], action_ids[2]
# Each batch of videos are performed by the same group of people
action_ids = [['1.1', '1.2', '1.3', '1.4', '1.5',
               '2.1', '2.2', '2.3', '2.4', '2.5',
               '3.1', '3.2', '3.3', '3.4', '3.5',
               '4.1', '4.2', '4.3', '4.4', '4.5',
               '5.1', '5.2', '5.3', '5.4', '5.5'],
              ['7.1', '7.2', '7.3', '7.4', '7.5',
               '8.1', '8.2', '8.3', '8.4', '8.5',
               '9.1', '9.2', '9.3', '9.4', '9.5',
               '10.1', '10.2', '10.3', '10.4', '10.5'],
              ['11.1', '11.2', '11.3', '11.4', '11.5',
               '12.1', '12.2', '12.3', '12.4', '12.5',
               '13.1', '13.2', '13.3', '13.4', '13.5',
               '14.1', '14.2', '14.3', '14.4', '14.5',
               '15.1', '15.2', '15.3', '15.4', '15.5']]

action_ids_bank = {'all': action_ids[0] + action_ids[1] + action_ids[2],
                   'train': action_ids[1] + action_ids[2],
                   'test': action_ids[0],
                   # 'test': action_ids[0] +
                   #             ['12.1', '12.2', '12.3', '12.4', '12.5'] +
                   #             ['14.1', '14.2', '14.3', '14.4', '14.5'],
                   'train_old': action_ids[0]}



class ActionVerificationDataset(data.Dataset):

    def __init__(self,
                 mode="train",
                 txt_path=None,
                 normalization=None,
                 num_clip=3,
                 len_clip=1,
                 augment=True,
                 num_sample=600):

        # assert mode in ['train', 'test']
        self.mode = mode

        self.normalization = normalization
        self.num_clip = num_clip  # num of samples selected from each video
        self.len_clip = len_clip  # length of each sample
        self.augment = augment
        if augment:
            self.aug_flip = True
            self.aug_crop = True
            self.aug_color = True
            self.aug_rot = True
        self.num_sample = num_sample  # num of pairs randomly selected from all train pairs

        self.data_list = [line.strip() for line in open(txt_path, 'r').readlines()]
        self.indices = list(range(len(self.data_list)))
        # assert num_sample < len(self.data_list), 'Cannot sample [%d] from [%d] files' % (num_sample, len(self.data_list))
        logger.info('Successfully construct dataset with [%s] mode and [%d] samples randomly selected from [%d] samples' % (mode, len(self), len(self.data_list)))




    def __getitem__(self, index):
        # print(index)

        # Strategy1: Reselect index randomly
        # if self.mode != 'test':
        #     np.random.seed(int(time.time()) + index)
        #     index = np.random.randint(0, len(self.data_list))
        #     while index in self.indices:
        #         index = np.random.randint(0, len(self.data_list))
        #     self.indices.append(index)

        if self.mode == 'train':
            np.random.seed(int(time.time()) + index)
            # np.random.seed(index)
            # print('index w as: ', index)
            index = self.indices.pop(np.random.randint(0, len(self.indices)))
            # print('index is: ', index)

        # Strategy2: Map the index to the larger range
        # index = int(index * (len(self.data_list) / self.num_sample))


        data_path = self.data_list[index]
        data_path_split = data_path.strip().split(' ')

        if len(data_path_split) == 4:
            # train_pairs or test_pairs
            sample = {
                'index': index,
                'frames_list1': self.sample_clips(data_path_split[0]),
                'frames_list2': self.sample_clips(data_path_split[2]),
                'label1': action_ids_bank[self.mode].index(data_path_split[1]),
                'label2': action_ids_bank[self.mode].index(data_path_split[3])
            }
        elif len(data_path_split) == 1:
            # pretrain
            sample = {
                'index': index,
                'frames_list':self.sample_pretrain_clips(data_path_split[0])
            }
        else:
            logger.info('*** Wrong data path! ***')
            exit(-1)

        return sample


    def __len__(self):
        if self.mode == 'train':
            return self.num_sample
        else:
            return len(self.data_list)


    def sample_clips(self, dir_path):
        all_frames = os.listdir(dir_path)
        all_frames = [x for x in all_frames if '_' not in x]

        # pdb.set_trace()

        segments = np.linspace(0, len(all_frames) - self.len_clip - 1, self.num_clip + 1, dtype=int)
        # print(dir_path, segments)
        # if '1.4/luoweixin' in dir_path:
        #     pdb.set_trace()

        if self.mode != 'test':
            sampled_clips = []
            for i in range(self.num_clip):
                start_index = np.random.randint(segments[i], segments[i + 1])
                frames = self.sample_frames(dir_path, start_index, self.augment)
                sampled_clips.append(frames.unsqueeze(-1))
            sampled_clips = torch.cat(sampled_clips, dim=-1)
            return sampled_clips
        elif self.mode == 'test':
            eval_per_segment = 3   # num of samples per segment while evaluating
            sampled_clips_list = []
            for j in range(eval_per_segment):
                sampled_clips = []
                for i in range(self.num_clip):
                    start_index = segments[i] + int((segments[i+1]-segments[i])/4) * (j+1)
                    frames = self.sample_frames(dir_path, start_index, self.augment)
                    sampled_clips.append(frames.unsqueeze(-1))
                sampled_clips = torch.cat(sampled_clips, dim=-1)
                sampled_clips_list.append(sampled_clips)
            # sampled_clips_list = torch.cat(sampled_clips_list, dim=-1)
            return sampled_clips_list
        else:
            logger.info('*** WRONG dataset mode, please check again! ***')
            exit(-1)


    def sample_pretrain_clips(self, dir_path):
        pass


    def sample_frames(self, data_path, start_index, augment):
        sampled_frames = []
        for i in range(self.len_clip):
            frame_index = start_index + i
            frame_path = os.path.join(data_path, str(frame_index + 1) + '.jpg')
            # print('Loading from:', frame_path)
            try:
                frame = Image.open(frame_path)
            except:
                logger.info('Wrong image path %s' % frame_path)
                # print(('Wrong image path %s' % frame_path))
                pdb.set_trace()
            sampled_frames.append(frame)

        # Data augmentation on the clip
        transforms = []
        if augment:
            # if self.seed != None:
            #     np.random.seed(self.seed)

            # Flip
            if np.random.random() > 0.5 and self.aug_flip:
                transforms.append(tf.RandomHorizontalFlip(1))

            # Random crop
            if np.random.random() > 0.5 and self.aug_crop:
                transforms.append(tf.RandomResizedCrop((180, 320), (0.7, 1.0)))

            # Color augmentation
            if np.random.random() > 0.5 and self.aug_color:
                transforms.append(tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5))

            # # Rotation
            # if np.random.random() > 0.5 and self.aug_rot:
            #     transforms.append(tf.RandomRotation(30))

        # PIL image to tensor
        transforms.append(tf.ToTensor())

        # Normalization
        if self.normalization is not None:
            transforms.append(tf.Normalize(self.normalization[0], self.normalization[1]))

        transforms = tf.Compose(transforms)

        for i in range(len(sampled_frames)):
            sampled_frames[i] = transforms(sampled_frames[i])

        # Concat all sampled frames
        for i in range(len(sampled_frames)):
            sampled_frames[i] = sampled_frames[i].unsqueeze(-1)

        sampled_frames = torch.cat(sampled_frames, dim=-1)

        return sampled_frames



def load_dataset(cfg):

    ImageNet_normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    loaders = data.DataLoader(
        ActionVerificationDataset(mode=cfg.DATASET.MODE,
                                  txt_path=cfg.DATASET.TXT_PATH,
                                  normalization=ImageNet_normalization,
                                  num_clip=cfg.DATASET.NUM_CLIP,
                                  len_clip=cfg.DATASET.LEN_CLIP,
                                  augment=cfg.DATASET.AUGMENT,
                                  num_sample=cfg.DATASET.NUM_SAMPLE),
        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.DATASET.SHUFFLE, drop_last=True, num_workers=cfg.DATASET.NUM_WORKERS)

    return loaders