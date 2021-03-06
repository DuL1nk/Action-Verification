import os
import random

import torch
from torch.utils import data
from PIL import Image
import cv2
import pdb
import time
import numpy as np
import logging
from data.label import LABELS
from torch.utils.data.sampler import Sampler

from torchvision import transforms as tf
# import tensorflow as tf

logger = logging.getLogger('ActionVerification')




class ActionVerificationDataset(data.Dataset):

    def __init__(self,
                 mode="train",
                 txt_path=None,
                 normalization=None,
                 num_clip=3,
                 len_clip=1,
                 augment=True,
                 num_sample=600,
                 use_prefetch=False):

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
        self.use_prefetch = use_prefetch

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

        # if self.mode == 'train':
            # np.random.seed(int(time.time()) + index)
            # np.random.seed(index)
            # print('index was: ', index)
            # index = self.indices.pop(np.random.randint(0, len(self.indices)))
            # print('index is: ', index)
            # print(len(self.indices))

        # Strategy2: Map the index to the larger range
        # index = int(index * (len(self.data_list) / self.num_sample))


        data_path = self.data_list[index]
        data_path_split = data_path.strip().split(' ')

        if len(data_path_split) == 4:
            # train_pairs or test_pairs

            sample = {
                'index': index,
                'data': data_path,
                'frames_list1': self.sample_clips(data_path_split[0]),
                'frames_list2': self.sample_clips(data_path_split[2]),
                # 'label1': LABELS['CV'][self.mode].index(data_path_split[1]) if self.mode == 'train' else data_path_split[1],
                # 'label2': LABELS['CV'][self.mode].index(data_path_split[3]) if self.mode == 'train' else data_path_split[3],
                'label1': LABELS['COIN_S2'][self.mode].index(data_path_split[1]) if self.mode == 'train' else data_path_split[1],
                'label2': LABELS['COIN_S2'][self.mode].index(data_path_split[3]) if self.mode == 'train' else data_path_split[3],
                # 'label1': LABELS['DIVING48_S1'][self.mode].index(data_path_split[1]) if self.mode == 'train' else data_path_split[1],
                # 'label2': LABELS['DIVING48_S1'][self.mode].index(data_path_split[3]) if self.mode == 'train' else data_path_split[3]
                # 'path1': data_path_split[2],
                # 'path2': data_path_split[3],
                # 'label1': data_path_split[1],
                # 'label2': data_path_split[3]
            }
        elif len(data_path_split) == 3:
            # Triplet train pairs
            sample = {
                'index': index,
                'frames_list1': self.sample_clips(data_path_split[0]),
                'frames_list2': self.sample_clips(data_path_split[1]),
                'frames_list3': self.sample_clips(data_path_split[2]),
            }
        elif len(data_path_split) == 6:
            # Triplet+cls train pairs
            sample = {
                'index': index,
                'frames_list1': self.sample_clips(data_path_split[0]),
                'label1': LABELS['TRIPLET'][self.mode].index(data_path_split[1]),
                'frames_list2': self.sample_clips(data_path_split[2]),
                'label2': LABELS['TRIPLET'][self.mode].index(data_path_split[3]),
                'frames_list3': self.sample_clips(data_path_split[4]),
                'label3': LABELS['TRIPLET'][self.mode].index(data_path_split[5]),
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

    def sample_tfrecords(self, tfrecord_path):
        # from data.tfrecord_utils import get_example_nums, read_records
        #
        # temp = tfrecord_path.split('/')
        # temp[4] += '_tfrecord'
        # temp.append(temp[-1] + '.tfrecord')
        # tfrecord_path = ''
        # for tmp in temp: tfrecord_path += tmp + '/'
        # tfrecord_path = tfrecord_path[:-1]
        #
        # # pdb.set_trace()
        #
        # segments = np.linspace(0, get_example_nums(tfrecord_path) - self.len_clip - 1, self.num_clip + 1, dtype=int)
        # start_indices = [np.random.randint(segments[i], segments[i + 1]) for i in range(self.num_clip)]
        # sampled_clips = read_records(tfrecord_path, start_indices)
        #
        # return sampled_clips

        return NotImplementedError


    def sample_clips(self, dir_path, apply_normalization=True):
        all_frames = os.listdir(dir_path)
        all_frames = [x for x in all_frames if '_' not in x]

        # pdb.set_trace()

        segments = np.linspace(0, len(all_frames) - self.len_clip - 1, self.num_clip + 1, dtype=int)
        # print(dir_path, segments)
        # if '1.4/luoweixin' in dir_path:
        #     pdb.set_trace()

        sampled_clips_list = []

        sampled_per_segment = 1 if self.mode == 'train' else 3


        if self.mode == 'train':
            # train mode
            for j in range(sampled_per_segment):
                sampled_clips = []
                for i in range(self.num_clip):
                    start_index = np.random.randint(segments[i], segments[i + 1])
                    frames = self.sample_frames(dir_path, start_index)
                    sampled_clips.append(frames)
                sampled_clips_list.append(self.preprocess_clips(sampled_clips))
                # print('Load img costs: ', spot2 - spot1)
                # print('Process img costs: ', spot3 - spot2)
                # print()

        elif self.mode == 'test' or self.mode == 'val':
            # test, val model
            for j in range(sampled_per_segment):
                sampled_clips = []
                for i in range(self.num_clip):
                    start_index = segments[i] + int((segments[i+1]-segments[i])/4) * (j+1)
                    frames = self.sample_frames(dir_path, start_index)
                    sampled_clips.append(frames)
                sampled_clips_list.append(self.preprocess_clips(sampled_clips))
        else:
            logger.info('*** WRONG dataset mode, please check again! ***')
            exit(-1)

        return sampled_clips_list

    def preprocess_clips(self, clips, apply_normalization=True):

        # Apply augmentation and normalization on a clip of frames

        # pdb.set_trace()

        # Data augmentation on the clip
        transforms = []
        if self.augment:
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

        # if apply_normalization == False:
        #     pdb.set_trace()

        # Normalization
        if self.normalization is not None and apply_normalization:
            transforms.append(tf.Normalize(self.normalization[0], self.normalization[1]))

        transforms = tf.Compose(transforms)


        clips = torch.cat([torch.cat([transforms(frame).unsqueeze(-1) for frame in clip], dim=-1).unsqueeze(-1) for clip in clips], dim=-1)


        return clips



    def sample_pretrain_clips(self, dir_path):
        pass


    def sample_frames(self, data_path, start_index):
        sampled_frames = []

        for i in range(self.len_clip):
            frame_index = start_index + i
            frame_path = os.path.join(data_path, str(frame_index + 1) + '.jpg')
            # print('Loading from:', frame_path)
            try:
                # PIL read
                # frame = Image.open(frame_path)

                # Opencv read
                frame = cv2.imread(frame_path)
                # Convert RGB to BGR and transform to PIL.Image
                frame = Image.fromarray(frame[:,:,[2,1,0]])



                # tfrecord read



                sampled_frames.append(frame)
            except:
                logger.info('Wrong image path %s' % frame_path)
                # print(('Wrong image path %s' % frame_path))
                pdb.set_trace()

        return sampled_frames


class TestSampler(Sampler):
    def __init__(self, dataset, txt_path, shuffle=False):
        self.dataset = dataset
        self.data_list = [line.strip() for line in open(txt_path, 'r').readlines()]
        self.shuffle = shuffle

    def __iter__(self):

        tmp = random.sample(range(len(self.data_list)), len(self.dataset))
        if not self.shuffle:
            tmp.sort()

        # print(tmp)
        return iter(tmp)

    def __len__(self):
        return len(self.dataset)




def load_dataset(cfg):

    ImageNet_normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


    dataset = ActionVerificationDataset(mode=cfg.DATASET.MODE,
                                  txt_path=cfg.DATASET.TXT_PATH,
                                  normalization=ImageNet_normalization,
                                  num_clip=cfg.DATASET.NUM_CLIP,
                                  len_clip=cfg.DATASET.LEN_CLIP,
                                  augment=cfg.DATASET.AUGMENT,
                                  num_sample=cfg.DATASET.NUM_SAMPLE,
                                  use_prefetch=cfg.TRAIN.PREFETCH)

    sampler = TestSampler(dataset, cfg.DATASET.TXT_PATH, cfg.DATASET.SHUFFLE)

    loaders = data.DataLoader(dataset=dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=False,
                              sampler=sampler,
                              drop_last=False,
                              num_workers=cfg.DATASET.NUM_WORKERS,
                              pin_memory=True)

    return loaders