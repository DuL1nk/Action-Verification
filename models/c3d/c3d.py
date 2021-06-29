import torch.nn as nn
import pdb
import torch

import logging
logger = logging.getLogger('ActionVerification')
class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self,
                 pretrain,
                 dim_embedding,
                 dropout):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 0))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 4, 4), stride=(2, 4, 4), padding=(0, 1, 0))

        # self.fc6 = nn.Linear(8192, 4096)
        self.fc6 = nn.Linear(7680, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, dim_embedding)

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        if pretrain:
            # pdb.set_trace()
            state_dict = torch.load(pretrain)
            self.load_state_dict({k:v for k,v in state_dict.items() if k[:2] != 'fc'}, strict=False)
            logger.info('Loading backbone state_dict from %s' % pretrain)

    def forward(self, x):

        # pdb.set_trace()

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        # h = h.view(-1, 8192)
        h = h.view(-1, 7680)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)

        return logits

        probs = self.softmax(logits)

        return probs