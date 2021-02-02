import torch
from torch import nn
from torch.nn.modules import Module

import numpy as np


class DQNModule(Module):
    def __init__(self, img_w: int, img_h: int, n_frames: int):
        super().__init__()
        self.img_w = img_w
        self.img_h = img_h

        self.conv_network = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=3),  # input => 80 x 80 x 4
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),  # output = 40 x 40 x 32

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)),  # output = 20 x 20 x 64

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)),  # output = 10 x 10 x 64

            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),  # output = 5 x 5 x 32
        )

        self.linear_input_size = self.get_linear_input_size(n_frames)

        self.linear_network = nn.Sequential(
            nn.Linear(self.linear_input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.conv_network(x)
        x = x.view(-1, self.flatten_shape(x))
        x = self.linear_network(x)

        return x

    def get_linear_input_size(self, n_frames: int):
        x = np.zeros((1, n_frames, self.img_w, self.img_h))
        x = torch.from_numpy(x)
        y = self.conv_network(x.float())

        return self.flatten_shape(y)

    @staticmethod
    def flatten_shape(x: torch.Tensor):
        dims = x.size()[1:]
        flat_shape = 1
        for dim in dims:
            flat_shape *= dim

        return flat_shape

