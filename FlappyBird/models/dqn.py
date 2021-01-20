from torch import nn
from torch.nn.modules import Module


class DQNModule(Module):
    def __init__(self, img_w: int, img_h: int, n_frames: int, batch_size):
        super().__init__()
        self.conv_network = nn.Sequential(
            nn.Conv2d(n_frames, 64,kernel_size=3),  # input => 80 x 80 x 4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),  # output = 40 x 40 x 32

            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),  # output = 20 x 20 x 64

            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),  # output = 10 x 10 x 64

            nn.Conv2d(64, 32, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),  # output = 5 x 5 x 32
        )

        self.linear_network = nn.Sequential(
            nn.Linear(5*5*32, 128),
            nn.ReLU(),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.conv_network(x)
        x = x.view(-1, 5*5*32)
        x = self.linear_network(x)

        return x
