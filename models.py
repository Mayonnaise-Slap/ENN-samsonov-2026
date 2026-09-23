import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=7 // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=5 // 2, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3 // 2, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=1 // 2, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=3 // 2, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=1 // 2, bias=False),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)
