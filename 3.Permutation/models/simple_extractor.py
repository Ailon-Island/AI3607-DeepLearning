from jittor import Module, nn


class SimpleExtractor(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool = nn.AvgPool2d(2, 2)
        self.linear = nn.Linear(16384, 512)

    def execute(self, x):
        x = self.pool(nn.relu(self.bn1(self.conv1(x))))
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.flatten(x, 1)
        x = nn.relu(self.linear(x))
        return x








