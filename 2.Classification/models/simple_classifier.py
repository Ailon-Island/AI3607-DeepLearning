from jittor import Module, nn


class SimpleClassifier(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 256, 5)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(256 * 5 * 5, 256)
        self.linear2 = nn.Linear(256, 96)
        self.linear3 = nn.Linear(96, num_classes)

    def execute(self, x):
        x = self.pool(nn.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.relu(self.bn2(self.conv2(x))))
        x = nn.flatten(x, 1)
        x = nn.relu(self.linear1(x))
        x = nn.relu(self.linear2(x))
        pred = self.linear3(x)
        return pred

