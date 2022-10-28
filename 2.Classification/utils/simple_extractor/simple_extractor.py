from jittor import Module, nn


class SimpleExtractor(Module):
    def __init__(self, patched=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool = nn.AvgPool2d(2, 2)
        self.linear = nn.Linear(16384, 512)
        self.patched = patched
        self.frozen = False

    def execute(self, x):
        x = self.pool(nn.relu(self.bn1(self.conv1(x))))
        x = nn.relu(self.bn2(self.conv2(x)))
        if not self.patched:
            x = self.pool(x)
        x = nn.flatten(x, 1)
        if self.patched:
            x = nn.relu(self.linear(x))
        return x

    def freeze(self):
        if self.frozen:
            return False
        self.frozen = True
        for param in self.parameters():
            param.stop_grad()

        return True

    def unfreeze(self):
        if not self.frozen:
            return False
        self.frozen = False
        for param in self.parameters():
            param.start_grad()

        return True










