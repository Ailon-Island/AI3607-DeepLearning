from jittor import nn, Module

from utils.simple_extractor.simple_extractor import SimpleExtractor


class TwoStagedClassifier(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.extractor = SimpleExtractor(patched=False)
        self.linear1 = nn.Linear(16384, 256)
        self.linear2 = nn.Linear(256, 96)
        self.linear3 = nn.Linear(96, num_classes)

    def execute(self, x):
        x = self.extractor(x)
        x = nn.relu(self.linear1(x))
        x = nn.relu(self.linear2(x))
        pred = self.linear3(x)
        return pred