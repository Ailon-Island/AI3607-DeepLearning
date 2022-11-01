from jittor import nn, Module

from .classifier import Classifier
from utils.simple_extractor.simple_extractor import SimpleExtractor


class TwoStageClassifier(Classifier):
    def __init__(self, num_classes=10):
        super().__init__()
        self.extractor = SimpleExtractor(patched=False)
        self.linears = nn.Sequential(
            nn.Linear(16384, 256),
            nn.ReLU(),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Linear(96, num_classes),
        )
        self.components = {'extractor': self.extractor, 'linears': self.linears}

    def execute(self, x):
        x = self.extractor(x)
        pred = self.linears(x)
        return pred




