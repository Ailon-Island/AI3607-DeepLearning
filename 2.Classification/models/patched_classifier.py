import jittor as jt
from jittor import nn

from .classifier import Classifier
from utils.simple_extractor.simple_extractor import SimpleExtractor


class PatchedClassifier(Classifier):
    def __init__(self, num_classes=10):
        super().__init__()
        self.extractor = SimpleExtractor(patched=True)
        self.linears = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
        self.components = {'extractor': self.extractor, 'linears': self.linears}

    def execute(self, x):
        bs = x.shape[0]
        H_2, W_2 = x.shape[2] // 2, x.shape[3] // 2

        x = jt.stack([x[:, :, :H_2, :W_2], x[:, :, :H_2, W_2:], x[:, :, H_2:, :W_2], x[:, :, H_2:, W_2:]], dim=1)
        x = self.extractor(x.view(-1, *x.shape[-3:]))
        x = x.view(bs, -1)
        pred = self.linears(x)
        return pred
