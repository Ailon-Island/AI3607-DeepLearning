import jittor as jt
from jittor import nn, Module

from utils.simple_extractor.simple_extractor import SimpleExtractor


class PatchedClassifier(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.extractor = SimpleExtractor(patched=True)
        self.linear1 = nn.Linear(2048, 256)
        self.linear2 = nn.Linear(256, num_classes)

    def execute(self, x):
        bs = x.shape[0]
        H_2, W_2 = x.shape[2] // 2, x.shape[3] // 2

        x = jt.stack([x[:, :, :H_2, :W_2], x[:, :, :H_2, W_2:], x[:, :, H_2:, :W_2], x[:, :, H_2:, W_2:]], dim=1)
        x = self.extractor(x.view(-1, *x.shape[-3:]))
        x = x.view(bs, -1)
        x = nn.relu(self.linear1(x))
        pred = self.linear2(x)
        return pred
