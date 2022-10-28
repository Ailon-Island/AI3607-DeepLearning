import os
import jittor as jt
from jittor import Module, nn
import pygmtools
from pygmtools import sinkhorn

from .resnet import ResNet50, ResNet18
from .simple_extractor import SimpleExtractor
from .aggregator import Aggregator

pygmtools.BACKEND = 'jittor'


class Net(Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        if opt.model == 'resnet50':
            Extractor = ResNet50
        if opt.model == 'resnet18':
            Extractor = ResNet18
        elif opt.model == 'simple_extractor':
            Extractor = SimpleExtractor
        else:
            raise ValueError('model not supported')
        self.extractor = Extractor()
        self.num_pos = opt.num_pos
        self.aggregator = Aggregator( num_pos=opt.num_pos)
        self.use_sinkhorn = opt.use_sinkhorn
        self.loss = nn.CrossEntropyLoss()

        if opt.resume:
            model_file = os.path.join(opt.checkpoint, opt.name, 'models', 'latest.pth')
            self.load(model_file)

    def execute(self, x, y=None):
        bs = x.shape[0]
        feat = self.extractor(x.view(-1, *x.shape[-3:]))
        out = self.aggregator(feat.view(bs, -1))
        out = out.view(-1, self.num_pos, self.num_pos)
        pred = sinkhorn(out) if self.use_sinkhorn else out

        if y is None:
            return pred
        else:
            loss = self.loss(pred.view(-1, self.num_pos), y)
            return pred, loss
