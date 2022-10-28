import os
import jittor as jt
from jittor import Module, nn
from .resnet import ResNet50
from .simple_classifier import SimpleClassifier
from .patched_classifier import PatchedClassifier
from .two_staged_classifier import TwoStagedClassifier


class Net(Module):
    def __init__(self, opt):
        super(Net, self).__init__()

        pretrained = False
        if opt.model == 'resnet50':
            Model = ResNet50
        elif opt.model == 'simple_classifier':
            Model = SimpleClassifier
        elif opt.model == 'patched_classifier':
            Model = PatchedClassifier
            pretrained = opt.pretrained
        elif opt.model == 'two_staged_classifier':
            Model = TwoStagedClassifier
            pretrained = opt.pretrained
        else:
            raise ValueError('model not supported')
        self.model = Model(opt.num_classes)
        if pretrained:
            self.model.load(opt.pretrained_dir)

        # weight = jt.ones(opt.num_classes, dtype='float32')
        # weight[:opt.num_classes // 2] = opt.inferior_weight
        # weight *= opt.num_classes / weight.sum()
        self.loss = nn.CrossEntropyLoss()

        if opt.resume:
            model_file = os.path.join(opt.checkpoint, opt.name, 'models', 'latest.pth')
            self.load(model_file)

    def execute(self, x, y=None):
        pred = self.model(x)
        if y is None:
            return pred
        else:
            loss = self.loss(pred, y)
            return pred, loss
