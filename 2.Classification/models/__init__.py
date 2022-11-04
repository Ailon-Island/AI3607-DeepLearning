import os
import jittor as jt
from jittor import Module, nn
from .resnet import ResNet50
from .simple_classifier import SimpleClassifier
from .patched_classifier import PatchedClassifier
from .two_stage_classifier import TwoStageClassifier


class Net(Module):
    def __init__(self, opt):
        super(Net, self).__init__()

        self.opt = opt
        pretrained = False
        if opt.model == 'resnet50':
            Model = ResNet50
        elif opt.model == 'simple_classifier':
            Model = SimpleClassifier
        elif opt.model == 'patched_classifier':
            Model = PatchedClassifier
            pretrained = opt.pretrainedj
        elif opt.model == 'two_stage_classifier':
            Model = TwoStageClassifier
            pretrained = opt.pretrained
        else:
            raise ValueError('model not supported')
        self.model = Model(opt.num_classes)
        if pretrained:
            self.model.load(opt.pretrained_dir)

        # weight = jt.ones(opt.num_classes, dtype='float32')
        # weight[:opt.num_classes // 2] = opt.inferior_weight
        # weight *= opt.num_classes / weight.sum()
        self.optimizer = self.model.get_optimizer(opt.lr, opt.weight_decay)
        self.loss = nn.CrossEntropyLoss()

        if opt.resume:
            model_file = os.path.join(opt.checkpoint, opt.name, 'models', 'latest.pth')
            if os.path.exists(model_file):
                self.model.load(model_file)
                print('Model loaded from {}.'.format(model_file))

    def execute(self, x, y=None):
        pred = self.model(x)
        if y is None:
            return pred
        else:
            loss = self.loss(pred, y)
            return pred, loss

    def update_lrs(self):
        for name, lr in self.opt.lr.items():
            new_lr = lr * self.opt.lr_decay
            self.opt.lr[name] = lr * self.opt.lr_decay
            print("Learning rate of {} decay to {}".format(name, new_lr))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.opt.lr_decay
