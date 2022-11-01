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
        self.optimizers = self.model.get_optimizers(opt.lr)
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

    def optimize_step(self, loss):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self.optimizers.values():
            optimizer.step()

    def update_lrs(self):
        new_lr_default = self.opt.lr['default'] * self.opt.lr_decay
        self.update_lr('default', new_lr_default)

        for name, lr in self.opt.lr.items():
            if name == 'default':
                continue
            if self.opt.lr[name] >= new_lr_default:
                self.update_lr(name, new_lr_default)

    def update_lr(self, name, lr):
        self.opt[name] = lr
        for param_group in self.optimizers[name].param_groups:
            param_group['lr'] = lr
        self.log('Learning rate of {} updates to {}'.format(name, lr))
