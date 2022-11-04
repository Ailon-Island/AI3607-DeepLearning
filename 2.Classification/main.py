import os
import time
import argparse
from matplotlib import pyplot as plt

import numpy as np
import jittor as jt
from jittor import transform

from options import Options
from dataset import CIFAR10
from models import Net


class Trainer:
    def __init__(self, opt, total_iters, model, loader, test_loader):
        self.opt = opt
        self.total_iters = total_iters
        self.model = model
        self.grad_clip = opt.use_gradient_clip
        self.loader = loader
        self.test_loader = test_loader
        self.losses = {'train': [], 'test': []}
        self.accs = {'train': [], 'test': []}
        self.iter_footprint = {'train': [], 'test': []}
        self.log_file = os.path.join(opt.checkpoint, opt.name, 'log.txt')

    def train(self, epoch_idx):
        self.model.train()

        num_iter = 0
        for batch_idx, (data, label) in enumerate(self.loader, start=1):
            self.total_iters += data.shape[0]
            num_iter += data.shape[0]
            pred, loss = self.model(data, label)
            self.model.optimizer.step(loss)

            acc = (pred.argmax(dim=1)[0] == label).float().mean()

            self.losses['train'] += [loss.item()]
            self.accs['train'] += [acc.item()]
            self.iter_footprint['train'] += [self.total_iters]

            if batch_idx % 10 == 0:
                self.log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                    epoch_idx, num_iter, len(self.loader),
                    100. * num_iter / len(self.loader), loss.item(), acc.item()
                ))

            # learning rate decay
            if opt.lr_decay_iter >= 0:
                if self.total_iters >= opt.lr_decay_iter:
                    if (self.total_iters - opt.lr_decay_iter) % opt.lr_decay_freq == 0:
                        self.model.update_lrs()

    def test(self, epoch_idx):
        self.model.eval()

        test_loss, test_acc = [], []
        total_loss, total_acc = 0., 0.
        num_iter = 0
        with jt.no_grad():
            for batch_idx, (data, label) in enumerate(self.test_loader, start=1):
                num_iter += data.shape[0]
                pred, loss = self.model(data, label)
                test_loss += [loss.item()]
                acc = (pred.argmax(dim=1)[0] == label).float().mean()
                test_acc += [acc.item()]
                total_loss += loss.item() * data.shape[0]
                total_acc += acc.item() * data.shape[0]

                if batch_idx % 10 == 0:
                    self.log('Testing Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                        epoch_idx, num_iter, len(self.test_loader),
                        100. * num_iter / len(self.test_loader), loss.item(), acc.item()))

        total_loss /= num_iter
        total_acc /= num_iter
        self.losses['test'] += [total_loss]
        self.accs['test'] += [total_acc]
        self.iter_footprint['test'] += [self.total_iters]

        self.log('Test Epoch: {}, Average loss: {:.6f}, Accuracy: {:.6f}'.format(epoch_idx, total_loss, total_acc))

    def save(self):
        model_dir = os.path.join(opt.checkpoint, opt.name, 'models')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_dir = os.path.join(model_dir)
        self.model.save(os.path.join(model_dir, 'iter_{}.pth'.format(self.total_iters)))
        self.model.save(os.path.join(model_dir, 'latest.pth'))
        self.log('Model saved to {}'.format(os.path.join(model_dir, 'iter_{}.pth'.format(self.total_iters))))

    def plot(self, save=False):
        results_dir = os.path.join(opt.checkpoint, opt.name, 'results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        train_losses, test_losses = self.losses['train'], self.losses['test']
        train_accs, test_accs = self.accs['train'], self.accs['test']

        iters = self.iter_footprint['train']
        epochs = self.iter_footprint['test']

        plt.figure(figsize=(10, 5))
        plt.plot(iters, train_losses, label='Train loss')
        plt.plot(epochs, test_losses, label='Test loss')
        plt.ylim(0, 2)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        if save:
            plt.savefig(os.path.join(results_dir, 'loss.png'))
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(iters, train_accs, label='Train accuracy')
        plt.plot(epochs, test_accs, label='Test accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()
        if save:
            plt.savefig(os.path.join(results_dir, 'accuracy.png'))
        plt.show()

    def log(self, msg):
        with open(self.log_file, 'a') as f:
            f.write(msg+'\n')
            print(msg)


def main():
    # Get Data Loaders
    print('Preparing data loaders...')
    train_loader = CIFAR10(opt=opt, train=True, shuffle=True, augment=augment, transform=transform)
    test_loader = CIFAR10(opt=opt, train=False, shuffle=False, augment=None, transform=transform)

    # Load the model
    print('Loading the model...')
    model = Net(opt)
    total_iters = 0
    trainer = Trainer(opt, total_iters, model, train_loader, test_loader)

    # Train & test the model
    print('Preparation done. Now start training.')
    for epoch_idx in range(1, opt.max_epoch+1):
        epoch_start_time = time.time()
        print('Epoch: {}'.format(epoch_idx))

        trainer.train(epoch_idx)
        trainer.test(epoch_idx)

        epoch_time = time.time() - epoch_start_time

        # visualization
        print(f'End of epoch: {epoch_idx}\ttime: {epoch_time:.2f}s')
        trainer.plot(save=True)

        # save model
        if epoch_idx % opt.save_freq == 0:
            trainer.save()

        if trainer.total_iters >= opt.max_iter:
            print('Reach max iteration. Stop training.')
            break

    # Save the results
    trainer.plot(save=True)


if __name__ == "__main__":
    jt.flags.use_cuda = 1
    jt.set_global_seed(0)

    opt = Options().parse()

    flip_prob = 1. - np.sqrt(1. - opt.flip_rate)
    augment = transform.Compose([
        transform.RandomHorizontalFlip(p=flip_prob),
        transform.RandomVerticalFlip(p=flip_prob),
        transform.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    ]) if opt.data_augment else None
    transform = transform.Compose([
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.5], std=[0.5]),
    ])

    main()


