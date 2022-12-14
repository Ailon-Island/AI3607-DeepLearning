import os
import argparse


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            '--train',
            type=bool,
            default=True,
            help='train or test'
        )
        self.parser.add_argument(
            '--test',
            action='store_false',
            dest='train',
            help='train or test'
        )
        self.parser.add_argument(
            '--name',
            required=True,
            help='name of experiment'
        )
        self.parser.add_argument(
            '--checkpoint',
            default='./checkpoint',
            help='path to save checkpoint'
        )

        # data
        self.parser.add_argument(
            '--data_root',
            default='./cifar_data',
            help='path to CIFAR10 dataset'
        )
        self.parser.add_argument(
            '--num_pos',
            type=int,
            default=4,
            help='number of possible positions in the dataset'
        )
        self.parser.add_argument(
            '--batch_size',
            type=int,
            default=1024,
            help='input batch size'
        )
        self.parser.add_argument(
            '--data_augment',
            action='store_true',
            help='toggle data augmentation'
        )
        self.parser.add_argument(
            '--flip_rate',
            type=float,
            default=0.5,
            help='probability of flipping an image'
        )

        # model
        self.parser.add_argument(
            '--model',
            default='resnet50',
            help='model to train'
        )
        self.parser.add_argument(
            '--use_sinkhorn',
            action='store_true',
            help='toggle sinkhorn'
        )
        self.parser.add_argument(
            '--resume',
            action='store_true',
            help='resume from checkpoint'
        )
        self.parser.add_argument(
            '--save_freq',
            type=int,
            default=1,
        )

        # training
        self.parser.add_argument(
            '--max_epoch',
            type=int,
            default=100,
            help='number of maximum epoch to train for'
        )
        self.parser.add_argument(
            '--max_iter',
            type=int,
            default=100*50000,
            help='number of maximum iteration to train for'
        )
        self.parser.add_argument(
            '--lr',
            type=float,
            default=1e-2,
            help='learning rate'
        )
        self.parser.add_argument(
            '--lr_decay_iter',
            type=int,
            default=50000,
            help='iterations to start learning rate decay'
        )
        self.parser.add_argument(
            '--lr_decay',
            type=float,
            default=0.8,
            help='learning rate decay ratio'
        )
        self.parser.add_argument(
            '--lr_decay_freq',
            type=int,
            default=200000,
            help='frequency of learning rate decay in iterations'
        )
        self.parser.add_argument(
            '--use_gradient_clip',
            action='store_true',
            help='toggle gradient clipping'
        )
        self.parser.add_argument(
            '--gradient_clip',
            type=float,
            default=0.1,
            help='gradient clipping'
        )
        self.parser.add_argument(
            '--weight_decay',
            type=float,
            default=1e-3,
            help='weight decay'
        )

        # visualization

        self.initialized = True


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)
        print('------------ self.options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.train:
            expr_dir = os.path.join(self.opt.checkpoint, self.opt.name)
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            if save and not self.opt.resume:
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as self.opt_file:
                    self.opt_file.write('------------ self.options -------------\n')
                    for k, v in sorted(args.items()):
                        self.opt_file.write('%s: %s\n' % (str(k), str(v)))
                    self.opt_file.write('-------------- End ----------------\n')

        return self.opt


