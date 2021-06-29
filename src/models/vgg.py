from functools import reduce

from torch import nn
from operator import mul

import utils.models as utils
from .local_loss_blocks import LocalLossBlockLinear, LocalLossBlockConv
from .local_loss_net import LocalLossNet


class FullyConnectedNet(LocalLossNet):
    '''
    A fully connected network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        num_layers (int): Number of hidden layers.
        num_hidden (int): Number of units in each hidden layer.
        input_dim (list): Iterable of input dimenstions, usually (ch, h, w).
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
    '''

    def __init__(self, args, num_layers, num_hidden, input_dim, num_classes, dropout=0):
        super().__init__()
        self.args = args

        self.num_hidden = num_hidden
        self.dropout = dropout
        self.num_layers = num_layers
        reduce_factor = 1
        self.layers = nn.ModuleList([utils.ViewLayer()])
        if num_layers > 0:
            self.layers.extend(
                [LocalLossBlockLinear(self.args, reduce(mul, input_dim, 1), num_hidden, num_classes, first_layer=True,
                                      dropout=dropout, print_stats=self.args.print_stats)])

            self.layers.extend([LocalLossBlockLinear(self.args, int(num_hidden // (reduce_factor ** (i - 1))),
                                                     int(num_hidden // (reduce_factor ** i)), num_classes,
                                                     print_stats=self.args.print_stats)
                                for i in range(1, num_layers)])

        self.layers.extend([nn.Linear(int(num_hidden // (reduce_factor ** (num_layers - 1))), num_classes)])

        # Was in original code, no clue why this was used
        """if not args.backprop:
            self.layers[-1].weight.data.zero_()"""


cfg = {
    'vgg6a': [128, 'M', 256, 'M', 512, 'M', 512],
    'vgg6b': [128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg8a': [128, 256, 'M', 256, 512, 'M', 512, 'M', 512],
    'vgg8b': [128, 256, 'M', 256, 512, 'M', 512, 'M', 512, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg11b': [128, 128, 128, 256, 'M', 256, 512, 'M', 512, 512, 'M', 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGn(LocalLossNet):
    '''
    VGG and VGG-like networks.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        vgg_name (str): The name of the network.
        input_dim (list): Iterable of input dimenstions, usually (ch, h, w)
        num_layers (int): Number of layers
        num_classes (int): Number of classes (used in local prediction loss).
        feat_mult (float): Multiply number of feature maps with this number.
    '''

    def __init__(self, local_loss_args, vgg_name, input_dim, num_layers, num_hidden, num_classes, dropout=0,
                 feat_mult=1):
        super(VGGn, self).__init__()
        self.cfg = cfg[vgg_name]
        self.args = local_loss_args
        self.input_dim = input_dim
        self.input_ch = input_dim[0]
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.layers = self._make_layers(self.cfg, self.input_ch, input_dim[1], feat_mult)

    def _make_layers(self, cfg, input_ch, input_dim, feat_mult):
        layers = []
        first_layer = True
        scale_cum = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                scale_cum *= 2
            elif x == 'M3':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
                scale_cum *= 2
            elif x == 'M4':
                layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
                scale_cum *= 4
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                scale_cum *= 2
            elif x == 'A3':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]
                scale_cum *= 2
            elif x == 'A4':
                layers += [nn.AvgPool2d(kernel_size=4, stride=4)]
                scale_cum *= 4
            else:
                x = int(x * feat_mult)
                if first_layer and input_dim > 64:
                    scale_cum = 2
                    layers += [LocalLossBlockConv(self.args, input_ch, x, kernel_size=7, stride=2, padding=3,
                                                  num_classes=self.num_classes,
                                                  dim_out=input_dim // scale_cum,
                                                  first_layer=first_layer)]
                else:
                    layers += [LocalLossBlockConv(self.args, input_ch, x, kernel_size=3, stride=1, padding=1,
                                                  num_classes=self.num_classes,
                                                  dim_out=input_dim // scale_cum,
                                                  first_layer=first_layer)]
                input_ch = x
                first_layer = False
            if isinstance(x, int):
                output_ch = x

        output_dim = input_dim // scale_cum
        layers += [
            FullyConnectedNet(self.args, self.num_layers, self.num_hidden, [output_ch, output_dim, output_dim], self.num_classes,
                              self.dropout)]

        return nn.ModuleList(layers)
