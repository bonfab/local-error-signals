import torch.nn
from torch import nn

from .local_loss_blocks import LocalLossBlockConv
import utils.models as utils
from .local_loss_net import LocalLossNet


class AllCNNBlock(LocalLossNet):

    def __init__(self, args, input_dim, channel_out, num_classes, kernel_size=3, first_layer=False):
        """
            for now only square input
            and
            only square, uneven kernel sizes
        """
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.padding = int(kernel_size/2)
        self.output_dim = utils.calc_conv_out_dim(input_dim[1], kernel_size, self.padding, 2)
        self.layers = nn.ModuleList([
            LocalLossBlockConv(args, input_dim[0], channel_out, kernel_size, 1, self.padding, num_classes, input_dim[1],
                               first_layer=first_layer),
            LocalLossBlockConv(args, channel_out, channel_out, kernel_size, 1, self.padding, num_classes, input_dim[1]),
            LocalLossBlockConv(args, channel_out, channel_out, kernel_size, 2, self.padding, num_classes, self.output_dim)
        ])


class AllCNNTail(LocalLossNet):

    def __init__(self, args, input_dim, num_classes, kernel_size=3):
        super().__init__()
        self.args = args
        out_dim = utils.calc_conv_out_dim(input_dim[1], kernel_size)
        self.layers = nn.ModuleList([
            LocalLossBlockConv(args, input_dim[0], input_dim[0], kernel_size, 1, 0, num_classes, out_dim),
            LocalLossBlockConv(args, input_dim[0], input_dim[0], 1, 1, 0, num_classes, out_dim),
            nn.Conv2d(input_dim[0], num_classes, 1),
            torch.nn.AvgPool2d(out_dim),
            utils.ViewLayer()
        ])

    def parameters(self):
        if not self.args.backprop:
            return self.layers[2].parameters()
        else:
            return nn.Module.parameters(self)


class AllCNN(LocalLossNet):
    """
    Assumes at the moment input width=height
    """

    def __init__(self, args, input_dim, num_classes):
        self.args = args
        super().__init__()
        self.layers = nn.ModuleList([
            AllCNNBlock(args, input_dim, 96, num_classes, first_layer=True),
        ])
        self.layers.extend([AllCNNBlock(args, [96, self.layers[0].output_dim, self.layers[0].output_dim], 192, num_classes)])
        self.layers.extend([AllCNNTail(args, [192, self.layers[1].output_dim, self.layers[1].output_dim], num_classes)])
