import torch.nn
from torch import nn

from .local_loss_blocks import LocalLossBlockConv
import utils.models as utils
from .local_loss_net import LocalLossNet


class AllCNNBlock(LocalLossNet):

    def __init__(self, args, input_dim, channel_out, num_classes, kernel_size=3, first_layer=False):
        super().__init__()
        self.args = args
        out_dim1 = utils.calc_conv_out_dim(input_dim[1], kernel_size)
        out_dim2 = utils.calc_conv_out_dim(out_dim1, kernel_size)
        out_dim3 = utils.calc_conv_out_dim(out_dim2, kernel_size, stride=2)
        self.out_dim = out_dim3
        self.layers = nn.ModuleList([
            LocalLossBlockConv(args, input_dim[0], channel_out, kernel_size, 1, 0, num_classes, out_dim1,
                               first_layer=first_layer),
            LocalLossBlockConv(args, channel_out, channel_out, kernel_size, 1, 0, num_classes, out_dim2),
            LocalLossBlockConv(args, channel_out, channel_out, kernel_size, 2, 0, num_classes, out_dim3)
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
        self.layers.extend([AllCNNBlock(args, [96, self.layers[0].out_dim, self.layers[0].out_dim], 192, num_classes)])
        self.layers.extend([AllCNNTail(args, [192, self.layers[1].out_dim, self.layers[1].out_dim], num_classes)])
