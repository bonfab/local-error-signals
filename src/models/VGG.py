from functools import reduce

from torch import nn
from operator import mul

from models.local_loss_blocks import LocalLossBlockLinear, LocalLossBlockConv


class Net(nn.Module):
    '''
    A fully connected network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        num_layers (int): Number of hidden layers.
        num_hidden (int): Number of units in each hidden layer.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
    '''

    def __init__(self, args, num_layers, num_hidden, input_dim, num_classes, dropout=0):
        super(Net, self).__init__()
        self.args = args

        self.num_hidden = num_hidden
        self.dropout = dropout
        self.num_layers = num_layers
        reduce_factor = 1
        self.layers = nn.ModuleList(
            [LocalLossBlockLinear(self.args, reduce(mul, input_dim, 1), num_hidden, num_classes, first_layer=True, dropout=dropout, print_stats=self.args.print_stats)])
        self.layers.extend([LocalLossBlockLinear(self.args, int(num_hidden // (reduce_factor ** (i - 1))),
                                                 int(num_hidden // (reduce_factor ** i)), num_classes,  print_stats=self.args.print_stats) for i in
                            range(1, num_layers)])
        self.layer_out = nn.Linear(int(num_hidden // (reduce_factor ** (num_layers - 1))), num_classes)
        if not args.backprop:
            self.layer_out.weight.data.zero_()

    def parameters(self):
        if not self.args.backprop:
            return self.layer_out.parameters()
        else:
            return super(Net, self).parameters()

    def set_learning_rate(self, lr):
        for i, layer in enumerate(self.layers):
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        for i, layer in enumerate(self.layers):
            layer.optim_zero_grad()

    def optim_step(self):
        for i, layer in enumerate(self.layers):
            layer.optim_step()

    def forward(self, x, y, y_onehot):
        x = x.view(x.size(0), -1)
        total_loss = 0.0
        for i, layer in enumerate(self.layers):
            x, loss = layer(x, y, y_onehot)
            total_loss += loss
        x = self.layer_out(x)

        return x, total_loss


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


class VGGn(nn.Module):
    '''
    VGG and VGG-like networks.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        vgg_name (str): The name of the network.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
        feat_mult (float): Multiply number of feature maps with this number.
    '''

    def __init__(self, local_loss_args, vgg_name, input_dim, num_layers, num_hidden, num_classes, dropout=0, feat_mult=1):
        super(VGGn, self).__init__()
        self.cfg = cfg[vgg_name]
        self.args = local_loss_args
        self.input_dim = input_dim
        self.input_ch = input_dim[0]
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.features, output_dim = self._make_layers(self.cfg, self.input_ch, input_dim[1], feat_mult)
        for layer in self.cfg:
            if isinstance(layer, int):
                output_ch = layer
        if self.num_layers > 0:
            self.classifier = Net(self.args, self.num_layers, num_hidden, [output_ch, output_dim, output_dim], num_classes, dropout)
        else:
            self.classifier = nn.Linear(output_dim * output_dim * int(output_ch * feat_mult), num_classes)

    def parameters(self):
        if not self.args.backprop:
            return self.classifier.parameters()
        else:
            return super(VGGn, self).parameters()

    def set_learning_rate(self, lr):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].set_learning_rate(lr)
        if self.args.num_layers > 0:
            self.classifier.set_learning_rate(lr)

    def optim_zero_grad(self):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].optim_zero_grad()
        if self.num_layers > 0:
            self.classifier.optim_zero_grad()

    def optim_step(self):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].optim_step()
        if self.num_layers > 0:
            self.classifier.optim_step()

    def forward(self, x, y, y_onehot):
        loss_total = 0
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                x, loss = self.features[i](x, y, y_onehot)
                loss_total += loss
            else:
                x = self.features[i](x)

        if self.num_layers > 0:
            x, loss = self.classifier(x, y, y_onehot)
        else:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        loss_total += loss

        return x, loss_total

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
                                                  first_layer=first_layer,
                                                  print_stats=self.args.print_stats)]
                else:
                    layers += [LocalLossBlockConv(self.args, input_ch, x, kernel_size=3, stride=1, padding=1,
                                                  num_classes=self.num_classes,
                                                  dim_out=input_dim // scale_cum,
                                                  first_layer=first_layer,
                                                  print_stats=self.args.print_stats)]
                input_ch = x
                first_layer = False

        return nn.Sequential(*layers), input_dim // scale_cum

