import models
from torch import nn


def get_model(cfg, logger=None):
    if logger is not None:
        logger.info(f'Selecting model {cfg.name}')
    if cfg.name.startswith('vgg'):
        return models.VGG(cfg.loss, cfg.name, cfg.input_dim, cfg.num_layers, cfg.num_hidden, cfg.num_classes,
                          cfg.dropout, 1)
    if cfg.name.__contains__('allcnn'):
        return models.AllCNN(cfg.loss, cfg.input_dim, cfg.num_classes)


def count_parameters(model):
    ''' Count number of parameters in model influenced by global loss. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_conv_out_dim(input_dim, kernel_size, padding=0, stride=1):
    return int(((input_dim - kernel_size + 2 * padding) / stride) + 1)


class ViewLayer(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)
