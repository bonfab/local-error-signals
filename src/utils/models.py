import math
import os

import pandas as pd
import torch
from omegaconf import OmegaConf

import models
from torch import nn
from .configuration import adjust_cfg, load_experiment_cfg


def get_model(cfg, logger=None):
    if logger is not None:
        logger.info(f'Selecting model {cfg.name}')
    if cfg.name.startswith('vgg'):
        return models.VGG(cfg.loss, cfg.name, cfg.input_dim, cfg.num_layers, cfg.num_hidden, cfg.num_classes,
                          cfg.dropout, 1)
    if cfg.name.__contains__('allcnn'):
        return models.AllCNN(cfg.loss, cfg.input_dim, cfg.num_classes)


def similarity_matrix(x, no_similarity_std=False):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if not no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0), -1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc ** 2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)
    return R


class LinearFAFunction(torch.autograd.Function):
    '''Autograd function for linear feedback alignment module.
    '''

    @staticmethod
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_variables
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight_fa)
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias


class LinearFA(nn.Module):
    '''Linear feedback alignment module.
    Args:
        input_features (int): Number of input features to linear layer.
        output_features (int): Number of output features from linear layer.
        bias (bool): True if to use trainable bias.
    '''

    def __init__(self, input_features, output_features, bias=True):
        super(LinearFA, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight_fa = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        if self.args.gpus:
            self.weight.data = self.weight.data.cuda()
            self.weight_fa.data = self.weight_fa.data.cuda()
            if bias:
                self.bias.data = self.bias.data.cuda()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_fa.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.input_features) \
               + ', out_features=' + str(self.output_features) \
               + ', bias=' + str(self.bias is not None) + ')'


def find_zero_grads(name):
    def hook(self, grad_in, grad_out):
        if len([1 for grad in grad_in if grad is not None and (grad == 0).all()]) > 0 or len([1 for grad in grad_out if grad is not None and (grad == 0).all()]) > 0:
            print(f"\ngrad of {name}:")
            print(f"Grad in: {[((grad == 0).all(), grad.shape) for grad in grad_in if grad is not None]}\n"
            f"Grad out {[((grad==0).all(), grad.shape) for grad in grad_out if grad is not None]}")
    return hook


def count_parameters(model):
    ''' Count number of parameters in model influenced by global loss. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_conv_out_dim(input_dim, kernel_size, padding=0, stride=1):
    return int(((input_dim - kernel_size + 2 * padding) / stride) + 1)


class ViewLayer(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def find_best_checkpoint_index(train_results_path):
    results = pd.read_csv(train_results_path)
    return results['valid_acc'].idxmax()


def load_model_params(model, path, cpu):
    if cpu:
        params = torch.load(path, map_location=torch.device('cpu'))
    else:
        params = torch.load(path, map_location=torch.device('cuda'))

    model.load_state_dict(params["model_state_dict"])


def load_best_model_from_exp_dir(exp_dir, cpu=not torch.cuda.is_available()):
    exp_cfg = OmegaConf.create(load_experiment_cfg(os.path.join(exp_dir, ".hydra", "config.yaml")))
    adjust_cfg(exp_cfg)
    if cpu:
        exp_cfg.train.gpus = 0
        exp_cfg.model.loss.gpus = 0
    model = get_model(exp_cfg.model)
    checkpoint_index = find_best_checkpoint_index(os.path.join(exp_dir, "training_results.csv"))
    params_path = os.path.join(exp_dir, "checkpoints", f"{checkpoint_index}.pt")
    load_model_params(model, params_path, not exp_cfg.train.gpus)
    return model
