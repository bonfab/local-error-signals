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
