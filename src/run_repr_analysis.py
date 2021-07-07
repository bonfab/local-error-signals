# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#from lib.conv_networks_AllCNN import DDTPConvAllCNNC
import sys
import argparse
import random
import wandb
import math
import os
import torch
import torch.optim as optim

import representation_analysis_tools.plots as plots
import representation_analysis_tools.logs as logs
import representation_analysis_tools.lazydict as lazydict

import representation_analysis_tools.rsa as rsa
import representation_analysis_tools.utils as repr_utils
import representation_analysis_tools.intrinsic_dimension as intrinsic_dimension
import representation_analysis_tools.centered_kernel_alignment as cka

from torch import nn
from torch.utils.data import Subset
from torchvision import datasets, transforms
from functools import partial
from pathlib import Path
from models.local_loss_blocks import LocalLossBlock
from models.local_loss_net import LocalLossNet
import utils.models

import matplotlib.pyplot as plt

'''
In this script you need to specify the variables below. 

1. subsample_test   :   fraction of the CIFAR10 testset to you use for the evaluation. My laptop can only handle 30% but maybe you can do
                        even more
2. model_name       :   the evaluation data will be stored in logs/similarity_metrics/<model_name>
                        To plot the evaluation data you have to specify this model_name in the mlp_representations.py script
3. model            :   Load the model with some model loader. Some possible loaders are outcommented below
4. modules          :   The list of layers to be analysed. For the local error model use the model.get_base_inference_layers()
                        For the target model use the list(model.named_modules())[1:] and del the 9th entry 
'''
# Select the network to evaluate here
#1.
subsample_test = 0.3

#2.
model_name = 'mlp_local_1'

#3.
# The imports for the target loss need to be changed. Maybe change the structure of the repository so the target stuff also is in the src folder

#import theoretical_framework_for_target_propagation.AllCNNC_backprop as cnnc

#model = cnnc.AllCNNC()
#model.load_state_dict(torch.load('theoretical_framework_for_target_propagation/results/pure_backprop/weights_backprop.pth',map_location=torch.device('cpu')))
model, _ = utils.models.load_best_model_from_exp_dir("../2021-06-18_12-50-30/1")

#4.
#modules = list(model.named_modules())[1:]
#del modules[9]
modules = model.get_base_inference_layers()


# %%
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model-name', type=str, default=model_name,
                    help='Name of the model (for logging).')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--use-bias', type=bool, default=False,
                    help='use biases')
parser.add_argument('--weight-init', type=str, default=None,
                    help='weight initialization')
parser.add_argument('--random-labels', type=bool, default=False,
                    help='train with random labels')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer for training')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum if used')
parser.add_argument('--step-lr', type=bool, default=False,
                    help='use step learning rate scheduler with gamma')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma if used')
parser.add_argument('--no-cuda', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu-id', type=int, default=2,
                    help='gpu device to use if used')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--log-interval-batch', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log-interval-epoch', type=int, default=1, metavar='N',
                    help='how many epoches to wait before logging similarity metrics')
parser.add_argument('--not-save-model', default=False,
                    help='For Not Saving the current Model')
parser.add_argument('--not-wandb', default=False,
                    help='For not using weights and biases.')

if any(['ipykernel' in arg for arg in sys.argv]):
    args = parser.parse_args(['--model-name', 'mlp', '--epochs', '2', '--log-interval-epoch', '1', '--not-wandb', 't', '--not-save-model', 't'])
else:
    args = parser.parse_args()


# %%
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu_id) if use_cuda else "cpu")

torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307, ), (0.3081, ))])


def gen_train_loader(dataset, batch_size=args.batch_size, **kwargs):
    dataset.transform = transform
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       **kwargs)


def gen_test_loader(dataset, batch_size=args.test_batch_size, **kwargs):
    dataset.transform = transform
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       **kwargs)


train_set = datasets.CIFAR10('../data', train=True, download=True)
train_loader = gen_train_loader(train_set, **kwargs)

test_set = datasets.CIFAR10('../data', train=False)
test_loader = gen_test_loader(test_set, **kwargs)

random.seed(1)

shuffled = random.sample(list(range(len(test_set))), len(test_set))
test_set_small = Subset(test_set,
                        shuffled[:int(len(test_set) * subsample_test)])
test_loader_small = gen_test_loader(test_set_small,
                                    batch_size=500)

activations = {}
input_rdms = {}
intrinsic_dims = {}
outer_prod_triu_arrays = {}

model_name = args.model_name

trackingflag = repr_utils.TrackingFlag(True, model_name, None, None)

model.training = False
trackingflag.epoch = 0
trackingflag.active = True
trackingflag.data_name = 'test'

activations_, handles = repr_utils.track_activations(modules, trackingflag)

if isinstance(model, LocalLossNet) or isinstance(model, LocalLossBlock):
    model.local_loss_eval()

model.eval()

for images, labels in test_loader_small:

    _ = model(images)

for handle in handles:
    handle.remove()

activations.update(activations_)
pline = "\n" + "-"*60 + "\n"
print(f"{pline}Start Logging Similarity Metrics.{pline}")

activations = repr_utils.flatten_activations(activations)

print(f"Input RDMS{pline}")
input_rdms.update(rsa.input_rdms_from_activations(activations))
logs.log_similarity_metric(input_rdms, "input_rdms", model_name)
input_rdms = {}

print(f"{pline}Intrinsic Dim{pline}")
intrinsic_dims.update(repr_utils.compute_from_activations(activations, intrinsic_dimension.computeID))
logs.log_similarity_metric(intrinsic_dims, "intrinsic_dims", model_name)
intrinsic_dims = {}

print(f"CKA Outer products{pline}")
outer_prod_triu_arrays.update(cka.outer_product_triu_array_from_activations(activations))
logs.log_similarity_metric(outer_prod_triu_arrays, "outer_prod_triu_arrays", model_name)
outer_prod_triu_arrays = {}

activations = {}

print(f"{pline}End Logging Similarity Metrics.{pline}")

