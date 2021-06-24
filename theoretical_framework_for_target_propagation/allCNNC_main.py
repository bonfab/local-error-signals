# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:25 2021

@author: rapha
"""
import json
import random
import numpy as np
import os
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image


from lib.conv_networks_AllCNN import DDTPConvAllCNNC, DDTPPureConvAllCNNC
from lib.conv_network import DDTPConvNetwork
from lib.train import train
from lib import utils
from lib import builders
import os.path
import pickle

    
#parameters:
args =  {'dataset': 'cifar10',
     'num_train': 1000,
     'num_test': 1000,
     'num_val': 1000,
     'epochs': 10,
     'batch_size': 128,
     'lr': '.1',
     'lr_fb': '.1', # learning rate for feedback parameters
     'target_stepsize': 0.01,
     'optimizer': 'Adam',
     'optimizer_fb': 'Adam',
     'momentum': 0.0,
     'sigma': 0.08,
     'forward_wd': 0.0,
     'feedback_wd': 0.0,
     'train_separate': False,
     'parallel': True,
     'normalize_lr': True,
     'not_randomized': True,
     'train_randomized': False,
     'normalize_lr': False,
     'train_only_feedback_parameters': False,
     'epochs_fb': 10,
     'soft_target': 0.9,
     'freeze_forward_weights': False,
     'freeze_fb_weights': False,
     'shallow_training': False,
     'norm_ratio': 1.0,
     'extra_fb_epochs': 1,
     'extra_fb_minibatches': 0,
     'freeze_output_layer': False,
     'gn_damping_training': 0.0,
     'not_randomized_fb': False,
     'train_randomized_fb': False,
     'only_train_first_layer': False,
     'no_val_set': False,
     'no_preprocessing_mnist': False,
     'loss_scale': 1.0,
     'only_train_last_two_layers': False,
     'only_train_last_three_layers': False,
     'only_train_last_four_layers': False,
     # from here 
     'beta1': 0.99,
     'beta2': 0.99,
     'epsilon': '1e-4',
     'beta1_fb': 0.99,
     'beta2_fb': 0.99,
     'epsilon_fb': '1e-4',
     'hidden_layers': None,
     'num_hidden': 2,
     'size_hidden': '500',
     'size_input': 784,
     'size_output': 10,
     'size_hidden_fb': 500,
     'hidden_activation': 'tanh',
     'output_activation': "softmax",
     'fb_activation': "linear",
     'no_bias': False,
     'network_type': 'DDTPConv',
     'initialization': 'xavier_normal',
     'size_mlp_fb': '100',
     'hidden_fb_activation': None,
     'recurrent_input': False,
     'no_cuda': False,
     'random_seed': 42,
     'cuda_deterministic': False,
     'freeze_BPlayers': False,
     'hpsearch': False,
     'multiple_hpsearch': False,
     'double_precision': False,
     'evaluate': True,
     'out_dir': 'logs/XXX5',
     'save_logs': False,
     'save_BP_angle': False,
     'save_GN_angle': False,
     'save_GNT_angle': False,
     'save_GN_activations_angle': False,
     'save_BP_activations_angle': False,
     'plots': 'save',
     'save_loss_plot': True,
     'create_plots': True,
     'gn_damping': '0.',
     'log_interval': 100,
     'output_space_plot': False,
     'output_space_plot_layer_idx': None,
     'output_space_plot_bp': False,
     'save_weights': True,
     'load_weights': False,
     'gn_damping_hpsearch': False,
     'save_nullspace_norm_ratio': False
     }

def load_network_w_weights(args, run_dir = "results/acnnc_1000_weights"):
    
    # function to load the AllCNNC Network according to the definition and load presaved weights
    from torchsummary import summary
    if type(args) != argparse.Namespace:
        args = argparse.Namespace(**args)
    forward_requires_grad = args.save_BP_angle or args.save_GN_angle or\
                            args.save_GN_activations_angle or \
                            args.save_BP_activations_angle or \
                            args.save_GNT_angle or \
                            args.network_type in ['GN', 'GN2'] or \
                            args.output_space_plot_bp or \
                            args.gn_damping_hpsearch or \
                            args.save_nullspace_norm_ratio
    net = DDTPPureConvAllCNNC(bias=not args.no_bias,
                                        hidden_activation=args.hidden_activation,
                                        feedback_activation=args.fb_activation,
                                        initialization=args.initialization,
                                        sigma=args.sigma,
                                        plots=args.plots,
                                        forward_requires_grad=forward_requires_grad)

    
    filename = os.path.normpath(os.path.join(run_dir, 'weights.pickle'))
    forward_parameters_loaded = pickle.load( open(filename, 'rb'))
    if len(net.layers) != len(forward_parameters_loaded)/2:
        print("the number of weights does not fit")
        return 
    for i in range(len(net.layers)):
        net.layers[i]._weights = forward_parameters_loaded[i*2]
        net.layers[i]._bias = forward_parameters_loaded[(i*2) + 1]
    print(summary(net.cuda(), (3,32,32)))
    return net


if __name__ == "__main__":
    args = argparse.Namespace(**args)

    batch_size = 128
        
    # code adapted from lib.builders build_network function

    forward_requires_grad = args.save_BP_angle or args.save_GN_angle or\
                            args.save_GN_activations_angle or \
                            args.save_BP_activations_angle or \
                            args.save_GNT_angle or \
                            args.network_type in ['GN', 'GN2'] or \
                            args.output_space_plot_bp or \
                            args.gn_damping_hpsearch or \
                            args.save_nullspace_norm_ratio

    allCNNC = DDTPConvAllCNNC(bias=not args.no_bias,
                                           hidden_activation=args.hidden_activation,
                                           feedback_activation=args.fb_activation,
                                           initialization=args.initialization,
                                           sigma=args.sigma,
                                           plots=args.plots,
                                           forward_requires_grad=forward_requires_grad)
                                           
    
    curdir = os.path.curdir
    out_dir = os.path.join(curdir, "log")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    if args.out_dir is None:
        out_dir = os.path.join(curdir, 'logs', )
        args.out_dir = out_dir
    else:
        out_dir = os.path.join(curdir, args.out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    print("Logging at {}".format(out_dir))
    
    with open(os.path.join(out_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    if args.dataset in ['mnist', 'fashion_mnist', 'cifar10']:
        args.classification = True
    else:
        args.classification = False
    
    if args.dataset in ['student_teacher', 'boston']:
        args.regression = True
    else:
        args.regression = False
    
    
    # initializing command line arguments if None
    if args.output_activation is None:
        if args.classification:
            args.output_activation = 'softmax'
        elif args.regression:
            args.output_activation = 'linear'
        else:
            raise ValueError('Dataset {} is not supported.'.format(
                args.dataset))
    
    if args.fb_activation is None:
        args.fb_activation = args.hidden_activation
    
    if args.hidden_fb_activation is None:
        args.hidden_fb_activation = args.hidden_activation
    
    if args.optimizer_fb is None:
        args.optimizer_fb = args.optimizer
    
    # Manipulating command line arguments if asked
    args.lr = utils.process_lr(args.lr)
    args.lr_fb = utils.process_lr(args.lr_fb)
    args.epsilon_fb = utils.process_lr(args.epsilon_fb)
    args.epsilon = utils.process_lr(args.epsilon)
    args.size_hidden = utils.process_hdim(args.size_hidden)
    if args.size_mlp_fb == 'None':
        args.size_mlp_fb = None
    else:
        args.size_mlp_fb = utils.process_hdim_fb(args.size_mlp_fb)
    
    if args.normalize_lr:
        args.lr = args.lr/args.target_stepsize
    
    if args.network_type in ['GN', 'GN2']:
        # if the GN variant of the network is used, the fb weights do not need
        # to be trained
        args.freeze_fb_weights = True
    
    if args.network_type == 'DFA':
        # manipulate cmd arguments such that we use a DMLPDTP2 network with
        # linear MLP's with fixed weights
        args.freeze_fb_weights = True
        args.network_type = 'DMLPDTP2'
        args.size_mlp_fb = None
        args.fb_activation = 'linear'
        args.train_randomized = False
    
    if args.network_type == 'DFAConv':
        args.freeze_fb_weights = True
        args.network_type = 'DDTPConv'
        args.fb_activation = 'linear'
        args.train_randomized = False
    
    
    if args.network_type == 'DFAConvCIFAR':
        args.freeze_fb_weights = True
        args.network_type = 'DDTPConvCIFAR'
        args.fb_activation = 'linear'
        args.train_randomized = False
    
    if args.network_type in ['DTPDR']:
        args.diff_rec_loss = True
    else:
        args.diff_rec_loss = False
    
    if args.network_type in ['DKDTP', 'DKDTP2', 'DMLPDTP', 'DMLPDTP2',
                             'DDTPControl', 'DDTPConv',
                             'DDTPConvCIFAR',
                             'DDTPConvControlCIFAR']:
        args.direct_fb = True
    else:
        args.direct_fb = False
    
    if ',' in args.gn_damping:
        args.gn_damping = utils.str_to_list(args.gn_damping, type='float')
    else:
        args.gn_damping = float(args.gn_damping)
    
    
    # Checking valid combinations of command line arguments
    if args.shallow_training:
        if not args.network_type == 'BP':
            raise ValueError('The shallow_training method is only implemented'
                             'in combination with BP. Make sure to set '
                             'the network_type argument on BP.')
    
    
    ### Ensure deterministic computation.
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Ensure that runs are reproducible even on GPU. Note, this slows down
    # training!
    # https://pytorch.org/docs/stable/notes/randomness.html
    if args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('Using cuda: ' + str(use_cuda))

    if args.double_precision:
        torch.set_default_dtype(torch.float64)
    
    if args.dataset == 'cifar10':
        print('### Training on CIFAR10')
        if args.multiple_hpsearch:
            data_dir = '../../../../../data'
        elif args.hpsearch:
            data_dir = '../../../../data'
        else:
            data_dir = './data'

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset_total = torchvision.datasets.CIFAR10(root=data_dir,
                                                      train=True,
                                                    download=True,
                                                    transform=transform)
        if args.no_val_set:
            train_loader = torch.utils.data.DataLoader(trainset_total,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=0)
            val_loader = None
        else:
            g_cuda = torch.Generator(device='cuda')
            trainset, valset = torch.utils.data.random_split(trainset_total,
                                                             [45000, 5000], generator = g_cuda)
            train_loader = torch.utils.data.DataLoader(trainset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True, num_workers=0)
            val_loader = torch.utils.data.DataLoader(valset,
                                                     batch_size=args.batch_size,
                                                     shuffle=False, num_workers=0)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                               download=True,
                                               transform=transform)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False, num_workers=0)

    summary = utils.setup_summary_dict(args)
    
    
    if use_cuda:
        net = allCNNC.cuda()
    else:
        net = allCNNC
    
    # adapted from lib.train train function
    summary = train(args=args,
                        device=device,
                        train_loader=train_loader,
                        net=net,
                        writer=None,
                        test_loader=val_loader,
                        summary=summary,
                        val_loader=val_loader)
    
    if (args.plots is not None and args.network_type != 'BP'):
        summary['bp_activation_angles'] = net.bp_activation_angles
        summary['gn_activation_angles'] = net.gn_activation_angles
        summary['bp_angles'] = net.bp_angles
        summary['gnt_angles'] = net.gnt_angles
        summary['nullspace_relative_norm_angles'] = net.nullspace_relative_norm


    # write final summary
    if summary['finished'] == 0:
        # if no error code in finished, put it on 1 to indicate succesful run
        summary['finished'] = 1
        utils.save_summary_dict(args, summary)

    if args.save_loss_plot:
        utils.plot_loss(summary, logdir=args.out_dir, logplot=True)

    # dump the whole summary in a pickle file
    filename = os.path.join(args.out_dir, 'results.pickle')
    with open(filename, 'wb') as f:
        pickle.dump(summary, f)

