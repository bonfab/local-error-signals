config = {
'lr': 4.6406733979941715e-05,
'target_stepsize': 0.13200208071240313,
'feedback_wd': 2.461969739133049e-07,
'beta1': 0.9,
'beta2': 0.99,
'epsilon': 1.5387405670994764e-07,
'lr_fb': 5.561288685940823e-05,
'sigma': 0.26549822198764983,
'beta1_fb': 0.9,
'beta2_fb': 0.9,
'epsilon_fb': 3.422970341759682e-06,
'out_dir': 'logs/fashion_mnist/DTP',
'network_type': 'DTP',
'initialization': 'xavier_normal',
'fb_activation': 'tanh',
'dataset': 'cifar10',
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.0,
'parallel': True,
'normalize_lr': True,
'batch_size': 128,
'forward_wd': 0.0,
'epochs_fb': 10,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'extra_fb_epochs': 2,
'epochs': 100,
'double_precision': True,
'no_val_set': True,
'num_hidden': 3,
'size_hidden': 1024,
'size_input': 3072,
'size_output': 10,
'hidden_activation': 'tanh',
'output_activation': 'softmax',
'no_bias': False,
'no_cuda': False,
'random_seed': 42,
'cuda_deterministic': False,
'freeze_BPlayers': False,
'multiple_hpsearch': False,
'save_logs': False,
'save_BP_angle': False,
'save_GN_angle': False,
'save_GN_activations_angle': False,
'save_BP_activations_angle': False,
'gn_damping': 0.0,
'hpsearch': False,
'log_interval': 80,
}