config = {
'lr': 2.2591298843888373e-06,
'target_stepsize': 0.016184143061123493,
'feedback_wd': 1.3449401889831889e-07,
'beta1': 0.9,
'beta2': 0.999,
'epsilon': 7.219082567713133e-08,
'lr_fb': 0.0009097783536288601,
'sigma': 0.027412564678360344,
'beta1_fb': 0.99,
'beta2_fb': 0.9,
'epsilon_fb': 2.050381900135163e-05,
'out_dir': 'logs/cifar10/DKDTP2_nonrec',
'network_type': 'DKDTP2',
'recurrent_input': False,
'hidden_fb_activation': 'tanh',
'fb_activation': 'tanh',
'initialization': 'xavier_normal',
'size_hidden_fb': 2048,
'dataset': 'cifar10',
'double_precision': True,
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
'log_interval': 5,
# 'gn_damping_hpsearch': True,
}