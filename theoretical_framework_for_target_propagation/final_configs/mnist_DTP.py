config = {
'lr': 6.185037324617764e-05,
'target_stepsize': 0.21951187497378802,
'beta1': 0.9,
'beta2': 0.999,
'epsilon': 1.0170111936290766e-08,
'lr_fb': 0.0019584756448113774,
'sigma': 0.08372453893037193,
'beta1_fb': 0.9,
'beta2_fb': 0.999,
'epsilon_fb': 7.541132645747324e-06,
'out_dir': 'logs/mnist/DTP',
'network_type': 'DTP',
'initialization': 'xavier_normal',
'fb_activation': 'tanh',
'dataset': 'mnist',
'double_precision': True,
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.0,
'parallel': True,
'normalize_lr': True,
'batch_size': 128,
'forward_wd': 0.0,
'epochs_fb': 0,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'extra_fb_epochs': 0,
'epochs': 100,
'only_train_first_layer': False,
'train_only_feedback_parameters': False,
'num_hidden': 5,
'size_hidden': 256,
'size_input': 784,
'size_output': 10,
'hidden_activation': 'tanh',
'output_activation': 'softmax',
'fb_activation': 'tanh',
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
'log_interval': 50,
}