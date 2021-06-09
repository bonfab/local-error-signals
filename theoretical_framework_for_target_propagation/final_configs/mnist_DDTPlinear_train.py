config = {
'lr': 5.445880112825726e-05,
'target_stepsize': 0.019894350490609045,
'beta1': 0.99,
'beta2': 0.9,
'epsilon': 1.1409034963508697e-08,
'lr_fb': 0.00046435267019415523,
'sigma': 0.011955442965055574,
'feedback_wd': 0.00044356554874093793,
'beta1_fb': 0.999,
'beta2_fb': 0.9,
'epsilon_fb': 7.73438668243311e-05,
'out_dir': 'logs/mnist/DMLPDTP2_linear',
'network_type': 'DMLPDTP2',
'recurrent_input': False,
'hidden_fb_activation': 'linear',
'size_mlp_fb': None,
'fb_activation': 'linear',
'initialization': 'xavier_normal',
'dataset': 'mnist',
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.0,
'parallel': True,
'normalize_lr': True,
'batch_size': 128,
'forward_wd': 0.0,
'epochs_fb': 6,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'extra_fb_epochs': 1,
'epochs': 100,
'only_train_first_layer': False,
'train_only_feedback_parameters': False,
'no_val_set': True,
'num_hidden': 5,
'size_hidden': 256,
'size_input': 784,
'size_output': 10,
'hidden_activation': 'tanh',
'output_activation': 'softmax',
'no_bias': False,
'no_cuda': False,
'random_seed': 42,
'cuda_deterministic': False,
'freeze_BPlayers': False,
'multiple_hpsearch': False,
'double_precision': True,
'save_logs': False,
'save_BP_angle': False,
'save_GN_angle': False,
'save_GN_activations_angle': False,
'save_BP_activations_angle': False,
'gn_damping': 0.0,
'hpsearch': False,
'log_interval': 80,
}
