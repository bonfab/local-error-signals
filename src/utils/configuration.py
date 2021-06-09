import yaml


def adjust_cfg(cfg):
    cfg.model['input_dim'] = cfg.dataset['input_dim']
    cfg.model['num_classes'] = cfg.dataset['num_classes']
    cfg.model.loss['optim'] = cfg.train['optim']
    cfg.model.loss['weight_decay'] = cfg.train['weight_decay']
    cfg.model.loss['gpus'] = cfg.train['gpus']


def load_experiment_cfg(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)