from src.models.VGG import VGGn


def get_model(cfg):
    if cfg.name.startswith('vgg'):
        return VGGn(cfg.loss, cfg.name, cfg.input_dim, cfg.num_layers, cfg.num_hidden, cfg.num_classes, cfg.dropout, 1)

def count_parameters(model):
    ''' Count number of parameters in model influenced by global loss. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)