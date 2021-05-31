import hydra
from omegaconf import OmegaConf

from src.train import Trainer
from utils.data import get_datasets
from utils.models import get_model


@hydra.main(config_path= "../configs", config_name="config.yaml")
def main(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    train_set, test_set = get_datasets(cfg.data, cfg.data_dir)
    cfg.model['input_dim'] = cfg.data['input_dim']
    cfg.model['num_classes'] = cfg.data['num_classes']
    cfg.model.loss['optim'] = cfg.train['optim']
    cfg.model.loss['weight_decay'] = cfg.train['weight_decay']
    model = get_model(cfg.model)
    print(model)
    trainer = Trainer(cfg.train, model, train_set, test_set)
    trainer.fit()




if __name__ == "__main__":
    main()