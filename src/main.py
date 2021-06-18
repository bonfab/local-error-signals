import hydra
from omegaconf import OmegaConf

from utils.eval import make_acc_plots
from train import Trainer
from utils.logging import get_logger, str_to_logging_level, shutdown_logging
from utils.data import get_datasets
from utils.models import get_model
from utils.configuration import adjust_cfg, set_seed


@hydra.main(config_path="../configs", config_name="config.yaml")
def main(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    set_seed(cfg.train.seed)
    logger = get_logger(__name__, level=str_to_logging_level(cfg.logging_level))
    logger.info(OmegaConf.to_yaml(cfg).__str__())
    adjust_cfg(cfg)
    train_set, test_set = get_datasets(cfg.dataset, cfg.data_dir, logger)
    model = get_model(cfg.model, logger)
    trainer = Trainer(cfg.train, model, train_set, test_set, logger)
    trainer.fit()
    make_acc_plots("./training_results.csv")
    shutdown_logging(logger)

if __name__ == "__main__":
    main()
