import logging

import hydra
from omegaconf import OmegaConf

from utils.eval import make_acc_plots
from train import Trainer
from utils.logging import get_logger, str_to_logging_level, retire_logger
from utils.data import get_datasets
from utils.models import get_model, load_best_model_from_exp_dir
from utils.configuration import adjust_cfg, set_seed
from evaluate_dimensions import Evaluation


@hydra.main(config_path="../configs/training", config_name="config.yaml")
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
    model = load_best_model_from_exp_dir("./")
    agent = Evaluation(cfg.evaluation, model=model, data_set=train_set)
    agent.evaluate()
    retire_logger(logger)
    logging.shutdown()


if __name__ == "__main__":
    main()
