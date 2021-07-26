import logging
import os


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    """
    logger.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    ch.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(ch)"""

    return logger


def get_csv_logger(name="csv", file_path='./results.csv', header='epoch,'
                                                                 'train_loss_local,train_loss_global,train_acc,'
                                                                 'valid_loss_local,valid_loss_global,valid_acc'):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info(header)

    return logger


def str_to_logging_level(str):
    if str.__contains__('debug') or str.__contains__('DEBUG'):
        return logging.DEBUG
    if str.__contains__('info') or str.__contains__('INFO'):
        return logging.INFO
    if str.__contains__('warning') or str.__contains__('WARNING'):
        return logging.WARNING
    if str.__contains__('error') or str.__contains__('ERROR'):
        return logging.ERROR
    if str.__contains__('critical') or str.__contains__('CRITICAL'):
        return logging.critical()

    raise ValueError(f'logging level {str} not known')


def retire_logger(logger):
    for handler in logger.handlers:
        try:
            handler.flush()
            handler.close()
        except AttributeError:
            pass
    logger.handlers.clear()
    del logger


def get_unique_save_path(path):

    split = path.split(".")
    if len(split) < 1:
        og_path = path
        og_end = ""
    else:
        og_path = ".".join(split[:-1])
        og_end = f".{split[-1]}"
    i = 1

    while os.path.exists(path):
        path = og_path + f"-{i}" + og_end
        i += 1

    return path
