import logging


def get_logger(name: str):
    logging.basicConfig()
    logger = logging.getLogger(name)
    logger.setLevel("DEBUG")
    return logger
