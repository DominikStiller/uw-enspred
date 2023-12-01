import logging


def get_logger(name: str):
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    logger = logging.getLogger(name)
    logger.setLevel("DEBUG")
    return logger
