import logging


def get_logger(name: str):
    logging.basicConfig(
        format="%(asctime)s %(levelname)7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel("DEBUG")
    return logger
