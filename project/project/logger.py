import logging
from contextlib import contextmanager


def get_logger(name: str):
    logging.basicConfig(
        format="%(asctime)s %(levelname)7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel("DEBUG")
    return logger


@contextmanager
def logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages triggered during the body from being processed.

    Args:
        highest_level: the maximum logging level in use. This would only need to be changed if a custom level
                        greater than CRITICALis defined.
    """
    # From https://gist.github.com/simon-weber/7853144

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
