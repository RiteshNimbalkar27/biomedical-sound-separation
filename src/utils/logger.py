import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger compatible with both local and Colab environments.
    Avoids fileno() call which raises UnsupportedOperation in Colab.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.propagate = False

    return logger