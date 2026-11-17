# Nothing here may import from elsewhere in the project or from a third-party package.
import os
import logging
import re


def get_logger(
    level: int = logging.INFO, filename: str = None, add_console: bool = True
) -> logging.Logger:
    fmt_str = "%(asctime)s, [%(levelname)s, %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt_str)
    logger = logging.getLogger("GameBoyWorlds-Client")
    if add_console:
        logger.handlers.clear()
        console_handler = logging.StreamHandler()
        log_formatter = logging.Formatter(fmt_str)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
    if filename is not None:
        file_handler = logging.FileHandler(filename, mode="a")
        log_formatter = logging.Formatter(fmt_str)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    if level is not None:
        logger.setLevel(level)
        logger.propagate = False
    return logger


def depathify(string: str) -> str:
    return re.sub(r"[^\w]", "_", string).strip("_")


def file_makedir(file_path: str) -> None:
    dirname = os.path.dirname(file_path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)
    return
