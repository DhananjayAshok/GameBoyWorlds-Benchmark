from typing import Any, Optional
from utils.parameter_handling import load_parameters


def log_error(message: str, parameters: Optional[dict[str, Any]] = None) -> None:
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    logger.error(message, stacklevel=2)
    raise RuntimeError(f"{message}")


def log_warn(message: str, parameters: Optional[dict[str, Any]] = None) -> None:
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    logger.warning(message, stacklevel=2)


def log_info(message: str, parameters: Optional[dict[str, Any]] = None) -> None:
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    logger.info(message, stacklevel=2)
