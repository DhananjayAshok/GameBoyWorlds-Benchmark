import os
import yaml
from typing import Any, Optional
from utils.fundamental import get_logger


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOT_PATH_KEYS = ("storage_dir", "results_dir")


def absolutize_roots(params: dict[str, Any]) -> None:
    for key in ROOT_PATH_KEYS:
        if key in params and isinstance(params[key], str):
            params[key] = os.path.abspath(os.path.join(PROJECT_ROOT, params[key]))


def load_yaml(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def compute_secondary_parameters(params: dict[str, Any]) -> None:
    absolutize_roots(params)
    params["data_dir"] = os.path.join(params["storage_dir"], "data")
    params["model_dir"] = os.path.join(params["storage_dir"], "models")
    params["tmp_dir"] = os.path.join(params["storage_dir"], "tmp")
    params["sync_dir"] = os.path.join(params["storage_dir"], "sync")
    params["log_dir"] = os.path.join(params["results_dir"], "logs")
    params["figure_dir"] = os.path.join(params["results_dir"], "figures")
    for dirname in [
        "data_dir",
        "model_dir",
        "log_dir",
        "figure_dir",
        "tmp_dir",
        "sync_dir",
    ]:
        if not os.path.exists(params[dirname]):
            os.makedirs(params[dirname])
    if "log_file" not in params:
        log_file = os.path.join(params["log_dir"], "log.txt")
        params["log_file"] = log_file
    else:
        log_dir_str = params["log_dir"].replace("//", "/")
        log_file_str = params["log_file"].replace("//", "/")
        if not log_file_str.startswith(log_dir_str):
            log_file = os.path.join(params["log_dir"], params["log_file"])
            params["log_file"] = log_file
    logger = get_logger(filename=params["log_file"])
    params["logger"] = logger


def load_parameters(parameters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if parameters is not None:
        if (
            "logger" not in parameters
        ):  # a missing logger means secondary parameters are not computed yet
            compute_secondary_parameters(parameters)
        return parameters
    essential_keys = ["storage_dir", "results_dir"]
    project_root = PROJECT_ROOT
    params = {"project_root": project_root}
    logger = get_logger()
    config_files = os.listdir(os.path.join(project_root, "configs"))

    def error(msg):
        logger.error(msg)
        raise ValueError(msg)

    if "private_vars.yaml" not in config_files:
        error("Please create private_vars.yaml in the configs directory")
    for file in config_files:
        if file.endswith(".yaml"):
            configs = load_yaml(os.path.join(project_root, "configs", file))
            for key in configs:
                if key in params:
                    error(
                        f"{key} is present in multiple config files. At least one of which is {file}. Please remove the duplicate"
                    )
            params.update(configs)
        else:
            pass

    for key in params:
        if params[key] == "PLACEHOLDER":
            error(
                f"{key} is currently the placeholder value in private_vars.yaml. Please set it"
            )
    for essential_key in essential_keys:
        if essential_key not in params:
            error(f"Please set {essential_key} in one of the config yamls")
    # Must precede the makedirs below, or a relative root creates a stray tree under the cwd.
    absolutize_roots(params)
    if os.path.exists(params["storage_dir"]):
        if any([f.endswith(".py") for f in os.listdir(params["storage_dir"])]):
            logger.warning(
                f"There are .py files in the storage_dir {params['storage_dir']}. It is recommended to set a path which has nothing else inside it to avoid issues."
            )
    else:
        os.makedirs(params["storage_dir"])
        logger.info(f"Created storage directory {params['storage_dir']}")
    compute_secondary_parameters(params)
    return params
