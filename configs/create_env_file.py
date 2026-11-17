import os
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Must match ROOT_PATH_KEYS in utils/parameter_handling.py, which cannot be imported from here.
ROOT_PATH_KEYS = ("storage_dir", "results_dir")


def absolutize_roots(params):
    for key in ROOT_PATH_KEYS:
        if key in params and isinstance(params[key], str):
            params[key] = os.path.abspath(os.path.join(PROJECT_ROOT, params[key]))


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    if not os.path.exists("configs"):
        raise FileNotFoundError("configs directory not found. Please run this script from the root of the project.")

    yaml_files = [f for f in os.listdir("configs") if f.endswith(".yaml")]
    if not yaml_files:
        raise FileNotFoundError("No YAML files found in configs directory.")

    all_params = {}
    for yaml_file in yaml_files:
        with open(os.path.join("configs", yaml_file), 'r') as f:
            data = yaml.safe_load(f)
            all_params.update(data)
    fail_keys = []
    for key in all_params:
        if all_params[key] == "PLACEHOLDER":
            fail_keys.append(key)
        if key == "env_dir" and not os.path.exists(all_params[key]+ "/bin"):
            raise ValueError(f"Config variable {key} must be set to a virtual environment directory (but could not find {all_params[key]}/bin. Check that the path is correct.")
    if fail_keys:
        raise ValueError(f"The following keys have the value 'PLACEHOLDER'. Please set them in the config files before creating the env file: {fail_keys}")

    flat_data = flatten_dict(all_params)
    absolutize_roots(flat_data)
    env_file = os.path.join("configs", "config.env")

    with open(env_file, 'w') as f:
        for key, value in flat_data.items():
            if isinstance(value, str):
                val_str = value.replace('"', '\\"')
                f.write(f'export {key}="{val_str}"\n')
            else:
                f.write(f'export {key}={value}\n')

    print(f"✅ Wrote {len(flat_data)} variables to {env_file}")
    print("Printing the env file contents below, check that everything looks correct:")
    with open(env_file, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    main()
