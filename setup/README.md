# Environment

This project uses Python 3.12 with [uv](https://docs.astral.sh/uv/guides/projects/). The uv project lives in this directory, not the repository root, so the virtual environment is always at `setup/.venv`.

The quick way, from the repository root:

```console
bash setup/create_env.sh --env_dir /path/to/env
```

## By hand

```console
pip install --upgrade pip uv
```

To keep the environment on another filesystem, create it there and symlink it. Set `UV_CACHE_DIR` to the same filesystem as the environment.

```console
uv venv /path/to/venv --python=3.12
ln -s /path/to/venv setup/.venv
```

Then install and activate:

```console
cd setup
uv sync
cd ..
source setup/.venv/bin/activate
python configs/create_env_file.py
python GameBoyWorlds/configs/create_env_file.py
```

## Adding a dependency

```console
cd setup
uv add <package>
```

Commit the updated `pyproject.toml` and `uv.lock`.
