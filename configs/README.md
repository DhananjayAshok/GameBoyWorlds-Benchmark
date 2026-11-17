## General Philosophy

You should put all private variables in the private_vars yaml (currently storage_dir is mandatory), the remaining can go into any .yaml file in this directory and will be read as parameters. Once you've defined the parameters, run the following command to create a config.env file in the configs folder. 

#### Important: All paths MUST be global paths

```bash
python configs/create_env_file.py
```

You can source this file to use config vars in bash scripts. 

## Splits (`configs/splits/`)

`split_config.yaml` sets the seed, the training games and the `val`/`test` fractions. `benchmark_splits.json` is generated from it and is what the environment server reads; it is checked in so everyone evaluates on the same held-out tasks. Regenerate it with:

```bash
python run_env_server.py generate_splits --show
```

The server stops with an error if the file no longer matches the benchmark's task list.
