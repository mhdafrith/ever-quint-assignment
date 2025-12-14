import os
import logging
import logging.config
try:
    import tomllib
except ImportError:
    import tomli as tomllib # for python < 3.11

def setup_logging(config_path="logging_config.toml"):
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

def load_env(env_path=".env"):
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
