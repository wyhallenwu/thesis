import yaml
import time


def get_config():
    with open(".../config.yaml", "r") as f:
        configs = yaml.safe_load(f)
    return configs


def get_mill_sec_timestamp():
    return (int(time.time() * 1000))
