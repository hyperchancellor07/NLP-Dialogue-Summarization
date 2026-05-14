import os
import yaml
import pickle
from box import ConfigBox
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads YAML file and returns ConfigBox for dot-access.
    """
    with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
        content = yaml.safe_load(yaml_file)
    return ConfigBox(content)

def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Create list of directories.
    """
    for path in path_to_directories:
        os.makedirs(str(path), exist_ok=True)
        if verbose:
            print(f"📁 Directory created at: {path}")

def save_pickle(file_path: Path, data: Any):
    """
    Save data as pickle file.
    """
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

def load_pickle(file_path: Path) -> Any:
    """
    Load pickle file.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)
