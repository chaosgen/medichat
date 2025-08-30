import yaml
import os
from pathlib import Path

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Configuration section '{name}' not found")
    
    def get_path(self, *path_keys):
        """Get an absolute path from the config, joining with the project root"""
        root = Path(__file__).parent.parent
        path = self._config
        for key in path_keys:
            path = path[key]
        return str(root / path)

# Create a global config instance
config = Config()
