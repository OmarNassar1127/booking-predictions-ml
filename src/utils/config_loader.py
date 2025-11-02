"""Configuration loader utility"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default=None) -> Any:
        """Get configuration value using dot notation (e.g., 'model.type')"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)

    @property
    def all(self) -> Dict:
        """Get entire configuration"""
        return self._config


# Global config instance
config = Config()
