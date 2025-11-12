"""Configuration management for the N-Queens experiment suite.

This module provides a thin, explicit wrapper around a JSON configuration file
to centralize experiment settings, timeouts, GA tuning grids, and persisted
optimal GA parameters per fitness function.

File format (high-level)
------------------------
- experiment_settings: default N values, run counts, and output directory.
- timeout_settings: per-algorithm time limits and global experiment timeout.
- tuning_grid: GA tuning search space for population/gen multipliers and rates.
- fitness_modes: list of fitness labels to consider (e.g., ["F1", ..., "F6"]).
- optimal_parameters: mapping fitness_mode -> { N: {params...} }

All methods return Python native types; the class does not validate semantics
beyond presence of keys to keep responsibilities minimal.
"""
import json
import os
from pathlib import Path


class ConfigManager:
    """Load, query, and persist configuration and optimal parameters.

    Parameters
    ----------
    config_path : str | os.PathLike, default "config.json"
        Path to the configuration file.
    """
    
    def __init__(self, config_path="config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self):
        """Load and parse the JSON configuration file.

        Returns
        -------
        dict
            Root configuration object.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Create it or use the default config.json template"
            )
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def save_config(self):
        """Persist the current in-memory configuration to disk."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_experiment_settings(self):
        """Return high-level experiment settings (sizes, runs, output dir)."""
        return self.config.get("experiment_settings", {})
    
    def get_timeout_settings(self):
        """Return per-algorithm and global timeout settings."""
        return self.config.get("timeout_settings", {})
    
    def get_tuning_grid(self):
        """Return GA tuning grid settings (population/gen multipliers, rates)."""
        return self.config.get("tuning_grid", {})
    
    def get_fitness_modes(self):
        """Return the list of fitness function labels to test (e.g., F1..F6)."""
        return self.config.get("fitness_modes", ["F1", "F2", "F3", "F4", "F5", "F6"])
    
    def get_optimal_parameters(self, fitness_mode=None):
        """Return stored optimal GA parameters.

        Parameters
        ----------
        fitness_mode : str | None
            If provided, return parameters for the specific fitness label;
            otherwise return the entire mapping.

        Returns
        -------
        dict
            Mapping of fitness -> per-N parameter dicts, or a single per-N dict
            when ``fitness_mode`` is specified.
        """
        optimal = self.config.get("optimal_parameters", {})
        if fitness_mode:
            return optimal.get(fitness_mode, {})
        return optimal
    
    def save_optimal_parameters(self, fitness_mode, parameters):
        """Persist optimal GA parameters for a specific fitness function.

        Parameters
        ----------
        fitness_mode : str
            Fitness label (e.g., "F1").
        parameters : dict
            Mapping ``{N: {params}}`` produced by tuning.
        """
        if "optimal_parameters" not in self.config:
            self.config["optimal_parameters"] = {}

        self.config["optimal_parameters"][fitness_mode] = parameters
        self.save_config()
        print(f"Optimal parameters for {fitness_mode} saved to {self.config_path}")
    
    def has_optimal_parameters(self, fitness_mode):
        """Return True if optimal parameters exist for ``fitness_mode``."""
        optimal = self.config.get("optimal_parameters", {})
        return fitness_mode in optimal and bool(optimal[fitness_mode])
    
    def update_setting(self, section, key, value):
        """Update a specific setting and persist the change immediately."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
