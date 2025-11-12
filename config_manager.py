"""
Configuration Manager for N-Queens Experiment Suite
Handles loading/saving configuration and optimal parameters from JSON
"""
import json
import os
from pathlib import Path


class ConfigManager:
    """Manages configuration and optimal parameters persistence"""
    
    def __init__(self, config_path="config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Create it or use the default config.json template"
            )
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def save_config(self):
        """Save current configuration to JSON file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_experiment_settings(self):
        """Get experiment configuration"""
        return self.config.get("experiment_settings", {})
    
    def get_timeout_settings(self):
        """Get timeout configuration"""
        return self.config.get("timeout_settings", {})
    
    def get_tuning_grid(self):
        """Get tuning grid parameters"""
        return self.config.get("tuning_grid", {})
    
    def get_fitness_modes(self):
        """Get list of fitness functions to test"""
        return self.config.get("fitness_modes", ["F1", "F2", "F3", "F4", "F5", "F6"])
    
    def get_optimal_parameters(self, fitness_mode=None):
        """
        Get optimal parameters for a specific fitness or all
        
        Args:
            fitness_mode: Optional fitness mode (e.g., "F1")
        
        Returns:
            dict: optimal parameters
        """
        optimal = self.config.get("optimal_parameters", {})
        if fitness_mode:
            return optimal.get(fitness_mode, {})
        return optimal
    
    def save_optimal_parameters(self, fitness_mode, parameters):
        """
        Save optimal parameters for a fitness function after tuning
        
        Args:
            fitness_mode: Fitness mode (e.g., "F1")
            parameters: Dict of optimal parameters {N: {params}}
        """
        if "optimal_parameters" not in self.config:
            self.config["optimal_parameters"] = {}

        self.config["optimal_parameters"][fitness_mode] = parameters
        self.save_config()
        print(f"Optimal parameters for {fitness_mode} saved to {self.config_path}")
    
    def has_optimal_parameters(self, fitness_mode):
        """Check if optimal parameters exist for a fitness mode"""
        optimal = self.config.get("optimal_parameters", {})
        return fitness_mode in optimal and bool(optimal[fitness_mode])
    
    def update_setting(self, section, key, value):
        """Update a specific setting"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
