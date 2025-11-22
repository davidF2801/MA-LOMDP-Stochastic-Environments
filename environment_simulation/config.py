"""Centralized configuration for simulation parameters."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class SimulationConfig:
    """Centralized configuration for all simulation parameters."""
    
    # Grid parameters
    grid_height: int = 30
    grid_width: int = 60
    
    # Orbit parameters
    orbit_period: int = 120  # Number of steps in one complete orbit
    
    # Agent parameters
    field_of_regard_deg: float = 20.0  # Field of regard in degrees
    communication_range_deg: float = 5.0  # Communication range for ground stations in degrees
    
    # Monte Carlo planning parameters
    num_rollouts: int = 30  # Number of Monte Carlo rollouts per action
    planning_horizon: int = 5  # Planning horizon for Monte Carlo rollouts
    discount_factor: float = 0.95  # Discount factor for future rewards
    
    # Ground station parameters
    num_ground_stations: int = 4
    min_satellites_per_station: int = 2
    
    # Simulation parameters
    num_steps: int = 100  # Number of simulation steps
    num_agents: int = 5  # Number of agents
    
    # Performance parameters
    use_parallel_planning: bool = True  # Enable parallel agent planning (one thread per agent)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.grid_height > 0, "grid_height must be positive"
        assert self.grid_width > 0, "grid_width must be positive"
        assert self.orbit_period > 0, "orbit_period must be positive"
        assert self.field_of_regard_deg > 0, "field_of_regard_deg must be positive"
        assert self.communication_range_deg > 0, "communication_range_deg must be positive"
        assert self.num_rollouts > 0, "num_rollouts must be positive"
        assert self.planning_horizon > 0, "planning_horizon must be positive"
        assert 0 < self.discount_factor <= 1, "discount_factor must be in (0, 1]"
        assert self.num_ground_stations > 0, "num_ground_stations must be positive"
        assert self.min_satellites_per_station > 0, "min_satellites_per_station must be positive"
        assert self.num_steps > 0, "num_steps must be positive"
        assert self.num_agents > 0, "num_agents must be positive"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Create SimulationConfig from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_path: str | Path) -> "SimulationConfig":
        """Load SimulationConfig from JSON file."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if "simulation" in data:
            return cls.from_dict(data["simulation"])
        else:
            return cls.from_dict(data)


def load_agent_configurations(json_path: str | Path | None = None) -> List[Dict[str, Any]]:
    """
    Load agent configurations from JSON file.
    
    Args:
        json_path: Path to JSON configuration file. If None, uses default.json
        
    Returns:
        List of agent configuration dictionaries
    """
    if json_path is None:
        # Default to configs/default.json relative to this file
        config_dir = Path(__file__).parent / "configs"
        json_path = config_dir / "default.json"
    
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if "agents" in data:
        return data["agents"]
    else:
        raise ValueError("JSON file must contain 'agents' key with list of agent configurations")


# Default configuration instance
DEFAULT_CONFIG = SimulationConfig()

# Try to load from default.json if it exists
try:
    config_dir = Path(__file__).parent / "configs"
    default_json = config_dir / "default.json"
    if default_json.exists():
        DEFAULT_CONFIG = SimulationConfig.from_json(default_json)
except Exception:
    # If loading fails, use hardcoded defaults
    pass
