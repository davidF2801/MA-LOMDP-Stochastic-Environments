"""Main file for running multiple Monte Carlo agents with visualization."""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set matplotlib backend to 'Agg' (non-interactive) to avoid tkinter issues
# This must be done before any matplotlib imports
import matplotlib
matplotlib.use('Agg')

import numpy as np

# Handle imports for both package and direct execution
try:
    if __package__:
        from .environment import Environment, RSPParams
        from .replay_environment import ReplayEnvironment
        from .agents.agent import Agent
        from .agents.greedy_agent import GreedyAgent
        from .agents.monte_carlo_agent import MonteCarloAgent
        from .agents.random_agent import RandomAgent
        from .agents.sharing_monte_carlo_agent import SharingMonteCarloAgent
        from .agents.dsb_abba_agent import DSBABBAAgent
        from .agents.satellite_attention_agent import SatelliteCentralizedAgent
        from .agents.ground_station import GroundStation
        from .agents.belief import Belief
        from .agents.trajectory import Trajectory
        from .utils import create_ground_stations_from_agent_configs, angular_distance_deg
        from .visualize_globe import GlobeFireAnimator
        from .config import SimulationConfig, DEFAULT_CONFIG, load_agent_configurations
    else:
        raise ImportError
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environment_simulation.environment import Environment, RSPParams
    from environment_simulation.replay_environment import ReplayEnvironment
    from environment_simulation.agents.agent import Agent
    from environment_simulation.agents.greedy_agent import GreedyAgent
    from environment_simulation.agents.monte_carlo_agent import MonteCarloAgent
    from environment_simulation.agents.random_agent import RandomAgent
    from environment_simulation.agents.sharing_monte_carlo_agent import SharingMonteCarloAgent
    from environment_simulation.agents.dsb_abba_agent import DSBABBAAgent
    from environment_simulation.agents.satellite_attention_agent import SatelliteCentralizedAgent
    from environment_simulation.agents.ground_station import GroundStation
    from environment_simulation.agents.belief import Belief
    from environment_simulation.agents.trajectory import Trajectory
    from environment_simulation.utils import create_ground_stations_from_agent_configs, angular_distance_deg
    from auxiliary_scripts.visualize_globe import GlobeFireAnimator
    from environment_simulation.config import SimulationConfig, DEFAULT_CONFIG, load_agent_configurations


def create_lat_lon_grids(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Create latitude and longitude grids."""
    lat_centers = 90.0 - (np.arange(height) + 0.5) * (180.0 / height)
    lon_centers = (np.arange(width) + 0.5) * (360.0 / width)
    lat_grid = np.repeat(lat_centers.reshape(-1, 1), width, axis=1)
    lon_grid = np.tile(lon_centers, (height, 1))
    return lat_grid, lon_grid


def _create_material_map_2d(
    height: int,
    width: int,
    blocked_mask: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict[tuple[int, int], "Material"]:
    """
    Create material assignments for 2D grid cells.
    
    Only uses WATER, GRASS, and WOOD (no GASOLINE).
    Blocked cells get WATER.
    
    Args:
        height: Grid height
        width: Grid width
        blocked_mask: Optional boolean (H,W) array indicating blocked cells
        rng: Random number generator
    
    Returns:
        Dictionary mapping (y, x) -> Material
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Import Material here to avoid circular imports and only when actually needed
    import sys
    import os
    
    # Mock window_option before importing Material to avoid favicon error
    FIRE_SIM_PATH = os.path.join(os.path.dirname(__file__), '..', 'fire-simulation')
    if FIRE_SIM_PATH not in sys.path:
        sys.path.insert(0, FIRE_SIM_PATH)
    
    # Mock src.window_option to avoid FileNotFoundError for favicon
    import types
    if 'src.window_option' not in sys.modules:
        window_option_mock = types.ModuleType('src.window_option')
        window_option_mock.CELL_WIDTH = 10
        window_option_mock.CELL_HEIGHT = 10
        sys.modules['src.window_option'] = window_option_mock
    
    from src.material import Material
    
    material_map = {}
    
    for y in range(height):
        for x in range(width):
            cell_key = (y, x)
            # Use blocked_mask to assign WATER (can't burn) to blocked cells
            if blocked_mask is not None and blocked_mask[y, x]:
                material_map[cell_key] = Material.WATER
            else:
                # Random material assignment: only WATER, GRASS, and WOOD
                rand = rng.random()
                if rand < 0.40:
                    material_map[cell_key] = Material.GRASS
                elif rand < 0.80:
                    material_map[cell_key] = Material.WOOD
                else:
                    material_map[cell_key] = Material.WATER
    
    return material_map


def create_environment(
    height: int,
    width: int,
    kernel_path: Optional[str] = None,
) -> Environment:
    """Create and configure the environment.
    
    Args:
        height: Grid height
        width: Grid width
        kernel_path: Optional path to learned kernel file. If provided, uses kernel mode.
    
    Returns:
        Environment instance
    """
    base_rng = np.random.default_rng(1)
    
    lat_grid, lon_grid = create_lat_lon_grids(height, width)
    
    # Create blocked mask (same for both RSP and kernel modes)
    blocked_mask = np.zeros((height, width), dtype=bool)
    blocked_mask[:, : width // 18] = True  # permanent ocean strip
    blocked_mask |= base_rng.random((height, width)) < 0.05
    
    # Load kernel if kernel_path provided
    kernel_learner = None
    material_map = None
    mode = "rsp"
    
    if kernel_path is not None:
        # Lazy import to avoid fire-simulation dependencies when not using kernel
        from environment_simulation.kernel_learning import TransitionKernelLearner
        kernel_learner = TransitionKernelLearner()
        kernel_learner.load_kernel(kernel_path)
        print(f"Loaded kernel from: {kernel_path}")
        mode = "kernel"
        
        # Create material map for kernel mode
        material_map = _create_material_map_2d(height, width, blocked_mask, base_rng)
        print(f"Created material map for {height}x{width} grid")
    
    # Create clustered ignition map (for RSP mode)
    lam_map = 1.5e-4 * np.ones((height, width))
    for center_lat, center_lon, amp, spread in [
        (-0.3, 1.2, 9e-4, 0.18),
        (0.55, 4.2, 8e-4, 0.22),
        (0.1, 2.7, 7e-4, 0.16),
    ]:
        dist = (lat_grid - center_lat) ** 2 + ((np.mod(lon_grid - center_lon + np.pi, 2 * np.pi) - np.pi)) ** 2
        lam_map += amp * np.exp(-dist / spread)
    lam_map += 1e-4 * base_rng.random((height, width))
    
    alpha_map = 8e-4 + 8e-4 * np.cos(lat_grid * np.pi) ** 4
    alpha_map += 4e-4 * base_rng.random((height, width))
    
    beta0_map = 3e-5 + 2e-5 * base_rng.random((height, width))
    
    persistence_map = 0.985 + 0.01 * np.cos(lat_grid * np.pi)
    persistence_map += 0.003 * base_rng.random((height, width))
    persistence_map = np.clip(persistence_map, 0.0, 0.999)
    
    # Create environment with appropriate mode
    env_kwargs = {
        "width": width,
        "height": height,
        "mode": mode,
        "topology": "sphere",
        "blocked_mask": blocked_mask,
        "seed": 7,
    }
    
    if mode == "kernel":
        env_kwargs["kernel_learner"] = kernel_learner
        env_kwargs["material_map"] = material_map
    else:
        env_kwargs["rsp_params"] = RSPParams(lam=0.02, beta0=0.001, alpha=0.03, delta=0.92)
        env_kwargs["ignition_map"] = lam_map
        env_kwargs["alpha_map"] = alpha_map
        env_kwargs["beta0_map"] = beta0_map
        env_kwargs["persistence_map"] = persistence_map
    
    env = Environment(**env_kwargs)
    
    env.reset(initial_events=8)
    return env


def create_greedy_agents(
    height: int, width: int, num_agents: int = 5, sim_config: SimulationConfig = DEFAULT_CONFIG,
    agent_configs_json: str | None = None, kernel_path: Optional[str] = None
) -> list[GreedyAgent]:
    """Create multiple greedy agents with different trajectories.
    
    Args:
        height: Grid height
        width: Grid width
        num_agents: Number of agents to create
        sim_config: Simulation configuration
        agent_configs_json: Path to JSON file with agent configurations. If None, uses default.json
    """
    agents = []
    
    # Load agent configurations from JSON
    try:
        agent_configs = load_agent_configurations(agent_configs_json)
    except (FileNotFoundError, ValueError):
        # Fallback to hardcoded configs if JSON loading fails
        agent_configs = [
            {
                "name": "Aurora-1",
                "inclination_deg": 25.0,
                "phase_deg": 0.0,
                "latitude_offset": 0.0,
                "color": "#ffd166",
            },
            {
                "name": "Borealis-2",
                "inclination_deg": 55.0,
                "phase_deg": 72.0,
                "latitude_offset": 5.0,
                "color": "#4ecdc4",
            },
            {
                "name": "Zenith-3",
                "inclination_deg": 10.0,
                "phase_deg": 144.0,
                "latitude_offset": -8.0,
                "color": "#ff6b6b",
            },
            {
                "name": "Polaris-4",
                "inclination_deg": 40.0,
                "phase_deg": 216.0,
                "latitude_offset": 3.0,
                "color": "#95e1d3",
            },
            {
                "name": "Vega-5",
                "inclination_deg": 30.0,
                "phase_deg": 288.0,
                "latitude_offset": -5.0,
                "color": "#f38181",
            },
        ]
    
    # Add simulation config values to each agent config
    for config in agent_configs:
        config["field_of_regard_deg"] = sim_config.field_of_regard_deg
    
    # Create lat/lon grids for pre-computing FOR locations
    lat_grid, lon_grid = create_lat_lon_grids(height, width)
    
    for i, config in enumerate(agent_configs[:num_agents]):
        trajectory = Trajectory.circular_orbit(
            period=sim_config.orbit_period,
            inclination_deg=config["inclination_deg"],
            phase_deg=config["phase_deg"],
            latitude_offset=config["latitude_offset"],
        )
        
        # Create belief with all FOR locations pre-initialized
        # Set num_states=3 for kernel mode, 2 for binary mode
        kernel_path_used = kernel_path or sim_config.kernel_path
        num_states = 3 if kernel_path_used else 2
        belief = Belief(height=height, width=width, prior_probability=0.5, num_states=num_states)
        
        # Create temporary agent to compute FOR locations
        temp_agent = GreedyAgent(
            name=config["name"],
            trajectory=trajectory,
            belief=belief,
            field_of_regard_deg=config["field_of_regard_deg"],
            color=config["color"],
        )
        
        # Pre-compute all locations that will be in FOR over trajectory period
        all_for_locations = temp_agent.compute_all_for_locations(lat_grid, lon_grid)
        
        # Initialize belief with all FOR locations at prior probability
        for lat, lon in all_for_locations:
            belief.set_probability(lat, lon, belief.prior_probability)
        
        # Determine event_utility based on mode
        event_utility_final = sim_config.event_utility.copy()
        if kernel_path_used and 2 not in event_utility_final:
            # Add state 2 utility for kernel mode if not present
            event_utility_final[2] = 0.5
        
        agent = GreedyAgent(
            name=config["name"],
            trajectory=trajectory,
            belief=belief,
            field_of_regard_deg=config["field_of_regard_deg"],
            color=config["color"],
            w_h=sim_config.w_h,
            w_v=sim_config.w_v,
            event_utility=event_utility_final,
            max_observation_ratio=0.2,
        )
        
        agents.append(agent)
    
    return agents


def create_random_agents(
    height: int, width: int, num_agents: int = 5, sim_config: SimulationConfig = DEFAULT_CONFIG,
    agent_configs_json: str | None = None, kernel_path: Optional[str] = None
) -> list[RandomAgent]:
    """Create multiple random agents with different trajectories.
    
    Args:
        height: Grid height
        width: Grid width
        num_agents: Number of agents to create
        sim_config: Simulation configuration
        agent_configs_json: Path to JSON file with agent configurations. If None, uses default.json
    """
    agents = []
    
    # Load agent configurations from JSON
    try:
        agent_configs = load_agent_configurations(agent_configs_json)
    except (FileNotFoundError, ValueError):
        # Fallback to hardcoded configs if JSON loading fails
        agent_configs = [
            {
                "name": "Aurora-1",
                "inclination_deg": 25.0,
                "phase_deg": 0.0,
                "latitude_offset": 0.0,
                "color": "#ffd166",
            },
            {
                "name": "Borealis-2",
                "inclination_deg": 55.0,
                "phase_deg": 72.0,
                "latitude_offset": 5.0,
                "color": "#4ecdc4",
            },
            {
                "name": "Zenith-3",
                "inclination_deg": 10.0,
                "phase_deg": 144.0,
                "latitude_offset": -8.0,
                "color": "#ff6b6b",
            },
            {
                "name": "Polaris-4",
                "inclination_deg": 40.0,
                "phase_deg": 216.0,
                "latitude_offset": 3.0,
                "color": "#95e1d3",
            },
            {
                "name": "Vega-5",
                "inclination_deg": 30.0,
                "phase_deg": 288.0,
                "latitude_offset": -5.0,
                "color": "#f38181",
            },
        ]
    
    # Add simulation config values to each agent config
    for config in agent_configs:
        config["field_of_regard_deg"] = sim_config.field_of_regard_deg
    
    # Create lat/lon grids for pre-computing FOR locations
    lat_grid, lon_grid = create_lat_lon_grids(height, width)
    
    for i, config in enumerate(agent_configs[:num_agents]):
        trajectory = Trajectory.circular_orbit(
            period=sim_config.orbit_period,
            inclination_deg=config["inclination_deg"],
            phase_deg=config["phase_deg"],
            latitude_offset=config["latitude_offset"],
        )
        
        # Create belief with all FOR locations pre-initialized
        # Set num_states=3 for kernel mode, 2 for binary mode
        num_states = 3 if kernel_path else 2
        belief = Belief(height=height, width=width, prior_probability=0.5, num_states=num_states)
        
        # Create temporary agent to compute FOR locations
        temp_agent = RandomAgent(
            name=config["name"],
            trajectory=trajectory,
            belief=belief,
            field_of_regard_deg=config["field_of_regard_deg"],
            color=config["color"],
        )
        
        # Pre-compute all locations that will be in FOR over trajectory period
        all_for_locations = temp_agent.compute_all_for_locations(lat_grid, lon_grid)
        
        # Initialize belief with all FOR locations at prior probability
        for lat, lon in all_for_locations:
            belief.set_probability(lat, lon, belief.prior_probability)
        
        agent = RandomAgent(
            name=config["name"],
            trajectory=trajectory,
            belief=belief,
            field_of_regard_deg=config["field_of_regard_deg"],
            color=config["color"],
            max_observation_ratio=0.2,
            seed=None,  # Use different seed for each agent if desired
        )
        
        agents.append(agent)
    
    return agents


def create_monte_carlo_agents(
    height: int, width: int, num_agents: int = 5, sim_config: SimulationConfig = DEFAULT_CONFIG,
    agent_configs_json: str | None = None, kernel_path: Optional[str] = None
) -> list[MonteCarloAgent]:
    """Create multiple Monte Carlo agents with different trajectories.
    
    Args:
        height: Grid height
        width: Grid width
        num_agents: Number of agents to create
        sim_config: Simulation configuration
        agent_configs_json: Path to JSON file with agent configurations. If None, uses default.json
    """
    agents = []
    
    # Load agent configurations from JSON
    try:
        agent_configs = load_agent_configurations(agent_configs_json)
    except (FileNotFoundError, ValueError):
        # Fallback to hardcoded configs if JSON loading fails
        agent_configs = [
            {
                "name": "Aurora-1",
                "inclination_deg": 25.0,
                "phase_deg": 0.0,
                "latitude_offset": 0.0,
                "color": "#ffd166",
            },
            {
                "name": "Borealis-2",
                "inclination_deg": 55.0,
                "phase_deg": 72.0,
                "latitude_offset": 5.0,
                "color": "#4ecdc4",
            },
            {
                "name": "Zenith-3",
                "inclination_deg": 10.0,
                "phase_deg": 144.0,
                "latitude_offset": -8.0,
                "color": "#ff6b6b",
            },
            {
                "name": "Polaris-4",
                "inclination_deg": 40.0,
                "phase_deg": 216.0,
                "latitude_offset": 3.0,
                "color": "#95e1d3",
            },
            {
                "name": "Vega-5",
                "inclination_deg": 30.0,
                "phase_deg": 288.0,
                "latitude_offset": -5.0,
                "color": "#f38181",
            },
        ]
    
    # Add simulation config values to each agent config
    for config in agent_configs:
        config["field_of_regard_deg"] = sim_config.field_of_regard_deg
        config["num_rollouts"] = sim_config.num_rollouts
    
    # Create lat/lon grids for pre-computing FOR locations
    lat_grid, lon_grid = create_lat_lon_grids(height, width)
    
    for i, config in enumerate(agent_configs[:num_agents]):
        trajectory = Trajectory.circular_orbit(
            period=sim_config.orbit_period,
            inclination_deg=config["inclination_deg"],
            phase_deg=config["phase_deg"],
            latitude_offset=config["latitude_offset"],
        )
        
        # Create belief with all FOR locations pre-initialized
        # Set num_states=3 for kernel mode, 2 for binary mode
        kernel_path_used = kernel_path or sim_config.kernel_path
        num_states = 3 if kernel_path_used else 2
        belief = Belief(height=height, width=width, prior_probability=0.5, num_states=num_states)
        
        # Create temporary agent to compute FOR locations
        temp_agent = MonteCarloAgent(
            name=config["name"],
            trajectory=trajectory,
            belief=belief,
            field_of_regard_deg=config["field_of_regard_deg"],
            color=config["color"],
            num_rollouts=config["num_rollouts"],
        )
        
        # Pre-compute all locations that will be in FOR over trajectory period
        all_for_locations = temp_agent.compute_all_for_locations(lat_grid, lon_grid)
        
        # Initialize belief with all FOR locations at prior probability
        for lat, lon in all_for_locations:
            belief.set_probability(lat, lon, belief.prior_probability)
        
        # Determine event_utility based on mode
        event_utility_final = sim_config.event_utility.copy()
        if kernel_path_used and 2 not in event_utility_final:
            event_utility_final[2] = 0.5
        
        agent = MonteCarloAgent(
            name=config["name"],
            trajectory=trajectory,
            belief=belief,
            field_of_regard_deg=config["field_of_regard_deg"],
            color=config["color"],
            num_rollouts=config["num_rollouts"],
            w_h=sim_config.w_h,
            w_v=sim_config.w_v,
            event_utility=event_utility_final,
            discount_factor=sim_config.discount_factor,
        )
        
        agents.append(agent)
    
    return agents


def create_sharing_monte_carlo_agents(
    height: int, width: int, num_agents: int = 5, sim_config: SimulationConfig = DEFAULT_CONFIG,
    agent_configs_json: str | None = None, kernel_path: Optional[str] = None
) -> list[SharingMonteCarloAgent]:
    """Create multiple Sharing Monte Carlo agents with different trajectories.
    
    Args:
        height: Grid height
        width: Grid width
        num_agents: Number of agents to create
        sim_config: Simulation configuration
        agent_configs_json: Path to JSON file with agent configurations. If None, uses default.json
    """
    agents = []
    
    # Load agent configurations from JSON
    try:
        agent_configs = load_agent_configurations(agent_configs_json)
    except (FileNotFoundError, ValueError):
        # Fallback to hardcoded configs if JSON loading fails
        agent_configs = [
            {
                "name": "Aurora-1",
                "inclination_deg": 25.0,
                "phase_deg": 0.0,
                "latitude_offset": 0.0,
                "color": "#ffd166",
            },
            {
                "name": "Borealis-2",
                "inclination_deg": 55.0,
                "phase_deg": 72.0,
                "latitude_offset": 5.0,
                "color": "#4ecdc4",
            },
            {
                "name": "Zenith-3",
                "inclination_deg": 10.0,
                "phase_deg": 144.0,
                "latitude_offset": -8.0,
                "color": "#ff6b6b",
            },
            {
                "name": "Polaris-4",
                "inclination_deg": 40.0,
                "phase_deg": 216.0,
                "latitude_offset": 3.0,
                "color": "#95e1d3",
            },
            {
                "name": "Vega-5",
                "inclination_deg": 30.0,
                "phase_deg": 288.0,
                "latitude_offset": -5.0,
                "color": "#f38181",
            },
        ]
    
    # Add simulation config values to each agent config
    for config in agent_configs:
        config["field_of_regard_deg"] = sim_config.field_of_regard_deg
        config["num_rollouts"] = sim_config.num_rollouts
    
    # Create lat/lon grids for pre-computing FOR locations
    lat_grid, lon_grid = create_lat_lon_grids(height, width)
    
    for i, config in enumerate(agent_configs[:num_agents]):
        trajectory = Trajectory.circular_orbit(
            period=sim_config.orbit_period,
            inclination_deg=config["inclination_deg"],
            phase_deg=config["phase_deg"],
            latitude_offset=config["latitude_offset"],
        )
        
        # Create belief with all FOR locations pre-initialized
        # Set num_states=3 for kernel mode, 2 for binary mode
        kernel_path_used = kernel_path or sim_config.kernel_path
        num_states = 3 if kernel_path_used else 2
        belief = Belief(height=height, width=width, prior_probability=0.5, num_states=num_states)
        
        # Create temporary agent to compute FOR locations
        temp_agent = SharingMonteCarloAgent(
            name=config["name"],
            trajectory=trajectory,
            belief=belief,
            field_of_regard_deg=config["field_of_regard_deg"],
            color=config["color"],
            num_rollouts=config["num_rollouts"],
        )
        
        # Pre-compute all locations that will be in FOR over trajectory period
        all_for_locations = temp_agent.compute_all_for_locations(lat_grid, lon_grid)
        
        # Initialize belief with all FOR locations at prior probability
        for lat, lon in all_for_locations:
            belief.set_probability(lat, lon, belief.prior_probability)
        
        # Determine event_utility based on mode
        event_utility_final = sim_config.event_utility.copy()
        if kernel_path_used and 2 not in event_utility_final:
            event_utility_final[2] = 0.5
        
        agent = SharingMonteCarloAgent(
            name=config["name"],
            trajectory=trajectory,
            belief=belief,
            field_of_regard_deg=config["field_of_regard_deg"],
            color=config["color"],
            num_rollouts=config["num_rollouts"],
            w_h=sim_config.w_h,
            w_v=sim_config.w_v,
            event_utility=event_utility_final,
            discount_factor=sim_config.discount_factor,
            communication_range_deg=sim_config.communication_range_deg,
        )
        
        agents.append(agent)
    
    return agents


def create_dsb_abba_agents(
    height: int, width: int, num_agents: int = 5, sim_config: SimulationConfig = DEFAULT_CONFIG,
    agent_configs_json: str | None = None, event_utility: Optional[dict[int, float]] = None,
    kernel_path: Optional[str] = None
) -> list[DSBABBAAgent]:
    """Create multiple DSB-ABBA agents with different trajectories.
    
    Args:
        height: Grid height
        width: Grid width
        num_agents: Number of agents to create
        sim_config: Simulation configuration
        agent_configs_json: Path to JSON file with agent configurations. If None, uses default.json
    """
    agents = []
    
    # Load agent configurations from JSON
    try:
        agent_configs = load_agent_configurations(agent_configs_json)
    except (FileNotFoundError, ValueError):
        # Fallback to hardcoded configs if JSON loading fails
        agent_configs = [
            {
                "name": "Aurora-1",
                "inclination_deg": 25.0,
                "phase_deg": 0.0,
                "latitude_offset": 0.0,
                "color": "#ffd166",
            },
            {
                "name": "Borealis-2",
                "inclination_deg": 55.0,
                "phase_deg": 72.0,
                "latitude_offset": 5.0,
                "color": "#4ecdc4",
            },
            {
                "name": "Zenith-3",
                "inclination_deg": 10.0,
                "phase_deg": 144.0,
                "latitude_offset": -8.0,
                "color": "#ff6b6b",
            },
            {
                "name": "Polaris-4",
                "inclination_deg": 40.0,
                "phase_deg": 216.0,
                "latitude_offset": 3.0,
                "color": "#95e1d3",
            },
            {
                "name": "Vega-5",
                "inclination_deg": 30.0,
                "phase_deg": 288.0,
                "latitude_offset": -5.0,
                "color": "#f38181",
            },
        ]
    
    # Add simulation config values to each agent config
    for config in agent_configs:
        config["field_of_regard_deg"] = sim_config.field_of_regard_deg
    
    # Create lat/lon grids for pre-computing FOR locations
    lat_grid, lon_grid = create_lat_lon_grids(height, width)
    
    for i, config in enumerate(agent_configs[:num_agents]):
        trajectory = Trajectory.circular_orbit(
            period=sim_config.orbit_period,
            inclination_deg=config["inclination_deg"],
            phase_deg=config["phase_deg"],
            latitude_offset=config["latitude_offset"],
        )
        
        # Create belief with all FOR locations pre-initialized
        # Set num_states=3 for kernel mode, 2 for binary mode
        kernel_path_used = kernel_path or sim_config.kernel_path
        num_states = 3 if kernel_path_used else 2
        belief = Belief(height=height, width=width, prior_probability=0.5, num_states=num_states)
        
        # Create temporary agent to compute FOR locations
        temp_agent = DSBABBAAgent(
            name=config["name"],
            trajectory=trajectory,
            belief=belief,
            field_of_regard_deg=config["field_of_regard_deg"],
            color=config["color"],
        )
        
        # Pre-compute all locations that will be in FOR over trajectory period
        all_for_locations = temp_agent.compute_all_for_locations(lat_grid, lon_grid)
        
        # Initialize belief with all FOR locations at prior probability
        for lat, lon in all_for_locations:
            belief.set_probability(lat, lon, belief.prior_probability)
        
        # Use config event_utility (already has state 2 if kernel mode)
        event_utility_final = event_utility if event_utility is not None else sim_config.event_utility.copy()
        if kernel_path_used and 2 not in event_utility_final:
            event_utility_final[2] = 0.5
        
        agent = DSBABBAAgent(
            name=config["name"],
            trajectory=trajectory,
            belief=belief,
            field_of_regard_deg=config["field_of_regard_deg"],
            color=config["color"],
            w_h=sim_config.w_h,
            w_v=sim_config.w_v,
            event_utility=event_utility_final,
            discount_factor=sim_config.discount_factor,
            communication_range_deg=sim_config.communication_range_deg,
            n_seed=sim_config.n_seed,
            n_roll=sim_config.n_roll,
            n_sweep=sim_config.n_sweep,
            horizon=sim_config.planning_horizon,
            gamma=sim_config.discount_factor,
        )
        
        agents.append(agent)
    
    return agents


class MonteCarloSimulation:
    """Simulation runner for agents (Monte Carlo, Random, etc.)."""
    
    def __init__(
        self,
        env: Environment,
        agents: list[Agent],
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        planning_horizon: int = 5,
        transition_env: Optional[Environment] = None,
        ground_stations: Optional[list[GroundStation]] = None,
        use_parallel_planning: bool = True,
    ):
        self.env = env
        self.transition_env = transition_env if transition_env is not None else env
        self.agents = agents
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.planning_horizon = planning_horizon
        self.ground_stations = ground_stations if ground_stations is not None else []
        self.use_parallel_planning = use_parallel_planning
        
        # Track statistics for each agent (per-agent metrics only)
        self.agent_stats = {
            agent.name: {
                "fires_observed": 0,  # Per-agent fire observations
                "total_reward": 0.0,
                "observations_count": 0,
                "detection_value": 0.0,  # Accumulated detection value from observed states
                # Planning time statistics (seconds)
                "planning_time": 0.0,    # Total time spent in planning/act() calls
                "planning_calls": 0,     # Number of planning/act() calls
            }
            for agent in agents
        }
        
        # Global fire event tracking
        # Track all fires that have ever existed: (lat, lon) -> {start_step, end_step, observed_by}
        self.fire_events: dict[tuple[float, float], dict] = {}
        
        # Constellation-wide fire tracking (not per-agent)
        # Set of (lat, lon) tuples representing unique fires observed by ANY agent in the constellation
        self.constellation_unique_fires: set[tuple[float, float]] = set()
        
        # Constellation-wide statistics
        self.constellation_stats = {
            "fires_observed": 0,  # Total fire observations (including reobservations)
            "unique_fires_observed": 0,  # Number of unique fires observed by constellation
            "reobservations": 0,  # Number of times a fire was reobserved (by any agent)
            "total_detection_value": 0.0,  # Accumulated detection value across all agents
        }
        
        # Track previous state to detect fire start/end events
        # Initialize with current state to track fires from step 0
        self.previous_state = self.env.get_state().copy()
        
        # Track current actions for visualization (step -> agent name -> observation_mask)
        self.current_actions: dict[int, dict[str, np.ndarray]] = {}
        
        # Threading support for parallel agent planning
        self.print_lock = threading.Lock()  # Lock for synchronized printing
    
    def step(self):
        """Execute one simulation step."""
        # Store the step BEFORE env.step() increments it
        # This ensures actions are stored at the correct step for visualization
        step = self.env.time
        
        # Environment evolves (increments env.time)
        self.env.step()
        
        if self.use_parallel_planning and len(self.agents) > 1:
            # Parallel planning: all agents plan simultaneously
            self._step_parallel(step)
        else:
            # Sequential planning: original behavior
            self._step_sequential(step)
    
    def _step_parallel(self, step: int):
        """Execute one simulation step with parallel agent planning."""
        # Prepare action arguments for each agent
        agent_tasks = []
        for agent in self.agents:
            act_kwargs = {
                "step": step,
                "lat_grid": self.lat_grid,
                "lon_grid": self.lon_grid,
                "env": self.env,
                "horizon": self.planning_horizon,
                "max_observation_cells": None,
            }
            if isinstance(agent, SharingMonteCarloAgent):
                act_kwargs["ground_stations"] = self.ground_stations
            if isinstance(agent, DSBABBAAgent):
                act_kwargs["all_agents"] = self.agents
                act_kwargs["ground_stations"] = self.ground_stations
            if isinstance(agent, SatelliteCentralizedAgent):
                act_kwargs["all_agents"] = self.agents
                act_kwargs["ground_stations"] = self.ground_stations
            agent_tasks.append((agent, act_kwargs))
        
        # Execute agent planning in parallel
        # Use agent name as key since agent objects are not hashable
        agent_results = {}
        agent_name_to_agent = {}  # Map agent name to agent object
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            # Submit all planning tasks
            future_to_agent_name = {
                executor.submit(self._agent_plan, agent, kwargs): agent.name
                for agent, kwargs in agent_tasks
            }
            
            # Store mapping from name to agent object
            for agent, _ in agent_tasks:
                agent_name_to_agent[agent.name] = agent
            
            # Wait for all agents to finish planning
            for future in as_completed(future_to_agent_name):
                agent_name = future_to_agent_name[future]
                agent = agent_name_to_agent[agent_name]
                try:
                    action_info = future.result()
                    agent_results[agent_name] = action_info
                except Exception as exc:
                    with self.print_lock:
                        print(f"Agent {agent_name} generated an exception: {exc}")
                    raise
        
        # Now process results sequentially (to maintain deterministic output order)
        # Sort by agent name to ensure consistent ordering
        for agent in sorted(self.agents, key=lambda a: a.name):
            action_info = agent_results[agent.name]
            self._process_agent_action(agent, step, action_info)
    
    def _agent_plan(self, agent: Agent, act_kwargs: dict) -> dict:
        """Execute agent planning (runs in parallel thread)."""
        start_time = time.perf_counter()
        action_info = agent.act(**act_kwargs)
        elapsed = time.perf_counter() - start_time
        
        # Update per-agent planning time statistics (thread-safe)
        with self.print_lock:
            stats = self.agent_stats.get(agent.name)
            if stats is not None:
                stats["planning_time"] += elapsed
                stats["planning_calls"] += 1
        
        return action_info
    
    def _step_sequential(self, step: int):
        """Execute one simulation step with sequential agent planning (original behavior)."""
        # Each agent takes action and updates belief
        for agent in self.agents:
            # Agent takes action (observes contiguous subset in field of regard)
            # Pass ground_stations if agent is a SharingMonteCarloAgent
            act_kwargs = {
                "step": step,
                "lat_grid": self.lat_grid,
                "lon_grid": self.lon_grid,
                "env": self.env,
                "horizon": self.planning_horizon,
                "max_observation_cells": None,  # Uses max_observation_ratio
            }
            
            # Add ground_stations parameter for sharing agents
            if isinstance(agent, SharingMonteCarloAgent):
                act_kwargs["ground_stations"] = self.ground_stations
            
            # Add all_agents for DSB-ABBA and CTDE agents
            if isinstance(agent, DSBABBAAgent) or isinstance(agent, SatelliteCentralizedAgent):
                act_kwargs["all_agents"] = self.agents
                act_kwargs["ground_stations"] = self.ground_stations
            
            # Time the planning/act call
            start_time = time.perf_counter()
            action_info = agent.act(**act_kwargs)
            elapsed = time.perf_counter() - start_time
            
            # Update per-agent planning time statistics
            stats = self.agent_stats.get(agent.name)
            if stats is not None:
                stats["planning_time"] += elapsed
                stats["planning_calls"] += 1
            
            self._process_agent_action(agent, step, action_info)
    
    def _process_agent_action(self, agent: Agent, step: int, action_info: dict):
        """Process agent action and update belief (shared by parallel and sequential paths)."""
        observation_points = action_info["observation_points"]
        expected_reward = action_info.get("expected_reward", 0.0)
        
        # Print selected action for this agent (with lock for parallel execution)
        num_cells = len(observation_points)
        if observation_points:
            # Calculate action bounds (lat/lon range)
            lats = [lat for lat, lon in observation_points]
            lons = [lon for lat, lon in observation_points]
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)
            # Calculate center
            lat_center = (lat_min + lat_max) / 2.0
            lon_center = (lon_min + lon_max) / 2.0
            
            with self.print_lock:
                print(f"Step {step:3d} | {agent.name:15s} | "
                      f"{num_cells:3d} cells | "
                      f"Reward: {expected_reward:7.3f} | "
                      f"Center: ({lat_center:6.2f}°, {lon_center:6.2f}°)")
        else:
            with self.print_lock:
                print(f"Step {step:3d} | {agent.name:15s} | "
                      f"{num_cells:3d} cells | "
                      f"Reward: {expected_reward:7.3f} | "
                      f"(Empty action)")
        
        # Convert observation_points to mask for visualization and statistics
        observation_mask = agent.points_to_mask(
            observation_points, self.lat_grid, self.lon_grid, self.env.height, self.env.width
        )
        
        # Store action for visualization
        # IMPORTANT: Store actions at the current step (after env.step() was called)
        # So actions taken at step N are stored at step N
        if step not in self.current_actions:
            self.current_actions[step] = {}
        self.current_actions[step][agent.name] = observation_mask.copy()
        
        # Verify action was stored correctly
        num_action_cells = np.sum(observation_mask)
        if num_action_cells == 0:
            with self.print_lock:
                print(f"  WARNING: {agent.name} has empty action mask at step {step}")
        
        # Get actual observations from environment (ground truth)
        actual_state = self.env.get_state()
        ground_truth_observation = actual_state.copy()
        
        # Extract only the observed cells (the action taken)
        observed_values = ground_truth_observation[observation_mask]
        
        # Track fire observations (unique vs reobservations)
        observed_fires = np.sum(observed_values == 1)
        self.agent_stats[agent.name]["fires_observed"] += observed_fires
        self.agent_stats[agent.name]["observations_count"] += len(observation_points)
        
        # Compute detection value from actual observed states
        # Get event_utility from agent (supports both 2-state and 3-state)
        event_utility = getattr(agent, 'event_utility', None)
        
        # If event_utility is not set or empty, determine default based on environment mode
        if not event_utility:
            # Check if we're in kernel mode (3 states) by checking environment mode or agent's belief
            is_kernel_mode = (
                hasattr(self.env, 'mode') and self.env.mode == "kernel"
            ) or (
                hasattr(agent, 'belief') and hasattr(agent.belief, 'num_states') and agent.belief.num_states == 3
            )
            if is_kernel_mode:
                event_utility = {0: 0.0, 1: 1.0, 2: 0.5}  # 3-state kernel mode
            else:
                event_utility = {0: 0.0, 1: 1.0}  # 2-state RSP/DBN-2 mode
        else:
            # Ensure event_utility has state 2 if we're in kernel mode (even if agent's utility doesn't have it)
            is_kernel_mode = (
                hasattr(self.env, 'mode') and self.env.mode == "kernel"
            ) or (
                hasattr(agent, 'belief') and hasattr(agent.belief, 'num_states') and agent.belief.num_states == 3
            )
            if is_kernel_mode and 2 not in event_utility:
                # Add state 2 utility if missing (default to 0.5 for burned state)
                event_utility = event_utility.copy()  # Don't modify agent's original dict
                event_utility[2] = 0.5
        
        # Sum detection value for all observed cells (including burned state 2)
        detection_value = 0.0
        for observed_state in observed_values:
            state = int(observed_state)
            utility = event_utility.get(state, 0.0)
            detection_value += utility
        
        self.agent_stats[agent.name]["detection_value"] += detection_value
        self.constellation_stats["total_detection_value"] += detection_value
        
        # Track constellation-wide unique fires and reobservations
        # Get the (lat, lon) coordinates of observed fire cells
        fire_cells = []
        for y in range(self.env.height):
            for x in range(self.env.width):
                if observation_mask[y, x] and ground_truth_observation[y, x] == 1:
                    lat, lon = float(self.lat_grid[y, x]), float(self.lon_grid[y, x])
                    fire_cells.append((lat, lon))
        
        # Count unique fires vs reobservations at constellation level
        unique_fires_this_step = 0
        reobservations_this_step = 0
        
        for fire_location in fire_cells:
            if fire_location in self.constellation_unique_fires:
                # This is a reobservation (by any agent in the constellation)
                reobservations_this_step += 1
            else:
                # This is a new unique fire for the constellation
                unique_fires_this_step += 1
                self.constellation_unique_fires.add(fire_location)
        
        # Update constellation-wide statistics
        self.constellation_stats["fires_observed"] += observed_fires
        self.constellation_stats["unique_fires_observed"] = len(self.constellation_unique_fires)
        self.constellation_stats["reobservations"] += reobservations_this_step
        
        # Track reward (use expected_reward from Monte Carlo evaluation)
        if "expected_reward" in action_info:
            # For immediate reward tracking, compute actual reward from observation
            immediate_reward = agent.compute_reward(
                self.lat_grid, self.lon_grid, step, observation_points=observation_points
            )
            self.agent_stats[agent.name]["total_reward"] += immediate_reward
        
        # For SharingMonteCarloAgent and DSBABBAAgent: save belief snapshot BEFORE observation update
        # This allows temporal belief reconstruction from this timestep
        if isinstance(agent, SharingMonteCarloAgent):
            # Save snapshot at the START of this timestep (before observation)
            # This represents the belief state before applying observation at step
            agent._save_belief_snapshot(step)
        
        # For DSBABBAAgent: also save snapshot
        if isinstance(agent, DSBABBAAgent):
            agent._save_belief_snapshot(step)
        
        # Update agent belief with ground truth observations
        # Only update cells in the observation_points (the action taken - FOV, subset of FOR)
        agent.update_belief_from_observation(
            self.lat_grid, self.lon_grid, step, ground_truth_observation, 
            observation_points=observation_points
        )
        
        # For SharingMonteCarloAgent, DSBABBAAgent, and SatelliteCentralizedAgent: record the observation at this timestep
        # This allows temporal belief reconstruction when data is shared
        if isinstance(agent, (SharingMonteCarloAgent, DSBABBAAgent, SatelliteCentralizedAgent)):
            # Convert ground truth observation to dict format: (lat, lon) -> value
            # observation_points is a set of (lat, lon) tuples
            # observation_mask tells us which grid cells correspond to these points
            observation_dict: dict[tuple[float, float], int] = {}
            
            # Create mapping from (lat, lon) to grid coordinates for fast lookup
            height, width = self.lat_grid.shape
            latlon_to_grid = {}
            for y in range(height):
                for x in range(width):
                    lat, lon = float(self.lat_grid[y, x]), float(self.lon_grid[y, x])
                    latlon_to_grid[(lat, lon)] = (y, x)
            
            # Extract observed values for each point in observation_points
            for lat, lon in observation_points:
                lat_key = float(lat)
                lon_key = float(lon)
                
                # Find grid coordinates
                grid_coords = latlon_to_grid.get((lat_key, lon_key))
                if grid_coords is None:
                    # Try tolerance-based search
                    for (g_lat, g_lon), (gy, gx) in latlon_to_grid.items():
                        if abs(g_lat - lat_key) < 1e-6 and abs(g_lon - lon_key) < 1e-6:
                            grid_coords = (gy, gx)
                            break
                
                if grid_coords is not None:
                    y, x = grid_coords
                    # Only include if this cell was actually observed (in observation_mask)
                    if observation_mask[y, x]:
                        observed_value = int(ground_truth_observation[y, x])
                        observation_dict[(lat_key, lon_key)] = observed_value
            
            # Record this observation at the current timestep
            if observation_dict:
                agent._record_own_observation(step, observation_dict)
        
        # Evolve belief for next step (for unobserved cells)
        # Use transition_env for transition kernel (original env when using ReplayEnvironment)
        agent.evolve_belief_with_environment(
            self.transition_env, self.lat_grid, self.lon_grid, step, observation_mask=observation_mask
        )
    
    def get_statistics(self) -> dict:
        """
        Get statistics for all agents as a dictionary.
        
        Returns:
            Dictionary containing simulation statistics
        """
        # Compute aggregate planning time across all agents
        total_planning_time = 0.0
        total_planning_calls = 0
        for stats in self.agent_stats.values():
            total_planning_time += stats.get("planning_time", 0.0)
            total_planning_calls += stats.get("planning_calls", 0)
        avg_planning_time = (
            total_planning_time / total_planning_calls if total_planning_calls > 0 else 0.0
        )
        
        stats_dict = {
            "simulation": {
                "total_steps": self.env.time,
                "planning_horizon": self.planning_horizon,
                "environment_size": {
                    "height": self.env.height,
                    "width": self.env.width,
                },
                "total_fire_events": len(self.fire_events),
                # Average planning time per plan call across all agents (seconds)
                "avg_planning_time": float(avg_planning_time),
            },
            "constellation": {
                "fires_observed": int(self.constellation_stats["fires_observed"]),  # Total (including reobservations)
                "unique_fires_observed": int(self.constellation_stats["unique_fires_observed"]),  # Unique fires
                "reobservations": int(self.constellation_stats["reobservations"]),  # Reobservation count
                "total_detection_value": float(self.constellation_stats["total_detection_value"]),  # Accumulated detection value
            },
            "agents": {},
        }
        
        for agent_name, stats in self.agent_stats.items():
            avg_reward = (stats["total_reward"] / stats["observations_count"] 
                         if stats["observations_count"] > 0 else 0.0)
            # Average planning time per call for this agent
            pt_calls = stats.get("planning_calls", 0)
            avg_planning_time_agent = (
                stats.get("planning_time", 0.0) / pt_calls if pt_calls > 0 else 0.0
            )
            stats_dict["agents"][agent_name] = {
                "fires_observed": int(stats["fires_observed"]),  # Per-agent total
                "total_reward": float(stats["total_reward"]),
                "observations_count": int(stats["observations_count"]),
                "avg_reward_per_observation": float(avg_reward),
                "detection_value": float(stats.get("detection_value", 0.0)),  # Per-agent detection value
                "avg_planning_time": float(avg_planning_time_agent),  # Per-agent average planning time (seconds)
            }
        
        return stats_dict
    
    def print_statistics(self):
        """Print statistics for all agents."""
        print("\n" + "="*70)
        print("SIMULATION RESULTS")
        print("="*70)
        
        # Constellation-wide statistics
        print("CONSTELLATION-WIDE METRICS:")
        print(f"  Fires Observed (Total): {self.constellation_stats['fires_observed']}")
        print(f"  Unique Fires Observed: {self.constellation_stats['unique_fires_observed']}")
        print(f"  Reobservations: {self.constellation_stats['reobservations']}")
        print(f"  Total Detection Value: {self.constellation_stats['total_detection_value']:.4f}")
        # Average planning time across all agents (seconds)
        total_planning_time = sum(
            stats.get("planning_time", 0.0) for stats in self.agent_stats.values()
        )
        total_planning_calls = sum(
            stats.get("planning_calls", 0) for stats in self.agent_stats.values()
        )
        avg_planning_time = (
            total_planning_time / total_planning_calls if total_planning_calls > 0 else 0.0
        )
        print(f"  Avg Planning Time per Call: {avg_planning_time:.6f} s")
        print()
        
        # Per-agent statistics
        print("PER-AGENT METRICS:")
        print(f"{'Agent':<20} {'Fires Observed':<18} {'Total Reward':<18} "
              f"{'Detection Value':<18} {'Avg Reward/Obs':<18} {'Avg Plan Time (s)':<18}")
        print("-"*100)
        
        for agent_name, stats in sorted(self.agent_stats.items()):
            avg_reward = (stats["total_reward"] / stats["observations_count"] 
                         if stats["observations_count"] > 0 else 0.0)
            detection_value = stats.get("detection_value", 0.0)
            pt_calls = stats.get("planning_calls", 0)
            avg_planning_time_agent = (
                stats.get("planning_time", 0.0) / pt_calls if pt_calls > 0 else 0.0
            )
            print(
                f"{agent_name:<20} {stats['fires_observed']:<18} "
                f"{stats['total_reward']:<18.4f} {detection_value:<18.4f} "
                f"{avg_reward:<18.4f} {avg_planning_time_agent:<18.6f}"
            )
        
        print("="*70)
        print(f"Total simulation steps: {self.env.time}")
        print(f"Total unique fire events: {len(self.fire_events)}")
        print("="*70 + "\n")
    
    def save_statistics(self, filepath: str):
        """
        Save statistics to a JSON file.
        
        Args:
            filepath: Path to save the statistics JSON file
        """
        import json
        from datetime import datetime
        
        stats_dict = self.get_statistics()
        
        # Add metadata - determine simulation type from agents
        agent_types = [type(agent).__name__ for agent in self.agents]
        simulation_type = f"Agent Simulation ({', '.join(set(agent_types))})"
        
        stats_dict["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "simulation_type": simulation_type,
            "agent_types": list(set(agent_types)),
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"Statistics saved to: {filepath}")
    
    def save_statistics_csv(self, filepath: str):
        """
        Save statistics to a CSV file.
        
        Args:
            filepath: Path to save the statistics CSV file
        """
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write constellation-wide statistics first
            writer.writerow(['CONSTELLATION-WIDE METRICS'])
            writer.writerow([
                'Metric', 'Value'
            ])
            writer.writerow([
                'Fires Observed (Total)', self.constellation_stats['fires_observed']
            ])
            writer.writerow([
                'Unique Fires Observed', self.constellation_stats['unique_fires_observed']
            ])
            writer.writerow([
                'Reobservations', self.constellation_stats['reobservations']
            ])
            writer.writerow([
                'Total Detection Value', f"{self.constellation_stats['total_detection_value']:.4f}"
            ])
            # Average planning time across all agents (seconds)
            total_planning_time = sum(
                stats.get("planning_time", 0.0) for stats in self.agent_stats.values()
            )
            total_planning_calls = sum(
                stats.get("planning_calls", 0) for stats in self.agent_stats.values()
            )
            avg_planning_time = (
                total_planning_time / total_planning_calls if total_planning_calls > 0 else 0.0
            )
            writer.writerow([
                'Avg Planning Time per Call (s)', f"{avg_planning_time:.6f}"
            ])
            writer.writerow([])  # Empty row separator
            
            # Write per-agent statistics
            writer.writerow(['PER-AGENT METRICS'])
            writer.writerow([
                'Agent', 'Fires Observed', 'Total Reward', 'Detection Value',
                'Observations Count', 'Avg Reward per Observation',
                'Avg Planning Time (s)'
            ])
            # Write data
            for agent_name, stats in self.agent_stats.items():
                avg_reward = (stats["total_reward"] / stats["observations_count"] 
                             if stats["observations_count"] > 0 else 0.0)
                detection_value = stats.get("detection_value", 0.0)
                pt_calls = stats.get("planning_calls", 0)
                avg_planning_time_agent = (
                    stats.get("planning_time", 0.0) / pt_calls if pt_calls > 0 else 0.0
                )
                writer.writerow([
                    agent_name,
                    stats["fires_observed"],
                    f"{stats['total_reward']:.4f}",
                    f"{detection_value:.4f}",
                    stats["observations_count"],
                    f"{avg_reward:.4f}",
                    f"{avg_planning_time_agent:.6f}",
                ])
            # Write simulation summary
            writer.writerow([])
            writer.writerow(['Simulation Summary'])
            writer.writerow(['Total Steps', self.env.time])
            writer.writerow(['Planning Horizon', self.planning_horizon])
            writer.writerow(['Environment Height', self.env.height])
            writer.writerow(['Environment Width', self.env.width])
        
        print(f"Statistics saved to CSV: {filepath}")


def get_method_folder_name(agent_type: str) -> str:
    """Convert agent_type to a standardized method folder name."""
    method_map = {
        "sharing": "sharing_monte_carlo",
        "monte_carlo": "monte_carlo",
        "random": "random",
        "greedy": "greedy",
        "dsb_abba": "dsb_abba",
    }
    return method_map.get(agent_type, agent_type.lower())


def create_timestamp_folder() -> str:
    """
    Create a timestamp folder for a run: results/YYYYMMDD_HHMMSS/
    Returns the timestamp folder path.
    """
    # Create timestamp folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create full path: results/timestamp/
    timestamp_path = os.path.join("results", timestamp)
    
    # Create directory if it doesn't exist
    os.makedirs(timestamp_path, exist_ok=True)
    
    return timestamp_path


def setup_method_directory(timestamp_path: str, agent_type: str) -> str:
    """
    Create method subdirectory within timestamp folder: timestamp_path/method_name/
    Returns the method folder path.
    """
    # Get method folder name
    method_name = get_method_folder_name(agent_type)
    
    # Create full path: timestamp_path/method_name/
    method_path = os.path.join(timestamp_path, method_name)
    
    # Create directory if it doesn't exist
    os.makedirs(method_path, exist_ok=True)
    
    return method_path


def main(
    config: SimulationConfig = DEFAULT_CONFIG,
    height: int | None = None,
    width: int | None = None,
    num_agents: int | None = None,
    planning_horizon: int | None = None,
    num_steps: int | None = None,
    animation_interval: int = 150,
    save_path: str | None = None,
    fps: int | None = None,
    dpi: int = 120,
    results_save_path: str | None = None,
    save_csv: bool = False,
    agent_type: str = "monte_carlo",
    env: Optional[Environment] = None,  # Optional pre-created environment (can be ReplayEnvironment)
    original_env: Optional[Environment] = None,  # Original env for transition kernel (for replay envs)
    results_timestamp_path: str | None = None,  # Optional timestamp folder path (if None, creates new one)
    kernel_path: Optional[str] = None,  # Optional path to learned kernel file (for kernel mode)
):
    """
    Main function to run agent simulation.
    
    Args:
        config: Simulation configuration object (defaults to DEFAULT_CONFIG)
        height: Grid height (latitude resolution). If None, uses config.grid_height
        width: Grid width (longitude resolution). If None, uses config.grid_width
        num_agents: Number of agents to create. If None, uses config.num_agents
        planning_horizon: Planning horizon for Monte Carlo rollouts. If None, uses config.planning_horizon
        num_steps: Number of simulation steps to run. If None, uses config.num_steps
        animation_interval: Animation frame interval in milliseconds
        save_path: Path to save the animation (e.g., 'animation.mp4' or 'animation.gif').
                  If None, animation is displayed but not saved.
        fps: Frames per second for saved animation. If None, calculated from animation_interval
        dpi: DPI for saved animation
        results_save_path: Path to save simulation results/statistics (JSON format).
                          If None, results are printed but not saved.
                          If not specified but save_path is provided, defaults to same name with .json extension
        save_csv: If True, also save statistics as CSV file
        agent_type: Type of agents to create ("monte_carlo", "random", "greedy", or "sharing")
        env: Optional pre-created environment. If None, creates a new environment.
             Can be a ReplayEnvironment for deterministic replay.
        original_env: Original Environment instance (for ReplayEnvironment cases).
                      Agents need access to the original environment's transition_probability
                      and _iter_neighbors methods for belief updates.
    """
    # Use config values if parameters are not provided
    height = height if height is not None else config.grid_height
    width = width if width is not None else config.grid_width
    num_agents = num_agents if num_agents is not None else config.num_agents
    planning_horizon = planning_horizon if planning_horizon is not None else config.planning_horizon
    num_steps = num_steps if num_steps is not None else config.num_steps
    
    # Use kernel_path from config if not provided as parameter
    kernel_path_used = kernel_path if kernel_path is not None else config.kernel_path
    
    # Use provided environment or create a new one
    if env is None:
        env = create_environment(height, width, kernel_path=kernel_path_used)
    
    # Use original_env for transition kernel if provided, otherwise use env
    transition_env = original_env if original_env is not None else env
    
    # Create agents based on type
    ground_stations = None
    agent_configs_for_summary = None  # Store for trajectory summary
    
    if agent_type == "random":
        agents = create_random_agents(height, width, num_agents=num_agents, sim_config=config, kernel_path=kernel_path_used)
    elif agent_type == "greedy":
        agents = create_greedy_agents(height, width, num_agents=num_agents, sim_config=config, kernel_path=kernel_path_used)
    elif agent_type == "sharing":
        # Load agent configs from JSON
        try:
            agent_configs = load_agent_configurations()
        except (FileNotFoundError, ValueError):
            # Fallback to hardcoded configs if JSON loading fails
            agent_configs = [
                {"name": "Aurora-1", "inclination_deg": 25.0, "phase_deg": 0.0, "latitude_offset": 0.0},
                {"name": "Borealis-2", "inclination_deg": 55.0, "phase_deg": 72.0, "latitude_offset": 5.0},
                {"name": "Zenith-3", "inclination_deg": 10.0, "phase_deg": 144.0, "latitude_offset": -8.0},
                {"name": "Polaris-4", "inclination_deg": 40.0, "phase_deg": 216.0, "latitude_offset": 3.0},
                {"name": "Vega-5", "inclination_deg": 30.0, "phase_deg": 288.0, "latitude_offset": -5.0},
            ]
        
        # Store for trajectory summary
        agent_configs_for_summary = agent_configs.copy()
        
        # Add field_of_regard_deg to agent configs for ground station creation
        for agent_config in agent_configs:
            agent_config["field_of_regard_deg"] = config.field_of_regard_deg
        
        agents = create_sharing_monte_carlo_agents(height, width, num_agents=num_agents, sim_config=config, kernel_path=kernel_path_used)
        ground_stations = create_ground_stations_from_agent_configs(
            agent_configs[:num_agents], 
            orbit_period=config.orbit_period, 
            num_stations=config.num_ground_stations, 
            min_satellites_per_station=config.min_satellites_per_station,
            communication_range_deg=config.communication_range_deg,
        )
        
        # DEBUG: Print ground station info
        if ground_stations:
            print(f"\n[GS Setup] Created {len(ground_stations)} ground stations:")
            for gs in ground_stations:
                # Get visited_by info from the ground station dict (we need to check the creation function)
                # For now, just print the station location
                print(f"  - {gs.name}: lat={gs.latitude:.2f}°, lon={gs.longitude:.2f}°")
            print()
    elif agent_type == "monte_carlo":
        # Load agent configs from JSON (same as sharing agents)
        try:
            agent_configs = load_agent_configurations()
        except (FileNotFoundError, ValueError):
            # Fallback to hardcoded configs if JSON loading fails
            agent_configs = [
                {"name": "Aurora-1", "inclination_deg": 25.0, "phase_deg": 0.0, "latitude_offset": 0.0},
                {"name": "Borealis-2", "inclination_deg": 55.0, "phase_deg": 72.0, "latitude_offset": 5.0},
                {"name": "Zenith-3", "inclination_deg": 10.0, "phase_deg": 144.0, "latitude_offset": -8.0},
                {"name": "Polaris-4", "inclination_deg": 40.0, "phase_deg": 216.0, "latitude_offset": 3.0},
                {"name": "Vega-5", "inclination_deg": 30.0, "phase_deg": 288.0, "latitude_offset": -5.0},
            ]
        agents = create_monte_carlo_agents(height, width, num_agents=num_agents, sim_config=config, kernel_path=kernel_path_used)
    elif agent_type == "dsb_abba":
        # Load agent configs from JSON
        try:
            agent_configs = load_agent_configurations()
        except (FileNotFoundError, ValueError):
            # Fallback to hardcoded configs if JSON loading fails
            agent_configs = [
                {"name": "Aurora-1", "inclination_deg": 25.0, "phase_deg": 0.0, "latitude_offset": 0.0},
                {"name": "Borealis-2", "inclination_deg": 55.0, "phase_deg": 72.0, "latitude_offset": 5.0},
                {"name": "Zenith-3", "inclination_deg": 10.0, "phase_deg": 144.0, "latitude_offset": -8.0},
                {"name": "Polaris-4", "inclination_deg": 40.0, "phase_deg": 216.0, "latitude_offset": 3.0},
                {"name": "Vega-5", "inclination_deg": 30.0, "phase_deg": 288.0, "latitude_offset": -5.0},
            ]
        # Use config event_utility (already configured for kernel mode if needed)
        event_utility_final = config.event_utility.copy()
        if kernel_path_used and 2 not in event_utility_final:
            event_utility_final[2] = 0.5
        agents = create_dsb_abba_agents(height, width, num_agents=num_agents, sim_config=config, event_utility=event_utility_final, kernel_path=kernel_path_used)
        # Create ground stations for DSB-ABBA (it needs communication)
        ground_stations = create_ground_stations_from_agent_configs(
            agent_configs[:num_agents],
            orbit_period=config.orbit_period,
            num_stations=config.num_ground_stations,
            min_satellites_per_station=config.min_satellites_per_station,
            communication_range_deg=config.communication_range_deg,
        )
    elif agent_type == "attentionCTDE":
        # CTDE agents require training first before execution
        print("\n" + "="*70)
        print("ATTENTION CTDE AGENTS: Training Phase")
        print("="*70)
        
        from environment_simulation.agents.satellite_attention_training import (
            train_satellite_centralized_agents,
            create_satellite_centralized_agents,
        )
        from environment_simulation.agents.satellite_attention_trainer import (
            CentralizedPPOTrainer,
        )
        
        # Create training environment (separate from evaluation environment)
        training_env = create_environment(height, width, kernel_path=kernel_path)
        training_env.reset()
        
        # Detect num_states from environment
        num_states = 3 if kernel_path_used else 2
        
        # Create agents with shared networks (untrained initially)
        agents = create_satellite_centralized_agents(
            height=height,
            width=width,
            num_agents=num_agents,
            sim_config=config,
            num_states=num_states,  # Pass num_states for proper belief and network initialization
        )
        # Create ground stations for communication during execution
        # During training, we use ideal centralized belief (no communication needed)
        # During execution, agents use ground station communication to build centralized belief
        try:
            agent_configs = load_agent_configurations()
        except (FileNotFoundError, ValueError):
            agent_configs = [
                {"name": "Aurora-1", "inclination_deg": 25.0, "phase_deg": 0.0, "latitude_offset": 0.0, "color": "#ffd166"},
                {"name": "Borealis-2", "inclination_deg": 55.0, "phase_deg": 72.0, "latitude_offset": 5.0, "color": "#4ecdc4"},
                {"name": "Zenith-3", "inclination_deg": 10.0, "phase_deg": 144.0, "latitude_offset": -8.0, "color": "#ff6b6b"},
                {"name": "Polaris-4", "inclination_deg": 40.0, "phase_deg": 216.0, "latitude_offset": 3.0, "color": "#95e1d3"},
                {"name": "Vega-5", "inclination_deg": 30.0, "phase_deg": 288.0, "latitude_offset": -5.0, "color": "#f38181"},
            ]
        
        agent_configs_for_gs = agent_configs[:num_agents].copy()
        for agent_config in agent_configs_for_gs:
            agent_config["field_of_regard_deg"] = config.field_of_regard_deg
        
        ground_stations = create_ground_stations_from_agent_configs(
            agent_configs_for_gs,
            orbit_period=config.orbit_period,
            num_stations=config.num_ground_stations,
            min_satellites_per_station=config.min_satellites_per_station,
            communication_range_deg=config.communication_range_deg,
        )
        
        # Get lat/lon grids for training
        lat_grid, lon_grid = create_lat_lon_grids(height, width)
        
        # Create trainer (standard PPO for fully centralized RL)
        shared_actor = agents[0].actor_network
        shared_critic = agents[0].critic_network
        
        # Detect GPU availability
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[AttentionCTDE] Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"[AttentionCTDE] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            device = "cpu"
            print("[AttentionCTDE] Using CPU (GPU not available)")
        
        trainer = CentralizedPPOTrainer(
            actor_network=shared_actor,
            critic_network=shared_critic,
            actor_lr=2e-3,  # Increased from 5e-4 for faster learning
            critic_lr=2e-3,  # Increased from 5e-4 for faster learning
            gamma=0.99,
            lambda_gae=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.1,  # Increased from 0.02 for more exploration
            value_coef=0.5,
            max_grad_norm=1.0,  # Increased from 0.5 (less aggressive clipping)
            device=device,
        )
        
        # Train agents
        centralized_episodes = max(200, num_steps)  # Longer training
        centralized_steps_per_episode = min(num_steps, 240)  # Longer episodes
        print(f"\n[AttentionCTDE] Training agents for {centralized_episodes} episodes...")
        print("[AttentionCTDE] (This may take a few minutes)")
        
        train_satellite_centralized_agents(
            env=training_env,
            agents=agents,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            trainer=trainer,
            num_episodes=centralized_episodes,
            episode_length=centralized_steps_per_episode,
            update_frequency=10,
            update_iterations=10,  # Increased from 5 for more updates per batch
            w_h=config.w_h,
            w_v=config.w_v,
            event_utility=config.event_utility,
            gamma=0.99,
            verbose=True,
        )
        
        # Set agents to execution mode
        for agent in agents:
            agent.training_mode = False
            agent.deterministic_execution = False
            # Verify networks are set
            if agent.actor_network is None:
                raise ValueError(f"Agent {agent.name} has no actor network after training!")
            if agent.critic_network is None:
                raise ValueError(f"Agent {agent.name} has no critic network after training!")
        
        print(f"\n[AttentionCTDE] Training complete!")
        print(f"[AttentionCTDE] Verified {len(agents)} agents have trained networks")
        print("[AttentionCTDE] Starting evaluation with trained agents...\n")
    else:
        # Fallback for other agent types
        agents = create_monte_carlo_agents(height, width, num_agents=num_agents, sim_config=config, kernel_path=kernel_path)
    
    # Create latitude/longitude grids
    lat_grid, lon_grid = create_lat_lon_grids(height, width)
    
    # Print trajectory summary for sharing agents
    if agent_type == "sharing" and ground_stations:
        print("\n" + "="*70)
        print("TRAJECTORY SUMMARY")
        print("="*70)
        
        # Create a mapping from agent name to config for easy lookup
        agent_config_map = {}
        if agent_type == "sharing":
            # Use agent_configs_for_summary if available, otherwise load from JSON
            if agent_configs_for_summary is not None:
                agent_configs_list = agent_configs_for_summary.copy()
            else:
                try:
                    agent_configs_list = load_agent_configurations()
                except (FileNotFoundError, ValueError):
                    agent_configs_list = [
                        {"name": "Aurora-1", "inclination_deg": 25.0, "phase_deg": 0.0, "latitude_offset": 0.0},
                        {"name": "Borealis-2", "inclination_deg": 55.0, "phase_deg": 72.0, "latitude_offset": 5.0},
                        {"name": "Zenith-3", "inclination_deg": 10.0, "phase_deg": 144.0, "latitude_offset": -8.0},
                        {"name": "Polaris-4", "inclination_deg": 40.0, "phase_deg": 216.0, "latitude_offset": 3.0},
                        {"name": "Vega-5", "inclination_deg": 30.0, "phase_deg": 288.0, "latitude_offset": -5.0},
                    ]
            # Add field_of_regard_deg for display
            for cfg in agent_configs_list:
                if "field_of_regard_deg" not in cfg:
                    cfg["field_of_regard_deg"] = config.field_of_regard_deg
            agent_config_map = {cfg["name"]: cfg for cfg in agent_configs_list[:num_agents]}
        
        for agent in agents:
            print(f"\n{agent.name}:")
            print(f"  Orbit Period: {agent.trajectory.period} steps")
            print(f"  Orbital Parameters:")
            # Get trajectory parameters from agent config
            agent_config = agent_config_map.get(agent.name, {})
            inc = agent_config.get('inclination_deg', 'N/A')
            phase = agent_config.get('phase_deg', 'N/A')
            offset = agent_config.get('latitude_offset', 'N/A')
            print(f"    - Inclination: {inc if inc == 'N/A' else f'{inc:.1f}°'}")
            print(f"    - Phase: {phase if phase == 'N/A' else f'{phase:.1f}°'}")
            print(f"    - Latitude Offset: {offset if offset == 'N/A' else f'{offset:.1f}°'}")
            print(f"  Field of Regard: {agent.field_of_regard_deg:.1f}°")
            if isinstance(agent, SharingMonteCarloAgent):
                print(f"  Communication Range: {agent.communication_range_deg:.1f}°")
            
            # Find which ground stations this agent will visit during its orbit
            # Use communication range for ground station contact (not FOR)
            if isinstance(agent, SharingMonteCarloAgent):
                contact_range = agent.communication_range_deg
            else:
                # For non-sharing agents, use FOR as fallback (though they shouldn't contact GS)
                contact_range = agent.field_of_regard_deg
            
            visited_stations = []
            for gs in ground_stations:
                # Check every step in the orbit to find when agent is in range
                contact_steps = []
                min_distance = float('inf')
                min_distance_step = -1
                
                for step in range(agent.trajectory.period):
                    agent_lat, agent_lon = agent.position(step)
                    distance = angular_distance_deg(
                        agent_lat, agent_lon, gs.latitude, gs.longitude
                    )
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_step = step
                    if distance <= contact_range:
                        contact_steps.append(step)
                
                if contact_steps:
                    visited_stations.append({
                        'station': gs,
                        'contact_steps': contact_steps,
                        'min_distance': min_distance,
                        'min_distance_step': min_distance_step,
                    })
            
            if visited_stations:
                print(f"  Ground Stations Visited: {len(visited_stations)}")
                for vs in visited_stations:
                    gs = vs['station']
                    contact_steps = vs['contact_steps']
                    print(f"    - {gs.name}:")
                    print(f"        Location: lat={gs.latitude:.2f}°, lon={gs.longitude:.2f}°")
                    print(f"        Contact at {len(contact_steps)} steps: {contact_steps[:10]}{'...' if len(contact_steps) > 10 else ''}")
                    print(f"        Min distance: {vs['min_distance']:.2f}° (at step {vs['min_distance_step']})")
            else:
                print(f"  Ground Stations Visited: NONE (agent never gets within {contact_range:.1f}° comm range of any GS)")
                # Show closest approach to each GS
                print(f"  Closest approaches to ground stations:")
                for gs in ground_stations:
                    min_dist = float('inf')
                    min_step = -1
                    for step in range(agent.trajectory.period):
                        agent_lat, agent_lon = agent.position(step)
                        distance = angular_distance_deg(
                            agent_lat, agent_lon, gs.latitude, gs.longitude
                        )
                        if distance < min_dist:
                            min_dist = distance
                            min_step = step
                    print(f"    - {gs.name}: min distance = {min_dist:.2f}° (at step {min_step}, need ≤{contact_range:.1f}° comm range)")
        
        print("\n" + "="*70 + "\n")
    
    # Create simulation runner
    # Pass transition_env so agents can use original environment's transition kernel
    # when using ReplayEnvironment
    # Pass ground_stations for sharing agents
    simulation = MonteCarloSimulation(
        env, agents, lat_grid, lon_grid, 
        planning_horizon=planning_horizon, 
        transition_env=transition_env,
        ground_stations=ground_stations,
        use_parallel_planning=config.use_parallel_planning,
    )
    
    # Create animator
    animator = GlobeFireAnimator(
        env,
        agents=agents,
        field_color="#ffd700",  # Gold/yellow for Field of Regard (FOR)
        observation_color="#9d00ff",  # Purple for Field of View (FOV/selected action)
        simulation=simulation,  # Pass simulation to track current actions
    )
    
    # Custom update function that includes agent actions
    # Wrap the original update to step the simulation first, then visualize
    original_update = animator._update
    
    def custom_update(frame: int):
        # Step the simulation BEFORE visualization (this handles env.step() and agent actions)
        # Frame 0: Show initial state (step 0), agents act at step 0
        # Frame 1: Step to step 1, agents act at step 1
        # ...
        # Frame N: Step to step N, agents act at step N
        # But we can only step num_steps times, so frame num_steps would try to step beyond recorded time
        # So we only step if frame < num_steps
        if frame > 0 and frame < num_steps:
            # Step the simulation (this increments env.time and agents act)
            try:
                simulation.step()
            except (IndexError, ValueError) as e:
                # If we can't step (e.g., ReplayEnvironment at max_time), skip stepping
                # This can happen on the last frame
                pass
        # For frame 0, we also need to step to get initial actions
        # (agents act at step 0, which is stored correctly now)
        elif frame == 0:
            try:
                simulation.step()
            except (IndexError, ValueError) as e:
                # If we can't step even at frame 0, skip it
                pass
        # Now call the original update which will visualize the current state
        # Since simulation is provided, the original _update will skip env.step()
        return original_update(frame)
    
    animator._update = custom_update
    
    # Run animation
    print("Starting Monte Carlo agent simulation...")
    print(f"Environment: {height}x{width}, {len(agents)} agents")
    print(f"Planning horizon: {simulation.planning_horizon} steps")
    print(f"Simulation length: {num_steps} steps")
    print("Press Ctrl+C to stop")
    print("-"*70)
    print(f"{'Step':>6s} | {'Agent':<15s} | {'Cells':<8s} | {'Reward':<12s} | {'Center (lat, lon)':<25s}")
    print("-"*70)
    
    # Set up results directory structure: results/timestamp/method_name/
    # The timestamp path MUST be provided to ensure all methods use the same timestamp folder
    # It should be created ONCE in __main__ before calling main() for each method
    if results_timestamp_path is None:
        raise ValueError("results_timestamp_path must be provided to ensure all methods use the same timestamp folder")
    
    # Create method subdirectory within the SAME timestamp folder (never create a new timestamp here)
    results_base_path = setup_method_directory(results_timestamp_path, agent_type)
    
    # Determine save paths within the results directory structure
    # If save_path is provided, use it as the base name; otherwise generate one
    if save_path is None:
        # Generate default animation filename
        animation_filename = f"animation_{agent_type}.gif"
        save_path = os.path.join(results_base_path, animation_filename)
    else:
        # If save_path is provided, place it in the results directory
        # Extract just the filename if a full path was provided
        animation_filename = os.path.basename(save_path)
        save_path = os.path.join(results_base_path, animation_filename)
    
    # Determine results save paths based on save_path
    if results_save_path is None:
        # Auto-generate results path from animation path
        base_name = os.path.splitext(save_path)[0]
        results_save_path = f"{base_name}_results.json"
    else:
        # If results_save_path is provided, place it in the results directory
        results_filename = os.path.basename(results_save_path)
        results_save_path = os.path.join(results_base_path, results_filename)
    
    # Determine CSV path - use save_path base name to ensure uniqueness
    csv_path = None
    if save_csv:
        if results_save_path:
            # Derive CSV path from JSON path (most reliable)
            csv_path = os.path.splitext(results_save_path)[0] + ".csv"
        elif save_path:
            # Derive CSV path directly from animation path
            base_name = os.path.splitext(save_path)[0]
            csv_path = f"{base_name}_results.csv"
        else:
            # Fallback: use agent type in filename
            csv_path = os.path.join(results_base_path, f"simulation_results_{agent_type}.csv")
    
    # With 'Agg' backend, tkinter is not used, so no cleanup warnings should occur
    # But we'll suppress any warnings just in case
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    try:
        animator.animate(
            frames=num_steps,
            interval=animation_interval,
            save_path=save_path,
            fps=fps,
            dpi=dpi,
        )
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        # Print statistics after simulation
        simulation.print_statistics()
        
        # Save statistics to file if requested
        if results_save_path or csv_path:
            try:
                if results_save_path:
                    simulation.save_statistics(results_save_path)
                    print(f"Statistics saved to: {os.path.abspath(results_save_path)}")
                
                # Save CSV if requested (path already determined above to ensure uniqueness)
                if csv_path:
                    simulation.save_statistics_csv(csv_path)
                    print(f"Statistics CSV saved to: {os.path.abspath(csv_path)}")
            except Exception as e:
                print(f"Warning: Failed to save statistics: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    # Configure simulation parameters using centralized config
    # All parameters can be configured in environment_simulation/configs/default.json
    # or by modifying DEFAULT_CONFIG programmatically
    config = DEFAULT_CONFIG
    
    # Override specific config values here if needed (or edit default.json)
    # config.w_h = 3.0  # Example: increase information gain weight
    # config.w_v = 1.0  # Example: event detection value weight
    # config.kernel_path = "learned_kernels/learned_kernel_1000_500.pkl"  # Enable kernel mode
    # config.event_utility = {0: 0.0, 1: 1.0, 2: 0.5}  # 3-state utilities for kernel mode
    
    animation_interval = 150  # Milliseconds between frames
    
    # Use kernel_path from config if not explicitly set
    kernel_path = config.kernel_path  # Can override here: kernel_path = "learned_kernels/learned_kernel_1000_500.pkl"
    
    # Create environment ONCE before the loop to record evolution
    # All agent types will replay the exact same sequence of environmental changes
    print("Creating and recording environment evolution for fair comparison...")
    recording_env = create_environment(config.grid_height, config.grid_width, kernel_path=kernel_path)
    
    # Record the environment evolution (this modifies recording_env in place)
    print(f"Recording {config.num_steps} steps of environment evolution...")
    replay_env = ReplayEnvironment.record_from(recording_env, num_steps=config.num_steps)
    
    print(f"Recorded environment evolution: {replay_env.max_time + 1} states "
          f"(initial + {config.num_steps} steps)")
    
    # Keep the original environment for transition kernel queries
    # Agents need access to the original environment's transition_probability
    # and _iter_neighbors methods for belief updates
    original_env = create_environment(config.grid_height, config.grid_width, kernel_path=kernel_path)
    
    # Create ONE timestamp folder for this entire run (all methods will use the same timestamp)
    timestamp_path = create_timestamp_folder()
    print(f"\nResults will be saved to: {os.path.abspath(timestamp_path)}")
    
    # Run simulation for each agent type with the SAME recorded evolution
    for agent_type in ["dsb_abba","greedy","monte_carlo", "sharing"]:
        print(f"\n{'='*70}")
        print(f"Running simulation for {agent_type.upper()} agents (replaying recorded evolution)")
        print(f"{'='*70}")
        
        # Reset replay environment to initial state
        replay_env.reset()
        
        # Run simulation with the replay environment
        # save_path=None will trigger automatic path generation in results/timestamp/method_name/
        # All methods use the same timestamp_path for this run
        main(
            config=config,
            animation_interval=animation_interval,
            save_path=None,  # Will be auto-generated in results/timestamp/method_name/
            save_csv=True,  # Also save results as CSV
            agent_type=agent_type,  # "monte_carlo", "random", or "greedy"
            env=replay_env,  # Use the replay environment (deterministic)
            original_env=original_env,  # Original env for transition kernel
            results_timestamp_path=timestamp_path,  # Use the same timestamp folder for all methods
            kernel_path=kernel_path,  # Pass kernel_path for kernel mode
        )

