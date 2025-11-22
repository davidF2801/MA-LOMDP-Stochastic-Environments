"""Ground station class for storing observations from satellites."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class GroundStation:
    """
    Ground station located at orbit intersection points.
    
    Stores observations received from satellites over time.
    Observations are stored as dictionaries mapping (lat, lon) -> observed state (0 or 1).
    """
    
    name: str
    latitude: float  # Latitude in degrees
    longitude: float  # Longitude in degrees
    # Observations: (agent_id, timestep) -> observation dictionary {(lat, lon): value}
    observations: Dict[Tuple[str, int], Dict[Tuple[float, float], int]] = field(
        default_factory=dict,
        init=False,
        repr=False
    )
    
    def __post_init__(self):
        """Initialize observations dictionary."""
        self.observations = {}
    
    def record_observation(
        self,
        agent_id: str,
        timestep: int,
        observation: Dict[Tuple[float, float], int],
    ) -> None:
        """
        Record an observation from a satellite at a specific timestep.
        
        Args:
            agent_id: Identifier of the satellite/agent making the observation
            timestep: Time step when the observation was made
            observation: Dictionary mapping (lat, lon) tuples to observed state values (0 or 1)
        """
        key = (agent_id, timestep)
        self.observations[key] = observation.copy()  # Store a copy to avoid reference issues
    
    def get_observation(
        self,
        agent_id: str,
        timestep: int,
    ) -> Dict[Tuple[float, float], int] | None:
        """
        Retrieve an observation from a specific satellite at a specific timestep.
        
        Args:
            agent_id: Identifier of the satellite/agent
            timestep: Time step of the observation
        
        Returns:
            Dictionary mapping (lat, lon) -> observed state, or None if not found
        """
        key = (agent_id, timestep)
        return self.observations.get(key)
    
    def get_all_observations_from_agent(
        self,
        agent_id: str,
    ) -> Dict[int, Dict[Tuple[float, float], int]]:
        """
        Get all observations from a specific satellite/agent across all timesteps.
        
        Args:
            agent_id: Identifier of the satellite/agent
        
        Returns:
            Dictionary mapping timestep -> observation dictionary
        """
        agent_observations = {}
        for (obs_agent_id, timestep), observation in self.observations.items():
            if obs_agent_id == agent_id:
                agent_observations[timestep] = observation
        return agent_observations
    
    def get_observations_at_timestep(
        self,
        timestep: int,
    ) -> Dict[str, Dict[Tuple[float, float], int]]:
        """
        Get all observations from all satellites at a specific timestep.
        
        Args:
            timestep: Time step of interest
        
        Returns:
            Dictionary mapping agent_id -> observation dictionary
        """
        timestep_observations = {}
        for (agent_id, obs_timestep), observation in self.observations.items():
            if obs_timestep == timestep:
                timestep_observations[agent_id] = observation
        return timestep_observations
    
    def has_observation(
        self,
        agent_id: str,
        timestep: int,
    ) -> bool:
        """
        Check if an observation exists from a specific satellite at a specific timestep.
        
        Args:
            agent_id: Identifier of the satellite/agent
            timestep: Time step of interest
        
        Returns:
            True if observation exists, False otherwise
        """
        key = (agent_id, timestep)
        return key in self.observations
    
    def clear_observations(self) -> None:
        """Clear all stored observations."""
        self.observations.clear()
    
    def get_all_observed_agents(self) -> set[str]:
        """
        Get set of all agent IDs that have made observations at this ground station.
        
        Returns:
            Set of agent IDs
        """
        return {agent_id for (agent_id, _) in self.observations.keys()}
    
    def get_all_observed_timesteps(self) -> set[int]:
        """
        Get set of all timesteps at which observations were made.
        
        Returns:
            Set of timesteps
        """
        return {timestep for (_, timestep) in self.observations.keys()}
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about observations stored at this ground station.
        
        Returns:
            Dictionary with statistics:
            - total_observations: Total number of observations
            - unique_agents: Number of unique agents
            - unique_timesteps: Number of unique timesteps
            - observations_per_agent: Dictionary mapping agent_id -> count
        """
        unique_agents = self.get_all_observed_agents()
        unique_timesteps = self.get_all_observed_timesteps()
        
        observations_per_agent = {}
        for agent_id in unique_agents:
            observations_per_agent[agent_id] = len(self.get_all_observations_from_agent(agent_id))
        
        return {
            "total_observations": len(self.observations),
            "unique_agents": len(unique_agents),
            "unique_timesteps": len(unique_timesteps),
            "observations_per_agent": observations_per_agent,
        }


def create_ground_stations_from_dicts(
    ground_station_dicts: List[Dict[str, Any]],
) -> List[GroundStation]:
    """
    Create GroundStation instances from ground station dictionaries.
    
    This function can be used to convert ground station dictionaries (as returned
    by find_ground_stations in visualize_orbits.py) into GroundStation objects.
    
    Args:
        ground_station_dicts: List of dictionaries with keys:
            - name: Ground station name
            - lat: Latitude in degrees
            - lon: Longitude in degrees
            - visited_by: (optional) List of satellite names that can visit this station
    
    Returns:
        List of GroundStation instances
    """
    ground_stations = []
    for gs_dict in ground_station_dicts:
        gs = GroundStation(
            name=gs_dict["name"],
            latitude=gs_dict["lat"],
            longitude=gs_dict["lon"],
        )
        ground_stations.append(gs)
    return ground_stations

