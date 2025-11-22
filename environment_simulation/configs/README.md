# Configuration Files

This folder contains JSON configuration files for the simulation.

## Usage

To change simulation parameters, edit the JSON files in this folder. The main configuration file is `default.json`.

## File Structure

Each JSON file should have the following structure:

```json
{
  "simulation": {
    "grid_height": 30,
    "grid_width": 60,
    "orbit_period": 120,
    "field_of_regard_deg": 20.0,
    "communication_range_deg": 5.0,
    "num_rollouts": 30,
    "planning_horizon": 5,
    "discount_factor": 0.95,
    "num_ground_stations": 4,
    "min_satellites_per_station": 2,
    "num_steps": 100,
    "num_agents": 5,
    "use_parallel_planning": true
  },
  "agents": [
    {
      "name": "Aurora-1",
      "inclination_deg": 25.0,
      "phase_deg": 0.0,
      "latitude_offset": 0.0,
      "color": "#ffd166"
    }
  ]
}
```

## Parameters

### Simulation Parameters

- `grid_height`: Grid height (latitude resolution)
- `grid_width`: Grid width (longitude resolution)
- `orbit_period`: Number of steps in one complete orbit
- `field_of_regard_deg`: Field of regard in degrees
- `communication_range_deg`: Communication range for ground stations in degrees
- `num_rollouts`: Number of Monte Carlo rollouts per action
- `planning_horizon`: Planning horizon for Monte Carlo rollouts
- `discount_factor`: Discount factor for future rewards (0 < value <= 1)
- `num_ground_stations`: Number of ground stations to create
- `min_satellites_per_station`: Minimum number of satellites that must visit each station
- `num_steps`: Number of simulation steps
- `num_agents`: Number of agents (should match number of agents in agents array)
- `use_parallel_planning`: Enable parallel agent planning (true/false)

### Agent Parameters

Each agent in the `agents` array should have:

- `name`: Agent name (string)
- `inclination_deg`: Orbit inclination in degrees
- `phase_deg`: Orbit phase in degrees
- `latitude_offset`: Latitude offset in degrees
- `color`: Color for visualization (hex color code)

## Creating Custom Configurations

1. Copy `default.json` to a new file (e.g., `my_config.json`)
2. Modify the parameters as needed
3. Load it in your code: `config = SimulationConfig.from_json("configs/my_config.json")`

