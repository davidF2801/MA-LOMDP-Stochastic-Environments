"""Main file for running graph-based Monte Carlo agents with visualization."""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set matplotlib backend to 'Agg' (non-interactive) to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')

import numpy as np

# Handle imports
try:
    if __package__:
        from .graph_environment import GraphEnvironment, RSPParams
        from .graph_replay_environment import GraphReplayEnvironment
        from .graph_agents.graph_monte_carlo_agent import GraphMonteCarloAgent
        from .graph_agents.graph_random_agent import GraphRandomAgent
        from .graph_agents.graph_greedy_agent import GraphGreedyAgent
        from .graph_agents.graph_sharing_monte_carlo_agent import GraphSharingMonteCarloAgent
        from .graph_agents.graph_belief import GraphBelief
        from .visualize_graph import GraphAnimator, plot_graph_state
        # CTDE imports (optional)
        try:
            from .graph_agents.graph_ctde_agent import GraphCTDEAgent
            from .graph_agents.graph_ctde_training import create_graph_ctde_agents, train_ctde_agents
            from .graph_agents.graph_ctde_trainer import CTDETrainer
            CTDE_AVAILABLE = True
        except ImportError:
            CTDE_AVAILABLE = False
            GraphCTDEAgent = None
            create_graph_ctde_agents = None
            train_ctde_agents = None
            CTDETrainer = None
    else:
        raise ImportError
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environment_simulation.graph_environment import GraphEnvironment, RSPParams
    from environment_simulation.graph_replay_environment import GraphReplayEnvironment
    from environment_simulation.graph_agents.graph_monte_carlo_agent import GraphMonteCarloAgent
    from environment_simulation.graph_agents.graph_random_agent import GraphRandomAgent
    from environment_simulation.graph_agents.graph_greedy_agent import GraphGreedyAgent
    from environment_simulation.graph_agents.graph_sharing_monte_carlo_agent import GraphSharingMonteCarloAgent
    from environment_simulation.graph_agents.graph_belief import GraphBelief
    from environment_simulation.visualize_graph import GraphAnimator, plot_graph_state
    # CTDE imports (optional)
    try:
        from environment_simulation.graph_agents.graph_ctde_agent import GraphCTDEAgent
        from environment_simulation.graph_agents.graph_ctde_training import create_graph_ctde_agents, train_ctde_agents
        from environment_simulation.graph_agents.graph_ctde_trainer import CTDETrainer
        CTDE_AVAILABLE = True
    except ImportError:
        CTDE_AVAILABLE = False
        GraphCTDEAgent = None
        create_graph_ctde_agents = None
        train_ctde_agents = None
        CTDETrainer = None


def create_graph_environment(
    num_nodes: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
    mode: str = "rsp",
    seed: int = 1,
) -> GraphEnvironment:
    """Create and configure the graph environment."""
    base_rng = np.random.default_rng(seed)
    
    # Infer grid dimensions if not provided
    if width is None or height is None:
        grid_size = int(np.sqrt(num_nodes))
        width = grid_size
        height = grid_size
    
    # Create parameter maps (2D grid format, will be flattened)
    if mode == "rsp":
        # Create clustered ignition map with slower evolution
        lam_map = 0.002 * np.ones((height, width))  # Reduced for slower evolution
        # Add some spatial variation
        y_coords, x_coords = np.mgrid[0:height, 0:width].astype(float)
        for center_y, center_x, amp, spread in [
            (height * 0.3, width * 0.2, 0.004, width * 0.18),  # Reduced
            (height * 0.7, width * 0.6, 0.003, width * 0.22),  # Reduced
            (height * 0.5, width * 0.45, 0.003, width * 0.16),  # Reduced
        ]:
            dist = (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2
            lam_map += amp * np.exp(-dist / spread)
        lam_map += 0.001 * base_rng.random((height, width))  # Reduced
        
        # Reduced alpha (contagion) for slower propagation
        alpha_map = 0.01 + 0.008 * np.cos(y_coords / height * np.pi) ** 4  # Reduced
        alpha_map += 0.004 * base_rng.random((height, width))  # Reduced
        
        # Reduced beta0 (spontaneous birth) for slower event creation
        beta0_map = 0.0003 + 0.0002 * base_rng.random((height, width))  # Reduced
        
        persistence_map = 0.94 + 0.04 * np.cos(y_coords / height * np.pi)  # Adjusted
        persistence_map += 0.01 * base_rng.random((height, width))  # Reduced variation
        persistence_map = np.clip(persistence_map, 0.90, 0.98)  # Adjusted range
        
        # Create blocked mask with more blocked cells that will never ignite
        blocked_mask = np.zeros((height, width), dtype=bool)
        # Add permanent blocked strip
        blocked_mask[:, : width // 18] = True
        # Add random blocked cells (increased from 0.03 to 0.08 for more blocked cells)
        blocked_mask |= base_rng.random((height, width)) < 0.08
        # Add some blocked regions (clusters)
        for center_y, center_x, radius in [
            (height * 0.25, width * 0.75, width * 0.12),
            (height * 0.75, width * 0.25, width * 0.10),
            (height * 0.6, width * 0.8, width * 0.08),
        ]:
            y_grid, x_grid = np.mgrid[0:height, 0:width]
            dist = np.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
            blocked_mask |= dist < radius
        
        # Reduced base RSP parameters for slower evolution
        env = GraphEnvironment(
            num_nodes=num_nodes,
            width=width,
            height=height,
            mode="rsp",
            rsp_params=RSPParams(lam=0.002, beta0=0.0003, alpha=0.01, delta=0.94),  # Reduced for slower evolution
            ignition_map=lam_map,
            alpha_map=alpha_map,
            beta0_map=beta0_map,
            persistence_map=persistence_map,
            blocked_mask=blocked_mask,
            seed=seed,
            connect_8_neighbors=True,  # 8-connected grid graph
        )
    else:  # dbn2 mode
        blocked_mask = None
        env = GraphEnvironment(
            num_nodes=num_nodes,
            width=width,
            height=height,
            mode="dbn2",
            birth_rate=0.02,
            death_rate=0.05,
            neighbor_influence=0.03,
            seed=seed,
            connect_8_neighbors=True,
        )
    
    return env


def create_graph_monte_carlo_agents(
    num_nodes: int,
    num_agents: int = 3,
    initial_nodes: Optional[list[int]] = None,
    num_rollouts: int = 50,
    planning_horizon: int = 5,
    w_h: float = 1.0,
    w_v: float = 1.0,
    discount_factor: float = 0.95,
    event_utility: Optional[dict[int, float]] = None,
    epsilon: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> list[GraphMonteCarloAgent]:
    """Create graph-based Monte Carlo agents."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    agents = []
    agent_names = [f"UAV_{i+1}" for i in range(num_agents)]
    colors = ["#ff5733", "#33ff57", "#3357ff", "#ff33f5", "#f5ff33"]
    
    # Initialize starting nodes
    if initial_nodes is None:
        initial_nodes = rng.choice(num_nodes, size=num_agents, replace=False).tolist()
    else:
        assert len(initial_nodes) == num_agents, "initial_nodes must have length num_agents"
    
    for i, (name, start_node) in enumerate(zip(agent_names, initial_nodes)):
        belief = GraphBelief(num_nodes=num_nodes, prior_probability=0.5)
        
        agent = GraphMonteCarloAgent(
            name=name,
            current_node=int(start_node),
            belief=belief,
            num_nodes=num_nodes,
            color=colors[i % len(colors)],
            num_rollouts=num_rollouts,
            w_h=w_h,
            w_v=w_v,
            discount_factor=discount_factor,
            event_utility=event_utility if event_utility is not None else {0: 0.0, 1: 1.0},
            epsilon=epsilon,
        )
        
        # Initialize belief for starting node (neighbors will be initialized during first observation)
        belief.initialize_nodes({start_node})
        
        agents.append(agent)
    
    return agents


def create_graph_random_agents(
    num_nodes: int,
    num_agents: int = 3,
    initial_nodes: Optional[list[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[GraphRandomAgent]:
    """Create graph-based random agents."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    agents = []
    agent_names = [f"UAV_{i+1}" for i in range(num_agents)]
    colors = ["#ff5733", "#33ff57", "#3357ff", "#ff33f5", "#f5ff33"]
    
    # Initialize starting nodes
    if initial_nodes is None:
        initial_nodes = rng.choice(num_nodes, size=num_agents, replace=False).tolist()
    else:
        assert len(initial_nodes) == num_agents, "initial_nodes must have length num_agents"
    
    for i, (name, start_node) in enumerate(zip(agent_names, initial_nodes)):
        belief = GraphBelief(num_nodes=num_nodes, prior_probability=0.5)
        
        agent = GraphRandomAgent(
            name=name,
            current_node=int(start_node),
            belief=belief,
            num_nodes=num_nodes,
            color=colors[i % len(colors)],
            seed=rng.integers(0, 2**31),  # Different seed for each agent
        )
        
        # Initialize belief for starting node
        belief.initialize_nodes({start_node})
        
        agents.append(agent)
    
    return agents


def create_graph_greedy_agents(
    num_nodes: int,
    num_agents: int = 3,
    initial_nodes: Optional[list[int]] = None,
    w_h: float = 1.0,
    w_v: float = 1.0,
    event_utility: Optional[dict[int, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[GraphGreedyAgent]:
    """Create graph-based greedy agents."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    agents = []
    agent_names = [f"UAV_{i+1}" for i in range(num_agents)]
    colors = ["#ff5733", "#33ff57", "#3357ff", "#ff33f5", "#f5ff33"]
    
    # Initialize starting nodes
    if initial_nodes is None:
        initial_nodes = rng.choice(num_nodes, size=num_agents, replace=False).tolist()
    else:
        assert len(initial_nodes) == num_agents, "initial_nodes must have length num_agents"
    
    for i, (name, start_node) in enumerate(zip(agent_names, initial_nodes)):
        belief = GraphBelief(num_nodes=num_nodes, prior_probability=0.5)
        
        agent = GraphGreedyAgent(
            name=name,
            current_node=int(start_node),
            belief=belief,
            num_nodes=num_nodes,
            color=colors[i % len(colors)],
            w_h=w_h,
            w_v=w_v,
            event_utility=event_utility if event_utility is not None else {0: 0.0, 1: 1.0},
        )
        
        # Initialize belief for starting node
        belief.initialize_nodes({start_node})
        
        agents.append(agent)
    
    return agents


def create_graph_sharing_monte_carlo_agents(
    num_nodes: int,
    num_agents: int = 3,
    initial_nodes: Optional[list[int]] = None,
    num_rollouts: int = 50,
    planning_horizon: int = 5,
    w_h: float = 1.0,
    w_v: float = 1.0,
    discount_factor: float = 0.95,
    event_utility: Optional[dict[int, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> list["GraphSharingMonteCarloAgent"]:
    """Create graph-based sharing Monte Carlo agents."""
    # GraphSharingMonteCarloAgent is imported at module level
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    agents = []
    agent_names = [f"UAV_{i+1}" for i in range(num_agents)]
    colors = ["#ff5733", "#33ff57", "#3357ff", "#ff33f5", "#f5ff33"]
    
    # Initialize starting nodes
    if initial_nodes is None:
        initial_nodes = rng.choice(num_nodes, size=num_agents, replace=False).tolist()
    else:
        assert len(initial_nodes) == num_agents, "initial_nodes must have length num_agents"
    
    for i, (name, start_node) in enumerate(zip(agent_names, initial_nodes)):
        belief = GraphBelief(num_nodes=num_nodes, prior_probability=0.5)
        
        agent = GraphSharingMonteCarloAgent(
            name=name,
            current_node=int(start_node),
            belief=belief,
            num_nodes=num_nodes,
            color=colors[i % len(colors)],
            num_rollouts=num_rollouts,
            w_h=w_h,
            w_v=w_v,
            discount_factor=discount_factor,
            event_utility=event_utility if event_utility is not None else {0: 0.0, 1: 1.0},
        )
        
        # Initialize belief for starting node (neighbors will be initialized during first observation)
        belief.initialize_nodes({start_node})
        
        agents.append(agent)
    
    return agents


class GraphSimulation:
    """Simulation runner for graph-based agents."""
    
    def __init__(
        self,
        env: GraphEnvironment,
        agents: list,
        planning_horizon: int = 5,
        use_parallel_planning: bool = True,
        transition_env: Optional[GraphEnvironment] = None,
    ):
        self.env = env  # Environment for state evolution (can be replay env)
        self.transition_env = transition_env if transition_env is not None else env  # Environment for transition kernel
        self.agents = agents
        self.planning_horizon = planning_horizon
        self.use_parallel_planning = use_parallel_planning
        
        # Track statistics for each agent
        self.agent_stats = {
            agent.name: {
                "events_observed": 0,  # Per-agent event observations
                "total_reward": 0.0,
                "observations_count": 0,
                "nodes_visited": set(),
            }
            for agent in agents
        }
        
        # Global event tracking
        # Track all events that have ever existed: event_id -> {node_id, start_step, end_step, first_detection_step, observed_by}
        # Each event instance gets a unique ID (node_id, start_step), even if multiple events occur at the same node
        self.event_events: dict[tuple[int, int], dict] = {}  # (node_id, start_step) -> event info
        
        # Multi-agent event tracking
        # Set of unique event IDs (node_id, start_step) representing unique events observed by ANY agent
        self.multi_agent_unique_events: set[tuple[int, int]] = set()
        
        # Multi-agent statistics
        self.multi_agent_stats = {
            "events_observed": 0,  # Total event observations (including reobservations)
            "unique_events_observed": 0,  # Number of unique events observed
            "reobservations": 0,  # Number of times an event was reobserved
            "avg_normalized_detection_delay": 0.0,  # Average normalized detection delay
        }
        
        # Track previous state to detect event start/end
        self.previous_state = self.env.get_state().copy()
        
        # Track agent positions over time
        self.agent_positions: dict[int, dict[str, int]] = {}  # step -> agent_name -> node
        
        # Debug tracking: track consecutive timesteps agents spend at same node
        # Format: {(agent1, agent2): {"node": node_id, "start_step": step, "consecutive_count": count}}
        # Keys are sorted agent name tuples to avoid duplicates (a1, a2) == (a2, a1)
        self._same_node_streaks: dict[tuple[str, str], dict] = {}
        
        # Threading support for parallel agent planning
        self.print_lock = threading.Lock()  # Lock for synchronized printing
    
    def step(self):
        """
        Execute one simulation step.
        
        For UAVs (action and observation are separate), the correct POMDP order is:
        1. Observe current node (where agent is from previous step) at t=k
        2. Update belief with observation
        3. Select action based on updated belief
        4. Evolve belief with transition model (to get belief at t=k+1)
        5. Execute action (move to new node)
        6. Environment evolves (env.step() to t=k+1)
        """
        step_before = self.env.time  # Current step (t=k)
        
        # STEP 1: Observe current node (where agents are from previous step)
        # STEP 2: Update belief with observations
        for agent in self.agents:
            # Observe the node the agent is currently at
            observed_state = agent.observe_current_node(self.env)
            observed_node = agent.current_node
            
            # Update agent statistics
            self.agent_stats[agent.name]["observations_count"] += 1
            if observed_state == 1:
                self.agent_stats[agent.name]["events_observed"] += 1
            
            # Update belief with observation
            agent.update_belief_from_observation(self.env, observed_node, observed_state)
            
            # Record observation for sharing agents
            if isinstance(agent, GraphSharingMonteCarloAgent):
                agent._record_own_observation(step_before, observed_node, observed_state)
            
            # Track events (for statistics)
            self._track_events(step_before, agent.name, observed_node, observed_state)
        
        # Handle communication for sharing agents (after observations are recorded)
        self._handle_agent_communication(step_before)
        
        # STEP 3: Select actions based on updated beliefs (ONLY selection, no execution yet)
        agent_actions = {}
        # Check if we have CTDE agents
        has_ctde_agents = CTDE_AVAILABLE and any(
            isinstance(agent, GraphCTDEAgent) for agent in self.agents
        )
        
        if has_ctde_agents:
            # CTDE agents: sequential (need access to other agents for peer info)
            for agent in self.agents:
                if isinstance(agent, GraphCTDEAgent):
                    # Update communication info before action selection
                    other_agents = [a for a in self.agents if a.name != agent.name]
                    agent.update_communication_info(other_agents, self.env, step_before)
                    # Use act_with_context
                    action_info = agent.act_with_context(self.env, other_agents, step_before)
                    agent_actions[agent.name] = action_info
                else:
                    # Non-CTDE agent mixed with CTDE agents
                    agent.compute_policy(self.env, self.planning_horizon)
                    action_info = agent.act(self.env)
                    agent_actions[agent.name] = action_info
        elif self.use_parallel_planning and len(self.agents) > 1:
            # Parallel planning: all agents plan simultaneously
            agent_tasks = [(agent, {}) for agent in self.agents]
            agent_results = {}
            agent_name_to_agent = {}
            
            with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                future_to_agent_name = {
                    executor.submit(self._agent_plan, agent, {}): agent.name
                    for agent, _ in agent_tasks
                }
                for agent, _ in agent_tasks:
                    agent_name_to_agent[agent.name] = agent
                
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
            
            for agent in sorted(self.agents, key=lambda a: a.name):
                agent_actions[agent.name] = agent_results[agent.name]
        else:
            # Sequential planning
            for agent in self.agents:
                agent.compute_policy(self.env, self.planning_horizon)
                action_info = agent.act(self.env)
                agent_actions[agent.name] = action_info
        
        # STEP 4: Evolve beliefs with transition model (after action selection, before execution)
        # This evolves beliefs from t=k to t=k+1
        for agent in self.agents:
            agent.evolve_belief_with_environment(self.transition_env, set())
        
        # STEP 5: Execute actions (move agents to target nodes)
        for agent in sorted(self.agents, key=lambda a: a.name):
            action_info = agent_actions[agent.name]
            self._process_agent_action(agent, step_before, action_info)
        
        # STEP 6: Environment evolves (increments env.time to t=k+1)
        self.env.step()
        step_after = self.env.time  # Step after env.step() (this is t=k+1)
        
        # Track event start/end from environment state transitions
        # Compares previous_state (state at step_before) with current_state (state at step_after)
        # to detect transitions that occurred during env.step()
        self._track_event_lifecycle(step_after)
        
        # Update previous state (after all processing)
        self.previous_state = self.env.get_state().copy()
        
        # Debug: Check for agents stuck at same node for multiple consecutive timesteps
        self._check_same_node_streaks(step_after)
    
    def _handle_agent_communication(self, step: int) -> None:
        """
        Handle peer-to-peer communication between sharing agents.
        
        Checks all pairs of agents and if they can communicate (same node or connected nodes),
        exchanges observations between them.
        
        Args:
            step: Current time step
        """
        # Find all sharing agents (GraphSharingMonteCarloAgent is imported at module level)
        sharing_agents = [
            agent for agent in self.agents
            if isinstance(agent, GraphSharingMonteCarloAgent)
        ]
        
        if len(sharing_agents) < 2:
            return  # Need at least 2 agents for communication
        
        # Check all pairs of sharing agents for communication
        # For each agent, check if it can communicate with any other agent
        for agent1 in sharing_agents:
            other_sharing_agents = [a for a in sharing_agents if a.name != agent1.name]
            if other_sharing_agents:
                agent1.communicate_with_agents(other_sharing_agents, self.env, step)
    
    def _step_parallel(self, step: int):
        """Execute one simulation step with parallel agent planning."""
        # Prepare action arguments for each agent
        agent_tasks = []
        for agent in self.agents:
            agent_tasks.append((agent, {}))
        
        # Execute agent planning in parallel
        agent_results = {}
        agent_name_to_agent = {}
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            # Submit all planning tasks
            future_to_agent_name = {
                executor.submit(self._agent_plan, agent, {}): agent.name
                for agent, _ in agent_tasks
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
        for agent in sorted(self.agents, key=lambda a: a.name):
            action_info = agent_results[agent.name]
            self._process_agent_action(agent, step, action_info)
    
    def _agent_plan(self, agent: "GraphAgent", kwargs: dict) -> dict:
        """Execute agent planning (runs in parallel thread)."""
        # Handle CTDE agents specially (they need other_agents and timestep)
        if CTDE_AVAILABLE and isinstance(agent, GraphCTDEAgent):
            # Get other agents and timestep from kwargs
            other_agents = kwargs.get('other_agents', [])
            timestep = kwargs.get('timestep', self.env.time)
            # CTDE agents use act_with_context
            action_info = agent.act_with_context(self.env, other_agents, timestep)
        else:
            # Standard agents
            agent.compute_policy(self.env, self.planning_horizon)
            action_info = agent.act(self.env)
        
        return action_info
    
    def _step_sequential(self, step: int):
        """Execute one simulation step with sequential agent planning."""
        # Each agent takes action and updates belief
        for agent in self.agents:
            # Agent computes policy
            agent.compute_policy(self.env, self.planning_horizon)
            
            # Agent chooses action (which node to move to)
            action_info = agent.act(self.env)
            
            self._process_agent_action(agent, step, action_info)
    
    def _process_agent_action(self, agent: "GraphAgent", step: int, action_info: dict):
        """
        Process agent action execution (shared by parallel and sequential paths).
        
        Note: Observations and belief updates now happen in step() before action selection.
        This method only executes the action (moves agent) and tracks statistics.
        """
        target_node = action_info["target_node"]
        expected_reward = action_info.get("expected_reward", 0.0)
        
        # Execute action: Move agent to target node
        agent.move_to_node(target_node, self.env)
        
        # Update agent statistics
        self.agent_stats[agent.name]["total_reward"] += expected_reward
        self.agent_stats[agent.name]["nodes_visited"].add(target_node)
        
        # Print action (with lock for parallel execution)
        with self.print_lock:
            # Get the state at the target node for display (but we already observed at current node)
            target_state = agent.observe_current_node(self.env)
            print(f"Step {step:3d} | {agent.name:15s} | "
                  f"Node {target_node:4d} | "
                  f"State {target_state} | "
                  f"Reward: {expected_reward:7.3f}")
        
        # Track positions
        if step not in self.agent_positions:
            self.agent_positions[step] = {}
        self.agent_positions[step][agent.name] = target_node
    
    def _track_event_lifecycle(self, step: int):
        """
        Track event start/end from environment state transitions.
        
        This runs AFTER env.step() to detect transitions properly.
        Compares previous_state (state at step t) with current_state (state at step t+1)
        to detect transitions that occurred during env.step().
        
        The step parameter is the new step (after env.step()), so events start/end at this step.
        
        Note: Events are only added to unique_events when they are first observed,
        not when they start in the environment.
        
        Each event instance gets a unique identifier (node_id, start_step), so multiple
        events at the same node (that start at different times) are tracked separately.
        """
        current_state = self.env.get_state()
        
        for node in range(self.env.num_nodes):
            prev_state = self.previous_state[node]
            curr_state = current_state[node]
            
            # Event start: transition from 0 to 1
            # This transition happened DURING env.step(), so the event started
            # at the step AFTER env.step() (the current step)
            if prev_state == 0 and curr_state == 1:
                # Create unique event ID: (node_id, start_step)
                event_id = (node, step)
                
                # Check if an active event already exists at this node
                # (shouldn't happen, but handle it by creating new event instance)
                if event_id not in self.event_events:
                    # New event instance - track its actual start time
                    # The event started at the current step (after env.step())
                    self.event_events[event_id] = {
                        "node_id": node,
                        "start_step": step,  # Event starts at this step (after env.step())
                        "end_step": None,
                        "first_detection_step": None,  # When first observed by any agent
                        "observed_by": set(),
                        "observation_count": 0,  # Track total observations of this event
                    }
                    # NOTE: We do NOT add to multi_agent_unique_events here
                    # Events are only counted as "unique events observed" when first observed
            
            # Event end: transition from 1 to 0
            # This transition happened DURING env.step(), so the event ended
            # at the step AFTER env.step() (the current step)
            if prev_state == 1 and curr_state == 0:
                # Find the active event at this node (the one that hasn't ended yet)
                # Look for events at this node that don't have an end_step set
                for event_id, event_info in self.event_events.items():
                    if event_info["node_id"] == node and event_info["end_step"] is None:
                        event_info["end_step"] = step
                        break  # Only mark the first active event as ended
    
    def _track_events(self, step: int, agent_name: str, node: int, state: int):
        """
        Track event observations by agents.
        
        Counting logic:
        - events_observed: Total number of times any agent observed state==1 (any node)
        - unique_events_observed: Number of unique event instances observed at least once
          (each event instance is uniquely identified by (node_id, start_step))
        - reobservations: Number of observations beyond the first observation of each unique event
          (i.e., events_observed - unique_events_observed)
        
        Handles the case where multiple events occur at the same node by tracking each
        event instance separately using (node_id, start_step) as the unique identifier.
        """
        # Track observations - count every time an agent observes state==1
        if state == 1:
            # Always increment total observations when state==1 is observed
            self.multi_agent_stats["events_observed"] += 1
            
            # Find the active event at this node
            # An active event is one that:
            # 1. Is at this node
            # 2. Has started (start_step <= step)
            # 3. Hasn't ended yet (end_step is None or end_step >= step)
            active_event_id = None
            for event_id, event_info in self.event_events.items():
                if (event_info["node_id"] == node and 
                    event_info["start_step"] <= step and
                    (event_info["end_step"] is None or event_info["end_step"] >= step)):
                    active_event_id = event_id
                    break
            
            # If no active event found, create one (shouldn't happen normally, but handle it)
            if active_event_id is None:
                # Create event with approximate start time (current step or earlier)
                active_event_id = (node, step)
                self.event_events[active_event_id] = {
                    "node_id": node,
                    "start_step": step,  # Approximate start (when first observed)
                    "end_step": None,
                    "first_detection_step": None,
                    "observed_by": set(),
                    "observation_count": 0,
                }
            
            # Increment observation count for this event instance
            event_info = self.event_events[active_event_id]
            event_info["observation_count"] = event_info.get("observation_count", 0) + 1
            
            # Track first detection time (when first observed by ANY agent)
            is_first_observation = event_info["first_detection_step"] is None
            if is_first_observation:
                event_info["first_detection_step"] = step
                # Add to unique events when first observed by ANY agent
                # Use unique event ID (node_id, start_step) to distinguish multiple events at same node
                self.multi_agent_unique_events.add(active_event_id)
            
            # Track which agents have observed this event
            event_info["observed_by"].add(agent_name)
        
        # Update counts: reobservations = total observations - unique events
        # Each unique event has at least 1 observation, so reobservations = observations beyond the first
        self.multi_agent_stats["unique_events_observed"] = len(self.multi_agent_unique_events)
        self.multi_agent_stats["reobservations"] = self.multi_agent_stats["events_observed"] - self.multi_agent_stats["unique_events_observed"]
    
    def _check_same_node_streaks(self, step: int):
        """
        Debug method: Check if any pairs of agents have been at the same node
        for more than 3 consecutive timesteps.
        
        Args:
            step: Current time step
        """
        if step not in self.agent_positions:
            return
        
        current_positions = self.agent_positions[step]
        
        # Get all pairs of agents
        agent_names = list(current_positions.keys())
        
        # Check each pair
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                agent1_name = agent_names[i]
                agent2_name = agent_names[j]
                
                # Create sorted tuple key to avoid duplicates
                pair_key = tuple(sorted([agent1_name, agent2_name]))
                
                node1 = current_positions[agent1_name]
                node2 = current_positions[agent2_name]
                
                # Check if they're at the same node
                if node1 == node2:
                    # They're at the same node
                    if pair_key in self._same_node_streaks:
                        # Continue existing streak
                        streak_info = self._same_node_streaks[pair_key]
                        if streak_info["node"] == node1:
                            # Same node as before, increment count
                            streak_info["consecutive_count"] += 1
                        else:
                            # Different node, reset streak
                            streak_info["node"] = node1
                            streak_info["start_step"] = step
                            streak_info["consecutive_count"] = 1
                    else:
                        # New streak
                        self._same_node_streaks[pair_key] = {
                            "node": node1,
                            "start_step": step,
                            "consecutive_count": 1,
                        }
                    
                    # Check if streak is > 3 timesteps (breakpoint here for debugging)
                    streak_info = self._same_node_streaks[pair_key]
                    if streak_info["consecutive_count"] > 3:
                        # Breakpoint location: agents stuck at same node for > 3 timesteps
                        # You can inspect:
                        # - agent1_name, agent2_name: names of agents
                        # - node1: node they're stuck at
                        # - streak_info["consecutive_count"]: how many timesteps
                        # - step: current timestep
                        pass  # Place VS Code breakpoint here
                else:
                    # They're not at the same node, reset streak if it exists
                    if pair_key in self._same_node_streaks:
                        # Only reset if streak was > 1 (don't spam for single-step collisions)
                        if self._same_node_streaks[pair_key]["consecutive_count"] > 1:
                            del self._same_node_streaks[pair_key]
    
    def run(self, num_steps: int):
        """Run simulation for specified number of steps."""
        print(f"\n{'='*80}")
        print(f"Starting graph-based simulation with {len(self.agents)} agents")
        print(f"Environment: {self.env.num_nodes} nodes, mode={self.env.mode}")
        print(f"{'='*80}\n")
        
        for step in range(num_steps):
            self.step()
        
        print(f"\n{'='*80}")
        print("Simulation complete")
        print(f"{'='*80}\n")
    
    def _calculate_normalized_detection_delay(self) -> float:
        """
        Calculate average normalized detection delay across all events.
        
        Normalized delay = (first_detection_step - event_start_step) / (event_end_step - event_start_step)
        
        This measures: timesteps between event start and detection / total timesteps event lasted
        
        Formula: (detection_step - start_step) / (end_step - start_step)
        
        Only includes events that:
        - Have been detected (first_detection_step is not None)
        - Have ended (end_step is not None)
        - Detection happened before or at end (first_detection <= end_step)
        - Have a valid lifetime (end_step > start_step)
        """
        delays = []
        
        # event_events now uses (node_id, start_step) tuples as keys
        for event_id, event_info in self.event_events.items():
            start_step = event_info.get("start_step")
            end_step = event_info.get("end_step")
            first_detection = event_info.get("first_detection_step")
            
            # Only calculate for events that:
            # 1. Were detected (first_detection_step is not None)
            # 2. Have ended (end_step is not None)
            # 3. Have valid start step (not None)
            # 4. Have valid numeric values
            # 5. Detection happened at or after start (first_detection >= start_step)
            # 6. Detection happened before or at end (first_detection <= end_step)
            # 7. Have a valid lifetime (end_step > start_step)
            if (first_detection is not None and 
                end_step is not None and 
                start_step is not None and
                isinstance(first_detection, (int, np.integer)) and
                isinstance(end_step, (int, np.integer)) and
                isinstance(start_step, (int, np.integer)) and
                first_detection >= start_step and  # Detection must be at or after start
                first_detection <= end_step and  # Detection must be at or before end
                end_step > start_step):
                
                # Calculate: timesteps between event start and detection
                timesteps_to_detection = int(first_detection) - int(start_step)
                
                # Calculate: total timesteps the event lasted
                total_timesteps_event_lasted = int(end_step) - int(start_step)
                
                # Normalized delay = timesteps to detection / total timesteps lasted
                # Formula: (detection_step - start_step) / (end_step - start_step)
                # Should be between 0 (detected immediately at start) and 1 (detected at end)
                if total_timesteps_event_lasted > 0:
                    normalized_delay = float(timesteps_to_detection) / float(total_timesteps_event_lasted)
                    # Ensure normalized delay is in [0, 1] range (safety check)
                    normalized_delay = max(0.0, min(1.0, normalized_delay))
                    delays.append(normalized_delay)
        
        if len(delays) == 0:
            return 0.0
        
        return float(np.mean(delays))
    
    def get_statistics(self) -> dict:
        """Get simulation statistics."""
        # Calculate normalized detection delay
        avg_delay = self._calculate_normalized_detection_delay()
        self.multi_agent_stats["avg_normalized_detection_delay"] = avg_delay
        
        stats = {
            "agent_stats": {},
            "multi_agent_stats": self.multi_agent_stats.copy(),
            "environment_stats": {
                "total_active": int(self.env.count_active()),
                "time": self.env.time,
            },
        }
        
        for agent_name, agent_stat in self.agent_stats.items():
            stats["agent_stats"][agent_name] = {
                "events_observed": agent_stat["events_observed"],
                "total_reward": agent_stat["total_reward"],
                "observations_count": agent_stat["observations_count"],
                "unique_nodes_visited": len(agent_stat["nodes_visited"]),
            }
        
        return stats
    
    def print_statistics(self):
        """Print simulation statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("SIMULATION STATISTICS")
        print("="*80)
        
        print("\nAgent Statistics:")
        for agent_name, agent_stat in stats["agent_stats"].items():
            print(f"  {agent_name}:")
            print(f"    Events observed: {agent_stat['events_observed']}")
            print(f"    Total reward: {agent_stat['total_reward']:.3f}")
            print(f"    Observations: {agent_stat['observations_count']}")
            print(f"    Unique nodes visited: {agent_stat['unique_nodes_visited']}")
        
        print("\nMulti-Agent Statistics:")
        print(f"  Total events observed: {stats['multi_agent_stats']['events_observed']}")
        print(f"  Unique events observed: {stats['multi_agent_stats']['unique_events_observed']}")
        print(f"  Reobservations: {stats['multi_agent_stats']['reobservations']}")
        print(f"  Avg normalized detection delay: {stats['multi_agent_stats']['avg_normalized_detection_delay']:.4f}")
        
        print("\nEnvironment Statistics:")
        print(f"  Current active events: {stats['environment_stats']['total_active']}")
        print(f"  Time steps: {stats['environment_stats']['time']}")
        print("="*80 + "\n")
    
    def save_statistics(self, filepath: str):
        """
        Save statistics to a JSON file.
        
        Args:
            filepath: Path to save the statistics JSON file
        """
        import json
        
        stats = self.get_statistics()
        
        # Convert sets and numpy types to JSON-serializable formats
        json_stats = {
            "agent_stats": {},
            "multi_agent_stats": {
                "events_observed": int(stats["multi_agent_stats"]["events_observed"]),
                "unique_events_observed": int(stats["multi_agent_stats"]["unique_events_observed"]),
                "reobservations": int(stats["multi_agent_stats"]["reobservations"]),
                "avg_normalized_detection_delay": float(stats["multi_agent_stats"]["avg_normalized_detection_delay"]),
            },
            "environment_stats": {
                "total_active": int(stats["environment_stats"]["total_active"]),
                "time": int(stats["environment_stats"]["time"]),
                "num_nodes": int(self.env.num_nodes),
            },
        }
        
        for agent_name, agent_stat in stats["agent_stats"].items():
            json_stats["agent_stats"][agent_name] = {
                "events_observed": int(agent_stat["events_observed"]),
                "total_reward": float(agent_stat["total_reward"]),
                "observations_count": int(agent_stat["observations_count"]),
                "unique_nodes_visited": int(agent_stat["unique_nodes_visited"]),
            }
        
        with open(filepath, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        print(f"Statistics saved to: {filepath}")
    
    def save_statistics_csv(self, filepath: str):
        """
        Save statistics to a CSV file.
        
        Args:
            filepath: Path to save the statistics CSV file
        """
        import csv
        
        stats = self.get_statistics()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write multi-agent statistics first
            writer.writerow(['MULTI-AGENT METRICS'])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Events Observed', stats['multi_agent_stats']['events_observed']])
            writer.writerow(['Unique Events Observed', stats['multi_agent_stats']['unique_events_observed']])
            writer.writerow(['Reobservations', stats['multi_agent_stats']['reobservations']])
            writer.writerow(['Avg Normalized Detection Delay', f"{stats['multi_agent_stats']['avg_normalized_detection_delay']:.4f}"])
            writer.writerow([])  # Empty row separator
            
            # Write per-agent statistics
            writer.writerow(['PER-AGENT METRICS'])
            writer.writerow([
                'Agent', 'Events Observed', 'Total Reward',
                'Observations Count', 'Unique Nodes Visited', 'Avg Reward per Observation'
            ])
            # Write data
            for agent_name, agent_stat in stats["agent_stats"].items():
                avg_reward = (agent_stat["total_reward"] / agent_stat["observations_count"]
                             if agent_stat["observations_count"] > 0 else 0.0)
                writer.writerow([
                    agent_name,
                    agent_stat["events_observed"],
                    f"{agent_stat['total_reward']:.4f}",
                    agent_stat["observations_count"],
                    agent_stat["unique_nodes_visited"],
                    f"{avg_reward:.4f}",
                ])
            # Write simulation summary
            writer.writerow([])
            writer.writerow(['Simulation Summary'])
            writer.writerow(['Total Steps', stats['environment_stats']['time']])
            writer.writerow(['Planning Horizon', self.planning_horizon])
            writer.writerow(['Number of Nodes', self.env.num_nodes])
            writer.writerow(['Number of Agents', len(self.agents)])
        
        print(f"Statistics saved to CSV: {filepath}")


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


def save_config_file(
    timestamp_path: str,
    num_nodes: int,
    num_agents: int,
    num_steps: int,
    planning_horizon: int,
    num_rollouts: int,
    mode: str,
    seed: int,
    animation_interval: int,
    w_h: float,
    w_v: float,
    event_utility: dict[int, float],
    epsilon: float,
    agent_types: list[str],
) -> str:
    """
    Save simulation configuration to a markdown file in the timestamp folder.
    
    Args:
        timestamp_path: Path to the timestamp folder
        num_nodes: Number of nodes in the graph
        num_agents: Number of agents
        num_steps: Number of simulation steps
        planning_horizon: Planning horizon for Monte Carlo agents
        num_rollouts: Number of rollouts for Monte Carlo agents
        mode: Environment mode ('rsp' or 'dbn2')
        seed: Random seed
        animation_interval: Animation interval in milliseconds
        w_h: Weight for information gain in reward calculation
        w_v: Weight for event value in reward calculation
        event_utility: Event utility mapping {state: utility}
        epsilon: Epsilon-greedy exploration probability
        agent_types: List of agent types run in this batch
        
    Returns:
        Path to the saved config file
    """
    config_path = os.path.join(timestamp_path, "config.md")
    
    # Format event_utility as a readable string
    event_utility_str = ", ".join([f"{state}: {utility}" for state, utility in sorted(event_utility.items())])
    
    config_content = f"""# Simulation Configuration

This file contains the configuration parameters used for this simulation run.

## General Parameters

- **Number of Nodes**: {num_nodes}
- **Number of Agents**: {num_agents}
- **Number of Steps**: {num_steps}
- **Random Seed**: {seed}
- **Environment Mode**: {mode}

## Agent Parameters

- **Planning Horizon**: {planning_horizon}
- **Number of Rollouts** (Monte Carlo agents): {num_rollouts}
- **Epsilon-Greedy Exploration Probability**: {epsilon}

## Reward Function Parameters

- **Weight for Information Gain (w_h)**: {w_h}
- **Weight for Event Value (w_v)**: {w_v}
- **Event Utility Mapping**: {{{event_utility_str}}}

## Visualization Parameters

- **Animation Interval**: {animation_interval} ms

## Agent Types

The following agent types were run in this batch:
{chr(10).join(f"- {agent_type}" for agent_type in agent_types)}

## Notes

- All agent types use the same recorded environment evolution for fair comparison.
- The environment evolution is recorded once and then replayed for each agent type.
- Results are saved in subdirectories: `{timestamp_path}/<agent_type>/`
"""
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    return config_path


def main(
    num_nodes: int = 100,
    num_agents: int = 4,
    num_steps: int = 20,
    planning_horizon: int = 5,
    num_rollouts: int = 50,
    mode: str = "rsp",
    seed: int = 42,
    agent_type: str = "monte_carlo",  # "monte_carlo", "random", or "greedy"
    save_animation: bool = True,
    animation_path: Optional[str] = None,
    animation_interval: int = 200,
    animation_fps: Optional[int] = None,
    env: Optional[GraphEnvironment] = None,  # Optional pre-created environment (can be GraphReplayEnvironment)
    original_env: Optional[GraphEnvironment] = None,  # Original env for transition kernel (for replay envs)
    results_timestamp_path: Optional[str] = None,  # Optional timestamp folder path (if None, creates new one)
    results_save_path: Optional[str] = None,  # Optional path to save statistics JSON
    save_csv: bool = False,  # If True, also save statistics as CSV
    w_h: float = 1.0,  # Weight for information gain (horizon) in reward calculation
    w_v: float = 1.0,  # Weight for event value in reward calculation
    event_utility: Optional[dict[int, float]] = None,  # Event utility mapping {state: utility}
    epsilon: float = 0.0,  # Epsilon-greedy exploration probability for Monte Carlo agents
):
    """
    Main function to run graph-based simulation with visualization.
    
    Args:
        num_nodes: Number of nodes in the graph
        num_agents: Number of agents
        num_steps: Number of simulation steps
        planning_horizon: Planning horizon for agents (only used for Monte Carlo agents)
        num_rollouts: Number of Monte Carlo rollouts per action (only used for Monte Carlo agents)
        mode: Environment mode ('rsp' or 'dbn2')
        seed: Random seed
        agent_type: Type of agents to use ('monte_carlo', 'random', 'greedy', or 'sharing')
        save_animation: Whether to save animation
        animation_path: Path to save animation (if None, auto-generates)
        animation_interval: Animation interval in milliseconds
        animation_fps: Frames per second for saved animation
        env: Optional pre-created environment (can be GraphReplayEnvironment for deterministic replay).
             If None, creates a new environment.
        original_env: Original GraphEnvironment instance (for GraphReplayEnvironment cases).
                      Used for transition kernel queries during belief updates.
                      If None and env is a replay environment, creates a new one.
        results_timestamp_path: Optional timestamp folder path for saving results.
                               If None, auto-generates a new timestamp folder.
        results_save_path: Optional path to save simulation results/statistics (JSON format).
                          If None, results are printed but not saved.
                          If not specified but animation_path is provided, defaults to same name with _results.json extension
        save_csv: If True, also save statistics as CSV file
        w_h: Weight for information gain (horizon) in reward calculation (used for monte_carlo, greedy, sharing agents)
        w_v: Weight for event value in reward calculation (used for monte_carlo, greedy, sharing agents)
        event_utility: Event utility mapping dict {state: utility}, e.g., {0: 0.0, 1: 1.0} (used for monte_carlo, greedy, sharing agents)
    """
    # Create or use provided environment
    if env is None:
        env = create_graph_environment(
            num_nodes=num_nodes,
            mode=mode,
            seed=seed,
        )
        # Initialize environment with more initial events for better propagation
        initial_events_count = max(10, int(num_nodes * 0.1))  # 10% of nodes or at least 10
        env.reset(initial_events=initial_events_count)
    else:
        # Use provided environment (e.g., GraphReplayEnvironment)
        # Make sure it's reset to initial state
        initial_events_count = max(10, int(num_nodes * 0.1))  # 10% of nodes or at least 10
        env.reset(initial_events=initial_events_count)
    
    # For replay environments, we need the original environment for transition kernel
    # If original_env is not provided and we're using a replay env, create one
    transition_env = env
    if isinstance(env, GraphReplayEnvironment):
        if original_env is None:
            # Create original environment for transition kernel queries
            original_env = create_graph_environment(
                num_nodes=num_nodes,
                mode=mode,
                seed=seed,
            )
        transition_env = original_env
    
    # Create agents based on agent_type
    # IMPORTANT: Use a different seed for agent creation to ensure agent randomness
    # is independent of environment randomness, even if environment seed is the same
    # Agents should always have their own randomness, regardless of environment seed
    import time
    agent_seed = int(time.time() * 1000000) % (2**31)  # Independent random seed for agents
    
    if agent_type == "monte_carlo":
        agents = create_graph_monte_carlo_agents(
            num_nodes=num_nodes,
            num_agents=num_agents,
            num_rollouts=num_rollouts,
            planning_horizon=planning_horizon,
            w_h=w_h,
            w_v=w_v,
            discount_factor=0.95,
            event_utility=event_utility,
            epsilon=epsilon,
            rng=np.random.default_rng(agent_seed),  # Use independent seed for agents
        )
    elif agent_type == "random":
        agents = create_graph_random_agents(
            num_nodes=num_nodes,
            num_agents=num_agents,
            rng=np.random.default_rng(agent_seed),  # Use independent seed for agents
        )
    elif agent_type == "greedy":
        agents = create_graph_greedy_agents(
            num_nodes=num_nodes,
            num_agents=num_agents,
            w_h=w_h,
            w_v=w_v,
            event_utility=event_utility,
            rng=np.random.default_rng(agent_seed),  # Use independent seed for agents
        )
    elif agent_type == "sharing":
        agents = create_graph_sharing_monte_carlo_agents(
            num_nodes=num_nodes,
            num_agents=num_agents,
            num_rollouts=num_rollouts,
            planning_horizon=planning_horizon,
            w_h=w_h,
            w_v=w_v,
            discount_factor=0.95,
            event_utility=event_utility,
            rng=np.random.default_rng(agent_seed),  # Use independent seed for agents
        )
    elif agent_type == "ctde":
        if not CTDE_AVAILABLE:
            raise ImportError(
                "CTDE agents require PyTorch. Install with: pip install torch\n"
                "CTDE agents are not available without PyTorch."
            )
        # CTDE agents require training first before execution
        print("\n" + "="*70)
        print("CTDE AGENTS: Training Phase")
        print("="*70)
        
        try:
            # Create training environment (separate from evaluation environment)
            # Handle None seed case
            training_env_seed = (seed + 10000) if seed is not None else None
            if training_env_seed is None:
                import time
                training_env_seed = int(time.time() * 1000000) % (2**31)
            
            training_env = create_graph_environment(
                num_nodes=num_nodes,
                mode=mode,
                seed=training_env_seed,  # Use different seed for training
            )
            training_env.reset(initial_events=5)
            
            # Create agents with shared networks
            agents = create_graph_ctde_agents(
                num_nodes=num_nodes,
                num_agents=num_agents,
                rng=np.random.default_rng(agent_seed),
            )
            
            # Set agents to training mode
            for agent in agents:
                agent.training_mode = True
                agent.deterministic_execution = False  # Sample during training
            
            # Create trainer
            # train_ctde_agents and CTDETrainer should already be imported at module level
            # But verify they're available (they should be if CTDE_AVAILABLE is True)
            if not CTDE_AVAILABLE or train_ctde_agents is None or CTDETrainer is None:
                raise ImportError("CTDE training functions not available. Check imports.")
            
            trainer = CTDETrainer(
                actor_network=agents[0].actor_network,
                critic_network=agents[0].critic_network,
                actor_lr=3e-4,
                critic_lr=3e-4,
                gamma=0.95,
                lambda_gae=0.95,
                clip_epsilon=0.2,
                entropy_coef=0.01,
            )
            
            # Train agents
            print(f"Training CTDE agents for 50 episodes ({num_steps} steps per episode)...")
            print(f"Update frequency: every 5 episodes, {5} update iterations per update")
            training_stats = train_ctde_agents(
                env=training_env,
                agents=agents,
                trainer=trainer,
                num_episodes=50,  # Training episodes
                steps_per_episode=num_steps,  # Same as evaluation
                update_frequency=5,  # Update every 5 episodes
                num_updates=5,  # 5 update iterations per update
                w_h=w_h,
                w_v=w_v,
                event_utility=event_utility,
                verbose=True,
            )
            
            print(f"\n" + "="*70)
            print("Training completed!")
            print("="*70)
            if len(training_stats['episode_rewards']) >= 10:
                avg_last_10 = np.mean(training_stats['episode_rewards'][-10:])
                print(f"Average episode reward (last 10 episodes): {avg_last_10:.2f}")
            if len(training_stats['episode_rewards']) > 0:
                print(f"Final episode reward: {training_stats['episode_rewards'][-1]:.2f}")
                print(f"Best episode reward: {max(training_stats['episode_rewards']):.2f}")
                if len(training_stats['policy_losses']) > 0:
                    print(f"Final policy loss: {training_stats['policy_losses'][-1]:.4f}")
                if len(training_stats['value_losses']) > 0:
                    print(f"Final value loss: {training_stats['value_losses'][-1]:.4f}")
            print("="*70)
            print("\nCTDE AGENTS: Execution Phase (using trained networks)")
            print("="*70)
            
            # Reset agent beliefs for execution (training may have modified them)
            for agent in agents:
                agent.belief.reset()
            
            # Set execution mode (deterministic argmax)
            for agent in agents:
                agent.training_mode = False
                agent.deterministic_execution = True
                
        except Exception as e:
            print(f"\n" + "="*70)
            print("FATAL ERROR during CTDE training!")
            print("="*70)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("="*70)
            print("\nTraining failed. Exiting...")
            raise RuntimeError(f"CTDE training failed: {e}") from e
    else:
        raise ValueError(
            f"Unknown agent_type: {agent_type}. "
            f"Must be 'monte_carlo', 'random', 'greedy', 'sharing', or 'ctde'"
        )
    
    # Create simulation
    sim = GraphSimulation(
        env=env,
        agents=agents,
        planning_horizon=planning_horizon,
        use_parallel_planning=True,  # Enable parallel planning for speed
        transition_env=transition_env,  # Use transition_env for belief updates (for replay envs)
    )
    
    # Create animator
    print("\nCreating graph animator...")
    animation_saved = False
    animation_path_used = None
    results_base_path = None
    csv_path = None  # Initialize CSV path variable
    
    # Create output directory early if saving animation
    if save_animation:
        if animation_path is None:
            # Auto-generate animation path based on agent type
            if results_timestamp_path is None:
                timestamp_dir = create_timestamp_folder()
            else:
                timestamp_dir = results_timestamp_path
                os.makedirs(timestamp_dir, exist_ok=True)
            
            # Create subdirectory for this agent type: results/timestamp/agent_type/
            agent_type_dir = os.path.join(timestamp_dir, agent_type)
            os.makedirs(agent_type_dir, exist_ok=True)
            animation_path = os.path.join(agent_type_dir, "animation_graph.gif")
            print(f"Created results directory: {agent_type_dir}")
        
        animation_path_used = os.path.abspath(animation_path)
        results_base_path = os.path.dirname(animation_path_used) if animation_path_used else None
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(animation_path_used), exist_ok=True)
        print(f"Will save animation to: {animation_path_used}")
        
        # Determine results save paths based on animation path
        if results_save_path is None and animation_path_used:
            # Auto-generate results path from animation path
            base_name = os.path.splitext(animation_path_used)[0]
            results_save_path = f"{base_name}_results.json"
        elif results_save_path and results_base_path:
            # If results_save_path is provided, place it in the results directory
            results_filename = os.path.basename(results_save_path)
            results_save_path = os.path.join(results_base_path, results_filename)
        
        # Determine CSV path
        if save_csv:
            if results_save_path:
                # Derive CSV path from JSON path (most reliable)
                csv_path = os.path.splitext(results_save_path)[0] + ".csv"
            elif animation_path_used:
                # Derive CSV path directly from animation path
                base_name = os.path.splitext(animation_path_used)[0]
                csv_path = f"{base_name}_results.csv"
            elif results_base_path:
                # Fallback: use agent type in filename
                csv_path = os.path.join(results_base_path, f"simulation_results_{agent_type}.csv")
        
        # Verify directory was created
        if not os.path.exists(os.path.dirname(animation_path_used)):
            print(f"✗ ERROR: Failed to create directory: {os.path.dirname(animation_path_used)}")
            print("Running simulation without visualization...")
            sim.run(num_steps)
            sim.print_statistics()
            return sim, env, agents
    
    try:
        # Calculate adaptive visualization parameters based on environment
        # Node size should scale with number of nodes and grid size
        # Check if env has width/height attributes (GraphEnvironment or GraphReplayEnvironment)
        has_width_height = hasattr(env, 'width') and hasattr(env, 'height') and env.width and env.height
        if has_width_height:
            # For grid-based graphs, scale node size based on grid dimensions
            base_node_size = 500
            # Scale down for larger grids
            scale_factor = min(1.0, 100.0 / max(env.width, env.height))
            node_size = max(50, int(base_node_size * scale_factor))
            
            # Adjust figure size based on grid aspect ratio
            aspect_ratio = env.width / env.height if env.height > 0 else 1.0
            base_fig_width = 14
            base_fig_height = 10
            if aspect_ratio > 1.0:
                # Wider than tall
                figsize = (base_fig_width, base_fig_height / aspect_ratio)
            else:
                # Taller than wide
                figsize = (base_fig_width * aspect_ratio, base_fig_height)
        else:
            # For non-grid graphs, use adaptive sizing based on number of nodes
            node_size = max(50, min(300, int(5000 / env.num_nodes)))
            figsize = (14, 10)
        
        # Edge width can be thinner for denser graphs
        edge_width = max(0.3, min(1.0, 5.0 / (env.num_nodes ** 0.3)))
        
        print(f"Visualization parameters: node_size={node_size}, figsize={figsize}, edge_width={edge_width:.2f}")
        
        animator = GraphAnimator(
            env=env,
            agents=agents,
            fire_color="#d1495b",
            calm_color="#e0e0e0",
            node_size=node_size,
            edge_width=edge_width,
            figsize=figsize,
        )
        
        # Custom update function that includes simulation stepping
        # Frame 0: Show initial state, then step to get agents' initial actions
        # Frame 1+: Step simulation (env.step() + agents act), then visualize
        original_update = animator._update
        
        def custom_update(frame: int):
            # Step simulation before visualization
            # Frame 0: Show initial state, agents act at step 0
            # Frame 1+: Step simulation (env.step() + agents act), then visualize
            # But we can only step num_steps times, so frame num_steps would try to step beyond recorded time
            # So we only step if frame < num_steps
            if frame > 0 and frame < num_steps:
                try:
                    # Step the simulation (env.step() + agents act)
                    sim.step()
                except (IndexError, ValueError, AttributeError) as e:
                    # Skip stepping if at end or error
                    pass
            # For frame 0, we also need to step to get initial actions
            # (agents act at step 0, which is stored correctly now)
            elif frame == 0:
                try:
                    sim.step()
                except (IndexError, ValueError, AttributeError) as e:
                    # If we can't step even at frame 0, skip it
                    pass
            return original_update(frame)
        
        animator._update = custom_update
        
        # Run animation
        if save_animation:
            print(f"\n{'='*80}")
            print(f"=== STARTING ANIMATION ===")
            print(f"Frames: {num_steps + 1}, Interval: {animation_interval}ms")
            print(f"Save path: {animation_path_used}")
            print(f"Path exists check: {os.path.exists(os.path.dirname(animation_path_used))}")
            print(f"{'='*80}\n")
            
            # Call animate - it will handle saving internally
            print("About to call animator.animate()...")
            try:
                anim = animator.animate(
                    frames=num_steps + 1,  # +1 to include initial state
                    interval=animation_interval,
                    repeat=False,
                    save_path=animation_path_used,
                    dpi=120,
                    fps=animation_fps,
                    simulation=None,  # We handle stepping manually above
                )
                print("animator.animate() returned successfully.")
                
                # Wait for file system to sync before checking
                import time
                time.sleep(1.0)  # Give filesystem time to sync
                
            except Exception as anim_error:
                print(f"\n{'='*80}")
                print(f"EXCEPTION in animator.animate():")
                print(f"Error: {type(anim_error).__name__}: {anim_error}")
                print(f"{'='*80}")
                import traceback
                traceback.print_exc()
                print(f"{'='*80}\n")
                # Don't re-raise - let it continue to check if file was saved anyway
                # Some errors might still result in a saved file
                pass
            
            # Final verification - wait for file system to sync
            import time
            time.sleep(1.0)  # Give filesystem time to sync
            print(f"\nChecking if file exists: {animation_path_used}")
            if os.path.exists(animation_path_used):
                animation_saved = True
                size = os.path.getsize(animation_path_used)
                print(f"\n{'='*80}")
                print(f"✓ ANIMATION SAVE CONFIRMED!")
                print(f"File: {animation_path_used}")
                print(f"Size: {size:,} bytes ({size/1024:.1f} KB)")
                print(f"{'='*80}\n")
            else:
                print(f"\n{'='*80}")
                print(f"✗ WARNING: Animation file NOT FOUND after animate() returned!")
                print(f"Expected: {animation_path_used}")
                dir_path = os.path.dirname(animation_path_used)
                print(f"Directory exists: {os.path.exists(dir_path)}")
                if os.path.exists(dir_path):
                    print(f"Files in directory: {os.listdir(dir_path)}")
                print(f"{'='*80}\n")
                # Don't set animation_saved = True if file doesn't exist
        else:
            anim = animator.animate(
                frames=num_steps + 1,
                interval=animation_interval,
                repeat=False,
                simulation=None,  # We handle stepping manually above
            )
        
        # Also save a final snapshot
        if save_animation and animation_path_used:
            snapshot_path = animation_path_used.replace('.gif', '_final.png')
            try:
                plot_graph_state(env, agents, save_path=snapshot_path)
                if os.path.exists(snapshot_path):
                    print(f"✓ Final snapshot saved to: {snapshot_path}")
            except Exception as e:
                print(f"✗ Warning: Could not save snapshot: {e}")
                
    except ImportError as e:
        print(f"\n{'='*80}")
        print(f"✗ ERROR: Visualization not available ({e})")
        print("Running simulation without visualization...")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        sim.run(num_steps)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ ERROR: Animation error ({e})")
        print(f"Error type: {type(e).__name__}")
        print("Full traceback:")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}")
        print("\nNOTE: The animation save may have failed above.")
        
        # Check if animation file was saved despite the error
        if save_animation and animation_path_used:
            import time
            time.sleep(1.0)  # Give filesystem time to sync
            if os.path.exists(animation_path_used):
                animation_saved = True
                size = os.path.getsize(animation_path_used)
                print(f"\n{'='*80}")
                print(f"✓ ANIMATION WAS SAVED DESPITE ERROR!")
                print(f"File: {animation_path_used}")
                print(f"Size: {size:,} bytes ({size/1024:.1f} KB)")
                print(f"{'='*80}\n")
            else:
                print("Animation file was NOT saved. Running simulation without visualization...\n")
                # Only run simulation if animation wasn't saved
                if not hasattr(sim, '_has_run'):
                    sim.run(num_steps)
                    sim._has_run = True
        else:
            print("Running simulation without visualization...\n")
            if not hasattr(sim, '_has_run'):
                sim.run(num_steps)
                sim._has_run = True
    
    # Print statistics
    sim.print_statistics()
    
    # Save statistics to file if requested
    if results_save_path or csv_path:
        try:
            if results_save_path:
                sim.save_statistics(results_save_path)
                print(f"Statistics saved to: {os.path.abspath(results_save_path)}")
            
            # Save CSV if requested
            if csv_path:
                sim.save_statistics_csv(csv_path)
                print(f"Statistics CSV saved to: {os.path.abspath(csv_path)}")
        except Exception as e:
            print(f"Warning: Failed to save statistics: {e}")
            import traceback
            traceback.print_exc()
    
    if save_animation and animation_path_used:
        print(f"\n{'='*80}")
        if animation_saved:
            print(f"Animation file location: {animation_path_used}")
        else:
            print(f"WARNING: Animation may not have been saved correctly.")
            print(f"Expected location: {animation_path_used}")
        print(f"{'='*80}\n")
    
    return sim, env, agents


if __name__ == "__main__":
    # Configure simulation parameters
    num_nodes = 100
    num_agents = 4
    num_steps = 200
    planning_horizon = 5
    num_rollouts = 100
    mode = "rsp"
    # Use a random seed for each independent run (based on current time)
    # Set seed = 42 to reproduce a specific run, or use None for random seed
    seed = None  # None = random seed based on current time, or set to an integer for reproducibility
    if seed is None:
        import time
        # Use current time in microseconds for better randomness across independent runs
        seed = int(time.time() * 1000000) % (2**31)  # Use microseconds since epoch as seed
        print(f"Using random seed: {seed} (set seed=<integer> in __main__ block for reproducibility)")
    animation_interval = 200  # Milliseconds between frames
    w_h = 0.75 # Weight for information gain (horizon) in reward calculation
    w_v = 0.25  # Weight for event value in reward calculation
    event_utility = {0: 0.1, 1: 0.9}  # Event utility mapping {state: utility}
    epsilon = 0.2  # Epsilon-greedy exploration probability for Monte Carlo agents (0.0 = no exploration, 1.0 = always random)
    
    # Create environment ONCE before the loop to record evolution
    # All agent types will replay the exact same sequence of environmental changes
    print("Creating and recording environment evolution for fair comparison...")
    recording_env = create_graph_environment(
        num_nodes=num_nodes,
        mode=mode,
        seed=seed,
    )
    recording_env.reset(initial_events=5)
    
    # Keep a fresh copy of the original environment BEFORE recording
    # (recording_env will be modified during recording)
    # This will be used for transition kernel queries
    original_env = create_graph_environment(
        num_nodes=num_nodes,
        mode=mode,
        seed=seed,
    )
    
    # Record the environment evolution (this modifies recording_env in place)
    print(f"Recording {num_steps} steps of environment evolution...")
    replay_env = GraphReplayEnvironment.record_from(
        recording_env, 
        num_steps=num_steps,
        original_env=original_env  # Pass original env so replay can delegate transition queries
    )
    
    print(f"Recorded environment evolution: {replay_env.max_time + 1} states "
          f"(initial + {num_steps} steps)")
    
    # Create ONE timestamp folder for this entire run (all methods will use the same timestamp)
    timestamp_path = create_timestamp_folder()
    print(f"\nResults will be saved to: {os.path.abspath(timestamp_path)}")
    
    # Define agent types to run
    agent_types = ["ctde", "random", "greedy", "sharing", "monte_carlo"]
    
    # Save configuration file
    config_path = save_config_file(
        timestamp_path=timestamp_path,
        num_nodes=num_nodes,
        num_agents=num_agents,
        num_steps=num_steps,
        planning_horizon=planning_horizon,
        num_rollouts=num_rollouts,
        mode=mode,
        seed=seed,
        animation_interval=animation_interval,
        w_h=w_h,
        w_v=w_v,
        event_utility=event_utility,
        epsilon=epsilon,
        agent_types=agent_types,
    )
    print(f"Configuration saved to: {os.path.abspath(config_path)}")
    
    # Run simulation for each agent type with the SAME recorded evolution
    for agent_type in agent_types:
        print(f"\n{'='*70}")
        print(f"Running simulation for {agent_type.upper()} agents (replaying recorded evolution)")
        print(f"{'='*70}")
        
        # Reset replay environment to initial state
        replay_env.reset()
        
        # Run simulation with the replay environment
        # animation_path=None will trigger automatic path generation in results/timestamp/agent_type/
        # All methods use the same timestamp_path for this run
        main(
            num_nodes=num_nodes,
            num_agents=num_agents,
            num_steps=num_steps,
            planning_horizon=planning_horizon,
            num_rollouts=num_rollouts,
            mode=mode,
            seed=seed,
            agent_type=agent_type,  # "monte_carlo", "random", or "greedy"
            save_animation=True,
            animation_path=None,  # Will be auto-generated in results/timestamp/agent_type/
            animation_interval=animation_interval,
            animation_fps=None,
            env=replay_env,  # Use the replay environment (deterministic)
            original_env=original_env,  # Original env for transition kernel
            results_timestamp_path=timestamp_path,  # Use the same timestamp folder for all methods
            save_csv=True,  # Also save results as CSV
            w_h=w_h,  # Weight for information gain (horizon)
            w_v=w_v,  # Weight for event value
            event_utility=event_utility,  # Event utility mapping
            epsilon=epsilon,  # Epsilon-greedy exploration probability
        )

