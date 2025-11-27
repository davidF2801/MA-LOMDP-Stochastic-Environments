"""CTDE (Centralized Training Decentralized Execution) agent for graph-based planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

from .graph_agent import GraphAgent
from .graph_belief import GraphBelief
from .graph_ctde_networks import ActorNetwork, CriticNetwork, TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    raise ImportError("PyTorch is required for CTDE agents. Install with: pip install torch")


@dataclass
class GraphCTDEAgent(GraphAgent):
    """
    CTDE agent that uses neural network policies (actor-critic).
    
    Training: Uses centralized critic with full state information
    Execution: Uses decentralized actor with only local information (belief, position, peer info)
    
    Implements the CTDE framework:
    - Local actor input: s_i(t) = (B_i(t), p_i(t), {(p̂_i^j(t), Δ_i(j))}_{j≠i})
    - Centralized critic: V_ψ(S_t) where S_t is the full environment state
    - MAPPO-style updates with PPO clipping and entropy regularization
    """
    
    # Neural network components (set during initialization)
    actor_network: Optional[ActorNetwork] = field(default=None, init=False, repr=False)
    critic_network: Optional[CriticNetwork] = field(default=None, init=False, repr=False)
    
    # Communication tracking
    # σ_i(j): Last time agent i received fresh information from agent j
    _last_sync_time: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    
    # p̂_i^j(t): Last known position of agent j as seen by agent i
    _peer_positions: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    
    # Training mode flag
    training_mode: bool = True
    
    # Action masking: when in execution mode, we may want to use deterministic argmax
    deterministic_execution: bool = False
    
    def __post_init__(self):
        """Initialize CTDE agent after dataclass creation."""
        # Networks will be set separately (shared across agents)
        # No need to call super().__post_init__() as GraphAgent doesn't have one
    
    def set_networks(
        self,
        actor_network: ActorNetwork,
        critic_network: Optional[CriticNetwork] = None,
    ) -> None:
        """
        Set the actor and critic networks for this agent.
        
        Args:
            actor_network: Actor network (shared across all agents)
            critic_network: Optional critic network (only needed during training)
        """
        self.actor_network = actor_network
        self.critic_network = critic_network
    
    def update_communication_info(
        self,
        other_agents: list["GraphCTDEAgent"],
        env: "GraphEnvironment",
        timestep: int,
    ) -> None:
        """
        Update communication information (AoI and peer positions).
        
        For each agent j that this agent can communicate with:
        - Update σ_i(j) = timestep (last sync time)
        - Update p̂_i^j(t) = p_j(t) (last known position)
        
        Args:
            other_agents: List of other CTDE agents
            env: GraphEnvironment object
            timestep: Current timestep
        """
        for other_agent in other_agents:
            if self._can_communicate_with(other_agent, env):
                self._last_sync_time[other_agent.name] = timestep
                self._peer_positions[other_agent.name] = other_agent.current_node
    
    def _can_communicate_with(
        self,
        other_agent: "GraphAgent",
        env: "GraphEnvironment",
    ) -> bool:
        """
        Check if this agent can communicate with another agent.
        
        Agents can communicate if:
        - They are at the same node, OR
        - Their nodes are connected (within one step distance)
        
        Args:
            other_agent: Another GraphAgent
            env: GraphEnvironment object
            
        Returns:
            True if agents can communicate
        """
        # Same node
        if self.current_node == other_agent.current_node:
            return True
        
        # Connected nodes
        self_neighbors = set(env.get_neighbors(self.current_node))
        self_neighbors.add(self.current_node)
        return other_agent.current_node in self_neighbors
    
    def get_age_of_information(self, other_agent_name: str, timestep: int) -> int:
        """
        Get Age of Information (AoI) for another agent.
        
        Δ_i(j) = t - σ_i(j)
        
        Args:
            other_agent_name: Name of the other agent
            timestep: Current timestep
            
        Returns:
            AoI value (0 if just synced, larger if stale information)
        """
        last_sync = self._last_sync_time.get(other_agent_name, -1)
        if last_sync < 0:
            return timestep + 1  # Never synced, very stale
        return timestep - last_sync
    
    def get_peer_position(self, other_agent_name: str) -> int:
        """
        Get last known position of another agent.
        
        p̂_i^j(t) = p_j(σ_i(j))
        
        Args:
            other_agent_name: Name of the other agent
            
        Returns:
            Last known node index of the other agent (or -1 if never synced)
        """
        return self._peer_positions.get(other_agent_name, -1)
    
    def build_local_state(
        self,
        other_agents: list["GraphCTDEAgent"],
        timestep: int,
    ) -> dict[str, Any]:
        """
        Build local state s_i(t) for actor input.
        
        s_i(t) = (B_i(t), p_i(t), {(p̂_i^j(t), Δ_i(j))}_{j≠i})
        
        Args:
            other_agents: List of other CTDE agents
            timestep: Current timestep
            
        Returns:
            Dictionary with keys:
            - 'belief': Belief tensor [num_nodes, num_states]
            - 'position': Current position (index)
            - 'peer_info': Tensor [num_other_agents, num_nodes+1] with (one-hot position, AoI) for each peer
        """
        # Convert belief to tensor
        belief_array = np.zeros((self.num_nodes, 2), dtype=np.float32)
        for node in range(self.num_nodes):
            belief_array[node, 0] = self.belief.get_probability(node, 0)
            belief_array[node, 1] = self.belief.get_probability(node, 1)
        belief_tensor = torch.from_numpy(belief_array)  # [num_nodes, num_states]
        
        # Current position
        position = self.current_node
        
        # Peer information: for each other agent, (one-hot position, AoI)
        num_other_agents = len(other_agents)
        peer_info = np.zeros((num_other_agents, self.num_nodes + 1), dtype=np.float32)
        
        for i, other_agent in enumerate(other_agents):
            # Last known position (one-hot encoded)
            peer_pos = self.get_peer_position(other_agent.name)
            if peer_pos >= 0:
                peer_info[i, peer_pos] = 1.0
            
            # AoI
            aoi = self.get_age_of_information(other_agent.name, timestep)
            peer_info[i, self.num_nodes] = float(aoi)
        
        peer_info_tensor = torch.from_numpy(peer_info)  # [num_other_agents, num_nodes+1]
        
        return {
            'belief': belief_tensor,
            'position': position,
            'peer_info': peer_info_tensor,
        }
    
    def get_action_mask(self, env: "GraphEnvironment") -> torch.Tensor:
        """
        Get action mask for valid actions (reachable nodes).
        
        Args:
            env: GraphEnvironment object
            
        Returns:
            Boolean tensor [num_nodes] indicating valid actions
        """
        reachable = self.get_reachable_nodes(env)
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        for node in reachable:
            mask[node] = True
        return mask
    
    def act(
        self,
        env: "GraphEnvironment",
        other_agents: list["GraphCTDEAgent"],
        timestep: int,
        return_log_prob: bool = False,
    ) -> dict[str, Any]:
        """
        Select action using the actor network.
        
        Args:
            env: GraphEnvironment object
            other_agents: List of other CTDE agents (for peer info)
            timestep: Current timestep
            return_log_prob: If True, also return log probability of selected action
            
        Returns:
            Dictionary with:
            - 'action': Selected action (node index to move to)
            - 'log_prob': Log probability of action (if return_log_prob=True)
            - 'action_probs': Action probabilities [num_nodes] (if return_log_prob=True)
        """
        if self.actor_network is None:
            raise ValueError("Actor network not set. Call set_networks() first.")
        
        # Build local state
        local_state = self.build_local_state(other_agents, timestep)
        
        # Add batch dimension
        belief_batch = local_state['belief'].unsqueeze(0)  # [1, num_nodes, num_states]
        position_batch = torch.tensor([local_state['position']], dtype=torch.long)  # [1]
        peer_info_batch = local_state['peer_info'].unsqueeze(0)  # [1, num_other_agents, num_nodes+1]
        
        # Get action mask
        action_mask = self.get_action_mask(env).unsqueeze(0)  # [1, num_nodes]
        
        # Forward through actor network
        self.actor_network.eval()
        with torch.no_grad():
            logits, probs = self.actor_network(
                belief_batch,
                position_batch,
                peer_info_batch,
                action_mask,
            )
        
        # Select action
        if self.training_mode or not self.deterministic_execution:
            # Sample from distribution during training
            dist = torch.distributions.Categorical(probs=probs)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx)).item()
        else:
            # Argmax during deterministic execution
            action_idx = probs.argmax(dim=1).item()
            log_prob = torch.log(probs[0, action_idx] + 1e-10).item()
        
        result = {
            'action': action_idx,  # Node index to move to
        }
        
        if return_log_prob:
            result['log_prob'] = log_prob
            result['action_probs'] = probs[0].cpu().numpy()
        
        return result
    
    def compute_reward(
        self,
        env: "GraphEnvironment",
        action: Optional[int] = None,
        w_h: float = 1.0,
        w_v: float = 1.0,
        event_utility: Optional[dict[int, float]] = None,
    ) -> float:
        """
        Compute reward for taking an action (individual agent reward).
        
        This is used to compute team reward as sum of individual rewards.
        
        Implements: R = w_h * G(b_k) + w_v * Σ_x b_k(x) f(x)
        where k is the target node (action).
        
        Args:
            env: GraphEnvironment object
            action: Target node to move to. If None, uses current node.
            w_h: Weight for information gain term
            w_v: Weight for event detection value term
            event_utility: Dictionary mapping state (0 or 1) to utility value
            
        Returns:
            Reward value
        """
        if action is None:
            target_node = self.current_node
        else:
            target_node = action
        
        if event_utility is None:
            event_utility = {0: 0.0, 1: 1.0}
        
        # Information gain = entropy (perfect observation reduces entropy to 0)
        info_gain = self.information_gain({target_node})
        
        # Expected event value
        event_value = self.expected_event_value({target_node}, event_utility)
        
        # Total reward
        reward = w_h * info_gain + w_v * event_value
        return float(reward)
    
    def compute_policy(self, env: "GraphEnvironment", horizon: int) -> None:
        """
        Compute policy (no-op for CTDE agents - policy is in neural network).
        
        This method exists for compatibility with the simulation interface.
        The policy is already encoded in the actor network.
        """
        pass  # Policy is in the neural network, no computation needed
    
    def act_with_context(
        self,
        env: "GraphEnvironment",
        other_agents: list["GraphCTDEAgent"],
        timestep: int,
    ) -> dict[str, Any]:
        """
        Select action using actor network with full context.
        
        This is the proper way to call act() for CTDE agents.
        Returns dict compatible with simulation interface:
        - 'target_node': Node index to move to
        
        Args:
            env: GraphEnvironment object
            other_agents: List of other CTDE agents (for peer info)
            timestep: Current timestep
            
        Returns:
            Dictionary with 'target_node' key (and optionally other info)
        """
        # Call the neural network-based act method directly
        if self.actor_network is None:
            raise ValueError("Actor network not set. Call set_networks() first.")
        
        # Build local state
        local_state = self.build_local_state(other_agents, timestep)
        
        # Add batch dimension
        belief_batch = local_state['belief'].unsqueeze(0)  # [1, num_nodes, num_states]
        position_batch = torch.tensor([local_state['position']], dtype=torch.long)  # [1]
        peer_info_batch = local_state['peer_info'].unsqueeze(0)  # [1, num_other_agents, num_nodes+1]
        
        # Get action mask
        action_mask = self.get_action_mask(env).unsqueeze(0)  # [1, num_nodes]
        
        # Forward through actor network
        self.actor_network.eval()
        with torch.no_grad():
            logits, probs = self.actor_network(
                belief_batch,
                position_batch,
                peer_info_batch,
                action_mask,
            )
        
        # Select action
        if self.training_mode or not self.deterministic_execution:
            # Sample from distribution during training
            dist = torch.distributions.Categorical(probs=probs)
            action_idx = dist.sample().item()
        else:
            # Argmax during deterministic execution
            action_idx = probs.argmax(dim=1).item()
        
        # Convert to simulation-compatible format
        return {
            'target_node': action_idx,
            'expected_reward': 0.0,  # Can be computed if needed
        }

