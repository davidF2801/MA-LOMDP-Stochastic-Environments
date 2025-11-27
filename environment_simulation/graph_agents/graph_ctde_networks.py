"""Neural networks for CTDE (Centralized Training Decentralized Execution) agents."""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Fallback: create dummy classes if torch not available
    class nn:
        class Module:
            pass
        class Linear:
            pass
        class ReLU:
            pass
    torch = None
    F = None


class ActorNetwork(nn.Module):
    """
    Actor network for decentralized policy.
    
    Input: Local state s_i(t) = (B_i(t), p_i(t), {(p̂_i^j(t), Δ_i(j))}_{j≠i})
    - B_i(t): Belief vector [num_nodes, num_states] flattened -> [num_nodes * num_states]
    - p_i(t): Current position (one-hot encoded or embedded) -> [num_nodes]
    - Peer info: For each other agent j: (p̂_i^j(t), Δ_i(j))
      - p̂_i^j(t): Last known position (one-hot) -> [num_nodes]
      - Δ_i(j): Age of Information -> [1]
    
    Output: Action probabilities over reachable nodes (variable action space)
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_states: int = 2,
        num_agents: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        """
        Initialize actor network.
        
        Args:
            num_nodes: Number of nodes in the graph
            num_states: Number of states per node (typically 2 for binary)
            num_agents: Number of agents (to size peer information)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CTDE agents. Install with: pip install torch")
        
        self.num_nodes = num_nodes
        self.num_states = num_states
        self.num_agents = num_agents
        
        # Input sizes
        belief_size = num_nodes * num_states  # Flattened belief
        position_size = num_nodes  # One-hot position
        # Peer info per agent: position (num_nodes) + AoI (1)
        peer_info_size = (num_agents - 1) * (num_nodes + 1) if num_agents > 1 else 0
        input_size = belief_size + position_size + peer_info_size
        
        # Build network
        layers = []
        prev_size = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_dim))
            layers.append(nn.ReLU())
            prev_size = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Output layer: outputs logits for all nodes (variable masking will be applied)
        self.output = nn.Linear(hidden_dim, num_nodes)
        
    def forward(
        self,
        belief: torch.Tensor,  # [batch, num_nodes, num_states] or [batch, num_nodes * num_states]
        position: torch.Tensor,  # [batch, num_nodes] (one-hot) or [batch] (index)
        peer_info: Optional[torch.Tensor] = None,  # [batch, num_agents-1, num_nodes+1] or flattened
        action_mask: Optional[torch.Tensor] = None,  # [batch, num_nodes] boolean mask of valid actions
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network.
        
        Args:
            belief: Belief tensor [batch, num_nodes, num_states] or [batch, num_nodes * num_states]
            position: Position tensor [batch, num_nodes] (one-hot) or [batch] (index, will be converted)
            peer_info: Peer information [batch, num_agents-1, num_nodes+1] or [batch, (num_agents-1)*(num_nodes+1)]
                      Each peer has: one-hot position [num_nodes] + AoI [1]
            action_mask: Boolean mask [batch, num_nodes] indicating valid actions
        
        Returns:
            action_logits: [batch, num_nodes] logits for all nodes
            action_probs: [batch, num_nodes] probabilities (masked and normalized)
        """
        batch_size = belief.shape[0]
        
        # Flatten belief if needed
        if belief.dim() == 3:
            belief_flat = belief.view(batch_size, -1)
        else:
            belief_flat = belief
        
        # Convert position index to one-hot if needed
        if position.dim() == 1:
            position_onehot = F.one_hot(position, num_classes=self.num_nodes).float()
        else:
            position_onehot = position
        
        # Process peer info
        if peer_info is None or self.num_agents == 1:
            peer_flat = torch.zeros(batch_size, 0, device=belief.device)
        else:
            if peer_info.dim() == 3:
                peer_flat = peer_info.view(batch_size, -1)
            else:
                peer_flat = peer_info
        
        # Concatenate inputs
        x = torch.cat([belief_flat, position_onehot, peer_flat], dim=1)
        
        # Forward through network
        hidden = self.backbone(x)
        logits = self.output(hidden)  # [batch, num_nodes]
        
        # Apply action mask and normalize
        if action_mask is not None:
            # Set invalid actions to very negative value
            masked_logits = logits.masked_fill(~action_mask, float('-inf'))
            probs = F.softmax(masked_logits, dim=1)
        else:
            probs = F.softmax(logits, dim=1)
        
        return logits, probs
    
    def get_action_distribution(
        self,
        belief: torch.Tensor,
        position: torch.Tensor,
        peer_info: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Categorical:
        """
        Get action distribution for sampling.
        
        Returns:
            Categorical distribution over actions
        """
        _, probs = self.forward(belief, position, peer_info, action_mask)
        return torch.distributions.Categorical(probs=probs)


class CriticNetwork(nn.Module):
    """
    Centralized critic network for value estimation.
    
    Input: Centralized state S_t = (E_t, p_1(t), ..., p_N(t))
    or ideal centralized belief: S_t = (B_ideal(t), p_1(t), ..., p_N(t))
    
    Output: Value estimate V(S_t)
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_states: int = 2,
        num_agents: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        """
        Initialize critic network.
        
        The critic always receives the ideal centralized belief (joint observations from all agents)
        as input, not the true state.
        
        Args:
            num_nodes: Number of nodes in the graph
            num_states: Number of states per node
            num_agents: Number of agents
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CTDE agents. Install with: pip install torch")
        
        self.num_nodes = num_nodes
        self.num_states = num_states
        self.num_agents = num_agents
        
        # Input sizes
        # State/belief: [num_nodes, num_states] -> flattened [num_nodes * num_states]
        state_size = num_nodes * num_states
        # Agent positions: each agent has one-hot position [num_nodes]
        positions_size = num_agents * num_nodes
        input_size = state_size + positions_size
        
        # Build network
        layers = []
        prev_size = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_dim))
            layers.append(nn.ReLU())
            prev_size = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Output: single value estimate
        self.output = nn.Linear(hidden_dim, 1)
        
        # Initialize output layer to output values around expected reward scale
        # Expected reward per step might be around 1-5, so initialize output near 0
        # But allow network to learn the scale through training
        nn.init.orthogonal_(self.output.weight, gain=0.01)
        nn.init.constant_(self.output.bias, 0.0)
        
    def forward(
        self,
        state_or_belief: torch.Tensor,  # [batch, num_nodes, num_states] or [batch, num_nodes * num_states]
        positions: torch.Tensor,  # [batch, num_agents, num_nodes] (one-hot) or [batch, num_agents] (indices)
    ) -> torch.Tensor:
        """
        Forward pass through critic network.
        
        Args:
            state_or_belief: Environment state or ideal belief [batch, num_nodes, num_states] or flattened
            positions: Agent positions [batch, num_agents, num_nodes] (one-hot) or [batch, num_agents] (indices)
        
        Returns:
            value: [batch, 1] value estimates
        """
        batch_size = state_or_belief.shape[0]
        
        # Flatten state/belief if needed
        if state_or_belief.dim() == 3:
            state_flat = state_or_belief.view(batch_size, -1)
        else:
            state_flat = state_or_belief
        
        # Convert position indices to one-hot if needed
        if positions.dim() == 2 and positions.dtype in (torch.long, torch.int):
            # [batch, num_agents] indices -> [batch, num_agents, num_nodes] one-hot
            positions_onehot = F.one_hot(positions, num_classes=self.num_nodes).float()
            positions_flat = positions_onehot.view(batch_size, -1)
        else:
            # Already one-hot or flattened
            if positions.dim() == 3:
                positions_flat = positions.view(batch_size, -1)
            else:
                positions_flat = positions
        
        # Concatenate inputs
        x = torch.cat([state_flat, positions_flat], dim=1)
        
        # Forward through network
        hidden = self.backbone(x)
        value = self.output(hidden)  # [batch, 1]
        
        return value

