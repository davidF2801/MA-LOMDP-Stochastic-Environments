"""Training infrastructure for CTDE agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None
    optim = None

from .graph_ctde_networks import ActorNetwork, CriticNetwork
from .graph_ctde_agent import GraphCTDEAgent

if not TORCH_AVAILABLE:
    raise ImportError("PyTorch is required for CTDE training. Install with: pip install torch")


@dataclass
class TrajectoryBuffer:
    """
    Buffer to store trajectory data for CTDE training.
    
    Stores for each timestep:
    - Centralized state S_t
    - Local states {s_i(t)}_i for all agents
    - Joint action a_t
    - Team reward r_t
    - Done flag
    - Value estimate V_ψ(S_t)
    - Action log probabilities for each agent
    - Action masks for each agent (for proper masking during updates)
    """
    
    max_size: int
    
    # Centralized states
    centralized_states: list[Any] = field(default_factory=list)
    
    # Local states for each agent: {agent_name: [s_i(0), s_i(1), ...]}
    local_states: dict[str, list[Any]] = field(default_factory=dict)
    
    # Joint actions: [a_0, a_1, ...] where each a_t is {agent_name: action}
    joint_actions: list[dict[str, int]] = field(default_factory=list)
    
    # Team rewards
    rewards: list[float] = field(default_factory=list)
    
    # Done flags
    dones: list[bool] = field(default_factory=list)
    
    # Value estimates
    values: list[float] = field(default_factory=list)
    
    # Action log probabilities for each agent: {agent_name: [log_prob_0, log_prob_1, ...]}
    action_log_probs: dict[str, list[float]] = field(default_factory=dict)
    
    # Action masks for each agent: {agent_name: [mask_0, mask_1, ...]}
    # Each mask is a boolean tensor [num_nodes] indicating valid actions
    action_masks: dict[str, list[Any]] = field(default_factory=dict)
    
    # Agent positions at each timestep: [[pos_0, pos_1, ...], ...] where each inner list is [agent_0_pos, agent_1_pos, ...]
    positions: list[Any] = field(default_factory=list)
    
    def add(
        self,
        centralized_state: Any,
        local_states: dict[str, Any],
        joint_action: dict[str, int],
        reward: float,
        done: bool,
        value: float,
        action_log_probs: dict[str, float],
        action_masks: Optional[dict[str, Any]] = None,
        positions: Optional[Any] = None,
    ) -> None:
        """
        Add a timestep to the buffer.
        
        Args:
            action_masks: Optional dictionary {agent_name: action_mask} where
                         action_mask is a boolean tensor/array [num_nodes] indicating valid actions
            positions: Optional tensor/array [num_agents] with agent positions at this timestep
        """
        self.centralized_states.append(centralized_state)
        
        for agent_name, local_state in local_states.items():
            if agent_name not in self.local_states:
                self.local_states[agent_name] = []
            self.local_states[agent_name].append(local_state)
        
        self.joint_actions.append(joint_action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        
        # Store positions if provided
        if positions is not None:
            self.positions.append(positions)
        
        for agent_name, log_prob in action_log_probs.items():
            if agent_name not in self.action_log_probs:
                self.action_log_probs[agent_name] = []
            self.action_log_probs[agent_name].append(log_prob)
        
        # Store action masks if provided
        if action_masks is not None:
            for agent_name, mask in action_masks.items():
                if agent_name not in self.action_masks:
                    self.action_masks[agent_name] = []
                self.action_masks[agent_name].append(mask)
        
        if len(self.rewards) > self.max_size:
            # Remove oldest entry
            self.centralized_states.pop(0)
            for agent_name in self.local_states:
                self.local_states[agent_name].pop(0)
            self.joint_actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.values.pop(0)
            if len(self.positions) > 0:
                self.positions.pop(0)
            for agent_name in self.action_log_probs:
                self.action_log_probs[agent_name].pop(0)
            for agent_name in self.action_masks:
                if agent_name in self.action_masks and len(self.action_masks[agent_name]) > 0:
                    self.action_masks[agent_name].pop(0)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.centralized_states.clear()
        self.local_states.clear()
        self.joint_actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.action_log_probs.clear()
        self.action_masks.clear()
        self.positions.clear()
    
    def length(self) -> int:
        """Get current buffer length."""
        return len(self.rewards)


def compute_gae_advantages(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    next_value: Optional[float] = None,
) -> tuple[list[float], list[float]]:
    """
    Compute Generalized Advantage Estimation (GAE) advantages and returns.
    
    δ_t = r_t + γ * V(S_{t+1}) - V(S_t)
    A_t = Σ_{k=0}^{T-1-t} (γ * λ)^k * δ_{t+k}
    R_t = A_t + V(S_t)
    
    Args:
        rewards: List of rewards [r_0, r_1, ..., r_{T-1}]
        values: List of value estimates [V_0, V_1, ..., V_{T-1}]
        dones: List of done flags
        gamma: Discount factor
        lambda_gae: GAE lambda parameter
        next_value: Value estimate for state at T (for bootstrap). If None, uses 0.
        
    Returns:
        advantages: List of advantages [A_0, A_1, ..., A_{T-1}]
        returns: List of returns [R_0, R_1, ..., R_{T-1}]
    """
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T
    
    # Use next_value for bootstrap if provided, otherwise 0
    next_val = next_value if next_value is not None else 0.0
    
    # Compute advantages backwards
    gae = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            if t == T - 1:
                delta = rewards[t] + gamma * next_val - values[t]
            else:
                delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lambda_gae * gae
        
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns


class CTDETrainer:
    """
    CTDE trainer for MAPPO-style updates.
    
    Implements:
    - Critic update: Minimize (V_ψ(S_t) - R_t)^2
    - Actor update: Maximize PPO clipped objective + entropy regularization
    """
    
    def __init__(
        self,
        actor_network: ActorNetwork,
        critic_network: CriticNetwork,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize CTDE trainer.
        
        Args:
            actor_network: Shared actor network
            critic_network: Centralized critic network
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on ('cpu' or 'cuda')
        """
        self.actor_network = actor_network
        self.critic_network = critic_network
        
        self.actor_optimizer = optim.Adam(actor_network.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic_network.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Move networks to device
        self.actor_network.to(device)
        self.critic_network.to(device)
        
        # Store old actor network for PPO
        self.old_actor_network = None
    
    def save_old_actor(self) -> None:
        """Save current actor network as old policy for PPO."""
        # Deep copy the actor network
        import copy
        self.old_actor_network = copy.deepcopy(self.actor_network)
    
    def update_critic(
        self,
        centralized_states: list[Any],
        returns: list[float],
        positions_batch: torch.Tensor,
    ) -> dict[str, float]:
        """
        Update critic network.
        
        Minimize: L_critic(ψ) = E[(V_ψ(S_t) - R_t)^2]
        
        Args:
            centralized_states: List of centralized states (environment states or ideal beliefs)
            returns: List of returns [R_0, R_1, ...]
            positions_batch: Tensor [batch, num_agents] with agent positions
        
        Returns:
            Dictionary with loss statistics
        """
        self.critic_network.train()
        self.critic_optimizer.zero_grad()
        
        # Convert states to tensor
        batch_size = len(centralized_states)
        if isinstance(centralized_states[0], np.ndarray):
            # Environment state: [num_nodes] binary states
            states_tensor = torch.from_numpy(np.array(centralized_states)).float()
            # Convert to [batch, num_nodes, num_states] format
            if states_tensor.dim() == 2:
                # [batch, num_nodes] -> [batch, num_nodes, 2] (binary states)
                states_tensor = F.one_hot(states_tensor.long(), num_classes=2).float()
        else:
            # Already tensor
            states_tensor = torch.stack(centralized_states).to(self.device)
        
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)  # [batch, 1]
        positions_batch = positions_batch.to(self.device)
        
        # Forward pass
        values_pred = self.critic_network(states_tensor, positions_batch)  # [batch, 1]
        
        # Value loss
        value_loss = F.mse_loss(values_pred, returns_tensor)
        
        # Backward pass
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        return {
            'value_loss': value_loss.item(),
            'value_mean': values_pred.mean().item(),
            'return_mean': returns_tensor.mean().item(),
        }
    
    def update_actor(
        self,
        agents: list[GraphCTDEAgent],
        env: "GraphEnvironment",
        local_states_list: list[dict[str, Any]],
        joint_actions: list[dict[str, int]],
        advantages: list[float],
        action_log_probs_old: dict[str, list[float]],
        action_masks_old: Optional[dict[str, list[Any]]] = None,
    ) -> dict[str, float]:
        """
        Update actor network using PPO.
        
        Maximize: L_actor = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)] + β * H(π)
        
        where r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio.
        
        Args:
            agents: List of CTDE agents
            env: GraphEnvironment object
            local_states_list: List of local states dictionaries [{agent_name: s_i(t)} for each t]
            joint_actions: List of joint actions [{agent_name: a_i(t)} for each t]
            advantages: List of advantages (shared across all agents)
            action_log_probs_old: Dictionary {agent_name: [old_log_prob_0, ...]} from old policy
        
        Returns:
            Dictionary with loss statistics
        """
        if self.old_actor_network is None:
            raise ValueError("Old actor network not saved. Call save_old_actor() first.")
        
        self.actor_network.train()
        self.actor_optimizer.zero_grad()
        
        batch_size = len(local_states_list)
        agent_names = [agent.name for agent in agents]
        
        # Prepare data for all agents
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_clipped_frac = 0.0
        
        # Stack data across timesteps
        all_advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        
        # Store raw advantage statistics before normalization
        adv_mean_raw = all_advantages.mean().item()
        adv_std_raw = all_advantages.std().item()
        adv_min_raw = all_advantages.min().item()
        adv_max_raw = all_advantages.max().item()
        
        # Normalize advantages (for stability, but keep a copy for loss computation)
        # Only normalize if std is meaningful (> 1e-6), otherwise keep as-is
        adv_std = all_advantages.std()
        if adv_std > 1e-6:
            all_advantages_normalized = (all_advantages - all_advantages.mean()) / (adv_std + 1e-8)
        else:
            # If all advantages are nearly identical, set them to small random values to allow exploration
            all_advantages_normalized = torch.randn_like(all_advantages) * 0.01
        
        # Use normalized advantages for updates (better gradient stability)
        all_advantages_for_update = all_advantages_normalized
        
        for agent_name in agent_names:
            # Get agent
            agent = next(a for a in agents if a.name == agent_name)
            
            # Prepare batch data
            belief_batch = []
            position_batch = []
            peer_info_batch = []
            action_batch = []
            old_log_probs_batch = []
            action_mask_batch = []
            
            for t in range(batch_size):
                local_state = local_states_list[t][agent_name]
                action = joint_actions[t][agent_name]
                old_log_prob = action_log_probs_old[agent_name][t]
                
                belief_batch.append(local_state['belief'])
                position_batch.append(local_state['position'])
                peer_info_batch.append(local_state['peer_info'])
                action_batch.append(action)
                old_log_probs_batch.append(old_log_prob)
                
                # Get action mask from stored masks if available, otherwise compute it
                if agent_name in action_masks_old and t < len(action_masks_old[agent_name]):
                    mask = action_masks_old[agent_name][t]
                else:
                    # Fallback: compute mask from current env state
                    mask = agent.get_action_mask(env)
                action_mask_batch.append(mask)
            
            # Stack into batches
            belief_batch = torch.stack(belief_batch).to(self.device)  # [batch, num_nodes, num_states]
            position_batch = torch.tensor(position_batch, dtype=torch.long, device=self.device)  # [batch]
            peer_info_batch = torch.stack(peer_info_batch).to(self.device)  # [batch, num_other_agents, num_nodes+1]
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=self.device)  # [batch]
            old_log_probs_batch = torch.tensor(old_log_probs_batch, dtype=torch.float32, device=self.device)  # [batch]
            action_mask_batch = torch.stack(action_mask_batch).to(self.device)  # [batch, num_nodes]
            
            # Forward pass with current policy
            _, probs_current = self.actor_network(
                belief_batch,
                position_batch,
                peer_info_batch,
                action_mask_batch,
            )
            
            # Forward pass with old policy
            self.old_actor_network.eval()
            with torch.no_grad():
                _, probs_old = self.old_actor_network(
                    belief_batch,
                    position_batch,
                    peer_info_batch,
                    action_mask_batch,
                )
            
            # Get log probabilities
            log_probs_current = torch.log(probs_current + 1e-10)
            log_probs_old = torch.log(probs_old + 1e-10)
            
            # Select actions
            log_prob_current = log_probs_current.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # [batch]
            log_prob_old = log_probs_old.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # [batch]
            
            # Compute ratio
            ratio = torch.exp(log_prob_current - log_prob_old)  # [batch]
            
            # Clipped surrogate loss (use normalized advantages for stability)
            surr1 = ratio * all_advantages_for_update
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * all_advantages_for_update
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            entropy = -(probs_current * log_probs_current).sum(dim=1).mean()
            
            # Clipped fraction (for monitoring)
            clipped_frac = ((ratio < (1.0 - self.clip_epsilon)) | (ratio > (1.0 + self.clip_epsilon))).float().mean()
            
            # Total loss for this agent
            agent_loss = policy_loss - self.entropy_coef * entropy
            
            # For logging: compute policy loss with unnormalized advantages to see true magnitude
            # (but still use normalized for gradient computation)
            surr1_raw = ratio * all_advantages
            surr2_raw = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * all_advantages
            policy_loss_raw = -torch.min(surr1_raw, surr2_raw).mean()
            
            total_policy_loss += policy_loss_raw.item()  # Log unnormalized loss
            total_entropy += entropy.item()
            total_clipped_frac += clipped_frac.item()
        
        # Average across agents
        num_agents = len(agents)
        avg_policy_loss = total_policy_loss / num_agents
        avg_entropy = total_entropy / num_agents
        avg_clipped_frac = total_clipped_frac / num_agents
        
        # Backward pass (sum losses across agents)
        # Need to recompute loss with gradients using normalized advantages
        total_loss = 0.0
        for agent_name in agent_names:
            agent = next(a for a in agents if a.name == agent_name)
            
            belief_batch = []
            position_batch = []
            peer_info_batch = []
            action_batch = []
            old_log_probs_batch = []
            action_mask_batch = []
            
            for t in range(batch_size):
                local_state = local_states_list[t][agent_name]
                action = joint_actions[t][agent_name]
                old_log_prob = action_log_probs_old[agent_name][t]
                
                belief_batch.append(local_state['belief'])
                position_batch.append(local_state['position'])
                peer_info_batch.append(local_state['peer_info'])
                action_batch.append(action)
                old_log_probs_batch.append(old_log_prob)
                
                # Get action mask from stored masks if available, otherwise compute it
                if action_masks_old is not None and agent_name in action_masks_old and t < len(action_masks_old[agent_name]):
                    mask = action_masks_old[agent_name][t]
                    # Convert numpy array to torch tensor if needed
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask).bool()
                else:
                    # Fallback: compute mask from current env state
                    mask = agent.get_action_mask(env)
                action_mask_batch.append(mask)
            
            belief_batch = torch.stack(belief_batch).to(self.device)
            position_batch = torch.tensor(position_batch, dtype=torch.long, device=self.device)
            peer_info_batch = torch.stack(peer_info_batch).to(self.device)
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=self.device)
            old_log_probs_batch = torch.tensor(old_log_probs_batch, dtype=torch.float32, device=self.device)
            action_mask_batch = torch.stack(action_mask_batch).to(self.device)
            
            _, probs = self.actor_network(
                belief_batch,
                position_batch,
                peer_info_batch,
                action_mask_batch,
            )
            
            log_probs = torch.log(probs + 1e-10)
            log_prob = log_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                _, probs_old = self.old_actor_network(
                    belief_batch,
                    position_batch,
                    peer_info_batch,
                    action_mask_batch,
                )
                log_probs_old = torch.log(probs_old + 1e-10)
                log_prob_old = log_probs_old.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            
            ratio = torch.exp(log_prob - log_prob_old)
            # Use normalized advantages for gradient computation
            surr1 = ratio * all_advantages_for_update
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * all_advantages_for_update
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = -(probs * log_probs).sum(dim=1).mean()
            agent_loss = policy_loss - self.entropy_coef * entropy
            total_loss += agent_loss
        
        # Average and backward
        total_loss = total_loss / num_agents
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        return {
            'policy_loss': avg_policy_loss,
            'entropy': avg_entropy,
            'clipped_fraction': avg_clipped_frac,
            'advantage_mean_raw': adv_mean_raw,
            'advantage_std_raw': adv_std_raw,
            'advantage_min_raw': adv_min_raw,
            'advantage_max_raw': adv_max_raw,
        }

