"""Graph-based agents for graph environment simulations."""

from .graph_agent import GraphAgent
from .graph_monte_carlo_agent import GraphMonteCarloAgent
from .graph_random_agent import GraphRandomAgent
from .graph_greedy_agent import GraphGreedyAgent
from .graph_sharing_monte_carlo_agent import GraphSharingMonteCarloAgent
from .graph_belief import GraphBelief

# CTDE imports (optional - require PyTorch)
try:
    from .graph_ctde_agent import GraphCTDEAgent
    from .graph_ctde_networks import ActorNetwork, CriticNetwork
    from .graph_ctde_trainer import CTDETrainer, TrajectoryBuffer
    from .graph_ctde_training import create_graph_ctde_agents, train_ctde_agents
    CTDE_AVAILABLE = True
except ImportError:
    CTDE_AVAILABLE = False
    GraphCTDEAgent = None
    ActorNetwork = None
    CriticNetwork = None
    CTDETrainer = None
    TrajectoryBuffer = None
    create_graph_ctde_agents = None
    train_ctde_agents = None

__all__ = [
    "GraphAgent",
    "GraphMonteCarloAgent",
    "GraphRandomAgent",
    "GraphGreedyAgent",
    "GraphSharingMonteCarloAgent",
    "GraphBelief",
]

if CTDE_AVAILABLE:
    __all__.extend([
        "GraphCTDEAgent",
        "ActorNetwork",
        "CriticNetwork",
        "CTDETrainer",
        "TrajectoryBuffer",
        "create_graph_ctde_agents",
        "train_ctde_agents",
    ])

