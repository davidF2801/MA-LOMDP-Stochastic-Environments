"""Visualization module for graph-based environment simulation."""

from __future__ import annotations

from typing import Any, Optional
import os

# Try to import matplotlib and networkx
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import networkx as nx
    _VISUALIZATION_AVAILABLE = True
except ImportError as e:
    _VISUALIZATION_AVAILABLE = False
    _IMPORT_ERROR = str(e)
    plt = None
    FuncAnimation = None
    nx = None


class GraphAnimator:
    """Animate graph-based fire propagation with agent movement."""

    def __init__(
        self,
        env: "GraphEnvironment",
        agents: Optional[list["GraphAgent"]] = None,
        fire_color: str = "#d1495b",
        calm_color: str = "#e0e0e0",
        agent_colors: Optional[list[str]] = None,
        node_size: int = 300,
        edge_width: float = 0.5,
        figsize: tuple[float, float] = (12, 8),
    ):
        """Initialize graph animator."""
        if not _VISUALIZATION_AVAILABLE:
            raise ImportError(
                "matplotlib and networkx are required for GraphAnimator "
                f"(encountered: {_IMPORT_ERROR})"
            )
        
        self.env = env
        self.agents = agents or []
        self.fire_color = fire_color
        self.calm_color = calm_color
        self.node_size = node_size
        self.edge_width = edge_width
        self.figsize = figsize
        
        # Default agent colors
        if agent_colors is None:
            agent_colors = ["#ff5733", "#33ff57", "#3357ff", "#ff33f5", "#f5ff33"]
        self.agent_colors = agent_colors
        
        # Build NetworkX graph
        self.G = nx.Graph()
        self.G.add_nodes_from(range(env.num_nodes))
        self.G.add_edges_from(env.edges)
        
        # Compute node positions
        if env.width and env.height:
            # Grid layout
            self.pos = {}
            for node in range(env.num_nodes):
                y, x = env.node_positions[node]
                self.pos[node] = (x, env.height - 1 - y)
        else:
            # Spring layout for non-grid graphs
            self.pos = nx.spring_layout(self.G, seed=42)
        
        # Track agent trajectories
        self.agent_trajectories: dict[str, list[int]] = {
            agent.name: [] for agent in self.agents
        }
        
        # Figure and axis holders
        self._figure = None
        self._axis = None
    
    def _init_plot(self):
        """Initialize the plot."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_aspect('equal')
        ax.axis('off')
        self._figure = fig
        self._axis = ax
        return fig, ax
    
    def _update(self, frame: int):
        """Update function for animation."""
        self._axis.clear()
        self._axis.set_aspect('equal')
        self._axis.axis('off')
        
        # Get current state
        state = self.env.get_state()
        
        # Draw edges
        nx.draw_networkx_edges(
            self.G,
            self.pos,
            ax=self._axis,
            width=self.edge_width,
            alpha=0.3,
            edge_color='gray',
        )
        
        # Color nodes based on state
        node_colors = [
            self.fire_color if state[node] == 1 else self.calm_color
            for node in range(self.env.num_nodes)
        ]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.G,
            self.pos,
            ax=self._axis,
            node_color=node_colors,
            node_size=self.node_size,
            alpha=0.8,
        )
        
        # Draw agents
        if self.agents:
            for i, agent in enumerate(self.agents):
                current_node = agent.position()
                if current_node in self.pos:
                    # Update trajectory
                    if len(self.agent_trajectories[agent.name]) == 0 or \
                       self.agent_trajectories[agent.name][-1] != current_node:
                        self.agent_trajectories[agent.name].append(current_node)
                    
                    # Draw trajectory (last 50 nodes)
                    trajectory = self.agent_trajectories[agent.name][-50:]
                    if len(trajectory) > 1:
                        path_positions = [self.pos[node] for node in trajectory if node in self.pos]
                        if len(path_positions) > 1:
                            xs = [p[0] for p in path_positions]
                            ys = [p[1] for p in path_positions]
                            color = self.agent_colors[i % len(self.agent_colors)]
                            self._axis.plot(xs, ys, color=color, alpha=0.5, linewidth=2, linestyle='--')
                    
                    # Draw agent marker
                    x, y = self.pos[current_node]
                    color = self.agent_colors[i % len(self.agent_colors)]
                    self._axis.scatter(
                        [x], [y],
                        s=500,
                        c=color,
                        marker='*',
                        edgecolors='black',
                        linewidths=2,
                        zorder=10,
                        label=agent.name,
                    )
        
        # Title
        active_count = self.env.count_active()
        title = f"Graph Simulation (t={self.env.time}) | Active Events: {active_count}"
        if self.agents:
            agent_positions = ", ".join([f"{a.name}: node {a.position()}" for a in self.agents])
            title += f"\n{agent_positions}"
        self._axis.set_title(title, fontsize=12)
        
        if self.agents and frame == 0:
            self._axis.legend(loc='upper right', fontsize=8)
        
        return []
    
    def animate(
        self,
        frames: int = 200,
        interval: int = 200,
        repeat: bool = False,
        save_path: Optional[str] = None,
        dpi: int = 120,
        writer: Optional[str] = None,
        fps: Optional[int] = None,
        simulation: Optional[Any] = None,
    ):
        """
        Create and save/display animation.
        """
        self.simulation = simulation
        
        # Initialize plot
        fig, _ = self._init_plot()
        
        # Create update function
        update_func = self._update
        
        # Create animation
        print(f"Creating FuncAnimation with {frames} frames...")
        anim = FuncAnimation(
            fig,
            update_func,
            frames=frames,
            interval=interval,
            blit=False,
            repeat=repeat,
        )
        print(f"FuncAnimation created.")
        
        # Save animation if path provided
        if save_path:
            # Make path absolute
            save_path = os.path.abspath(save_path)
            
            # Ensure directory exists
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                print(f"Directory ensured: {save_dir}")
            
            # Determine writer
            if writer is None:
                ext = os.path.splitext(save_path)[1].lower()
                if ext == '.gif':
                    writer = 'pillow'
                elif ext in ['.mp4', '.avi', '.mov']:
                    writer = 'ffmpeg'
                else:
                    writer = 'pillow'
            
            # Calculate fps
            if fps is None:
                fps = 1000.0 / interval if interval > 0 else 30.0
            
            print(f"\n{'='*60}")
            print(f"SAVING ANIMATION")
            print(f"{'='*60}")
            print(f"Path: {save_path}")
            print(f"Writer: {writer}")
            print(f"FPS: {fps:.2f}")
            print(f"Frames: {frames}")
            print(f"DPI: {dpi}")
            print(f"{'='*60}\n")
            
            # SAVE THE ANIMATION
            try:
                print("Calling anim.save()...")
                anim.save(save_path, writer=writer, fps=fps, dpi=dpi)
                print("anim.save() completed successfully!")
                
                # Wait a moment for file system
                import time
                time.sleep(1.0)
                
                # Verify file was created
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    print(f"\n{'='*60}")
                    print(f"SUCCESS! Animation saved!")
                    print(f"File: {save_path}")
                    print(f"Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                    print(f"{'='*60}\n")
                else:
                    print(f"\n{'='*60}")
                    print(f"ERROR! File not found after save!")
                    print(f"Expected: {save_path}")
                    print(f"Directory exists: {os.path.exists(save_dir)}")
                    if os.path.exists(save_dir):
                        print(f"Files in directory: {os.listdir(save_dir)}")
                    print(f"{'='*60}\n")
                    raise RuntimeError(f"Animation file was not created at {save_path}")
                    
            except Exception as e:
                print(f"\n{'='*60}")
                print(f"ERROR SAVING ANIMATION!")
                print(f"Error: {type(e).__name__}: {e}")
                print(f"{'='*60}\n")
                import traceback
                traceback.print_exc()
                raise
        else:
            # Display animation
            plt.show()
        
        return anim


def plot_graph_state(
    env: "GraphEnvironment",
    agents: Optional[list["GraphAgent"]] = None,
    save_path: Optional[str] = None,
):
    """Plot a single snapshot of the graph state."""
    if not _VISUALIZATION_AVAILABLE:
        raise ImportError(
            "matplotlib and networkx are required for graph visualization "
            f"(encountered: {_IMPORT_ERROR})"
        )
    
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(env.num_nodes))
    G.add_edges_from(env.edges)
    
    # Compute positions
    if env.width and env.height:
        pos = {}
        for node in range(env.num_nodes):
            y, x = env.node_positions[node]
            pos[node] = (x, env.height - 1 - y)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Get state
    state = env.get_state()
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, alpha=0.3, edge_color='gray')
    
    node_colors = [
        "#d1495b" if state[node] == 1 else "#e0e0e0"
        for node in range(env.num_nodes)
    ]
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=300, alpha=0.8)
    
    # Draw agents
    if agents:
        agent_colors = ["#ff5733", "#33ff57", "#3357ff", "#ff33f5", "#f5ff33"]
        for i, agent in enumerate(agents):
            current_node = agent.position()
            if current_node in pos:
                x, y = pos[current_node]
                color = agent_colors[i % len(agent_colors)]
                ax.scatter(
                    [x], [y],
                    s=500,
                    c=color,
                    marker='*',
                    edgecolors='black',
                    linewidths=2,
                    zorder=10,
                    label=agent.name,
                )
        ax.legend(loc='upper right')
    
    # Title
    active_count = env.count_active()
    ax.set_title(f"Graph State (t={env.time}) | Active Events: {active_count}", fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Snapshot saved to: {save_path}")
    
    plt.close(fig)

