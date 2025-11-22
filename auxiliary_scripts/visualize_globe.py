"""3D globe visualization for the spherical wildfire environment."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Matplotlib is optional; guard import so the module can be imported without it
try:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
except Exception as exc:  # pragma: no cover - handled gracefully at runtime
    plt = None
    FuncAnimation = None
    mcolors = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    if __package__:
        from .environment import Environment, RSPParams
    else:
        raise ImportError
except ImportError:
    import os
    import sys

    # Add parent directory to path to access environment_simulation module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environment_simulation.environment import Environment, RSPParams

try:
    if __package__:
        from .agents.agent import Agent
        from .agents.simple_agent import SimpleAgent
        from .agents.trajectory import Trajectory
        from .agents.belief import Belief
    else:
        raise ImportError
except ImportError:
    # Add parent directory to path to access environment_simulation module
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environment_simulation.agents.agent import Agent
    from environment_simulation.agents.simple_agent import SimpleAgent
    from environment_simulation.agents.trajectory import Trajectory
    from environment_simulation.agents.belief import Belief


class GlobeFireAnimator:
    """Animate fire propagation on a spherical grid."""

    def __init__(
        self,
        env: Environment,
        radius: float = 1.0,
        fire_color: str = "#d1495b",
        calm_color: str = "#1f77b4",
        field_color: str = "#ffd700",  # Gold/yellow for Field of Regard (FOR)
        observation_color: str = "#9d00ff",  # Purple for Field of View (FOV/selected action)
        agents: list[Agent] | None = None,
        simulation: Any | None = None,  # Optional simulation object to get current actions
    ):
        if env.topology != "sphere":
            raise ValueError("Environment topology must be 'sphere' for globe visualization.")
        if plt is None or FuncAnimation is None:
            raise ImportError(
                "matplotlib is required for GlobeFireAnimator "
                f"(encountered: {_IMPORT_ERROR})"
            )

        self.env = env
        self.radius = radius
        self.fire_color = fire_color
        self.calm_color = calm_color
        self.field_color = field_color
        self.observation_color = observation_color
        self.agents = agents or []
        self.simulation = simulation  # Store simulation reference for action tracking

        # Pre-compute coordinates for cell centers on the sphere
        self.lat_centers = 90.0 - (np.arange(env.height) + 0.5) * (180.0 / env.height)
        self.lon_centers = (np.arange(env.width) + 0.5) * (360.0 / env.width)
        lat_grid, lon_grid = np.meshgrid(self.lat_centers, self.lon_centers, indexing="ij")

        self.x_coords, self.y_coords, self.z_coords = self._latlon_to_xyz(lat_grid, lon_grid)
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid

        # Holder for the scatter object once created
        self._scatter = None
        self._coverage_scatter = None
        self._observation_scatter = None
        self._figure = None
        self._axis = None
        self._agent_markers: list = []
        self._agent_traces: list = []

    def _latlon_to_xyz(self, lat: np.ndarray, lon: np.ndarray):
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        x = self.radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = self.radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = self.radius * np.sin(lat_rad)
        return x, y, z

    def _init_plot(self):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect((1, 1, 1))
        ax.axis("off")

        # Base sphere surface for context
        u = np.linspace(0, 2 * np.pi, 80)
        v = np.linspace(0, np.pi, 40)
        xs = self.radius * np.outer(np.cos(u), np.sin(v))
        ys = self.radius * np.outer(np.sin(u), np.sin(v))
        zs = self.radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, color="#dddddd", alpha=0.2, linewidth=0, zorder=0)

        # Initial scatter of active/inactive cells
        state = self.env.get_state().astype(bool).flatten()
        base_colors = np.where(state, self.fire_color, self.calm_color)

        self._scatter = ax.scatter(
            self.x_coords.flatten(),
            self.y_coords.flatten(),
            self.z_coords.flatten(),
            c=base_colors,
            s=40,
            depthshade=False,
            edgecolor="none",
            zorder=1,
        )

        if self.agents and mcolors is not None:
            empty_rgba = np.zeros((state.size, 4))
            # Field of Regard (FOR) overlay - gold/yellow, semi-transparent
            self._coverage_scatter = ax.scatter(
                self.x_coords.flatten(),
                self.y_coords.flatten(),
                self.z_coords.flatten(),
                c=empty_rgba,
                s=75,  # Larger size for FOR visibility
                depthshade=False,
                edgecolor="none",
                zorder=1.5,
            )
            
            # Field of View (FOV) - selected action overlay - purple, prominent
            self._observation_scatter = ax.scatter(
                self.x_coords.flatten(),
                self.y_coords.flatten(),
                self.z_coords.flatten(),
                c=empty_rgba,
                s=80,  # Even larger size for FOV to make it stand out
                depthshade=False,
                edgecolor="none",
                zorder=1.7,  # Higher z-order to render on top of FOR
            )

            for agent in self.agents:
                traj_x, traj_y, traj_z = self._latlon_to_xyz(agent.trajectory.latitudes, agent.trajectory.longitudes)
                trace, = ax.plot(traj_x, traj_y, traj_z, color=agent.color, linewidth=1.2, alpha=0.6, zorder=2)
                self._agent_traces.append(trace)

                lat, lon = agent.position(self.env.time)
                px, py, pz = self._latlon_to_xyz(np.array([lat]), np.array([lon]))
                marker = ax.scatter(
                    px,
                    py,
                    pz,
                    color=agent.color,
                    s=120,
                    depthshade=False,
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=3,
                )
                self._agent_markers.append(marker)

        ax.set_title(f"Fire propagation (t={self.env.time})")

        self._figure = fig
        self._axis = ax
        return fig, ax

    def _update(self, frame: int):
        # Skip env.step() if simulation is provided (simulation handles stepping)
        if frame > 0 and self.simulation is None:
            self.env.step()

        state = self.env.get_state().astype(bool).flatten()
        base_colors = np.where(state, self.fire_color, self.calm_color)
        self._scatter.set_color(base_colors)

        artists = [self._scatter]

        if self.agents and self._coverage_scatter is not None and mcolors is not None:
            coverage = np.zeros(state.size, dtype=bool)
            current_actions = np.zeros(state.size, dtype=bool)
            
            for agent, marker in zip(self.agents, self._agent_markers):
                # Current field of regard
                for_mask = agent.coverage_mask(self.lat_grid, self.lon_grid, self.env.time)
                coverage |= for_mask.reshape(-1)
                
                lat, lon = agent.position(self.env.time)
                px, py, pz = self._latlon_to_xyz(np.array([lat]), np.array([lon]))
                marker._offsets3d = (px.flatten(), py.flatten(), pz.flatten())
                
                # Get the action taken at current step from simulation
                if self.simulation is not None:
                    current_step = self.env.time
                    # Debug: print available steps (only for first few steps)
                    if current_step <= 2 and len(self.simulation.current_actions) > 0:
                        available_steps = sorted(list(self.simulation.current_actions.keys()))
                        print(f"  DEBUG: Step {current_step}, Available action steps: {available_steps[:10]}")
                    
                    if current_step in self.simulation.current_actions:
                        if agent.name in self.simulation.current_actions[current_step]:
                            action_mask = self.simulation.current_actions[current_step][agent.name]
                            # Ensure action_mask is the correct shape
                            if action_mask.shape == (self.env.height, self.env.width):
                                # Flatten and combine with other agents' actions
                                action_flat = action_mask.flatten()
                                num_cells = np.sum(action_flat)
                                if num_cells > 0:
                                    current_actions |= action_flat
                                    if current_step <= 2:
                                        print(f"  DEBUG: Added {num_cells} FOV cells for {agent.name} at step {current_step}")
                                else:
                                    if current_step <= 2:
                                        print(f"  DEBUG: Empty action mask for {agent.name} at step {current_step}")
                            else:
                                # If shape doesn't match, try to reshape
                                try:
                                    action_mask_flat = action_mask.reshape(self.env.height, self.env.width).flatten()
                                    current_actions |= action_mask_flat
                                except ValueError:
                                    # Shape mismatch - skip this agent's action
                                    if current_step <= 2:
                                        print(f"  DEBUG: Shape mismatch for {agent.name}: {action_mask.shape} vs ({self.env.height}, {self.env.width})")
                        else:
                            if current_step <= 2:
                                agent_names = list(self.simulation.current_actions[current_step].keys())
                                print(f"  DEBUG: {agent.name} not in step {current_step}. Available agents: {agent_names}")
                    else:
                        if current_step <= 2:
                            available = sorted(list(self.simulation.current_actions.keys()))
                            print(f"  DEBUG: Step {current_step} not found. Available steps: {available[:5]}")

            # Field of Regard (FOR) overlay - gold/yellow, semi-transparent
            # Show all FOR cells, but make cells that are also in FOV slightly darker
            highlight_rgba = np.zeros((state.size, 4))
            if np.any(coverage):
                # FOR cells that are NOT in FOV - lighter gold
                for_only = coverage & (~current_actions)
                if np.any(for_only):
                    highlight_rgba[for_only] = mcolors.to_rgba(self.field_color, alpha=0.5)
                # FOR cells that ARE in FOV - darker gold (will be overlaid by red)
                for_and_fov = coverage & current_actions
                if np.any(for_and_fov):
                    highlight_rgba[for_and_fov] = mcolors.to_rgba("#ffaa00", alpha=0.3)  # Darker gold
            self._coverage_scatter.set_color(highlight_rgba)
            artists.append(self._coverage_scatter)
            
            # Field of View (FOV) - selected action overlay - purple, highly visible
            if self._observation_scatter is not None:
                obs_rgba = np.zeros((state.size, 4))
                num_total_fov_cells = np.sum(current_actions)
                if num_total_fov_cells > 0:
                    # Purple for the selected action (FOV) - fully opaque
                    # Set color to purple for cells in current_actions
                    purple_rgba = mcolors.to_rgba(self.observation_color, alpha=1.0)
                    obs_rgba[current_actions] = purple_rgba
                # Always set the scatter color, even if empty (to clear previous frame)
                self._observation_scatter.set_color(obs_rgba)
                artists.append(self._observation_scatter)
            
            artists.extend(self._agent_markers)

        # Update title with legend information
        title = f"Fire propagation (t={self.env.time}) | FOR: Gold, FOV: Purple"
        self._axis.set_title(title)
        return tuple(artists)

    def animate(
        self,
        frames: int = 200,
        interval: int = 200,
        repeat: bool = False,
        save_path: str | None = None,
        dpi: int = 120,
        writer: str | None = None,
        fps: int | None = None,
    ):
        """
        Create and display (and optionally save) the animation.
        
        Args:
            frames: Number of frames to animate
            interval: Interval between frames in milliseconds
            repeat: Whether to repeat the animation
            save_path: Path to save the animation (e.g., 'animation.mp4' or 'animation.gif')
            dpi: DPI for saved animation
            writer: Writer to use for saving (e.g., 'ffmpeg', 'pillow', 'html'). 
                   If None, automatically selects based on file extension
            fps: Frames per second for saved animation. If None, calculated from interval
        """
        fig, _ = self._init_plot()

        anim = FuncAnimation(
            fig,
            self._update,
            frames=frames,
            interval=interval,
            blit=False,  # 3D artists do not support blitting well
            repeat=repeat,
        )

        if save_path:
            # Verify save_path is a valid string
            if not isinstance(save_path, str) or not save_path.strip():
                print(f"WARNING: Invalid save_path provided: {save_path}")
                print("Animation will be displayed but not saved.")
                save_path = None
        
        if save_path:
            # Calculate FPS from interval if not provided
            if fps is None:
                fps = 1000 / interval if interval > 0 else 10
            
            # Auto-detect writer from file extension if not specified
            if writer is None:
                ext = save_path.lower().split('.')[-1]
                if ext in ['mp4', 'avi', 'mov']:
                    writer = 'ffmpeg'
                elif ext in ['gif']:
                    writer = 'pillow'
                elif ext in ['html']:
                    writer = 'html'
                else:
                    # Default to ffmpeg for video formats
                    writer = 'ffmpeg'
            
            print(f"\n{'='*70}")
            print(f"Saving animation to {save_path}...")
            print(f"  Writer: {writer}")
            print(f"  FPS: {fps:.2f}")
            print(f"  DPI: {dpi}")
            print(f"  Frames: {frames}")
            print(f"{'='*70}\n")
            
            try:
                if writer == 'ffmpeg':
                    # Use ffmpeg writer (requires ffmpeg installed)
                    # Create writer instance first, then pass to save()
                    from matplotlib.animation import FFMpegWriter
                    # FFMpegWriter automatically handles codec and pixel format for MP4
                    ffmpeg_writer = FFMpegWriter(fps=fps, metadata=dict(artist='Matplotlib'))
                    anim.save(save_path, writer=ffmpeg_writer, dpi=dpi)
                elif writer == 'pillow':
                    # Use pillow writer for GIF
                    # Note: GIF files can be large for many frames
                    anim.save(
                        save_path,
                        writer='pillow',
                        fps=fps,
                        dpi=dpi
                    )
                elif writer == 'html':
                    # Save as HTML5 video
                    anim.save(
                        save_path,
                        writer='html',
                        fps=fps,
                        dpi=dpi
                    )
                else:
                    # Fallback to default writer
                    anim.save(save_path, writer=writer, fps=fps, dpi=dpi)
                
                print(f"\n{'='*70}")
                print(f"Animation saved successfully to: {save_path}")
                # Verify file was created
                abs_path = os.path.abspath(save_path)
                if os.path.exists(abs_path):
                    file_size = os.path.getsize(abs_path)
                    print(f"File size: {file_size / (1024*1024):.2f} MB")
                    print(f"Full path: {abs_path}")
                else:
                    print(f"WARNING: File {abs_path} was not created!")
                    print(f"Current working directory: {os.getcwd()}")
                print(f"{'='*70}\n")
            except FileNotFoundError as e:
                # FFmpeg not found - offer to fallback to GIF
                print(f"\n{'='*70}")
                print(f"Error: FFmpeg not found on your system")
                print(f"{'='*70}")
                if writer == 'ffmpeg':
                    print("\nFFmpeg is required for MP4/AVI/MOV video formats.")
                    print("Options:")
                    print("  1. Install FFmpeg:")
                    print("     - Windows: Download from https://ffmpeg.org/download.html")
                    print("       Add ffmpeg.exe to your system PATH")
                    print("     - macOS: brew install ffmpeg")
                    print("     - Linux: sudo apt-get install ffmpeg")
                    print("\n  2. Use GIF format instead (no ffmpeg needed):")
                    print(f"     Change save_path to '{save_path.rsplit('.', 1)[0]}.gif'")
                    print(f"     Or modify the code to use: save_path='animation.gif'")
                    
                    # Offer automatic fallback to GIF
                    gif_path = save_path.rsplit('.', 1)[0] + '.gif'
                    print(f"\nAttempting to save as GIF instead: {gif_path}")
                    try:
                        anim.save(gif_path, writer='pillow', fps=fps, dpi=dpi)
                        print(f"\n{'='*70}")
                        print(f"Animation saved successfully as GIF: {gif_path}")
                        abs_gif_path = os.path.abspath(gif_path)
                        if os.path.exists(abs_gif_path):
                            file_size = os.path.getsize(abs_gif_path)
                            print(f"File size: {file_size / (1024*1024):.2f} MB")
                            print(f"Full path: {abs_gif_path}")
                        else:
                            print(f"WARNING: File {abs_gif_path} was not created!")
                            print(f"Current working directory: {os.getcwd()}")
                        print(f"{'='*70}\n")
                        # Return early since we successfully saved as GIF
                        # Close figure before returning
                        plt.close(fig)
                        return anim
                    except Exception as gif_error:
                        print(f"Failed to save as GIF: {gif_error}")
                        import traceback
                        traceback.print_exc()
                        raise e  # Re-raise original ffmpeg error
                else:
                    raise
            except Exception as e:
                print(f"\n{'='*70}")
                print(f"Error saving animation: {e}")
                print(f"{'='*70}")
                if writer == 'ffmpeg':
                    print("\nNote: For video formats (MP4, AVI, MOV), you need to install ffmpeg:")
                    print("  - Windows: Download from https://ffmpeg.org/download.html")
                    print("    Make sure ffmpeg.exe is in your system PATH")
                    print("  - macOS: brew install ffmpeg")
                    print("  - Linux: sudo apt-get install ffmpeg")
                elif writer == 'pillow':
                    print("\nNote: For GIF format, pillow should be available with matplotlib.")
                    print("If not, install it with: pip install pillow")
                raise
            finally:
                # Only close figure if we successfully saved or if there was an error
                # Don't close if we're also showing the animation
                if save_path:
                    plt.close(fig)
        
        # Show the animation (if not saving)
        if save_path is None:
            plt.show()

        return anim


def demo():
    """Run a demo animation with a spherical RSP environment."""
    height, width = 30, 60
    base_rng = np.random.default_rng(1)
    lat = np.linspace(-1.0, 1.0, height).reshape(-1, 1)
    lon = np.linspace(0.0, 2 * np.pi, width)
    lat_grid = np.repeat(lat, width, axis=1)
    lon_grid = np.tile(lon, (height, 1))

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

    blocked_mask = np.zeros((height, width), dtype=bool)
    blocked_mask[:, : width // 18] = True  # permanent ocean strip
    blocked_mask |= base_rng.random((height, width)) < 0.05

    env = Environment(
        width=width,
        height=height,
        mode="rsp",
        topology="sphere",
        rsp_params=RSPParams(lam=0.02, beta0=0.001, alpha=0.03, delta=0.92),
        ignition_map=lam_map,
        alpha_map=alpha_map,
        beta0_map=beta0_map,
        persistence_map=persistence_map,
        blocked_mask=blocked_mask,
        seed=7,
    )
    env.reset(initial_events=8)

    orbit_period = 240
    agents = [
        SimpleAgent(
            name="Aurora-1",
            trajectory=Trajectory.circular_orbit(period=orbit_period, inclination_deg=25.0, phase_deg=0.0),
            belief=Belief(height=height, width=width, prior_probability=0.0),
            field_of_regard_deg=14.0,
            color="#ffd166",
        ),
        SimpleAgent(
            name="Borealis-2",
            trajectory=Trajectory.circular_orbit(period=orbit_period, inclination_deg=55.0, phase_deg=120.0, latitude_offset=5.0),
            belief=Belief(height=height, width=width, prior_probability=0.0),
            field_of_regard_deg=18.0,
            color="#4ecdc4",
        ),
        SimpleAgent(
            name="Zenith-3",
            trajectory=Trajectory.circular_orbit(period=orbit_period, inclination_deg=10.0, phase_deg=240.0, latitude_offset=-8.0),
            belief=Belief(height=height, width=width, prior_probability=0.0),
            field_of_regard_deg=20.0,
            color="#ff6b6b",
        ),
    ]

    animator = GlobeFireAnimator(env, agents=agents)
    animator.animate(frames=300, interval=120)


if __name__ == "__main__":
    demo()

