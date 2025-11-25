"""
VIIRS → grid → local transition model (512-pattern table).

- Loads one or more VIIRS CSV files.
- Discretizes time into bins of length `bin_hours`.
- Builds a 30x60 lat-lon grid, with cells of 6° x 6°.
- Marks a cell as 'fire' (1) in a time bin if any detection in that cell
  exceeds a FRP or brightness threshold.
- Learns p(s_k(t+1) = 1 | s_k(t), s_{N(k)}(t)) as a 512-entry table
  over all 9-bit local patterns (center + 8 neighbors).
"""

from __future__ import annotations

import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# Matplotlib is optional for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# 1. Time parsing
# ---------------------------------------------------------------------------

def parse_viirs_datetime(row: pd.Series) -> pd.Timestamp:
    """
    Parse VIIRS acq_date + acq_time into a pandas Timestamp.

    Handles both:
      - 'YYYY-MM-DD' and 'MM/DD/YYYY' formats for acq_date.
      - acq_time as integer HHMM (e.g. 5 -> '0005', 123 -> '0123').
    """
    time_str = str(int(row["acq_time"])).zfill(4)
    hh = time_str[:2]
    mm = time_str[2:]
    date_str = str(row["acq_date"]).strip()

    # Try ISO format first, then fallback to US-style
    for fmt in ("%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M"):
        try:
            return pd.to_datetime(f"{date_str} {hh}:{mm}", format=fmt)
        except ValueError:
            continue

    # Last resort: let pandas guess
    return pd.to_datetime(f"{date_str} {hh}:{mm}", infer_datetime_format=True)


# ---------------------------------------------------------------------------
# 2. Load VIIRS CSVs and build time-binned, binary grids
# ---------------------------------------------------------------------------

def load_viirs_data(csv_paths: List[str]) -> pd.DataFrame:
    """
    Load and concatenate one or more VIIRS CSV files.

    Expects at least the following columns:
      - latitude, longitude, acq_date, acq_time, frp, brightness (optional)
    """
    dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No valid VIIRS CSV files found.")
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def build_binary_fire_frames(
    df: pd.DataFrame,
    bin_hours: int = 12,
    grid_shape: Tuple[int, int] = (30, 60),
    frp_threshold: float = 5.0,
    brightness_threshold: float | None = None,
) -> Dict[int, np.ndarray]:
    """
    Convert raw VIIRS point detections into a time-indexed sequence of
    binary fire grids.

    Args:
        df: VIIRS dataframe with columns: latitude, longitude,
            acq_date, acq_time, frp, brightness (optional).
        bin_hours: length of time bin in hours (e.g., 12 for two frames/day).
        grid_shape: (num_lat_cells, num_lon_cells), default (30, 60).
        frp_threshold: minimum FRP to consider a detection as fire.
        brightness_threshold: optional brightness threshold; if provided,
                             a detection is fire if frp >= frp_threshold OR
                             brightness >= brightness_threshold.

    Returns:
        frames: dict mapping integer time index t -> (H, W) numpy array of 0/1.
    """
    H, W = grid_shape

    # Parse datetime and sort
    df = df.copy()
    df["dt"] = df.apply(parse_viirs_datetime, axis=1)
    df = df.dropna(subset=["dt"])
    df = df.sort_values("dt")

    # Define time bin index
    t0 = df["dt"].min()
    seconds_per_bin = bin_hours * 3600
    df["t_idx"] = ((df["dt"] - t0).dt.total_seconds() // seconds_per_bin).astype(int)

    frames: Dict[int, np.ndarray] = {}
    
    print(f"\nBuilding fire frames:")
    print(f"  Time bins: {df['t_idx'].min()} to {df['t_idx'].max()}")
    print(f"  Unique time bins: {df['t_idx'].nunique()}")

    for t, group in df.groupby("t_idx"):
        grid = np.zeros((H, W), dtype=np.int8)

        for _, row in group.iterrows():
            frp = row.get("frp", np.nan)
            brightness = row.get("brightness", np.nan)

            # Fire condition
            fire_flag = False
            if not np.isnan(frp) and frp >= frp_threshold:
                fire_flag = True
            if brightness_threshold is not None and not np.isnan(brightness):
                if brightness >= brightness_threshold:
                    fire_flag = True

            if not fire_flag:
                continue

            lat = float(row["latitude"])
            lon = float(row["longitude"])

            # Map lat, lon to grid indices
            # Lat in [-90, 90], Lon in [-180, 180], cell size determined by H,W
            lat_step = 180.0 / H
            lon_step = 360.0 / W

            lat_idx = int((lat + 90.0) // lat_step)
            lon_idx = int((lon + 180.0) // lon_step)

            if 0 <= lat_idx < H and 0 <= lon_idx < W:
                grid[lat_idx, lon_idx] = 1

        frames[t] = grid
        if t < 5 or t == max(df['t_idx']):
            fire_count = grid.sum()
            print(f"  Time bin {t}: {len(group)} detections, {fire_count} grid cells with fire")

    print(f"  Total frames created: {len(frames)}")
    return frames


# ---------------------------------------------------------------------------
# 3. Learn 512-pattern local transition probabilities
# ---------------------------------------------------------------------------

NEIGHBOR_OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def encode_local_pattern(center: int, neighbors: List[int]) -> int:
    """
    Encode a 9-bit pattern (center + 8 neighbors) into an integer in [0, 511].

    Bit order: [center, n0, n1, ..., n7]  (each 0 or 1).
    """
    bits = [int(center)] + [int(b) for b in neighbors]
    r = 0
    for b in bits:
        r = (r << 1) | b
    return r


def decode_local_pattern(pattern_idx: int) -> Tuple[int, List[int]]:
    """
    Decode a pattern index into center and 8 neighbors.
    
    Args:
        pattern_idx: Integer in [0, 511]
    
    Returns:
        (center, neighbors) where center is 0 or 1, and neighbors is list of 8 bits
    """
    bits = []
    val = pattern_idx
    for _ in range(9):
        bits.append(val & 1)
        val >>= 1
    bits.reverse()  # Most significant bit first
    center = bits[0]
    neighbors = bits[1:]
    return center, neighbors


def pattern_to_string(pattern_idx: int) -> str:
    """
    Convert pattern index to a human-readable string representation.
    
    Format: 3x3 grid with center in the middle:
    Top row:    neighbors[0] neighbors[1] neighbors[2]
    Middle row: neighbors[3] center      neighbors[4]
    Bottom row: neighbors[5] neighbors[6] neighbors[7]
    
    Neighbor order: (-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)
    """
    center, neighbors = decode_local_pattern(pattern_idx)
    # Arrange neighbors in 3x3 grid (center is at position 1,1)
    # Neighbors order: (-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)
    grid = [
        [neighbors[0], neighbors[1], neighbors[2]],  # Top row
        [neighbors[3], center, neighbors[4]],         # Middle row (center in middle)
        [neighbors[5], neighbors[6], neighbors[7]],   # Bottom row
    ]
    return "\n".join([" ".join(str(cell) for cell in row) for row in grid])


def learn_location_specific_transition_table(
    frames: Dict[int, np.ndarray],
    grid_shape: Tuple[int, int],
    smoothing_alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Learn location-specific transition probabilities for each grid cell.
    
    For each (lat_idx, lon_idx) location, learns a 512-entry transition table
    p(s_k(t+1) = 1 | local pattern at t, location).

    Args:
        frames: dict t -> (H, W) binary grids.
        grid_shape: (H, W) grid dimensions.
        smoothing_alpha: Laplace smoothing parameter alpha.

    Returns:
        p_fire: array of shape (H, W, 512), p_fire[i,j,r] = p(s(t+1)=1 | pattern r at location i,j).
        count: array of shape (H, W, 512), total occurrences of pattern r at location i,j.
        count_fire: array of shape (H, W, 512), fire occurrences for pattern r at location i,j.
    """
    if not frames:
        raise ValueError("No frames provided.")

    H, W = grid_shape
    
    # Initialize 3D arrays: [lat_idx, lon_idx, pattern_idx]
    count = np.zeros((H, W, 512), dtype=np.int64)
    count_fire = np.zeros((H, W, 512), dtype=np.int64)

    # Sort time indices and keep only consecutive pairs
    times = sorted(frames.keys())
    
    print(f"\nDebug: Found {len(times)} time frames")
    if len(times) > 0:
        print(f"  Time range: {times[0]} to {times[-1]}")
        print(f"  First few time indices: {times[:10]}")
    
    consecutive_pairs = 0
    skipped_pairs = 0
    all_pairs = 0

    for idx in range(len(times) - 1):
        t = times[idx]
        t_next = times[idx + 1]
        all_pairs += 1
        
        # Only use consecutive bins (no gaps)
        if t_next != t + 1:
            skipped_pairs += 1
            if skipped_pairs <= 5:  # Show first few skipped pairs
                print(f"  Warning: Skipping non-consecutive pair: {t} -> {t_next} (gap: {t_next - t})")
            continue
        
        consecutive_pairs += 1

        grid_t = frames[t]
        grid_next = frames[t_next]

        # Sanity check shapes
        assert grid_t.shape == (H, W), f"Expected shape ({H}, {W}), got {grid_t.shape}"
        assert grid_next.shape == (H, W), f"Expected shape ({H}, {W}), got {grid_next.shape}"

        # For each location (lat_idx, lon_idx) in the grid
        for lat_idx in range(H):
            for lon_idx in range(W):
                center = grid_t[lat_idx, lon_idx]

                # Get neighbors (handle boundaries)
                neighbors = []
                for di, dj in NEIGHBOR_OFFSETS_8:
                    ni = lat_idx + di
                    nj = lon_idx + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        neighbors.append(grid_t[ni, nj])
                    else:
                        # Outside grid: treat as 0 (no fire)
                        neighbors.append(0)

                # Encode the local pattern
                pattern_idx = encode_local_pattern(center, neighbors)
                
                # Get next state at this location
                next_state = int(grid_next[lat_idx, lon_idx])

                # Update counts for this specific location and pattern
                count[lat_idx, lon_idx, pattern_idx] += 1
                if next_state == 1:
                    count_fire[lat_idx, lon_idx, pattern_idx] += 1
    
    print(f"\nTime pair statistics:")
    print(f"  Total pairs available: {all_pairs}")
    print(f"  Consecutive pairs used: {consecutive_pairs}")
    print(f"  Skipped pairs (non-consecutive): {skipped_pairs}")
    if all_pairs > 0:
        print(f"  Consecutive pair rate: {100*consecutive_pairs/all_pairs:.1f}%")
    print(f"  Total pattern observations: {count.sum():,}")
    
    if consecutive_pairs == 0 and all_pairs > 0:
        print(f"\nWARNING: No consecutive time pairs found!")
        print(f"  This means all time bins have gaps. Consider:")
        print(f"  - Using larger time bins (increase bin_hours)")
        print(f"  - Or modify code to allow non-consecutive pairs")

    # Compute probabilities with Laplace smoothing for each location
    p_fire = np.zeros((H, W, 512), dtype=np.float64)
    
    for lat_idx in range(H):
        for lon_idx in range(W):
            for pattern_idx in range(512):
                if count[lat_idx, lon_idx, pattern_idx] == 0:
                    # If pattern never appears at this location, set probability to 0
                    p_fire[lat_idx, lon_idx, pattern_idx] = 0.0
                else:
                    # Compute probability with Laplace smoothing
                    p_fire[lat_idx, lon_idx, pattern_idx] = (
                        count_fire[lat_idx, lon_idx, pattern_idx] + smoothing_alpha
                    ) / (count[lat_idx, lon_idx, pattern_idx] + 2 * smoothing_alpha)

    return p_fire, count, count_fire


# ---------------------------------------------------------------------------
# 4. Export and visualization
# ---------------------------------------------------------------------------

def save_location_specific_transition_table_to_csv(
    p_fire: np.ndarray,
    count: np.ndarray,
    count_fire: np.ndarray,
    grid_shape: Tuple[int, int],
    output_path: str = "local_transition_table.csv",
) -> None:
    """
    Save location-specific transition table to CSV.
    
    Args:
        p_fire: Array of shape (H, W, 512) with transition probabilities
        count: Array of shape (H, W, 512) with total pattern occurrences
        count_fire: Array of shape (H, W, 512) with fire occurrences
        grid_shape: (H, W) grid dimensions
        output_path: Path to save CSV file
    """
    H, W = grid_shape
    
    # Compute lat/lon centers for each grid cell
    lat_step = 180.0 / H
    lon_step = 360.0 / W
    lat_centers = -90.0 + (np.arange(H) + 0.5) * lat_step
    lon_centers = -180.0 + (np.arange(W) + 0.5) * lon_step
    
    data = []
    for lat_idx in range(H):
        for lon_idx in range(W):
            lat_center = lat_centers[lat_idx]
            lon_center = lon_centers[lon_idx]
            
            for pattern_idx in range(512):
                center, neighbors = decode_local_pattern(pattern_idx)
                active_neighbors = sum(neighbors)
                
                # Create pattern string representation
                pattern_str = "".join([str(center)] + [str(n) for n in neighbors])
                
                # Create 3x3 grid representation
                grid_str = pattern_to_string(pattern_idx).replace("\n", " | ")
                
                data.append({
                    "lat_idx": lat_idx,
                    "lon_idx": lon_idx,
                    "lat_center": lat_center,
                    "lon_center": lon_center,
                    "pattern_index": pattern_idx,
                    "pattern_binary": pattern_str,
                    "pattern_grid": grid_str,
                    "center_state": center,
                    "active_neighbors": active_neighbors,
                    "total_count": int(count[lat_idx, lon_idx, pattern_idx]),
                    "fire_count": int(count_fire[lat_idx, lon_idx, pattern_idx]),
                    "p_fire": float(p_fire[lat_idx, lon_idx, pattern_idx]),
                    "p_no_fire": 1.0 - float(p_fire[lat_idx, lon_idx, pattern_idx]),
                })
    
    df = pd.DataFrame(data)
    
    # Add a comment header explaining the table
    header_comment = (
        "# Location-Specific Transition Probability Table for Fire Spreading\n"
        "# This table contains transition probabilities for each grid cell location.\n"
        "# Each row represents: (lat_idx, lon_idx, pattern_index) -> transition probabilities\n"
        "# - p_fire: P(fire at t+1 | pattern at t, location)\n"
        "# - p_no_fire: P(no fire at t+1 | pattern at t, location) = 1 - p_fire\n"
        "# Pattern grid format: top row | middle row (center in middle) | bottom row\n"
        "# Neighbor order: (-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)\n"
    )
    
    # Write comment header, then CSV data
    with open(output_path, 'w') as f:
        f.write(header_comment)
        df.to_csv(f, index=False)
    
    print(f"Saved location-specific transition table to {output_path}")
    print(f"  Total entries: {len(df):,} (H={H}, W={W}, patterns=512)")
    print(f"  Locations with data: {(count.sum(axis=2) > 0).sum():,} / {H*W}")


def visualize_location_specific_transition_table(
    p_fire: np.ndarray,
    count: np.ndarray,
    count_fire: np.ndarray,
    grid_shape: Tuple[int, int],
    save_path: str | None = None,
    figsize: Tuple[int, int] = (16, 12),
) -> None:
    """
    Create visualizations of the location-specific transition table.
    
    Args:
        p_fire: Array of shape (H, W, 512) with transition probabilities
        count: Array of shape (H, W, 512) with total pattern occurrences
        count_fire: Array of shape (H, W, 512) with fire occurrences
        grid_shape: (H, W) grid dimensions
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return
    
    H, W = grid_shape
    
    # Flatten across all locations for global statistics
    p_fire_flat = p_fire.flatten()
    count_flat = count.flatten()
    count_fire_flat = count_fire.flatten()
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Transition probabilities histogram (all locations)
    ax = axes[0, 0]
    observed_probs = p_fire_flat[count_flat > 0]
    if len(observed_probs) > 0:
        ax.hist(observed_probs, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Transition Probability P(fire | pattern, location)")
        ax.set_ylabel("Number of (location, pattern) pairs")
        ax.set_title(f"Distribution of Transition Probabilities\n({len(observed_probs):,} observed)")
    else:
        ax.text(0.5, 0.5, "No patterns observed", ha='center', va='center')
    ax.grid(True, alpha=0.3)
    
    # 2. Spatial map of average transition probability per location
    ax = axes[0, 1]
    # Average p_fire across all patterns for each location (weighted by count)
    avg_p_fire = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            total = count[i, j, :].sum()
            if total > 0:
                # Weighted average
                avg_p_fire[i, j] = np.average(p_fire[i, j, :], weights=count[i, j, :])
            else:
                avg_p_fire[i, j] = np.nan
    
    im = ax.imshow(avg_p_fire, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel("Longitude Index")
    ax.set_ylabel("Latitude Index")
    ax.set_title("Average Transition Probability by Location")
    plt.colorbar(im, ax=ax, label='Avg P(fire)')
    
    # 3. Spatial map of data coverage (locations with observations)
    ax = axes[0, 2]
    coverage = (count.sum(axis=2) > 0).astype(float)
    im = ax.imshow(coverage, aspect='auto', cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel("Longitude Index")
    ax.set_ylabel("Latitude Index")
    ax.set_title(f"Data Coverage\n({coverage.sum():.0f}/{H*W} locations)")
    plt.colorbar(im, ax=ax, label='Has Data')
    
    # 4. Pattern counts distribution (log scale)
    ax = axes[1, 0]
    non_zero_count = count_flat[count_flat > 0]
    if len(non_zero_count) > 0:
        ax.hist(non_zero_count, bins=50, edgecolor='black', alpha=0.7)
        ax.set_yscale('log')
        ax.set_xlabel("Pattern Occurrence Count")
        ax.set_ylabel("Number of (location, pattern) pairs (log scale)")
        ax.set_title("Distribution of Pattern Occurrences")
    else:
        ax.text(0.5, 0.5, "No patterns observed", ha='center', va='center')
    ax.grid(True, alpha=0.3)
    
    # 5. Transition probability vs active neighbors (aggregated)
    ax = axes[1, 1]
    neighbor_counts = []
    probabilities = []
    for pattern_idx in range(512):
        center, neighbors = decode_local_pattern(pattern_idx)
        active_neighbors = sum(neighbors)
        # Get probabilities for this pattern across all locations
        pattern_probs = p_fire[:, :, pattern_idx].flatten()
        pattern_counts = count[:, :, pattern_idx].flatten()
        # Only include locations where this pattern was observed
        observed = pattern_counts > 0
        if observed.any():
            neighbor_counts.extend([active_neighbors] * observed.sum())
            probabilities.extend(pattern_probs[observed].tolist())
    
    if len(probabilities) > 0:
        neighbor_counts = np.array(neighbor_counts)
        probabilities = np.array(probabilities)
        unique_neighbors = np.unique(neighbor_counts)
        box_data = [probabilities[neighbor_counts == n] for n in unique_neighbors]
        
        bp = ax.boxplot(box_data, labels=[f"{int(n)}" for n in unique_neighbors], 
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_xlabel("Number of Active Neighbors")
        ax.set_ylabel("Transition Probability")
        ax.set_title("Transition Probability vs Active Neighbors\n(All Locations)")
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Total observations per location
    ax = axes[1, 2]
    total_obs = count.sum(axis=2)
    # Handle zeros for log scale: replace 0 with a small value for visualization
    total_obs_plot = total_obs.copy().astype(float)
    if (total_obs_plot == 0).any():
        # Replace zeros with 0.5 so they appear on log scale
        total_obs_plot[total_obs_plot == 0] = 0.5
        # Only use LogNorm if we have non-zero values
        if total_obs.max() > 0:
            im = ax.imshow(total_obs_plot, aspect='auto', cmap='viridis', interpolation='nearest', norm=mcolors.LogNorm(vmin=0.5, vmax=total_obs.max()))
        else:
            im = ax.imshow(total_obs_plot, aspect='auto', cmap='viridis', interpolation='nearest')
    else:
        im = ax.imshow(total_obs_plot, aspect='auto', cmap='viridis', interpolation='nearest', norm=mcolors.LogNorm())
    ax.set_xlabel("Longitude Index")
    ax.set_ylabel("Latitude Index")
    ax.set_title("Total Observations per Location\n(log scale)")
    plt.colorbar(im, ax=ax, label='Total Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_location_specific_transition_table_summary(
    p_fire: np.ndarray,
    count: np.ndarray,
    count_fire: np.ndarray,
    grid_shape: Tuple[int, int],
) -> None:
    """Print a summary of the location-specific transition table statistics."""
    H, W = grid_shape
    
    print("\n" + "=" * 70)
    print("Location-Specific Transition Table Summary")
    print("=" * 70)
    
    total_entries = H * W * 512
    observed_entries = np.sum(count > 0)
    locations_with_data = (count.sum(axis=2) > 0).sum()
    
    print(f"\nGrid Statistics:")
    print(f"  Grid size: {H} × {W} = {H*W} locations")
    print(f"  Total entries: {total_entries:,} (locations × patterns)")
    print(f"  Observed entries: {observed_entries:,} ({100*observed_entries/total_entries:.2f}%)")
    print(f"  Locations with data: {locations_with_data} / {H*W} ({100*locations_with_data/(H*W):.1f}%)")
    
    # Flatten for statistics
    p_fire_flat = p_fire.flatten()
    count_flat = count.flatten()
    count_fire_flat = count_fire.flatten()
    
    observed_mask = count_flat > 0
    observed_probs = p_fire_flat[observed_mask]
    
    print(f"\nTransition Probability Statistics (observed only):")
    if len(observed_probs) > 0:
        print(f"  Min: {observed_probs.min():.6f}")
        print(f"  Max: {observed_probs.max():.6f}")
        print(f"  Mean: {observed_probs.mean():.6f}")
        print(f"  Median: {np.median(observed_probs):.6f}")
        print(f"  Std: {observed_probs.std():.6f}")
    else:
        print("  No observed patterns")
    
    print(f"\nPattern Count Statistics:")
    print(f"  Total occurrences: {count.sum():,}")
    print(f"  Total fire occurrences: {count_fire.sum():,}")
    print(f"  Global fire rate: {count_fire.sum() / max(count.sum(), 1):.6f}")
    if count_flat[count_flat > 0].size > 0:
        print(f"  Max pattern count: {count.max():,}")
        print(f"  Mean pattern count (observed): {count_flat[count_flat > 0].mean():.2f}")
    
    # Find top patterns across all locations
    print(f"\nTop 10 (location, pattern) pairs by Occurrence Count:")
    flat_indices = np.argsort(count_flat)[::-1][:10]
    for flat_idx in flat_indices:
        lat_idx = flat_idx // (W * 512)
        remainder = flat_idx % (W * 512)
        lon_idx = remainder // 512
        pattern_idx = remainder % 512
        
        center, neighbors = decode_local_pattern(pattern_idx)
        active_neighbors = sum(neighbors)
        print(f"  Location ({lat_idx:2d}, {lon_idx:2d}), Pattern {pattern_idx:3d}: "
              f"count={count[lat_idx, lon_idx, pattern_idx]:8,}, "
              f"fire_count={count_fire[lat_idx, lon_idx, pattern_idx]:6,}, "
              f"p_fire={p_fire[lat_idx, lon_idx, pattern_idx]:.6f}, "
              f"center={center}, active_neighbors={active_neighbors}")
    
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# 5. Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: adapt these paths to your setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_paths = [
        os.path.join(script_dir, "fire_nrt_J1V-C2_685922.csv"),
        os.path.join(script_dir, "fire_nrt_J2V-C2_689024.csv"),
    ]

    # 1) Load VIIRS
    df_viirs = load_viirs_data(csv_paths)

    # 2) Build binary grids with configurable bin length (hours)
    bin_hours = 1/60  # <-- change here if you want 6, 24, etc.
    frp_threshold = 30
    frames = build_binary_fire_frames(
        df_viirs,
        bin_hours=bin_hours,
        grid_shape=(30, 60),
        frp_threshold=frp_threshold,
        brightness_threshold=None,  # or e.g. 330.0
    )

    # 3) Learn location-specific 512-pattern transition table
    grid_shape = (30, 60)
    p_fire, count, count_fire = learn_location_specific_transition_table(
        frames, 
        grid_shape=grid_shape,
        smoothing_alpha=1.0
    )

    # 4) Save results as .npy files
    project_root = os.path.dirname(script_dir)
    
    np.save(os.path.join(project_root, "local_transition_p_fire.npy"), p_fire)
    np.save(os.path.join(project_root, "local_transition_count.npy"), count)
    np.save(os.path.join(project_root, "local_transition_count_fire.npy"), count_fire)
    print(f"\nSaved .npy files to {project_root}")
    print(f"  Shape: {p_fire.shape} (H={grid_shape[0]}, W={grid_shape[1]}, patterns=512)")

    # 5) Save results as CSV
    csv_path = os.path.join(project_root, "local_transition_table.csv")
    save_location_specific_transition_table_to_csv(
        p_fire, count, count_fire, grid_shape, output_path=csv_path
    )

    # 6) Print summary statistics
    print_location_specific_transition_table_summary(p_fire, count, count_fire, grid_shape)

    # 7) Create visualizations
    viz_path = os.path.join(project_root, "local_transition_table_visualization.png")
    visualize_location_specific_transition_table(
        p_fire, count, count_fire, grid_shape, save_path=viz_path
    )
    
    print("\nDone! Check the generated files:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Visualization: {viz_path}")
    print(f"  - .npy files: {project_root}/local_transition_*.npy")
