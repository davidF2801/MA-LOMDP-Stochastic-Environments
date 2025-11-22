"""Post-processing script to plot simulation results from CSV files."""

from __future__ import annotations

import argparse
import os
import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_csv_results(csv_path: str) -> Dict:
    """
    Read results from a CSV file and extract agent statistics.
    
    Args:
        csv_path: Path to the CSV results file
    
    Returns:
        Dictionary with agent statistics and simulation metadata
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Extract agent statistics
    # The CSV format has columns: Agent, Fires Observed, Total Reward, Observations Count, Avg Reward/Obs
    agents_data = {}
    
    # Find the row where agent data starts (skip header and any metadata rows)
    for idx, row in df.iterrows():
        agent_name = str(row.get('Agent', ''))
        # Skip empty rows, summary rows, etc.
        if pd.isna(agent_name) or agent_name.strip() == '' or 'Agent' in agent_name:
            continue
        if 'Simulation Summary' in agent_name or 'Total Steps' in agent_name:
            break
        
        try:
            fires_observed = int(row.get('Fires Observed', 0))
            total_reward = float(row.get('Total Reward', 0.0))
            observations_count = int(row.get('Observations Count', 0))
            agents_data[agent_name] = {
                'fires_observed': fires_observed,
                'total_reward': total_reward,
                'observations_count': observations_count,
            }
        except (ValueError, TypeError):
            continue
    
    # Extract simulation metadata from the bottom of the CSV
    metadata = {}
    for idx in range(len(df) - 1, -1, -1):  # Search from bottom
        row = df.iloc[idx]
        key = str(row.iloc[0]) if len(row) > 0 else ''
        if 'Total Steps' in key:
            metadata['total_steps'] = int(row.iloc[1]) if len(row) > 1 else 0
        elif 'Planning Horizon' in key:
            metadata['planning_horizon'] = int(row.iloc[1]) if len(row) > 1 else 0
        elif 'Environment Height' in key:
            metadata['height'] = int(row.iloc[1]) if len(row) > 1 else 0
        elif 'Environment Width' in key:
            metadata['width'] = int(row.iloc[1]) if len(row) > 1 else 0
    
    return {
        'agents': agents_data,
        'metadata': metadata,
    }


def aggregate_results_by_method(csv_files: List[str]) -> Dict[str, Dict]:
    """
    Aggregate results from multiple CSV files, grouping by agent type (method).
    
    Args:
        csv_files: List of CSV file paths
    
    Returns:
        Dictionary mapping method name -> aggregated statistics
    """
    method_results = {}
    
    for csv_path in csv_files:
        # Extract method name from filename (e.g., "animation_monte_carlo_results.csv" -> "monte_carlo")
        filename = os.path.basename(csv_path)
        if '_results.csv' in filename:
            method_name = filename.replace('_results.csv', '').replace('animation_', '')
        else:
            method_name = os.path.splitext(filename)[0]
        
        try:
            results = read_csv_results(csv_path)
            
            # Aggregate agent statistics for this method
            total_fires = sum(agent['fires_observed'] for agent in results['agents'].values())
            total_reward = sum(agent['total_reward'] for agent in results['agents'].values())
            total_observations = sum(agent['observations_count'] for agent in results['agents'].values())
            num_agents = len(results['agents'])
            
            method_results[method_name] = {
                'total_fires_detected': total_fires,
                'total_reward': total_reward,
                'total_observations': total_observations,
                'num_agents': num_agents,
                'avg_fires_per_agent': total_fires / num_agents if num_agents > 0 else 0,
                'avg_reward_per_agent': total_reward / num_agents if num_agents > 0 else 0,
                'metadata': results['metadata'],
            }
        except Exception as e:
            print(f"Warning: Failed to read {csv_path}: {e}")
            continue
    
    return method_results


def plot_fires_detected(method_results: Dict[str, Dict], save_path: str | None = None):
    """
    Create a bar plot comparing total fires detected by each method.
    
    Args:
        method_results: Dictionary mapping method name -> statistics
        save_path: Optional path to save the plot
    """
    methods = list(method_results.keys())
    total_fires = [method_results[m]['total_fires_detected'] for m in methods]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot total fires detected
    colors = ['#4ecdc4', '#ffd166', '#ff6b6b', '#95e1d3', '#aa96da', '#fcbad3']
    bars = ax.bar(methods, total_fires, color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Fires Detected', fontsize=12, fontweight='bold')
    ax.set_title('Total Fires Detected by Method', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {os.path.abspath(save_path)}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison_comprehensive(method_results: Dict[str, Dict], save_path: str | None = None):
    """
    Create comprehensive comparison plots for all metrics (method-level only).
    
    Args:
        method_results: Dictionary mapping method name -> statistics
        save_path: Optional path to save the plot
    """
    methods = list(method_results.keys())
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    colors = ['#4ecdc4', '#ffd166', '#ff6b6b', '#95e1d3', '#aa96da', '#fcbad3']
    
    # Plot 1: Total fires detected
    ax1 = fig.add_subplot(gs[0, 0])
    total_fires = [method_results[m]['total_fires_detected'] for m in methods]
    bars1 = ax1.bar(methods, total_fires, color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Fires Detected', fontsize=11, fontweight='bold')
    ax1.set_title('Total Fires Detected', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Total reward
    ax2 = fig.add_subplot(gs[0, 1])
    total_reward = [method_results[m]['total_reward'] for m in methods]
    bars2 = ax2.bar(methods, total_reward, color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Total Reward', fontsize=11, fontweight='bold')
    ax2.set_title('Total Cumulative Reward', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Total observations
    ax3 = fig.add_subplot(gs[1, 0])
    total_obs = [method_results[m]['total_observations'] for m in methods]
    bars3 = ax3.bar(methods, total_obs, color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Total Observations', fontsize=11, fontweight='bold')
    ax3.set_title('Total Cells Observed', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Detection efficiency (fires per observation)
    ax4 = fig.add_subplot(gs[1, 1])
    efficiency = []
    for m in methods:
        total_obs = method_results[m]['total_observations']
        total_fires = method_results[m]['total_fires_detected']
        eff = total_fires / total_obs if total_obs > 0 else 0
        efficiency.append(eff)
    bars4 = ax4.bar(methods, efficiency, color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Detection Efficiency', fontsize=11, fontweight='bold')
    ax4.set_title('Fires Detected per Observation', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Simulation Results Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comprehensive plot saved to: {os.path.abspath(save_path)}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to plot results from CSV files."""
    parser = argparse.ArgumentParser(description='Plot simulation results from CSV files')
    parser.add_argument(
        '--csv-dir',
        type=str,
        default='.',
        help='Directory containing CSV result files (default: current directory)'
    )
    parser.add_argument(
        '--csv-files',
        type=str,
        nargs='+',
        default=None,
        help='Specific CSV files to plot (default: auto-detect *_results.csv files)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results_comparison.png',
        help='Output path for the plot (default: results_comparison.png)'
    )
    parser.add_argument(
        '--comprehensive',
        action='store_true',
        help='Create comprehensive comparison with multiple metrics'
    )
    
    args = parser.parse_args()
    
    # Find CSV files
    if args.csv_files:
        csv_files = args.csv_files
    else:
        csv_pattern = os.path.join(args.csv_dir, '*_results.csv')
        csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {args.csv_dir}")
        print("Looking for files matching pattern: *_results.csv")
        return
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f}")
    
    # Read and aggregate results
    print("\nReading CSV files...")
    method_results = aggregate_results_by_method(csv_files)
    
    if not method_results:
        print("No valid results found in CSV files.")
        return
    
    print("\nAggregated results:")
    for method, stats in method_results.items():
        print(f"  {method:15s}: {stats['total_fires_detected']:4d} fires, "
              f"{stats['total_reward']:7.2f} reward, {stats['num_agents']} agents")
    
    # Create plots
    print(f"\nCreating plot(s)...")
    if args.comprehensive:
        plot_comparison_comprehensive(method_results, save_path=args.output)
    else:
        plot_fires_detected(method_results, save_path=args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()

