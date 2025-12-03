"""
Visualization functions for cache experiment results.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import pandas as pd
from typing import Dict, List


def create_delta_table_image(results_df: pd.DataFrame, output_path: str):
    """Create a publication-ready summary table image of average Delta values."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.45, 0.18, 0.18, 0.18]  # First column wider
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 2.0)
    
    # Color header cells
    for i in range(len(results_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors and color-code delta values
    for i in range(len(results_df)):
        for j in range(len(results_df.columns)):
            if i % 2 == 0:
                table[(i + 1, j)].set_facecolor('#E9EDF4')
            else:
                table[(i + 1, j)].set_facecolor('#FFFFFF')
            
            # First column (trace name) in bold
            if j == 0:
                table[(i + 1, j)].set_text_props(fontweight='bold')
    
    plt.title('Average Δ(j, c, Proposed) Across All Cache Sizes (%)\n' + 
              '(Positive = Proposed outperforms baseline)',
              fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Delta table saved to: {output_path}")


def create_sliding_window_graph(results: Dict[str, Dict[str, List[float]]], 
                                 trace_names: List[str],
                                 output_path: str):
    """Create sliding window hit ratio graphs for non-stationary traces."""
    n_traces = len(trace_names)
    fig, axes = plt.subplots(1, n_traces, figsize=(7 * n_traces, 5))
    
    if n_traces == 1:
        axes = [axes]
    
    colors = {
        'Proposed': '#2E86AB',
        'LRU': '#A23B72', 
        'LFU': '#F18F01',
        'ARC': '#C73E1D'
    }
    
    for ax, trace_name in zip(axes, trace_names):
        trace_results = results[trace_name]
        
        for algo_name, hit_ratios in trace_results.items():
            # Subsample for plotting (every 100th point)
            step = max(1, len(hit_ratios) // 500)
            x = list(range(0, len(hit_ratios), step))
            y = [hit_ratios[i] for i in x]
            
            ax.plot(x, y, label=algo_name, color=colors[algo_name], 
                   linewidth=1.5, alpha=0.9)
        
        ax.set_xlabel('Request Number', fontsize=11)
        ax.set_ylabel('Sliding Window Hit Ratio', fontsize=11)
        ax.set_title(f'{trace_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.suptitle('Algorithm Adaptability: Sliding Window Hit Ratio Over Time\n' +
                 '(Non-Stationary Traces with Phase Changes)', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Sliding window graph saved to: {output_path}")
