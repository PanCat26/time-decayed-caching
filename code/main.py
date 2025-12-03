"""
Time-Decayed Caching (TDC) - Main Experiment Runner

Run this script to execute all experiments and generate output images.
"""

import numpy as np
import pandas as pd

from trace_generator import TraceGenerator
from experiment import ExperimentRunner, calculate_delta
from visualization import create_delta_table_image, create_sliding_window_graph


def run_all_experiments():
    """Run all experiments and generate output images."""
    
    print("=" * 60)
    print("Time-Decayed Caching - Experiment Suite")
    print("=" * 60)
    
    # Parameters
    NUM_ITEMS = 1000  # Total unique items
    NUM_REQUESTS = 50000  # Requests per trace
    CACHE_SIZES_PERCENT = list(range(1, 17))  # 1% to 16%
    SEED = 42
    
    # Generate synthetic traces
    print("\n[1/4] Generating synthetic traces...")
    
    traces = {}
    
    # Stationary traces with different Zipf alphas
    traces['Stationary (α=0.8)'] = TraceGenerator.zipf_stationary(
        NUM_ITEMS, NUM_REQUESTS, alpha=0.8, seed=SEED
    )
    traces['Stationary (α=1.2)'] = TraceGenerator.zipf_stationary(
        NUM_ITEMS, NUM_REQUESTS, alpha=1.2, seed=SEED + 1
    )
    
    # Non-stationary traces with different p_hot values
    traces['Non-Stationary (p_hot=0.7)'] = TraceGenerator.non_stationary_phases(
        NUM_ITEMS, num_phases=10, phase_length=NUM_REQUESTS // 10,
        alpha=1.0, p_hot=0.7, seed=SEED + 2
    )
    traces['Non-Stationary (p_hot=0.9)'] = TraceGenerator.non_stationary_phases(
        NUM_ITEMS, num_phases=10, phase_length=NUM_REQUESTS // 10,
        alpha=1.0, p_hot=0.9, seed=SEED + 3
    )
    
    print(f"  Generated {len(traces)} traces with {NUM_REQUESTS} requests each")
    
    # Run Experiment 1: Hit ratios for all combinations
    print("\n[2/4] Running Experiment 1: Computing hit ratios...")
    
    runner = ExperimentRunner()
    all_results = {}  # trace -> cache_size -> algorithm -> hit_ratio
    
    for trace_name, trace in traces.items():
        print(f"  Processing: {trace_name}")
        all_results[trace_name] = {}
        
        for cache_pct in CACHE_SIZES_PERCENT:
            cache_size = max(1, int(NUM_ITEMS * cache_pct / 100))
            all_results[trace_name][cache_pct] = {}
            
            for algo_name in runner.algorithms:
                stats = runner.run_single_experiment(trace, cache_size, algo_name)
                all_results[trace_name][cache_pct][algo_name] = stats['hit_ratio']
    
    # Calculate Delta values and create summary table
    print("\n[3/4] Computing Delta values and generating table...")
    
    # Create summary table with average deltas per trace
    summary_data = []
    for trace_name in traces:
        row = [trace_name]
        for algo in ['LRU', 'LFU', 'ARC']:
            deltas = []
            for cache_pct in CACHE_SIZES_PERCENT:
                proposed_hr = all_results[trace_name][cache_pct]['Proposed']
                algo_hr = all_results[trace_name][cache_pct][algo]
                deltas.append(calculate_delta(proposed_hr, algo_hr))
            avg_delta = np.mean(deltas)
            row.append(f"{avg_delta:+.2f}%")
        summary_data.append(row)
    
    summary_df = pd.DataFrame(
        summary_data,
        columns=['Trace', 'Δ vs LRU', 'Δ vs LFU', 'Δ vs ARC']
    )
    
    # Save table as image
    create_delta_table_image(summary_df, 'delta_table.png')
    
    # Run Experiment 2: Sliding window analysis on non-stationary traces
    print("\n[4/4] Running Experiment 2: Sliding window analysis...")
    
    non_stationary_traces = {
        name: trace for name, trace in traces.items() 
        if 'Non-Stationary' in name
    }
    
    sliding_results = {}
    cache_pct = 5  # Use 5% cache size for sliding window experiment
    cache_size = max(1, int(NUM_ITEMS * cache_pct / 100))
    
    for trace_name, trace in non_stationary_traces.items():
        print(f"  Processing: {trace_name}")
        sliding_results[trace_name] = runner.run_sliding_window_experiment(
            trace, cache_size, window_size=500
        )
    
    # Create sliding window graph
    create_sliding_window_graph(
        sliding_results, 
        list(non_stationary_traces.keys()),
        'sliding_window_graph.png'
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("\nOutput files generated:")
    print("  1. delta_table.png - Delta values table for Experiment 1")
    print("  2. sliding_window_graph.png - Adaptability graph for Experiment 2")
    print("\nSummary of results:")
    
    # Print average Delta values
    print("\nAverage Δ(j, c, Proposed) across all cache sizes:")
    for trace_name in traces:
        print(f"\n  {trace_name}:")
        for algo in ['LRU', 'LFU', 'ARC']:
            deltas = []
            for cache_pct in CACHE_SIZES_PERCENT:
                proposed_hr = all_results[trace_name][cache_pct]['Proposed']
                algo_hr = all_results[trace_name][cache_pct][algo]
                deltas.append(calculate_delta(proposed_hr, algo_hr))
            avg_delta = np.mean(deltas)
            print(f"    vs {algo}: {avg_delta:+.2f}%")


if __name__ == '__main__':
    run_all_experiments()
