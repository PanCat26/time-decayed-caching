"""
Time-Decayed Caching (TDC) - Real-World Trace Experiments

Run experiments on the NASA HTTP access log (July 1995).
"""

import numpy as np
import pandas as pd
import re
import os

from experiment import ExperimentRunner, calculate_delta
from visualization import create_delta_table_image, create_sliding_window_graph


def parse_nasa_log(filepath: str, max_requests: int = None) -> tuple:
    """
    Parse NASA HTTP access log and extract URL requests.
    
    Returns:
        trace: numpy array of integer item IDs
        num_unique: number of unique items
    """
    print(f"  Loading: {filepath}")
    
    # Regex to extract the URL from each log line
    # Format: host - - [timestamp] "METHOD /url HTTP/x.x" status size
    url_pattern = re.compile(r'"[A-Z]+\s+([^\s]+)\s+HTTP')
    
    urls = []
    with open(filepath, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            if max_requests and i >= max_requests:
                break
            match = url_pattern.search(line)
            if match:
                urls.append(match.group(1))
    
    print(f"  Parsed {len(urls)} requests")
    
    # Convert URLs to integer IDs
    unique_urls = list(set(urls))
    url_to_id = {url: idx for idx, url in enumerate(unique_urls)}
    trace = np.array([url_to_id[url] for url in urls])
    
    print(f"  Unique items: {len(unique_urls)}")
    
    return trace, len(unique_urls)


def run_realworld_experiments():
    """Run experiments on real-world NASA trace."""
    
    print("=" * 60)
    print("Time-Decayed Caching - Real-World Trace Experiments")
    print("=" * 60)
    
    # Load NASA trace
    print("\n[1/4] Loading NASA HTTP access log...")
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'access_log_Jul95')
    
    # Use first 100,000 requests for reasonable runtime
    trace, num_unique = parse_nasa_log(data_path, max_requests=100000)
    
    traces = {'NASA HTTP Log': trace}
    
    # Parameters
    CACHE_SIZES_PERCENT = list(range(1, 17))  # 1% to 16%
    
    # Run Experiment 1: Hit ratios for all combinations
    print("\n[2/4] Running Experiment 1: Computing hit ratios...")
    
    runner = ExperimentRunner()
    all_results = {}  # trace -> cache_size -> algorithm -> hit_ratio
    
    for trace_name, tr in traces.items():
        print(f"  Processing: {trace_name}")
        all_results[trace_name] = {}
        
        for cache_pct in CACHE_SIZES_PERCENT:
            cache_size = max(1, int(num_unique * cache_pct / 100))
            all_results[trace_name][cache_pct] = {}
            
            for algo_name in runner.algorithms:
                stats = runner.run_single_experiment(tr, cache_size, algo_name)
                all_results[trace_name][cache_pct][algo_name] = stats['hit_ratio']
            
            # Print progress
            print(f"    Cache {cache_pct}%: Proposed={all_results[trace_name][cache_pct]['Proposed']:.3f}, "
                  f"LRU={all_results[trace_name][cache_pct]['LRU']:.3f}, "
                  f"LFU={all_results[trace_name][cache_pct]['LFU']:.3f}, "
                  f"ARC={all_results[trace_name][cache_pct]['ARC']:.3f}")
    
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
    create_delta_table_image(summary_df, 'delta_table_realworld.png')
    
    # Run Experiment 2: Sliding window analysis
    print("\n[4/4] Running Experiment 2: Sliding window analysis...")
    
    sliding_results = {}
    cache_pct = 5  # Use 5% cache size for sliding window experiment
    cache_size = max(1, int(num_unique * cache_pct / 100))
    
    for trace_name, tr in traces.items():
        print(f"  Processing: {trace_name} (cache size = {cache_size})")
        sliding_results[trace_name] = runner.run_sliding_window_experiment(
            tr, cache_size, window_size=1000
        )
    
    # Create sliding window graph
    create_sliding_window_graph(
        sliding_results, 
        list(traces.keys()),
        'sliding_window_realworld.png'
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("\nOutput files generated:")
    print("  1. delta_table_realworld.png - Delta values table")
    print("  2. sliding_window_realworld.png - Adaptability graph")
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
    
    # Open the images
    print("\nOpening result images...")
    import subprocess
    subprocess.Popen(['start', '', 'delta_table_realworld.png'], shell=True)
    subprocess.Popen(['start', '', 'sliding_window_realworld.png'], shell=True)


if __name__ == '__main__':
    run_realworld_experiments()
