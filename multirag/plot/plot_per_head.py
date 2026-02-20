# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# author:
# Yi Zhu

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os


def load_results(
    per_head_path: str,
    regular_path: str,
    category_path: str | None = None,
    category_dist_path: str | None = None
) -> tuple[dict, dict, dict | None, dict | None]:
    """
    Load per-head, regular results, category coverage, and category distribution JSON files.

    :param per_head_path: Path to per-head data.
    :type per_head_path: str
    :param regular_path: Path to overall strategy results, that are always generated.
    :type regular_path: str
    :param category_path: Path to the category coverage data. Defaults to None.
    :type category_path: str | None
    :param category_dist_path: Path to the category distribution data. Defaults to None.
    :type category_dist_path: str | None
    :return: Tuple of the per-head, regular results, category coverage, and category distribution data.
    :rtype: tuple[dict, dict, dict | None, dict | None]
    """
    with open(per_head_path, 'r') as f:
        per_head_data = json.load(f)
    with open(regular_path, 'r') as f:
        regular_data = json.load(f)
    
    category_data = None
    if category_path and os.path.exists(category_path):
        with open(category_path, 'r') as f:
            category_data = json.load(f)
    
    category_dist_data = None
    if category_dist_path and os.path.exists(category_dist_path):
        with open(category_dist_path, 'r') as f:
            category_dist_data = json.load(f)
    
    return per_head_data, regular_data, category_data, category_dist_data


def plot_per_head_coverage(
    per_head_data: dict,
    regular_data: dict,
    category_data: dict | None,
    category_dist_data: dict | None,
    strategy_name: str,
    output_dir: str,
    metric: str = 'success_ratio'
) -> None:
    """
    Plot per-head coverage metrics for a strategy.
    
    :param per_head_data: Dictionary with per-head results.
    :type per_head_data: dict
    :param regular_data: Dictionary with overall strategy results.
    :type regular_data: dict
    :param category_data: Dictionary with per-head category coverage (optional).
    :type category_data: dict | None
    :param category_dist_data: Dictionary with per-head category distribution (optional).
    :type category_dist_data: dict | None
    :param strategy_name: Name of the strategy (e.g., 'multirag').
    :type strategy_name: str
    :param output_dir: Directory to store plots.
    :type output_dir: str
    :param metric: Metric to plot ('success_ratio', 'category_success_ratio', 'success', 'category_success'). Defaults to 'success_ratio'.
    :type metric: str
    """
    if strategy_name not in per_head_data:
        print(f"Strategy {strategy_name} not found in per-head data")
        return
    
    if strategy_name not in regular_data:
        print(f"Strategy {strategy_name} not found in regular data")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    per_head_strategy = per_head_data[strategy_name]
    regular_strategy = regular_data[strategy_name]
    
    # Process each n_rel (number of relevant documents)
    for n_rel_str, per_head_results in per_head_strategy.items():
        n_rel = int(n_rel_str)
        
        # Get regular results for this n_rel
        regular_results = regular_strategy[n_rel_str]
        total_coverage = regular_results[metric]  # List of lists [n_docs][query_idx]
        
        # Extract data for each head
        num_heads = len([k for k in per_head_results.keys() if k.startswith('head_')])
        num_n_docs = len(per_head_results['head_0'][metric])  # Number of different n values
        num_queries = len(per_head_results['head_0'][metric][0])  # Number of queries
        
        # Collect per-head data: [n_docs, query_idx, head_idx]
        head_data = np.zeros((num_n_docs, num_queries, num_heads))
        for head_idx in range(num_heads):
            head_key = f'head_{head_idx}'
            head_metric = per_head_results[head_key][metric]  # [n_docs][query_idx]
            for n_idx in range(num_n_docs):
                for q_idx in range(num_queries):
                    head_data[n_idx, q_idx, head_idx] = head_metric[n_idx][q_idx]
        
        # Calculate statistics for each query at each n_docs
        avg_per_head = np.mean(head_data, axis=2)  # [n_docs, query_idx]
        std_per_head = np.std(head_data, axis=2)   # [n_docs, query_idx]
        min_per_head = np.min(head_data, axis=2)   # [n_docs, query_idx]
        max_per_head = np.max(head_data, axis=2)   # [n_docs, query_idx]
        
        # Convert total_coverage to numpy array [n_docs, query_idx]
        total_coverage_arr = np.array(total_coverage)
        
        # Create an aggregate plot showing average across all queries
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(1, num_n_docs + 1)
        
        # Average across all queries
        avg_across_queries = np.mean(avg_per_head, axis=1)
        std_across_queries = np.mean(std_per_head, axis=1)
        min_across_queries = np.mean(min_per_head, axis=1)
        max_across_queries = np.mean(max_per_head, axis=1)
        total_across_queries = np.mean(total_coverage_arr, axis=1)
        
        ax.errorbar(x, avg_across_queries, yerr=std_across_queries, 
                   label='Average per head (±std)', marker='o', capsize=5, linewidth=2)
        ax.plot(x, min_across_queries, label='Min coverage (head)', 
               marker='s', linestyle='--', linewidth=2)
        ax.plot(x, max_across_queries, label='Max coverage (head)', 
               marker='^', linestyle='--', linewidth=2)
        ax.plot(x, total_across_queries, label='Total coverage (MultiRAG)', 
               marker='D', linestyle='-', linewidth=2.5, color='red')
        
        ax.set_xlabel('Number of Documents Retrieved', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{strategy_name} - Average across queries (n_rel={n_rel})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, num_n_docs + 0.5)
        
        if 'ratio' in metric:
            ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        output_filename = f'{strategy_name}_n_rel_{n_rel}_average_{metric}.pdf'
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved aggregate plot: {output_path}")
        
        # Create histogram plots showing distribution of relevant documents covered per head
        create_coverage_histogram(head_data, n_rel, num_n_docs, num_queries, num_heads, 
                                strategy_name, output_dir, metric)
        
        # Create per-head contribution plot
        category_results = None
        if category_data and strategy_name in category_data and n_rel_str in category_data[strategy_name]:
            category_results = category_data[strategy_name][n_rel_str]
        
        create_accumulated_coverage_plot(per_head_results, category_results, n_rel, num_n_docs, num_queries, num_heads,
                                        strategy_name, output_dir, metric)
        
        # Create transposed plot for specific heads
        if category_results:
            create_transposed_category_plot(category_results, n_rel, strategy_name, output_dir, selected_heads=[7, 10, 21])
            create_single_head_plot(category_results, n_rel, strategy_name, output_dir, head_idx=7)
        
        # Create mutual information heatmap if category distribution data is available
        if category_dist_data and strategy_name in category_dist_data and n_rel_str in category_dist_data[strategy_name]:
            category_dist_results = category_dist_data[strategy_name][n_rel_str]
            create_mutual_information_heatmap(category_dist_results, num_heads, n_rel, strategy_name, output_dir)
            create_category_head_correlation_heatmap(category_dist_results, num_heads, n_rel, strategy_name, output_dir)


def create_accumulated_coverage_plot(
    per_head_results: dict,
    category_results: dict | None,
    n_rel: int,
    num_n_docs: int,
    num_queries: int,
    num_heads: int,
    strategy_name: str,
    output_dir: str,
    metric: str
) -> None:
    """
    Create a plot showing the total coverage contribution per head, colored by category.

    :param per_head_results: Dictionary with per-head results for this n_rel.
    :type per_head_results: dict
    :param category_results: Dictionary with per-head category distribution for this n_rel (optional).
    :type category_results: dict | None
    :param n_rel: Number of relevant documents.
    :type n_rel: int
    :param num_n_docs: Number of different retrieval sizes.
    :type num_n_docs: int
    :param num_queries: Number of queries.
    :type num_queries: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param strategy_name: Name of the strategy.
    :type strategy_name: str
    :param output_dir: Directory to store plots.
    :type output_dir: str
    :param metric: Metric to be plotted.
    :type metric: str
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(num_heads)
    
    if category_results:
        # Create stacked bar chart with categories
        # First, collect all unique categories
        all_categories = set()
        for head_idx in range(num_heads):
            head_key = f'head_{head_idx}'
            if head_key in category_results:
                all_categories.update(category_results[head_key].keys())
        
        all_categories = sorted(list(all_categories))
        
        # Create a color map
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_categories)))
        category_colors = {cat: colors[i] for i, cat in enumerate(all_categories)}
        
        # Build the stacked data
        category_counts = {cat: np.zeros(num_heads) for cat in all_categories}
        
        for head_idx in range(num_heads):
            head_key = f'head_{head_idx}'
            if head_key in category_results:
                for category, count in category_results[head_key].items():
                    category_counts[category][head_idx] = count
        
        # Create stacked bars
        bottom = np.zeros(num_heads)
        for category in all_categories:
            ax.bar(x, category_counts[category], bottom=bottom, 
                  label=category, color=category_colors[category], 
                  edgecolor='black', linewidth=0.5)
            bottom += category_counts[category]
        
        # Add legend
        ax.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        total = np.sum(bottom)
    else:
        # Fallback to simple bar chart if no category data is available
        total_coverage_per_head = np.zeros(num_heads)
        
        for head_idx in range(num_heads):
            head_key = f'head_{head_idx}'
            head_metric = per_head_results[head_key][metric]
            # Get the final n_docs value for all queries and sum
            final_coverage = head_metric[-1]  # List of coverage for each query
            
            # Convert to counts if ratio metric
            if 'ratio' in metric:
                counts = np.array(final_coverage) * n_rel
            else:
                counts = np.array(final_coverage)
            
            total_coverage_per_head[head_idx] = np.sum(counts)
        
        ax.bar(x, total_coverage_per_head, color='steelblue', alpha=0.7, edgecolor='black')
        total = np.sum(total_coverage_per_head)
    
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Total Documents Retrieved (across queries)', fontsize=12)
    ax.set_title(f'{strategy_name} - Document Retrieval per Head by Category (n_rel={n_rel})', fontsize=14)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text showing total
    ax.text(0.02, 0.98, f'Total documents: {total:.0f}\n({num_queries} queries)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', 
            horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_filename = f'{strategy_name}_n_rel_{n_rel}_per_head_contribution_{metric}.pdf'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved per-head contribution plot: {output_path}")


def create_transposed_category_plot(
    category_results: dict,
    n_rel: int,
    strategy_name: str,
    output_dir: str,
    selected_heads: list[int] = [7, 10, 21]
) -> None:
    """
    Create a transposed plot showing category distribution for selected heads.
    
    :param category_results: Dictionary with per-head category distribution for this n_rel.
    :type category_results: dict
    :param n_rel: Number of relevant documents.
    :type n_rel: int
    :param strategy_name: Name of the strategy.
    :type strategy_name: str
    :param output_dir: Directory to store plots.
    :type output_dir: str
    :param selected_heads: List of head indices to plot. Defaults to [7, 10, 21].
    :type selected_heads: list[int]
    """
    # Collect all unique categories
    all_categories = set()
    for head_idx in selected_heads:
        head_key = f'head_{head_idx}'
        if head_key in category_results:
            all_categories.update(category_results[head_key].keys())
    
    all_categories = sorted(list(all_categories))
    
    if len(all_categories) == 0:
        print(f"No categories found for selected heads {selected_heads}")
        return
    
    # Create 3 subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    
    for i, head_idx in enumerate(selected_heads):
        ax = axes[i]
        head_key = f'head_{head_idx}'
        
        # Get counts for this head
        counts = []
        for category in all_categories:
            if head_key in category_results:
                count = category_results[head_key].get(category, 0)
            else:
                count = 0
            counts.append(count)
        
        # Create thin bars with minimal spacing
        x = np.arange(len(all_categories))
        ax.bar(x, counts, width=0.9, color=colors[i], edgecolor='none')
        
        # Remove category names, show just the tendency
        ax.set_xticks([])
        ax.set_xlabel(f'Categories', fontsize=10)
        ax.set_title(f'Head {head_idx}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Only show y-label on the leftmost plot
        if i == 0:
            ax.set_ylabel('Number of Documents', fontsize=11)
    
    fig.suptitle(f'{strategy_name} - Category Distribution Tendency (n_rel={n_rel})', fontsize=14)
    plt.tight_layout()
    
    output_filename = f'{strategy_name}_n_rel_{n_rel}_selected_heads_categories.pdf'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved transposed category plot for selected heads: {output_path}")


def create_single_head_plot(
    category_results: dict,
    n_rel: int,
    strategy_name: str,
    output_dir: str,
    head_idx: int = 7
) -> None:
    """
    Create a compact plot showing category distribution for a single head.
    
    :param category_results: Dictionary with per-head category distribution for this n_rel.
    :type category_results: dict
    :param n_rel: Number of relevant documents.
    :type n_rel: int
    :param strategy_name: Name of the strategy.
    :type strategy_name: str
    :param output_dir: Directory to store plots.
    :type output_dir: str
    :param head_idx: Head index to display. Defaults to 7.
    :type head_idx: int
    """
    head_key = f'head_{head_idx}'
    
    if head_key not in category_results:
        print(f"No data found for head {head_idx}")
        return
    
    # Get categories and counts for this head
    category_counts = category_results[head_key]
    all_categories = sorted(category_counts.keys())
    
    if len(all_categories) == 0:
        print(f"No categories found for head {head_idx}")
        return
    
    counts = [category_counts[cat] for cat in all_categories]
    
    # Create a small, vertically compact plot
    fig, ax = plt.subplots(figsize=(10, 2))
    
    x = np.arange(len(all_categories))
    ax.bar(x, counts, width=0.9, color='#1f77b4', edgecolor='none')
    
    # Remove category names
    ax.set_xticks([])
    ax.set_yticks([100, 200, 300])
    # Set y ticks font size larger
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('Categories', fontsize=20)
    ax.set_ylabel('Count', fontsize=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_filename = f'{strategy_name}_n_rel_{n_rel}_head_{head_idx}_categories.pdf'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved single head plot for head {head_idx}: {output_path}")


def create_coverage_histogram(
    head_data: np.ndarray,
    n_rel: int,
    num_n_docs: int,
    num_queries: int,
    num_heads: int,
    strategy_name: str,
    output_dir: str,
    metric: str
) -> None:
    """
    Create histogram plot showing how many relevant documents are covered per head.
    
    :param head_data: Numpy array [n_docs, query_idx, head_idx].
    :type head_data: np.ndarray
    :param n_rel: Number of relevant documents.
    :type n_rel: int
    :param num_n_docs: Number of different retrieval sizes.
    :type num_n_docs: int
    :param num_queries: Number of queries.
    :type num_queries: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param strategy_name: Name of the strategy.
    :type strategy_name: str
    :param output_dir: Directory to store plots.
    :type output_dir: str
    :param metric: Metric to be plotted.
    :type metric: str
    """
    # For ratio metrics, we need to convert back to counts
    # For binary metrics, we just count successes
    
    # Focus on the final retrieval size (maximum n_docs)
    final_data = head_data[-1, :, :]  # [num_queries, num_heads]
    
    # Convert ratios to counts if needed
    if 'ratio' in metric:
        # Multiply by n_rel to get actual counts
        counts_data = np.round(final_data * n_rel).astype(int)
    else:
        # For binary success, count how many were successful
        counts_data = (final_data * n_rel).astype(int)
    
    # Flatten to get all head-query combinations
    all_counts = counts_data.flatten()  # Total: num_queries * num_heads values
    
    # Create single histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.arange(0, n_rel + 2) - 0.5  # Bins centered on integers
    
    ax.hist(all_counts, bins=bins, edgecolor='black', color='steelblue', alpha=0.7)
    
    ax.set_xlabel('Number of Relevant Documents Covered', fontsize=12)
    ax.set_ylabel('Frequency (total head count)', fontsize=12)
    ax.set_title(f'{strategy_name} - Coverage Distribution across all Heads (n_rel={n_rel})', fontsize=14)
    ax.set_xticks(range(0, n_rel + 1))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text showing total count
    total_count = len(all_counts)
    ax.text(0.98, 0.98, f'Total: {total_count} head-query pairs\n({num_heads} heads × {num_queries} queries)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', 
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_filename = f'{strategy_name}_n_rel_{n_rel}_coverage_histogram_{metric}.pdf'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved histogram plot: {output_path}")


def create_mutual_information_heatmap(
    category_dist_results: dict,
    num_heads: int,
    n_rel: int,
    strategy_name: str,
    output_dir: str
) -> None:
    """
    Create a heatmap showing mutual information between pairs of heads based on category distributions.
    
    :param category_dist_results: Dictionary with per-head category distribution for this n_rel.
    :type category_dist_results: dict
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param n_rel: Number of relevant documents.
    :type n_rel: int
    :param strategy_name: Name of the strategy.
    :type strategy_name: str
    :param output_dir: Directory to store plots.
    :type output_dir: str
    """
    from sklearn.metrics import mutual_info_score
    
    # Collect all unique categories
    all_categories = set()
    for head_idx in range(num_heads):
        head_key = f'head_{head_idx}'
        if head_key in category_dist_results:
            all_categories.update(category_dist_results[head_key].keys())
    
    all_categories = sorted(list(all_categories))
    
    # Build category distributions for each head
    # Each head's distribution is a vector of counts per category
    head_distributions = []
    for head_idx in range(num_heads):
        head_key = f'head_{head_idx}'
        distribution = []
        if head_key in category_dist_results:
            for category in all_categories:
                count = category_dist_results[head_key].get(category, 0)
                distribution.append(count)
        else:
            distribution = [0] * len(all_categories)
        head_distributions.append(distribution)
    
    # Calculate mutual information matrix
    mi_matrix = np.zeros((num_heads, num_heads))
    
    for i in range(num_heads):
        for j in range(num_heads):
            if i == j:
                # Self-information (entropy)
                dist_i = head_distributions[i]
                total_i = sum(dist_i)
                if total_i > 0:
                    # Calculate entropy
                    probs = np.array(dist_i) / total_i
                    probs = probs[probs > 0]  # Remove zeros to avoid log(0)
                    entropy = -np.sum(probs * np.log2(probs))
                    mi_matrix[i][j] = entropy
                else:
                    mi_matrix[i][j] = 0
            else:
                # Mutual information between two heads
                dist_i = head_distributions[i]
                dist_j = head_distributions[j]
                
                # Skip if either distribution is empty
                if sum(dist_i) == 0 or sum(dist_j) == 0:
                    mi_matrix[i][j] = 0
                    continue
                
                # Create category labels for MI calculation
                # Replicate category indices according to their counts
                labels_i = []
                labels_j = []
                
                for cat_idx, category in enumerate(all_categories):
                    count_i = dist_i[cat_idx]
                    count_j = dist_j[cat_idx]
                    # Use the minimum count to create paired observations
                    min_count = min(count_i, count_j)
                    labels_i.extend([cat_idx] * min_count)
                    labels_j.extend([cat_idx] * min_count)
                    
                    # Add unpaired observations
                    if count_i > min_count:
                        labels_i.extend([cat_idx] * (count_i - min_count))
                        labels_j.extend([-1] * (count_i - min_count))  # Placeholder
                    if count_j > min_count:
                        labels_i.extend([-1] * (count_j - min_count))  # Placeholder
                        labels_j.extend([cat_idx] * (count_j - min_count))
                
                if len(labels_i) > 0 and len(labels_j) > 0:
                    mi = mutual_info_score(labels_i, labels_j)
                    mi_matrix[i][j] = mi
                else:
                    mi_matrix[i][j] = 0
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(mi_matrix, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mutual Information (bits)', fontsize=12)
    
    # Set ticks
    ax.set_xticks(np.arange(num_heads))
    ax.set_yticks(np.arange(num_heads))
    ax.set_xticklabels(np.arange(num_heads))
    ax.set_yticklabels(np.arange(num_heads))
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Head Index', fontsize=12)
    ax.set_title(f'{strategy_name} - Mutual Information between Heads (n_rel={n_rel})', fontsize=14)
    
    plt.tight_layout()
    
    output_filename = f'{strategy_name}_n_rel_{n_rel}_mutual_information_heatmap.pdf'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved mutual information heatmap: {output_path}")


def create_category_head_correlation_heatmap(
    category_dist_results: dict,
    num_heads: int,
    n_rel: int,
    strategy_name: str,
    output_dir: str
) -> None:
    """
    Create a heatmap showing the correlation between categories and heads.
    
    :param category_dist_results: Dictionary with per-head category distribution for this n_rel.
    :type category_dist_results: dict
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param n_rel: Number of relevant documents.
    :type n_rel: int
    :param strategy_name: Name of the strategy.
    :type strategy_name: str
    :param output_dir: Directory to store plots.
    :type output_dir: str
    """
    # Collect all unique categories
    all_categories = set()
    for head_idx in range(num_heads):
        head_key = f'head_{head_idx}'
        if head_key in category_dist_results:
            all_categories.update(category_dist_results[head_key].keys())
    
    all_categories = sorted(list(all_categories))
    num_categories = len(all_categories)
    
    # Build a matrix: categories × heads
    category_head_matrix = np.zeros((num_categories, num_heads))
    
    for head_idx in range(num_heads):
        head_key = f'head_{head_idx}'
        if head_key in category_dist_results:
            for cat_idx, category in enumerate(all_categories):
                count = category_dist_results[head_key].get(category, 0)
                category_head_matrix[cat_idx, head_idx] = count
    
    # Normalize by head (column-wise) to see what proportion each category contributes to each head
    head_totals = np.sum(category_head_matrix, axis=0)
    head_totals[head_totals == 0] = 1  # Avoid division by zero
    normalized_matrix = category_head_matrix / head_totals
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, num_categories * 0.3)))
    
    im = ax.imshow(normalized_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Documents per Head', fontsize=12)
    
    # Set ticks
    ax.set_xticks(np.arange(num_heads))
    ax.set_yticks(np.arange(num_categories))
    ax.set_xticklabels(np.arange(num_heads))
    ax.set_yticklabels(all_categories)
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    ax.set_title(f'{strategy_name} - Category Distribution per Head (n_rel={n_rel})', fontsize=14)
    
    # Add text annotations for non-zero values if the matrix is not too large
    if num_categories * num_heads <= 500:  # Only annotate if reasonably sized
        for cat_idx in range(num_categories):
            for head_idx in range(num_heads):
                value = normalized_matrix[cat_idx, head_idx]
                if value > 0.01:  # Only show if > 1%
                    text_color = 'white' if value > 0.5 else 'black'
                    ax.text(head_idx, cat_idx, f'{value:.2f}',
                           ha="center", va="center", color=text_color, fontsize=6)
    
    plt.tight_layout()
    
    output_filename = f'{strategy_name}_n_rel_{n_rel}_category_head_correlation.pdf'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved category-head correlation heatmap: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot per-head coverage metrics for MultiRAG')
    parser.add_argument('--per_head', type=str, required=True, 
                       help='Path to per-head results JSON file')
    parser.add_argument('--regular', type=str, required=True, 
                       help='Path to regular results JSON file')
    parser.add_argument('--categories', type=str, default=None,
                       help='Path to per-head category coverage JSON file (optional)')
    parser.add_argument('--category_dist', type=str, default=None,
                       help='Path to per-head category distribution JSON file (optional, for MI heatmap)')
    parser.add_argument('--strategy', type=str, default='multirag',
                       help='Strategy name to plot (default: multirag)')
    parser.add_argument('--output_dir', type=str, default='per_head_plots',
                       help='Output directory for plots (default: per_head_plots)')
    parser.add_argument('--metric', type=str, default='success_ratio',
                       choices=['success_ratio', 'category_success_ratio', 'success', 'category_success'],
                       help='Metric to plot (default: success_ratio)')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.per_head} and {args.regular}...")
    per_head_data, regular_data, category_data, category_dist_data = load_results(
        args.per_head, args.regular, args.categories, args.category_dist)
    
    if category_data:
        print("Category coverage data loaded successfully.")
    else:
        print("No category coverage data provided or file not found.")
    
    if category_dist_data:
        print("Category distribution data loaded successfully.")
    else:
        print("No category distribution data provided or file not found.")
    
    print(f"Plotting per-head coverage for strategy '{args.strategy}'...")
    plot_per_head_coverage(per_head_data, regular_data, category_data, category_dist_data,
                          args.strategy, args.output_dir, args.metric)
    
    print("Done!")


if __name__ == '__main__':
    main()
