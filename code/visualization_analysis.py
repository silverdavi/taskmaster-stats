import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from itertools import combinations
import os
from scipy.stats import rankdata

# Set seaborn style and custom parameters
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load the data
imdb_df = pd.read_csv('../data/TaskmasterUK_imdb.csv')
imdb_df['Series_Num'] = imdb_df['Series'].str.extract('(\d+)').astype(int)
imdb_df['Episode_Num'] = imdb_df['Episode'].str.extract('(\d+)').astype(int)

# Load episode metrics
episodes_df = pd.read_csv('../output/episode_level_analysis.csv')

def calculate_correlations():
    """Calculate and save all correlations between features."""
    # Get all numeric features
    numeric_features = episodes_df.select_dtypes(include=[np.number]).columns
    numeric_features = [f for f in numeric_features if f not in ['Series', 'Episode']]
    
    # Calculate correlations
    correlations = []
    for f1, f2 in combinations(numeric_features, 2):
        corr, pval = stats.pearsonr(episodes_df[f1], episodes_df[f2])
        correlations.append({
            'Feature 1': f1,
            'Feature 2': f2,
            'Correlation': corr,
            'P-value': pval
        })
    
    # Create DataFrame and sort by absolute correlation
    corr_df = pd.DataFrame(correlations)
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    # Save to CSV
    corr_df.to_csv('plots/correlation_report.csv', index=False)
    
    # Print top correlations
    print("\nTop 10 Correlations:")
    print(corr_df.head(10).to_string(index=False))
    
    return corr_df

def format_p_value(pval):
    """Format p-value for display in plots."""
    if pval < 0.05:
        if pval < 0.001:
            return f"$10^{{ {int(np.log10(pval))} }}$"
        return f"< {pval:.3f}"
    return f"> {pval:.3f}"

def get_significance_stars(pval):
    """Get significance stars for p-value."""
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    return ""

def plot_series_distributions():
    """Plot 18 subplots (6x3) showing distributions of scores and metrics per series."""
    fig, axes = plt.subplots(6, 3, figsize=(15, 30))
    fig.suptitle('Series-Level Distributions', fontsize=16, y=1.02)
    
    metrics = ['IMDb Rating', 'Volatility', 'Average Points', 'Point Variance', 
              'Rank Spread', 'Lead Changes']
    colors = sns.color_palette('pastel', n_colors=len(metrics))
    
    for i, series in enumerate(sorted(episodes_df['Series'].unique())):
        row = i % 6
        col = i // 6
        
        series_data = episodes_df[episodes_df['Series'] == series]
        
        # Create boxplots for each metric, distributed on x-axis
        positions = np.arange(len(metrics))
        for j, (metric, color) in enumerate(zip(metrics, colors)):
            # Create boxplot
            bp = axes[row, col].boxplot(series_data[metric], positions=[j],
                                      patch_artist=True,
                                      medianprops=dict(color="black"),
                                      flierprops=dict(marker='o', markerfacecolor=color),
                                      widths=0.8)  # Wider boxplots
            
            # Set box color
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color='black')
            plt.setp(bp['boxes'], facecolor=color)
        
        axes[row, col].set_title(f'Series {series}', fontweight='bold')
        axes[row, col].set_xticks(positions)
        axes[row, col].set_xticklabels(metrics, rotation=45, ha='right')
        
        # Set y-axis limits based on 5-95 percentile range
        for metric in metrics:
            if metric == axes[row, col].get_ylabel():
                if metric == 'IMDb Rating':
                    # Create twin axis for IMDb Rating
                    ax2 = axes[row, col].twinx()
                    metric_min = episodes_df[metric].quantile(0.05)
                    metric_max = episodes_df[metric].quantile(0.95)
                    axes[row, col].set_ylim(metric_min - 0.1 * (metric_max - metric_min),
                                          metric_max + 0.1 * (metric_max - metric_min))
                    # Set second y-axis to 1-10
                    ax2.set_ylim(2 * (metric_min - 0.1 * (metric_max - metric_min)),
                               2 * (metric_max + 0.1 * (metric_max - metric_min)))
                    ax2.set_ylabel('IMDb Rating (1-10)')
                else:
                    metric_min = episodes_df[metric].quantile(0.05)
                    metric_max = episodes_df[metric].quantile(0.95)
                    axes[row, col].set_ylim(metric_min - 0.1 * (metric_max - metric_min),
                                          metric_max + 0.1 * (metric_max - metric_min))
    
    plt.tight_layout()
    plt.savefig('plots/series_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_series_trends():
    """Plot trends of various metrics as function of series."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Series-Level Trends', fontsize=16, y=1.02)
    
    metrics = ['IMDb Rating', 'Volatility', 'Average Points', 
              'Point Variance', 'Rank Spread', 'Lead Changes']
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        
        # Calculate series means
        series_means = episodes_df.groupby('Series')[metric].mean()
        
        # Plot raw data points
        sns.scatterplot(data=episodes_df, x='Series', y=metric, 
                       ax=axes[row, col], alpha=0.5, color='gray',
                       label='Individual Episodes' if i == 0 else None)
        
        # Plot smoothed trend with thinner line
        x = series_means.index
        y = series_means.values
        y_smooth = gaussian_filter1d(y, sigma=0.8)  # Less smoothing
        axes[row, col].plot(x, y_smooth, color='red', linewidth=1.5,
                          label='Smoothed Trend' if i == 0 else None)
        
        # Plot series means
        sns.scatterplot(x=x, y=y, ax=axes[row, col], 
                       color='blue', s=100, label='Series Mean' if i == 0 else None)
        
        # Add trend line and correlation
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        trend_line = slope * x + intercept
        axes[row, col].plot(x, trend_line, 'g-', linewidth=2, label='Linear Trend' if i == 0 else None)
        
        # Add correlation and p-value text with significance stars
        stars = get_significance_stars(p_value)
        corr_text = f"r = {r_value:.3f}\np {format_p_value(p_value)}{stars}"
        axes[row, col].text(0.05, 0.95, corr_text, transform=axes[row, col].transAxes,
                          bbox=dict(facecolor='white', alpha=0.8))
        
        axes[row, col].set_title(f'{metric} Trend', fontweight='bold')
        if i == 0:
            axes[row, col].set_xlabel('Series Number')
        else:
            axes[row, col].set_xlabel('')
        axes[row, col].set_ylabel(metric)
        
        # Set integer x-ticks
        x_ticks = np.arange(int(min(x)), int(max(x)) + 1)
        axes[row, col].set_xticks(x_ticks)
        
        # Set y-axis limits based on 5-95 percentile range
        metric_min = episodes_df[metric].quantile(0.05)
        metric_max = episodes_df[metric].quantile(0.95)
        axes[row, col].set_ylim(metric_min - 0.1 * (metric_max - metric_min),
                              metric_max + 0.1 * (metric_max - metric_min))
        
        # Add legend only to first subplot
        if i == 0:
            axes[row, col].legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('plots/series_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report():
    """Generate a report with numerical results from the plots."""
    report = []
    
    # Series level trends
    report.append("Series Level Trends:")
    report.append("===================")
    metrics = ['IMDb Rating', 'Volatility', 'Average Points', 'Point Variance', 
              'Rank Spread', 'Lead Changes', 'Comeback Index', 'Tension', 
              'Score Skewness', 'Winner Dominance', 'Winner Margin Slope']
    
    for metric in metrics:
        series_means = episodes_df.groupby('Series')[metric].mean()
        x = series_means.index
        y = series_means.values
        
        # Drop NaN values
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 2:  # Need at least 2 points for correlation
            report.append(f"\n{metric}:")
            report.append("  Not enough data for trend analysis")
            continue
            
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        slope, intercept, r_value, p_value, _ = stats.linregress(x_valid, y_valid)
        report.append(f"\n{metric}:")
        report.append(f"  Best fit: y = {slope:.3f}x + {intercept:.3f}")
        report.append(f"  r = {r_value:.3f}")
        report.append(f"  p {format_p_value(p_value)}{get_significance_stars(p_value)}")
        report.append(f"  N = {len(x_valid)} series")
    
    # Episode feature correlations
    report.append("\n\nEpisode Feature Correlations:")
    report.append("===========================")
    features = ['Volatility', 'Average Points', 'Point Variance', 'Rank Spread', 
               'Lead Changes', 'Comeback Index', 'Tension', 'Score Skewness',
               'Winner Dominance', 'Winner Margin Slope']
    
    for f1, f2 in combinations(features, 2):
        # Drop NaN values
        valid_data = episodes_df[[f1, f2]].dropna()
        if len(valid_data) < 2:  # Need at least 2 points for correlation
            report.append(f"\n{f1} vs {f2}:")
            report.append("  Not enough data for correlation analysis")
            continue
            
        corr, pval = stats.pearsonr(valid_data[f1], valid_data[f2])
        report.append(f"\n{f1} vs {f2}:")
        report.append(f"  r = {corr:.3f}")
        report.append(f"  p {format_p_value(pval)}{get_significance_stars(pval)}")
        report.append(f"  N = {len(valid_data)} episodes")
    
    # Save report
    with open('plots/analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Print report
    print("\nAnalysis Report:")
    print("===============")
    print('\n'.join(report))

def plot_feature_importance():
    """Plot feature effect on IMDb score at series level."""
    # Calculate series-level means
    series_means = episodes_df.groupby('Series').mean()
    
    # Calculate correlations and p-values
    features = ['Volatility', 'Average Points', 'Point Variance', 'Rank Spread', 
               'Lead Changes', 'Comeback Index', 'Tension', 'Score Skewness']
    
    correlations = []
    p_values = []
    
    for feature in features:
        corr, pval = stats.pearsonr(series_means[feature], series_means['IMDb Rating'])
        correlations.append(corr)
        p_values.append(pval)
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': features,
        'Correlation': correlations,
        'P-value': p_values
    })
    
    # Sort by correlation value (descending)
    importance_df = importance_df.sort_values('Correlation', ascending=False)
    
    plt.figure(figsize=(12, 6))
    # Use pastel colors
    colors = ['#FF9999' if x < 0 else '#99FF99' for x in importance_df['Correlation']]
    bars = plt.barh(importance_df['Feature'], importance_df['Correlation'], 
                    color=colors)
    
    # Add correlation values
    for i, corr in enumerate(importance_df['Correlation']):
        plt.text(corr, i, f' {corr:.3f}', 
                ha='left' if corr > 0 else 'right', va='center')
    
    plt.title('Feature Effect on IMDb Rating (Series Level)\nNo Statistical Significance', 
              fontweight='bold', pad=20)
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Feature')
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_series_progression():
    """Plot 2D progression of series using two features, colored by series number."""
    # Select two features for visualization
    feature1 = 'Volatility'
    feature2 = 'Average Points'
    
    # Calculate series means
    series_means = episodes_df.groupby('Series').mean()
    
    plt.figure(figsize=(10, 8))
    # Use JET colormap
    scatter = plt.scatter(series_means[feature1], series_means[feature2], 
                         c=series_means.index, cmap='jet', s=200)
    
    # Add series numbers as text
    for i, txt in enumerate(series_means.index):
        plt.annotate(txt, (series_means[feature1].iloc[i], series_means[feature2].iloc[i]),
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(scatter, label='Series Number')
    plt.title('Series Progression in Feature Space', fontweight='bold', pad=20)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/series_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_episode_feature_correlations():
    """Plot 2D relationships between features, colored by relative IMDb score."""
    # Calculate series-relative IMDb scores
    series_means = episodes_df.groupby('Series')['IMDb Rating'].mean()
    episodes_df['Relative_IMDb'] = episodes_df.apply(
        lambda x: x['IMDb Rating'] - series_means[x['Series']], axis=1
    )
    
    # Select three features for visualization
    features = ['Volatility', 'Average Points', 'Rank Spread']
    
    # Create subplots for each feature pair
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Episode-Level Feature Correlations (Colored by Relative IMDb Score)', 
                fontsize=16, y=1.02)
    
    for i, (f1, f2) in enumerate(combinations(features, 2)):
        # Calculate correlation and p-value
        corr, pval = stats.pearsonr(episodes_df[f1], episodes_df[f2])
        
        # Create scatter plot
        scatter = axes[i].scatter(episodes_df[f1], episodes_df[f2], 
                                c=episodes_df['Relative_IMDb'], 
                                cmap='coolwarm', alpha=0.7)
        
        # Add trend line
        sns.regplot(data=episodes_df, x=f1, y=f2, ax=axes[i],
                   scatter=False, color='black', line_kws={'alpha': 0.5})
        
        # Add correlation and p-value text with significance stars
        stars = get_significance_stars(pval)
        corr_text = f"r = {corr:.3f}\np {format_p_value(pval)}{stars}"
        axes[i].text(0.05, 0.95, corr_text, transform=axes[i].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        axes[i].set_title(f'{f1} vs {f2}', fontweight='bold')
        axes[i].set_xlabel(f1)
        axes[i].set_ylabel(f2)
        axes[i].grid(alpha=0.3)
        
        # Set axis limits based on 5-95 percentile range
        for axis, feature in zip(['x', 'y'], [f1, f2]):
            min_val = episodes_df[feature].quantile(0.05)
            max_val = episodes_df[feature].quantile(0.95)
            if axis == 'x':
                axes[i].set_xlim(min_val - 0.1 * (max_val - min_val),
                               max_val + 0.1 * (max_val - min_val))
            else:
                axes[i].set_ylim(min_val - 0.1 * (max_val - min_val),
                               max_val + 0.1 * (max_val - min_val))
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[i])
        cbar.set_label('Relative IMDb Score')
    
    plt.tight_layout()
    plt.savefig('plots/episode_feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_boxplot():
    """Create a comprehensive boxplot of all metrics at series level with improved visuals."""
    print("Creating metrics boxplot...")
    
    # All available metrics
    all_metrics = ['Volatility', 'Average Points', 'Point Variance', 'Rank Spread', 
               'Lead Changes', 'Comeback Index', 'Tension', 'Score Skewness',
               'Winner Dominance', 'Winner Margin Slope']
    
    # Group by series and compute mean for each metric
    series_metrics = episodes_df.groupby('Series').mean()
    
    # Check which metrics have sufficient non-NaN values
    valid_metrics = []
    valid_data = []
    for metric in all_metrics:
        valid_count = series_metrics[metric].notna().sum()
        print(f"Metric {metric}: {valid_count} valid values out of {len(series_metrics)}")
        if valid_count >= len(series_metrics) * 0.5:  # At least 50% of series have values
            valid_metrics.append(metric)
            # Get values, dropping NaNs
            metric_values = series_metrics[metric].dropna().values
            valid_data.append(metric_values)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Use a single color for all boxplots - more elegant and less cluttered
    box_color = '#3498db'  # A nice blue
    
    # Create boxplot with custom style
    bp = plt.boxplot(valid_data, patch_artist=True, 
                    medianprops=dict(color="white", linewidth=2),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='white', 
                                  markeredgecolor=box_color,
                                  markersize=8, alpha=0.7),
                    widths=0.7)
    
    # Apply consistent color to all boxes
    for patch in bp['boxes']:
        patch.set_facecolor(box_color)
        patch.set_alpha(0.7)
    
    # Add subtle grid for only y-axis
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Customize plot
    plt.xticks(range(1, len(valid_metrics) + 1), valid_metrics, rotation=45, ha='right', fontsize=11)
    plt.ylabel('Value', fontsize=14, fontweight='bold')
    plt.title('Distribution of Metrics Across Series', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Add subtle background color
    plt.gca().set_facecolor('#f8f9fa')
    
    # Add horizontal lines at important y-values (0 and mean)
    plt.axhline(y=0, color='#e74c3c', linestyle='-', alpha=0.3, linewidth=1.5)
    
    # Annotate with mean values
    for i, data in enumerate(valid_data):
        mean_val = np.mean(data)
        # Add a subtle point to mark the mean
        plt.plot(i+1, mean_val, 'ro', markersize=6, alpha=0.7)
        # Annotate with value (with some offset)
        plt.annotate(f"{mean_val:.2f}", (i+1, mean_val), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Adjust layout for a tighter fit
    plt.tight_layout()
    
    # Save plot with higher DPI for publication
    plt.savefig('plots/metrics_boxplot.png', dpi=300, bbox_inches='tight')
    
    # Also save as PDF for publication
    plt.savefig('plots/metrics_boxplot.pdf', bbox_inches='tight')
    
    plt.close()
    print("Metrics boxplot completed.")

def plot_series_deep_dive(series_num):
    """Create detailed visualization of rank changes and cumulative points for a specific series."""
    print(f"Creating deep dive for Series {series_num}...")
    
    # Create subfolder for series deep dives if it doesn't exist
    os.makedirs('plots/series_deep_dives', exist_ok=True)
    
    # Load the series data
    series_file = f"taskmaster_series_tables/taskmaster_uk_series_{series_num}.csv"
    try:
        df = pd.read_csv(series_file)
        print(f"Successfully loaded series data with {df.shape[1]-1} tasks.")
    except Exception as e:
        print(f"Error loading series data: {e}")
        return
    
    # Get contestant names and task columns
    contestants = df['Contestant Name'].values
    tasks = [col for col in df.columns if col != 'Contestant Name']
    
    # Handle NaN values in the scores
    scores = df[tasks].fillna(0).values  # Replace NaN with 0
    
    # Calculate cumulative scores and ranks for each task
    cumulative_scores = np.zeros_like(scores, dtype=float)
    ranks = np.zeros_like(scores, dtype=float)
    
    # Calculate cumulative scores manually
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            if j == 0:
                cumulative_scores[i, j] = scores[i, j]
            else:
                cumulative_scores[i, j] = cumulative_scores[i, j-1] + scores[i, j]
    
    # Calculate ranks - higher score gets better rank (rank 1 is best)
    for j in range(cumulative_scores.shape[1]):
        task_scores = cumulative_scores[:, j]
        # Calculate ranks (higher score = better rank)
        for i in range(len(contestants)):
            ranks[i, j] = np.argsort(np.argsort(-task_scores))[i] + 1
    
    # Create figure with two subplots - increase height for better task name visibility
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1.5])
    fig.suptitle(f'Taskmaster UK Series {series_num} Progression', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Nice colors for contestants
    colors = sns.color_palette('husl', n_colors=len(contestants))
    
    # Plot rank changes
    for i, contestant in enumerate(contestants):
        ax1.plot(np.arange(len(tasks)), ranks[i], label=contestant, color=colors[i], 
                marker='o', markersize=6, linewidth=1.5)
    
    # Customize rank plot
    ax1.set_ylabel('Rank', fontsize=12, fontweight='bold')
    ax1.set_ylim(len(contestants) + 0.5, 0.5)  # Reverse y-axis to have rank 1 at the top
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    # Add episode separators and labels
    episode_tasks = {}
    for task in tasks:
        # Extract episode number from the task name
        parts = task.split(' - ')
        if len(parts) >= 2:
            episode = int(parts[0])
            if episode not in episode_tasks:
                episode_tasks[episode] = []
            episode_tasks[episode].append(task)
    
    # Add vertical lines for episodes
    task_num = 0
    for episode in sorted(episode_tasks.keys()):
        task_num += len(episode_tasks[episode])
        if task_num < len(tasks):  # Don't add line after last episode
            ax1.axvline(task_num - 0.5, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(task_num - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Plot cumulative points
    for i, contestant in enumerate(contestants):
        ax2.plot(np.arange(len(tasks)), cumulative_scores[i], label=contestant, color=colors[i],
                marker='o', markersize=6, linewidth=1.5)
    
    # Customize points plot
    ax2.set_ylabel('Cumulative Points', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add task names as x-ticks (show every nth task to avoid crowding)
    n = max(len(tasks) // 10, 1)  # Show approximately 10 task names for a narrower plot
    task_indices = list(range(0, len(tasks), n))
    ax2.set_xticks(task_indices)
    
    # Extract just the task description for x-tick labels
    task_labels = []
    for i in task_indices:
        parts = tasks[i].split(' - ')
        if len(parts) >= 2:
            task_labels.append(parts[1][:20] + '...' if len(parts[1]) > 20 else parts[1])  # Allow longer task names
        else:
            task_labels.append(tasks[i][:20] + '...' if len(tasks[i]) > 20 else tasks[i])
    
    ax2.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=9)
    
    # Add episode numbers above the tick labels
    for episode in sorted(episode_tasks.keys()):
        start_idx = sum(len(episode_tasks[ep]) for ep in episode_tasks if ep < episode)
        middle_idx = start_idx + len(episode_tasks[episode]) / 2
        ax2.text(middle_idx, ax2.get_ylim()[1] * 1.02, f'Ep {episode}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Set x-axis limits to show all tasks
    ax1.set_xlim(-0.5, len(tasks) - 0.5)
    ax2.set_xlim(-0.5, len(tasks) - 0.5)
    
    # Remove x-axis labels from top plot
    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    
    # Add x-axis label to bottom plot
    ax2.set_xlabel('Tasks', fontsize=12, fontweight='bold')
    
    # Adjust layout for a tighter fit
    plt.tight_layout()
    
    # Save plot with higher DPI for publication
    plt.savefig(f'plots/series_deep_dives/series_{series_num}_deep_dive.png', 
                dpi=300, bbox_inches='tight')
    
    # Also save as PDF for publication
    plt.savefig(f'plots/series_deep_dives/series_{series_num}_deep_dive.pdf', 
                bbox_inches='tight')
    
    plt.close()
    print(f"Series {series_num} deep dive completed.")

def main():
    """Generate all visualizations."""
    print("Generating visualizations...")
    
    # Calculate and save correlations
    corr_df = calculate_correlations()
    
    plot_series_distributions()
    print("✓ Series distributions plot generated")
    
    plot_series_trends()
    print("✓ Series trends plot generated")
    
    plot_feature_importance()
    print("✓ Feature importance plot generated")
    
    plot_series_progression()
    print("✓ Series progression plot generated")
    
    plot_episode_feature_correlations()
    print("✓ Episode feature correlations plot generated")
    
    plot_metrics_boxplot()
    print("✓ Metrics boxplot generated")
    
    # Generate deep dive plots for all series
    print("\nGenerating deep dive plots for all series...")
    # Create subfolder for series deep dives
    os.makedirs('plots/series_deep_dives', exist_ok=True)
    
    # Get list of available series from the series columns in episodes_df
    available_series = sorted(episodes_df['Series'].unique())
    print(f"Found {len(available_series)} series to plot")
    
    for series_num in available_series:
        # Extract numeric series number if it's a string (e.g., "Series 7" -> 7)
        if isinstance(series_num, str) and 'Series' in series_num:
            series_num = int(series_num.replace('Series', '').strip())
        try:
            plot_series_deep_dive(series_num)
        except Exception as e:
            print(f"Error generating deep dive for Series {series_num}: {e}")
    
    print("✓ All series deep dives generated")
    
    # Generate report
    generate_report()
    print("✓ Analysis report generated")
    
    print("\nAll visualizations have been generated and saved to the 'plots' directory.")
    print("Correlation report saved to 'plots/correlation_report.csv'")
    print("Analysis report saved to 'plots/analysis_report.txt'")

if __name__ == "__main__":
    main() 