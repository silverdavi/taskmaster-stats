import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from itertools import combinations
from scipy.stats import rankdata

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)
os.makedirs('output/series_deep_dives', exist_ok=True)

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

print("Starting Taskmaster UK data analysis...")

# First, run the correlation analysis to generate episode_level_analysis.csv
print("Running correlation analysis to generate metrics...")
sys.path.append('code')
import corr_analysis

# Load the data
imdb_df = pd.read_csv('data/TaskmasterUK_imdb.csv')
imdb_df['Series_Num'] = imdb_df['Series'].str.extract('(\d+)').astype(int)
imdb_df['Episode_Num'] = imdb_df['Episode'].str.extract('(\d+)').astype(int)

# Load episode metrics
episodes_df = pd.read_csv('output/episode_level_analysis.csv')

# Import the visualization_analysis module functions
from visualization_analysis import (
    calculate_correlations, 
    plot_series_distributions,
    plot_series_trends,
    plot_feature_importance,
    plot_series_progression,
    plot_episode_feature_correlations,
    plot_metrics_boxplot,
    plot_series_deep_dive,
    generate_report,
    format_p_value,
    get_significance_stars
)

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
    os.makedirs('output/series_deep_dives', exist_ok=True)
    
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
    
    print("\nAll visualizations have been generated and saved to the 'output' directory.")
    print("Correlation report saved to 'output/correlation_report.csv'")
    print("Analysis report saved to 'output/analysis_report.txt'")

if __name__ == "__main__":
    main() 