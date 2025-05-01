# Taskmaster UK Data Analysis

This project analyzes data from the TV show "Taskmaster UK", including episode metrics, IMDb ratings, and visualizations.

## Project Structure

```
TaskmasterAnalysis/
├── code/                       # Analysis scripts
│   ├── visualization_analysis.py  # Creates visualizations and reports
│   ├── corr_analysis.py           # Correlation analysis of metrics
│   ├── analyze_single_table.py    # Analyzes a single series table
│   ├── collect_tmi.py             # Data collection utilities
│   └── scrapeTMI.py               # Web scraper for Taskmaster data
│
├── data/                       # Raw data files
│   ├── TaskmasterUK_imdb.csv      # IMDb ratings for episodes
│   └── taskmaster_series_tables/  # Task scores for each series
│       ├── taskmaster_uk_series_1.csv
│       ├── taskmaster_uk_series_2.csv
│       ├── ...
│       └── taskmaster_uk_series_18.csv
│
└── output/                     # Generated visualizations and reports
    ├── analysis_report.txt        # Text report of metrics and correlations
    ├── correlation_report.csv     # CSV of all metric correlations
    ├── episode_level_analysis.csv # Processed metrics for each episode
    ├── series_distributions.png   # Boxplots of metrics by series
    ├── series_trends.png          # Trends of metrics across series
    ├── feature_importance.png     # Impact of metrics on IMDb ratings
    ├── series_progression.png     # 2D visualization of series progression
    ├── metrics_boxplot.png        # Distribution of metrics across all series
    ├── episode_feature_correlations.png # Relationships between metrics
    └── series_deep_dives/         # Detailed analyses of each series
        ├── series_1_deep_dive.png
        ├── ...
        └── series_18_deep_dive.png
```

## Running the Analysis

To generate all visualizations and reports:

```bash
python code/visualization_analysis.py
```

This will read the raw data files from `data/` and generate all visualizations in the `output/` directory.

## Analysis Components

1. **Correlation Analysis**: Examines relationships between different metrics (volatility, points, etc.) and IMDb ratings.

2. **Series Trends**: Analyzes how different metrics change across the 18 series.

3. **Series Deep Dives**: Detailed analysis of each series showing rank changes and cumulative points.

4. **Metrics Comparisons**: Compares the distribution and relationships between different performance metrics.

## Metrics Analyzed

- **IMDb Rating**: Episode ratings from IMDb
- **Volatility**: Measure of rank changes within an episode
- **Average Points**: Mean points awarded per task
- **Point Variance**: Variance in points across contestants
- **Rank Spread**: Spread between highest and lowest ranked contestants
- **Lead Changes**: Number of times the leader changed during an episode
- **Comeback Index**: Measure of comebacks from behind
- **Tension**: Closeness of the competition
- **Score Skewness**: Distribution skewness of scores
- **Winner Dominance**: How dominant the winner was
- **Winner Margin Slope**: Trend in winner's margin across tasks

## Data Sources

- Task scores from Taskmaster UK episodes
- IMDb ratings from episode listings 
