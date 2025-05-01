import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

def analyze_series(series_num):
    # Load the wide-format CSV
    filepath = f"taskmaster_series_tables/taskmaster_uk_series_{series_num}.csv"
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return
        
    print(f"\nAnalyzing Taskmaster UK Series {series_num}")
    print("=" * 50)
    
    df = pd.read_csv(filepath)
    contestants = df["Contestant Name"]
    scores = df.drop(columns=["Contestant Name"])
    
    # Basic statistics
    total_scores = scores.sum(axis=1)
    avg_scores = scores.mean(axis=1)
    tasks_won = (scores == scores.max(axis=0)).sum(axis=1)
    
    stats_df = pd.DataFrame({
        'Contestant': contestants,
        'Total Score': total_scores,
        'Average Score': avg_scores.round(2),
        'Tasks Won': tasks_won,
        'Final Rank': total_scores.rank(ascending=False).astype(int)
    }).sort_values('Final Rank')
    
    print("\nFinal Statistics:")
    print(stats_df.to_string(index=False))
    
    # Compute running sum (cumulative score)
    cumulative_scores = scores.cumsum(axis=1)
    
    # Compute running rank (lower rank = better)
    running_ranks = cumulative_scores.rank(axis=0, method="min", ascending=False)
    running_ranks["Contestant Name"] = contestants
    
    # Plot 1: Running Ranks
    plt.figure(figsize=(15, 10))
    
    # Create first subplot for running ranks
    plt.subplot(2, 1, 1)
    for _, row in running_ranks.iterrows():
        name = row["Contestant Name"]
        ranks = row.drop("Contestant Name")
        plt.plot(range(len(ranks)), ranks.values, label=name, marker='o', markersize=4)
    
    plt.gca().invert_yaxis()  # Rank 1 at the top
    plt.title(f"Taskmaster UK Series {series_num}: Running Rank Over Tasks")
    plt.xlabel("Task Number")
    plt.ylabel("Rank (1 = Leading)")
    plt.legend(title="Contestant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Plot 2: Score Distribution
    plt.subplot(2, 1, 2)
    melted_scores = pd.melt(scores)
    melted_scores['Contestant'] = np.repeat(contestants.values, len(scores.columns))
    sns.violinplot(data=melted_scores, x="value", y="Contestant")
    plt.title("Score Distribution by Contestant")
    plt.xlabel("Score")
    plt.ylabel("Contestant")
    
    plt.tight_layout()
    plt.show()
    
    # Additional Statistics
    print("\nTask Statistics:")
    print("-" * 30)
    print(f"Total Tasks: {len(scores.columns)}")
    print(f"Average Score Across All Tasks: {scores.mean().mean():.2f}")
    print(f"Most Common Score: {scores.mode().mode().iloc[0]}")
    
    # Find closest contests
    score_diffs = abs(cumulative_scores.iloc[:, -1] - cumulative_scores.iloc[:, -1].mean())
    closest_contestant = contestants[score_diffs.idxmin()]
    print(f"\nMost Average Contestant: {closest_contestant}")
    
    # Find most consistent contestant
    consistency = scores.std(axis=1)
    most_consistent = contestants[consistency.idxmin()]
    print(f"Most Consistent Contestant: {most_consistent} (SD: {consistency.min():.2f})")
    
    # Find most volatile contestant
    most_volatile = contestants[consistency.idxmax()]
    print(f"Most Volatile Contestant: {most_volatile} (SD: {consistency.max():.2f})")

if __name__ == "__main__":
    # Default to series 7 if no argument provided
    series_num = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    analyze_series(series_num)
