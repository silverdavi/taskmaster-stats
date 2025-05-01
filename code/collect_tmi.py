import pandas as pd
import os

# Load the CSV
df = pd.read_csv("taskmaster_results.csv")

# Filter for Taskmaster UK only
uk_df = df[df["Show Title"] == "Taskmaster UK"]

# Ensure consistent sorting
uk_df = uk_df.sort_values(by=["Series", "Episode", "Task Title", "Task ID", "Contestant Name"])

# Create output directory if it doesn't exist
output_dir = "taskmaster_series_tables"
os.makedirs(output_dir, exist_ok=True)

# Process each series
for series_num, group in uk_df.groupby("Series"):
    # Create a unique identifier for each task: Episode number and Task Title
    group["Task Column"] = group["Episode"].astype(str).str.zfill(2) + " - " + group["Task Title"]

    # Pivot: rows = contestants, columns = tasks, values = scores
    pivot_df = group.pivot_table(
        index="Contestant Name",
        columns="Task Column",
        values="Total Score",
        aggfunc="first"
    )

    # Sort columns (tasks) by episode-task order
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

    # Save to CSV
    filename = f"taskmaster_uk_series_{series_num}.csv"
    filepath = os.path.join(output_dir, filename)
    pivot_df.to_csv(filepath)

    print(f"Saved: {filepath}")
