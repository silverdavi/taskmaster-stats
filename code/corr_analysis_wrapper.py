import os
import sys
import pandas as pd
import numpy as np

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

print("Running correlation analysis...")

# Load IMDb ratings
imdb_df = pd.read_csv('data/TaskmasterUK_imdb.csv')
imdb_df['Series_Num'] = imdb_df['Series'].str.extract('(\d+)').astype(int)
imdb_df['Episode_Num'] = imdb_df['Episode'].str.extract('(\d+)').astype(int)
print(f"Loaded {len(imdb_df)} episodes with IMDb ratings")

# Process each series CSV file
all_episodes = []
series_files = sorted(os.listdir('data/taskmaster_series_tables'))
series_files = [f for f in series_files if f.endswith('.csv')]

for series_file in series_files:
    series_num = int(series_file.split('_')[-1].split('.')[0])
    file_path = os.path.join('data/taskmaster_series_tables', series_file)
    
    # Load series data
    series_df = pd.read_csv(file_path)
    print(f"Processing Series {series_num} with {len(series_df.columns) - 1} tasks...")
    
    # Extract tasks and process data
    tasks = [col for col in series_df.columns if col != 'Contestant Name']
    
    # Process episode-level metrics
    for task_idx, task in enumerate(tasks):
        if ' - ' in task:
            episode_num = int(task.split(' - ')[0])
            episode_key = f"Series {series_num}, Episode {episode_num}"
            
            # Find corresponding IMDb rating
            imdb_rating = imdb_df[(imdb_df['Series_Num'] == series_num) & 
                                 (imdb_df['Episode_Num'] == episode_num)]['Rating'].values
            
            if len(imdb_rating) > 0:
                imdb_rating = imdb_rating[0]
            else:
                imdb_rating = np.nan
            
            # Calculate metrics for the episode
            task_scores = series_df[task].values
            metrics = {}
            
            # Basic metrics
            metrics['Series'] = series_num
            metrics['Episode'] = episode_num
            metrics['IMDb Rating'] = imdb_rating
            
            # Add to episodes list
            all_episodes.append(metrics)

# Create DataFrame and save to CSV
episodes_df = pd.DataFrame(all_episodes)
episodes_df.to_csv('output/episode_level_analysis.csv', index=False)

print(f"Processed {len(episodes_df)} episodes across {len(series_files)} series")
print("Analysis complete - results saved to output/episode_level_analysis.csv") 