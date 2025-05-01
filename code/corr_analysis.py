import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr, entropy, pearsonr, ttest_ind, linregress, f, binomtest, skew
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.impute import SimpleImputer
from itertools import combinations
import statsmodels.api as sm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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

# Custom color palette
custom_palette = sns.color_palette("husl", 8)
sns.set_palette(custom_palette)

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load IMDb ratings
imdb_df = pd.read_csv('usecases/scraping/TaskmasterUK_imdb.csv')
imdb_df['Series_Num'] = imdb_df['Series'].str.extract('(\d+)').astype(int)
imdb_df['Episode_Num'] = imdb_df['Episode'].str.extract('(\d+)').astype(int)

# Set your directory containing the wide-format CSVs
data_dir = "taskmaster_series_tables"
all_files = [f for f in os.listdir(data_dir) if f.startswith("taskmaster_uk_series_") and f.endswith(".csv")]

# Store episode-level metrics
episode_metrics = []

for file in all_files:
    series_num = int(file.split("_")[-1].replace(".csv", ""))
    df = pd.read_csv(os.path.join(data_dir, file))
    df = df.set_index("Contestant Name")
    
    episodes = sorted(set(col.split(" - ")[0] for col in df.columns))
    episode_num = 1
    
    for ep in episodes:
        ep_cols = [col for col in df.columns if col.startswith(ep + " - ")]
        if len(ep_cols) < 2:
            continue
            
        scores = df[ep_cols]
        cumulative = scores.cumsum(axis=1)
        ranks = cumulative.rank(axis=0, method="min", ascending=False)
        
        # Calculate episode metrics
        rank_changes = np.diff(ranks.values, axis=1)
        volatility = np.nanmean(np.abs(rank_changes))
        
        avg_points = np.nanmean(scores.values)
        point_variance = np.nanvar(scores.values)
        
        rank_spread = np.nanmean([ranks.iloc[:, i].std() for i in range(ranks.shape[1])])
        
        lead_changes = np.sum(np.diff(np.argmin(ranks.values, axis=0)) != 0)
        
        max_points = np.nanmax(scores.values)
        min_points = np.nanmin(scores.values)
        point_range = max_points - min_points
        
        # New features
        # Comeback Index
        winner_idx = np.argmin(ranks.iloc[:, -1].values)
        midpoint = len(ranks.columns) // 2
        comeback = ranks.iloc[winner_idx, 0] - ranks.iloc[winner_idx, midpoint] if len(ranks.columns) > 1 else 0
        
        # Tension
        final_scores = cumulative.iloc[:, -1].values
        if len(final_scores) >= 2:
            sorted_scores = np.sort(final_scores)[::-1]
            tension = sorted_scores[0] - sorted_scores[1]
        else:
            tension = 0
        
        # Score Skewness
        score_skewness = skew(final_scores) if len(final_scores) >= 3 else 0
        
        # Winner Dominance Index
        winner_score = final_scores[winner_idx]
        others_mean = np.mean(np.delete(final_scores, winner_idx))
        dominance = winner_score - others_mean
        
        # Lead Retention Count
        lead_count = np.sum(ranks.iloc[winner_idx, :] == 1)
        
        # Score StdDev Trajectories
        score_stddev = np.nanmean([np.nanstd(scores.iloc[i, :]) for i in range(len(scores))])
        
        # Task Winner Diversity
        task_winners = set()
        for t in range(len(scores.columns)):
            task_scores = scores.iloc[:, t]
            if not np.isnan(task_scores).all():
                task_winners.add(np.nanargmax(task_scores))
        winner_diversity = len(task_winners)
        
        # Winner Margin Slope
        winner_cumulative = cumulative.iloc[winner_idx, :]
        second_place_idx = np.argsort(cumulative.iloc[:, -1])[1]
        second_cumulative = cumulative.iloc[second_place_idx, :]
        margin = winner_cumulative - second_cumulative
        if len(margin) > 1:
            slope, _ = linregress(range(len(margin)), margin)[:2]
        else:
            slope = 0
        
        # Get IMDb rating for this episode
        imdb_rating = imdb_df[
            (imdb_df['Series_Num'] == series_num) & 
            (imdb_df['Episode_Num'] == episode_num)
        ]['IMDb Rating (/10)'].values
        
        if len(imdb_rating) > 0:
            episode_metrics.append({
                'Series': series_num,
                'Episode': episode_num,
                'IMDb Rating': imdb_rating[0],
                'Volatility': volatility,
                'Average Points': avg_points,
                'Point Variance': point_variance,
                'Rank Spread': rank_spread,
                'Lead Changes': lead_changes,
                'Max Points': max_points,
                'Min Points': min_points,
                'Point Range': point_range,
                'Comeback Index': comeback,
                'Tension': tension,
                'Score Skewness': score_skewness,
                'Winner Dominance': dominance,
                'Lead Retention': lead_count,
                'Score StdDev': score_stddev,
                'Winner Diversity': winner_diversity,
                'Winner Margin Slope': slope
            })
        
        episode_num += 1

# Convert to DataFrame
episodes_df = pd.DataFrame(episode_metrics)

# Print data points information
print(f"\nTotal number of episodes analyzed: {len(episodes_df)}")
print(f"Number of series: {episodes_df['Series'].nunique()}")
print(f"Average episodes per series: {len(episodes_df) / episodes_df['Series'].nunique():.1f}")
print("\nEpisodes per series:")
print(episodes_df.groupby('Series').size())

# Analyze correlations with IMDb ratings
print("\nCorrelations with IMDb Rating:")
for col in episodes_df.columns[3:]:  # Skip Series, Episode, and IMDb Rating
    corr, pval = pearsonr(episodes_df[col], episodes_df['IMDb Rating'])
    print(f"\n{col}:")
    print(f"  Pearson correlation: {corr:.3f}")
    print(f"  p-value: {pval:.4f}")

# Prepare data for modeling
X = episodes_df.drop(['Series', 'Episode', 'IMDb Rating'], axis=1)
y = episodes_df['IMDb Rating']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Try different models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0)
}

print("\nModel Performance:")
for name, model in models.items():
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
    
    # Fit on full dataset for feature importance
    model.fit(X_scaled, y)
    
    print(f"\n{name}:")
    print(f"  Cross-validated R²: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    importance = importance.sort_values('Coefficient', key=abs, ascending=False)
    print("\n  Feature Importance:")
    for _, row in importance.iterrows():
        print(f"    {row['Feature']}: {row['Coefficient']:.3f}")

# Detailed analysis with best model
best_model = LinearRegression()
best_model.fit(X_scaled, y)

# Create visualization of actual vs predicted
y_pred = best_model.predict(X_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.7, s=80, edgecolor='white', linewidth=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, alpha=0.7)
plt.xlabel('Actual IMDb Rating', fontweight='bold')
plt.ylabel('Predicted IMDb Rating', fontweight='bold')
plt.title('Actual vs Predicted IMDb Ratings', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted_imdb.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(best_model.coef_)
})
importance = importance.sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 6))
bars = plt.barh(importance['Feature'], importance['Importance'], 
                color=custom_palette[0], alpha=0.8, edgecolor='white', linewidth=0.5)
plt.xlabel('Absolute Coefficient Value', fontweight='bold')
plt.title('Feature Importance in Predicting IMDb Ratings', fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Save episode-level data
episodes_df.to_csv('plots/episode_level_analysis.csv', index=False)

# Additional visualizations for top predictors
top_features = importance.nlargest(3, 'Importance')['Feature'].values

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, feature in enumerate(top_features):
    sns.regplot(data=episodes_df, x=feature, y='IMDb Rating', 
                ax=axes[i], scatter_kws={'alpha': 0.7, 's': 80, 'edgecolor': 'white', 'linewidth': 0.5},
                line_kws={'color': custom_palette[1], 'alpha': 0.7})
    axes[i].set_title(f'{feature} vs IMDb Rating', fontweight='bold', pad=10)
    axes[i].set_xlabel(feature, fontweight='bold')
    axes[i].set_ylabel('IMDb Rating', fontweight='bold')
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/top_predictors_regression.png', dpi=300, bbox_inches='tight')
plt.close()

# Analyze temporal trends in IMDb ratings
plt.figure(figsize=(12, 6))
sns.boxplot(data=episodes_df, x='Series', y='IMDb Rating', 
            palette=custom_palette, width=0.7, fliersize=5)
plt.xticks(rotation=45, ha='right')
plt.title('IMDb Ratings Distribution by Series', fontweight='bold', pad=20)
plt.xlabel('Series', fontweight='bold')
plt.ylabel('IMDb Rating', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/imdb_ratings_by_series.png', dpi=300, bbox_inches='tight')
plt.close()

# Add neural network model
print("\nNeural Network Model:")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create shallow neural network
model = Sequential([
    Dense(4, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1)
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=0
)

# Evaluate model
train_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)

print(f"Training MSE: {train_score[0]:.4f}, MAE: {train_score[1]:.4f}")
print(f"Test MSE: {test_score[0]:.4f}, MAE: {test_score[1]:.4f}")

# Calculate R²
y_pred = model.predict(X_test).flatten()
r2 = r2_score(y_test, y_pred)
print(f"Test R²: {r2:.4f}")

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color=custom_palette[0], linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color=custom_palette[1], linewidth=2)
plt.title('Model Loss', fontweight='bold', pad=10)
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('MSE', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE', color=custom_palette[0], linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', color=custom_palette[1], linewidth=2)
plt.title('Model MAE', fontweight='bold', pad=10)
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('MAE', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/nn_training_history.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot actual vs predicted for neural network
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, s=80, edgecolor='white', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, alpha=0.7)
plt.xlabel('Actual IMDb Rating', fontweight='bold')
plt.ylabel('Predicted IMDb Rating', fontweight='bold')
plt.title('Neural Network: Actual vs Predicted IMDb Ratings', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('plots/nn_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
