import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

print("=========================================================================")
print("NBA Per-Player Model Training: Temporal Backtest Architecture")
print("=========================================================================\n")

KNOWN_PLAYERS = [
    "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
    "Luka Doncic", "Nikola Jokic", "Jayson Tatum", "Joel Embiid",
    "Shai Gilgeous-Alexander", "Anthony Edwards"
]

try:
    df = pd.read_csv("nba_historical_data.csv")
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("Computing rolling form and fatigue features...")

df['DAYS_SINCE_LAST_GAME'] = df.groupby('PLAYER_NAME')['GAME_DATE'].diff().dt.days
df['IS_B2B'] = np.where(df['DAYS_SINCE_LAST_GAME'] <= 1, 1, 0)
df['IS_HOME'] = df['MATCHUP'].str.contains('vs').astype(int)

for stat in ['PTS', 'REB', 'AST', 'FG_PCT']:
    df[f'ROLLING_{stat}_5'] = (
        df.groupby('PLAYER_NAME')[stat]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

train_mask = df['GAME_DATE'] < '2025-06-01'
test_mask  = df['GAME_DATE'] >= '2025-10-01'

train_df = df[train_mask].copy()
test_df  = df[test_mask].copy()

player_avgs = train_df.groupby('PLAYER_NAME')[['PTS', 'REB', 'AST', 'FG_PCT']].mean()
player_avgs.columns = ['HIST_PTS_AVG', 'HIST_REB_AVG', 'HIST_AST_AVG', 'HIST_FG_PCT_AVG']

for split in [train_df, test_df]:
    split['HIST_PTS_AVG']    = split['PLAYER_NAME'].map(player_avgs['HIST_PTS_AVG'])
    split['HIST_REB_AVG']    = split['PLAYER_NAME'].map(player_avgs['HIST_REB_AVG'])
    split['HIST_AST_AVG']    = split['PLAYER_NAME'].map(player_avgs['HIST_AST_AVG'])
    split['HIST_FG_PCT_AVG'] = split['PLAYER_NAME'].map(player_avgs['HIST_FG_PCT_AVG'])

# Shared model uses all 10 features including player-identity features (HIST averages)
feature_cols = [
    'IS_HOME', 'IS_B2B',
    'HIST_PTS_AVG', 'HIST_REB_AVG', 'HIST_AST_AVG', 'HIST_FG_PCT_AVG',
    'ROLLING_PTS_5', 'ROLLING_REB_5', 'ROLLING_AST_5', 'ROLLING_FG_PCT_5'
]
# Per-player models only use features that vary game-to-game within a single player.
# HIST_*_AVG are constants per player, so they contribute zero importance — excluded here.
per_player_feature_cols = [
    'IS_HOME', 'IS_B2B',
    'ROLLING_PTS_5', 'ROLLING_REB_5', 'ROLLING_AST_5', 'ROLLING_FG_PCT_5'
]
target_cols = ['PTS', 'REB', 'AST', 'FG_PCT']

print(f"Training split: {len(train_df)} rows  |  Test split: {len(test_df)} rows")
print(f"Shared features: {len(feature_cols)}  |  Per-player features: {len(per_player_feature_cols)}  |  Targets: {len(target_cols)}\n")
print("-" * 73)
print("Per-Player Model Training")
print("-" * 73)

models              = {}
player_mae          = {}
player_r2           = {}
player_importances  = {}
player_train_games  = {}
algo_comparison     = {}

COMPARISON_ALGOS = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression',  Ridge(alpha=1.0)),
    ('Decision Tree',     DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)),
]

for player in KNOWN_PLAYERS:
    p_train = train_df[train_df['PLAYER_NAME'] == player].copy()
    p_test  = test_df[test_df['PLAYER_NAME'] == player].copy()

    if len(p_train) < 20:
        print(f"  {player:<30} SKIP  ({len(p_train)} training games — need 20+)")
        continue

    X_tr = p_train[per_player_feature_cols].fillna(0).values
    y_tr = p_train[target_cols].values

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)

    models[player]             = rf
    player_train_games[player] = len(p_train)
    player_importances[player] = dict(zip(per_player_feature_cols, rf.feature_importances_.tolist()))

    if len(p_test) >= 3:
        X_te = p_test[per_player_feature_cols].fillna(0).values
        y_te = p_test[target_cols].values
        preds = rf.predict(X_te)
        mae = mean_absolute_error(y_te, preds, multioutput='raw_values')
        r2  = r2_score(y_te, preds, multioutput='raw_values')
        player_mae[player] = mae.tolist()
        player_r2[player]  = r2.tolist()
        print(f"  {player:<30} train={len(p_train):>3}  test={len(p_test):>3} | "
              f"PTS {mae[0]:.2f}  REB {mae[1]:.2f}  AST {mae[2]:.2f}  FG% {mae[3]:.4f}")

        algo_comparison[player] = {'Random Forest': mae.tolist()}
        for algo_name, algo in COMPARISON_ALGOS:
            algo.fit(X_tr, y_tr)
            algo_mae = mean_absolute_error(y_te, algo.predict(X_te), multioutput='raw_values')
            algo_comparison[player][algo_name] = algo_mae.tolist()
    else:
        player_mae[player] = [None, None, None, None]
        player_r2[player]  = [None, None, None, None]
        print(f"  {player:<30} train={len(p_train):>3}  test={len(p_test):>3} | "
              f"insufficient test data")

# Algorithm comparison averages across all players that have test data
ALL_ALGO_NAMES = ['Random Forest', 'Linear Regression', 'Ridge Regression', 'Decision Tree']
algo_comparison_avg = {}
for algo_name in ALL_ALGO_NAMES:
    maes = [algo_comparison[p][algo_name] for p in algo_comparison if algo_name in algo_comparison.get(p, {})]
    if maes:
        algo_comparison_avg[algo_name] = np.mean(maes, axis=0).tolist()

# Average feature importances across all trained player models (per-player feature set)
all_imp_values = list(player_importances.values())
avg_importances = {
    feat: float(np.mean([imp[feat] for imp in all_imp_values]))
    for feat in per_player_feature_cols
}

# Average MAE / R2 across players that have test data
valid_mae_rows = [player_mae[p] for p in player_mae if all(v is not None for v in player_mae[p])]
valid_r2_rows  = [player_r2[p]  for p in player_r2  if all(v is not None for v in player_r2[p])]
avg_mae = np.mean(valid_mae_rows, axis=0).tolist() if valid_mae_rows else [None] * 4
avg_r2  = np.mean(valid_r2_rows,  axis=0).tolist() if valid_r2_rows  else [None] * 4

print("\n" + "-" * 73)
print("Cross-Player Average (2025-26 Out-of-Sample)")
print("-" * 73)
print(f"  PTS  MAE: {avg_mae[0]:.2f} pts  |  R2: {avg_r2[0]:.3f}")
print(f"  REB  MAE: {avg_mae[1]:.2f} reb  |  R2: {avg_r2[1]:.3f}")
print(f"  AST  MAE: {avg_mae[2]:.2f} ast  |  R2: {avg_r2[2]:.3f}")
print(f"  FG%  MAE: {avg_mae[3]:.4f}      |  R2: {avg_r2[3]:.3f}")

print("\n" + "-" * 73)
print("Algorithm Comparison — Average MAE on 2025-26 Holdout")
print("-" * 73)
for algo_name in ALL_ALGO_NAMES:
    if algo_name in algo_comparison_avg:
        m = algo_comparison_avg[algo_name]
        marker = "  ← SELECTED" if algo_name == 'Random Forest' else ""
        print(f"  {algo_name:<22}  PTS {m[0]:.2f}  REB {m[1]:.2f}  AST {m[2]:.2f}  FG% {m[3]:.4f}{marker}")

print("\nAverage Feature Importances:")
for feat, imp in sorted(avg_importances.items(), key=lambda x: -x[1]):
    bar = "#" * int(imp * 40)
    print(f"  {feat:<22} {imp:.3f}  {bar}")

model_bundle = {
    "models":                    models,
    "feature_cols":              feature_cols,           # shared model features (10)
    "per_player_feature_cols":   per_player_feature_cols, # per-player features (6)
    "player_avgs":               player_avgs.to_dict(),
    "player_mae":                player_mae,
    "player_r2":                 player_r2,
    "player_importances":        player_importances,
    "player_train_games":        player_train_games,
    # Aggregate stats kept for backward compatibility with Tab 3 overview
    "mae":                       avg_mae,
    "r2":                        avg_r2,
    "feature_importances":       avg_importances,
    "algo_comparison":           algo_comparison,
    "algo_comparison_avg":       algo_comparison_avg,
}

with open("nba_best_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print(f"\nSUCCESS: Per-player bundle saved as 'nba_best_model.pkl'")
print(f"  Players trained: {len(models)} / {len(KNOWN_PLAYERS)}")
print("=========================================================================")
