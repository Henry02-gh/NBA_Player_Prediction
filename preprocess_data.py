import pandas as pd
import numpy as np

df = pd.read_csv("nba_historical_data.csv")
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# Sort by player and date so rolling calculations are chronologically accurate
df = df.sort_values(by=['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)

print("Processing raw features...")

# Home vs Away flag
df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 0 if '@' in str(x) else 1)

# Back-to-back flag (same logic as train_models.py for consistency)
df['DAYS_SINCE_LAST'] = df.groupby('PLAYER_NAME')['GAME_DATE'].diff().dt.days
df['IS_B2B'] = np.where(df['DAYS_SINCE_LAST'] <= 1, 1, 0)

for stat in ['PTS', 'REB', 'AST']:
    df[f'RECENT_{stat}_AVG'] = (
        df.groupby('PLAYER_NAME')[stat]
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )

# Fill NaN from each player's first games with their own career mean
for stat in ['PTS', 'REB', 'AST']:
    player_means = df.groupby('PLAYER_NAME')[stat].transform('mean')
    df[f'RECENT_{stat}_AVG'] = df[f'RECENT_{stat}_AVG'].fillna(player_means)

features_df = df[[
    'PLAYER_NAME', 'GAME_DATE', 'IS_HOME', 'IS_B2B',
    'RECENT_PTS_AVG', 'RECENT_REB_AVG', 'RECENT_AST_AVG',
    'PTS', 'REB', 'AST'
]]
features_df.to_csv("nba_features.csv", index=False)

print(f"✅ Saved 'nba_features.csv' ({len(features_df)} rows)")
print("   Features: IS_HOME, IS_B2B, RECENT_PTS_AVG (5-game rolling), RECENT_REB_AVG, RECENT_AST_AVG")
