import pandas as pd
import time
from nba_api.stats.endpoints import playergamelog

TARGET_PLAYERS = {
    "LeBron James": "2544",
    "Stephen Curry": "201939",
    "Kevin Durant": "201142",
    "Giannis Antetokounmpo": "203507",
    "Luka Doncic": "1629029",
    "Nikola Jokic": "203999",
    "Jayson Tatum": "1628369",
    "Joel Embiid": "203954",
    "Shai Gilgeous-Alexander": "1628983",
    "Anthony Edwards": "1630162"
}

TARGET_SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']
SEASON_TYPES = ['Regular Season', 'Playoffs']

all_game_logs = []

print("=========================================================================")
print("🏀 ProBall Data Pipeline: Multi-Season Historical Ingestion Starting")
print(f"📂 Seasons: {', '.join(TARGET_SEASONS)}  |  Types: Regular Season + Playoffs")
print("=========================================================================\n")

for player_name, player_id in TARGET_PLAYERS.items():
    print(f"🎬 Extracting: {player_name}")

    for season in TARGET_SEASONS:
        for season_type in SEASON_TYPES:
            print(f"  📡 {season} — {season_type}...")
            df = None
            for attempt in range(1, 4):
                try:
                    log = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season,
                        season_type_all_star=season_type
                    )
                    df = log.get_data_frames()[0]
                    break
                except Exception as e:
                    print(f"  ⚠️ Attempt {attempt}/3 failed: {e}")
                    if attempt < 3:
                        time.sleep(6 * attempt)
                    else:
                        print(f"  ❌ All 3 attempts failed — skipping {season} {season_type} for {player_name}.")

            if df is not None:
                if not df.empty:
                    df['PLAYER_NAME'] = player_name
                    df['SEASON_TRACK'] = season
                    df['SEASON_TYPE'] = season_type
                    all_game_logs.append(df)
                    print(f"  ✅ {len(df)} games added.")
                else:
                    print(f"  ⚠️ No data returned (player may not have participated).")
            time.sleep(2.5)

    print(f"✨ Done: {player_name}\n")

print("-------------------------------------------------------------------------")
print("💾 Saving consolidated database...")
print("-------------------------------------------------------------------------")

if all_game_logs:
    final_dataset = pd.concat(all_game_logs, ignore_index=True)

    # Sort so 'Playoffs' rows come before 'Regular Season' rows for the same game
    # ('Playoffs' < 'Regular Season' alphabetically, so ascending sort puts Playoffs first)
    # This ensures the Finals/playoff game is correctly tagged when it appears in both API responses
    final_dataset = final_dataset.sort_values(
        by=['PLAYER_NAME', 'GAME_DATE', 'MATCHUP', 'SEASON_TYPE'],
        ascending=[True, True, True, True]
    )
    final_dataset = final_dataset.drop_duplicates(
        subset=['PLAYER_NAME', 'GAME_DATE', 'MATCHUP'], keep='first'
    ).reset_index(drop=True)

    # NBA regular season never runs into May or June — fix any Finals/late-playoff games
    # that the API incorrectly returns under the Regular Season endpoint
    parsed_dates = pd.to_datetime(final_dataset['GAME_DATE'])
    mis_tagged = (final_dataset['SEASON_TYPE'] == 'Regular Season') & (parsed_dates.dt.month.isin([5, 6]))
    if mis_tagged.any():
        print(f"  Correcting {mis_tagged.sum()} mis-tagged May/June game(s) from Regular Season to Playoffs")
        final_dataset.loc[mis_tagged, 'SEASON_TYPE'] = 'Playoffs'

    final_dataset.to_csv("nba_historical_data.csv", index=False)

    reg = final_dataset[final_dataset['SEASON_TYPE'] == 'Regular Season']
    pla = final_dataset[final_dataset['SEASON_TYPE'] == 'Playoffs']
    print("\n🚀 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📊 Total rows saved: {len(final_dataset)}")
    print(f"   Regular Season: {len(reg)} rows")
    print(f"   Playoffs:       {len(pla)} rows")

    print("\n📋 2025-26 Season Breakdown per Player:")
    ds26 = final_dataset[final_dataset['SEASON_TRACK'] == '2025-26']
    for player in TARGET_PLAYERS:
        p26 = ds26[ds26['PLAYER_NAME'] == player]
        rs = len(p26[p26['SEASON_TYPE'] == 'Regular Season'])
        po = len(p26[p26['SEASON_TYPE'] == 'Playoffs'])
        status = "⚠️ LOW" if rs < 20 else "✅"
        print(f"  {status} {player}: {rs} Regular Season + {po} Playoffs")
    print("=========================================================================")
else:
    print("\n❌ Critical Failure: No data collected.")
    print("=========================================================================")
