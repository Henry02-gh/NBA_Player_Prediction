import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ProBall Analytics Engine", layout="wide", page_icon="🏀")
st.title("🏀 ProBall Analytics & Predictive Scouting Engine")
st.write("See how your favorite NBA players are likely to perform in their next game — and check how accurate the predictions really are.")
st.markdown("---")

TEAM_ABBREVIATIONS = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

KNOWN_PLAYERS = [
    "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
    "Luka Doncic", "Nikola Jokic", "Jayson Tatum", "Joel Embiid",
    "Shai Gilgeous-Alexander", "Anthony Edwards"
]

# 2025-26 Regular Season DEF_RATING (points allowed per 100 possessions, lower = better defense)
# Used as a reliable fallback when the live NBA API is unavailable
DEFENSIVE_RANKINGS_2025_26 = {
    "OKC": (1,  107.9), "DET": (2,  109.8), "SAS": (3,  111.4),
    "BOS": (4,  112.7), "TOR": (5,  113.3), "HOU": (6,  113.3),
    "NYK": (7,  113.4), "MIN": (8,  113.7), "ATL": (9,  113.8),
    "PHX": (10, 114.1), "CHA": (11, 114.5), "ORL": (12, 114.5),
    "MIA": (13, 114.5), "POR": (14, 114.7), "CLE": (15, 115.2),
    "PHI": (16, 115.6), "GSW": (17, 115.6), "LAC": (18, 116.3),
    "LAL": (19, 116.5), "DAL": (20, 116.7), "DEN": (21, 117.5),
    "CHI": (22, 118.2), "MEM": (23, 119.0), "IND": (24, 119.0),
    "NOP": (25, 119.1), "BKN": (26, 119.1), "MIL": (27, 119.4),
    "SAC": (28, 121.6), "UTA": (29, 122.4), "WAS": (30, 122.8),
}


@st.cache_resource
def load_predictive_brain():
    try:
        with open("nba_best_model.pkl", "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj
        return {
            "model": obj,
            "feature_cols": ['IS_HOME', 'IS_B2B', 'HIST_PTS_AVG', 'HIST_REB_AVG', 'HIST_AST_AVG']
        }
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_defensive_rankings():
    for season in ['2025-26', '2024-25']:
        try:
            from nba_api.stats.endpoints import leaguedashteamstats
            import time
            time.sleep(0.6)
            response = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Advanced',
                per_mode_simple='PerGame'
            )
            df = response.get_data_frames()[0]
            if df.empty or 'DEF_RATING' not in df.columns:
                continue
            df = df.sort_values('DEF_RATING').reset_index(drop=True)
            df['DEF_RANK'] = range(1, len(df) + 1)
            return (
                dict(zip(df['TEAM_ABBREVIATION'], df['DEF_RANK'].tolist())),
                dict(zip(df['TEAM_ABBREVIATION'], df['DEF_RATING'].round(1).tolist())),
                season,
                True
            )
        except Exception:
            continue
    rank_map = {abbr: v[0] for abbr, v in DEFENSIVE_RANKINGS_2025_26.items()}
    rating_map = {abbr: v[1] for abbr, v in DEFENSIVE_RANKINGS_2025_26.items()}
    return rank_map, rating_map, '2025-26', False


model_bundle = load_predictive_brain()


@st.cache_data
def get_all_player_data(player_name):
    try:
        raw_df = pd.read_csv("nba_historical_data.csv")
        player_df = raw_df[raw_df['PLAYER_NAME'] == player_name].copy()
        player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'])
        player_df = player_df.sort_values('GAME_DATE').reset_index(drop=True)
        player_df['IS_B2B'] = np.where(player_df['GAME_DATE'].diff().dt.days <= 1, 1, 0)
        return player_df
    except Exception:
        return pd.DataFrame()


@st.cache_data
def get_historical_matchups(player_name, opponent_name):
    try:
        player_df = get_all_player_data(player_name)
        if player_df.empty:
            raise ValueError("No player data available")

        opp_code = TEAM_ABBREVIATIONS.get(opponent_name, "")
        matchup_mask = player_df['MATCHUP'].str.contains(opp_code, case=False, na=False)
        h2h_df = player_df[matchup_mask].sort_values(by='GAME_DATE', ascending=False)

        display_cols = ['GAME_DATE', 'SEASON_TRACK', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST', 'FG_PCT', 'PLUS_MINUS', 'IS_B2B']

        # All pre-2025-26 player games for training baseline
        historical_train = player_df[player_df['GAME_DATE'] < '2025-06-01']
        # Tab 1: historical H2H games only (2022-23 through 2024-25)
        h2h_historical = h2h_df[h2h_df['GAME_DATE'] < '2025-10-01']
        # Tab 2: 2025-26 H2H games vs this specific opponent only
        current_validation = h2h_df[h2h_df['GAME_DATE'] >= '2025-10-01']

        out_df = h2h_historical[display_cols].copy()
        out_df['GAME_DATE'] = out_df['GAME_DATE'].dt.strftime('%Y-%m-%d')
        out_df.columns = ['Game Date', 'Season', 'Matchup', 'Result', 'MIN', 'PTS', 'REB', 'AST', 'FG%', '+/-', 'B2B Setting']
        return out_df, historical_train, current_validation, False
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), True


def run_model_prediction(model_bundle, feature_map, player_name=None):
    if not model_bundle:
        return None
    # Per-player model takes priority; fall back to shared/legacy model
    per_player_models = model_bundle.get("models", {})
    if player_name and player_name in per_player_models:
        model = per_player_models[player_name]
        cols = model_bundle.get(
            "per_player_feature_cols",
            ['IS_HOME', 'IS_B2B', 'ROLLING_PTS_5', 'ROLLING_REB_5', 'ROLLING_AST_5', 'ROLLING_FG_PCT_5']
        )
    else:
        model = model_bundle.get("model")
        cols = model_bundle.get(
            "feature_cols",
            ['IS_HOME', 'IS_B2B', 'HIST_PTS_AVG', 'HIST_REB_AVG', 'HIST_AST_AVG']
        )
    if model is None:
        return None
    input_vector = np.array([[feature_map.get(f, 0.0) for f in cols]])
    return model.predict(input_vector)[0]


@st.cache_data
def compute_overall_validation(_model_bundle):
    if not _model_bundle:
        return pd.DataFrame()

    player_avgs_dict = _model_bundle.get("player_avgs", {})

    records = []
    for player in KNOWN_PLAYERS:
        player_df = get_all_player_data(player)
        if player_df.empty:
            continue

        valid_df = player_df[player_df['GAME_DATE'] >= '2025-10-01'].copy()
        if valid_df.empty:
            records.append({
                'Player': player,
                '2025-26 Games': 0,
                'Actual Avg PTS': '—',
                'MAE PTS': '—',
                'MAE REB': '—',
                'MAE AST': '—',
                'MAE FG%': '—',
                'PTS Accuracy': 'No data — re-run fetch_historical_data.py'
            })
            continue

        hp = player_avgs_dict.get("HIST_PTS_AVG", {}).get(player, player_df['PTS'].mean())
        hr = player_avgs_dict.get("HIST_REB_AVG", {}).get(player, player_df['REB'].mean())
        ha = player_avgs_dict.get("HIST_AST_AVG", {}).get(player, player_df['AST'].mean())
        hfg = player_avgs_dict.get("HIST_FG_PCT_AVG", {}).get(player, player_df['FG_PCT'].mean() if 'FG_PCT' in player_df.columns else 0.45)

        pt_errors, rb_errors, as_errors, fg_errors = [], [], [], []
        for _, row in valid_df.iterrows():
            feature_map = {
                'IS_HOME': 1 if 'vs' in str(row['MATCHUP']) else 0,
                'IS_B2B': int(row['IS_B2B']),
                'HIST_PTS_AVG': hp, 'HIST_REB_AVG': hr, 'HIST_AST_AVG': ha, 'HIST_FG_PCT_AVG': hfg,
                'ROLLING_PTS_5': hp, 'ROLLING_REB_5': hr, 'ROLLING_AST_5': ha, 'ROLLING_FG_PCT_5': hfg
            }
            pred = run_model_prediction(_model_bundle, feature_map, player)
            if pred is not None:
                pt_errors.append(abs(pred[0] - row['PTS']))
                rb_errors.append(abs(pred[1] - row['REB']))
                as_errors.append(abs(pred[2] - row['AST']))
                if len(pred) >= 4 and 'FG_PCT' in row.index:
                    fg_errors.append(abs(pred[3] - row['FG_PCT']))

        if pt_errors:
            actual_avg = valid_df['PTS'].mean()
            mae_pts = np.mean(pt_errors)
            records.append({
                'Player': player,
                '2025-26 Games': len(valid_df),
                'Actual Avg PTS': round(actual_avg, 1),
                'MAE PTS': round(mae_pts, 2),
                'MAE REB': round(np.mean(rb_errors), 2),
                'MAE AST': round(np.mean(as_errors), 2),
                'MAE FG%': round(np.mean(fg_errors) * 100, 2) if fg_errors else '—',
                'PTS Accuracy': f"{max(0.0, 100.0 - mae_pts / actual_avg * 100):.1f}%" if actual_avg > 0 else "N/A"
            })

    return pd.DataFrame(records)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("🎯 Target Selection Setup")
selected_player = st.sidebar.selectbox("Select Target Player", KNOWN_PLAYERS)
sorted_teams = sorted(list(TEAM_ABBREVIATIONS.keys()))
selected_opponent = st.sidebar.selectbox("Select Opponent Team", sorted_teams)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Opponent Defensive Strength")

def_ranks, def_ratings, def_season, is_live_api = get_defensive_rankings()
opp_abbrev = TEAM_ABBREVIATIONS.get(selected_opponent, "")
auto_rank = def_ranks.get(opp_abbrev)
auto_rating = def_ratings.get(opp_abbrev)

default_rank = auto_rank if auto_rank else 15
col_a, col_b = st.sidebar.columns(2)
col_a.metric("DEF Rank", f"#{default_rank} / 30")
col_b.metric("DEF Rating", f"{auto_rating}" if auto_rating else "—")
if is_live_api:
    st.sidebar.caption(f"📡 Live NBA API · {def_season} · refreshes hourly")
else:
    st.sidebar.caption(f"📋 2025-26 season data (API offline — local fallback)")

opponent_def_rank = st.sidebar.slider(
    "Adjust Rank for Prediction (simulate future changes)",
    min_value=1, max_value=30, value=default_rank,
    help=f"Default = actual {def_season} rank. Drag to simulate how a different defensive strength would affect the projection."
)

with st.sidebar.expander("📋 All 30 Teams — 2025-26 Rankings"):
    ref_rows = []
    for team_name, abbr in sorted(TEAM_ABBREVIATIONS.items(), key=lambda x: DEFENSIVE_RANKINGS_2025_26.get(x[1], (15, 0))[0]):
        rank, rating = DEFENSIVE_RANKINGS_2025_26.get(abbr, (15, 111.0))
        ref_rows.append({"Rank": rank, "Team": team_name, "DEF Rating": rating})
    st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True, height=300)

if opponent_def_rank <= 8:
    tier_label = "Elite Defense 🔒"
    def_mod = 0.93
elif opponent_def_rank >= 22:
    tier_label = "Weak Defense 🔓"
    def_mod = 1.05
else:
    tier_label = "Average Defense ⚖️"
    def_mod = 1.00
st.sidebar.caption(f"Tier: **{tier_label}** → **{def_mod:.2f}x** modifier on PTS/AST projections")

try:
    csv_mtime = os.path.getmtime("nba_historical_data.csv")
    csv_date = datetime.fromtimestamp(csv_mtime).strftime('%Y-%m-%d %H:%M')
    st.sidebar.markdown("---")
    st.sidebar.caption(f"📁 Dataset last updated: **{csv_date}**")
except Exception:
    pass


full_table, train_split, valid_split, is_fallback = get_historical_matchups(selected_player, selected_opponent)
all_player_df = get_all_player_data(selected_player)

tab1, tab2, tab3 = st.tabs(["📊 Player Stats & Prediction", "📉 How Accurate Is It?", "⚙️ Model Info (Advanced)"])


# =========================================================
# TAB 1: SPLIT TRACKING & PROJECTIONS
# =========================================================
with tab1:
    st.info(
        "Browse the player's stats and history against this opponent, then scroll down to "
        "**predict their next game**. ➡️ Afterwards, open the **How Accurate Is It?** tab to see how "
        "the predictions compare against real game results."
    )

    if not all_player_df.empty:
        recent_season = all_player_df[all_player_df['GAME_DATE'] >= '2024-10-01']
        if recent_season.empty:
            recent_season = all_player_df

        st.subheader(f"📊 {selected_player} — Current Season Overview")
        ov1, ov2, ov3, ov4, ov5 = st.columns(5)
        ov1.metric("Games Played", len(recent_season))
        ov2.metric("Season PTS Avg", f"{recent_season['PTS'].mean():.1f}")
        ov3.metric("Season REB Avg", f"{recent_season['REB'].mean():.1f}")
        ov4.metric("Season AST Avg", f"{recent_season['AST'].mean():.1f}")
        wins = (recent_season['WL'] == 'W').sum() if 'WL' in recent_season.columns else 0
        ov5.metric("Win Rate", f"{wins / len(recent_season) * 100:.0f}%")
        st.markdown("---")

    st.subheader(f"📋 Head-to-Head History: {selected_player} vs. {selected_opponent} (2022-23 to 2024-25)")
    if is_fallback:
        st.error(
            f"No player data found for {selected_player}. "
            "Run `fetch_historical_data.py` to populate the dataset, then restart the app."
        )
    elif full_table.empty:
        st.info(
            f"No historical H2H games found for {selected_player} vs {selected_opponent} "
            "across the 2022-23 through 2024-25 seasons. "
            "Check Tab 2 to see if they faced each other in 2025-26."
        )
    else:
        st.caption(f"✅ Loaded {len(full_table)} H2H matchup records from 2022-23 through 2024-25.")
        st.dataframe(full_table.drop(columns=['B2B Setting'], errors='ignore'), use_container_width=True, hide_index=True)
        h2h_wins = (full_table['Result'] == 'W').sum()
        h2h_losses = (full_table['Result'] == 'L').sum()
        wl1, wl2, wl3 = st.columns(3)
        wl1.metric("Wins vs Opponent", int(h2h_wins))
        wl2.metric("Losses vs Opponent", int(h2h_losses))
        wl3.metric("H2H Win Rate", f"{h2h_wins / len(full_table) * 100:.0f}%" if len(full_table) > 0 else "—")

    # Baseline always computed — falls back to overall player averages when no H2H history
    if not full_table.empty:
        baseline_pts = full_table["PTS"].mean()
        baseline_reb = full_table["REB"].mean()
        baseline_ast = full_table["AST"].mean()
        baseline_fgp = full_table["FG%"].mean() if "FG%" in full_table.columns else 0.45
        st.markdown(
            f"📈 **H2H Split Averages vs {selected_opponent}:** "
            f"**{baseline_pts:.1f} PTS** | **{baseline_reb:.1f} REB** | **{baseline_ast:.1f} AST** | **{baseline_fgp*100:.1f}% FG**"
        )
    else:
        player_hist = all_player_df[all_player_df['GAME_DATE'] < '2025-10-01'] if not all_player_df.empty else pd.DataFrame()
        baseline_pts = player_hist['PTS'].mean() if not player_hist.empty else 25.0
        baseline_reb = player_hist['REB'].mean() if not player_hist.empty else 6.0
        baseline_ast = player_hist['AST'].mean() if not player_hist.empty else 5.0
        baseline_fgp = player_hist['FG_PCT'].mean() if not player_hist.empty and 'FG_PCT' in player_hist.columns else 0.45
        if not all_player_df.empty:
            st.caption(
                f"No H2H history found — using {selected_player}'s overall pre-2025-26 averages as baseline: "
                f"**{baseline_pts:.1f} PTS** / **{baseline_reb:.1f} REB** / **{baseline_ast:.1f} AST**"
            )

    if not is_fallback and not full_table.empty and 'Season' in full_table.columns:
        st.markdown("---")
        st.subheader("📅 Season-by-Season H2H Breakdown")
        season_rows = []
        for season_name, grp_df in full_table.groupby('Season', sort=True):
            season_rows.append({
                'Season': season_name,
                'Games': len(grp_df),
                'PTS Avg': round(grp_df['PTS'].mean(), 1),
                'REB Avg': round(grp_df['REB'].mean(), 1),
                'AST Avg': round(grp_df['AST'].mean(), 1),
                'FG% Avg': f"{grp_df['FG%'].mean()*100:.1f}%" if 'FG%' in grp_df.columns else '—',
                'Win Rate': f"{(grp_df['Result'] == 'W').sum() / len(grp_df) * 100:.0f}%"
            })
        st.dataframe(pd.DataFrame(season_rows), hide_index=True, use_container_width=True)

    if not is_fallback and len(full_table) >= 3:
        st.markdown("---")
        st.subheader(f"📈 H2H Performance Trend vs. {selected_opponent}")
        chart_df = full_table[['Game Date', 'PTS', 'REB', 'AST']].copy().set_index('Game Date').sort_index()
        st.line_chart(chart_df, height=260)

    if not is_fallback and not full_table.empty:
        st.markdown("---")
        st.subheader("🔍 Contextual Split Breakdown")
        sc1, sc2 = st.columns(2)

        with sc1:
            home_rows = full_table[full_table['Matchup'].str.contains('vs', case=False, na=False)]
            away_rows = full_table[~full_table['Matchup'].str.contains('vs', case=False, na=False)]
            ha_df = pd.DataFrame({
                'Context': ['Home', 'Away'],
                'Games': [len(home_rows), len(away_rows)],
                'PTS Avg': [
                    round(home_rows['PTS'].mean(), 1) if not home_rows.empty else 0.0,
                    round(away_rows['PTS'].mean(), 1) if not away_rows.empty else 0.0
                ],
                'REB Avg': [
                    round(home_rows['REB'].mean(), 1) if not home_rows.empty else 0.0,
                    round(away_rows['REB'].mean(), 1) if not away_rows.empty else 0.0
                ],
                'AST Avg': [
                    round(home_rows['AST'].mean(), 1) if not home_rows.empty else 0.0,
                    round(away_rows['AST'].mean(), 1) if not away_rows.empty else 0.0
                ]
            })
            st.write("**Home vs Away (this matchup)**")
            st.dataframe(ha_df, hide_index=True, use_container_width=True)

        with sc2:
            if 'B2B Setting' in full_table.columns:
                b2b_rows = full_table[full_table['B2B Setting'] == 1]
                rest_rows = full_table[full_table['B2B Setting'] == 0]
                fatigue_df = pd.DataFrame({
                    'Rest Status': ['Back-to-Back', 'Normal Rest'],
                    'Games': [len(b2b_rows), len(rest_rows)],
                    'PTS Avg': [
                        round(b2b_rows['PTS'].mean(), 1) if not b2b_rows.empty else 0.0,
                        round(rest_rows['PTS'].mean(), 1) if not rest_rows.empty else 0.0
                    ],
                    'REB Avg': [
                        round(b2b_rows['REB'].mean(), 1) if not b2b_rows.empty else 0.0,
                        round(rest_rows['REB'].mean(), 1) if not rest_rows.empty else 0.0
                    ]
                })
                st.write("**Fatigue Impact (B2B vs Normal Rest)**")
                st.dataframe(fatigue_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    with st.container(border=True):
        dc1, dc2, dc3, dc4 = st.columns([2.5, 1, 1, 1])
        dc1.markdown(f"**🛡️ {selected_opponent} — Defensive Strength**")
        dc2.metric("DEF Rank", f"#{opponent_def_rank} / 30")
        dc3.metric("Tier", tier_label.split(" ")[0] + " " + tier_label.split(" ")[1] if len(tier_label.split(" ")) > 1 else tier_label)
        if auto_rating:
            dc4.metric("DEF Rating", f"{auto_rating}", help="Points allowed per 100 possessions — lower = better defense")
        else:
            dc4.metric("Score Modifier", f"{def_mod:.2f}x")
        source_note = (
            f"📡 Auto-fetched from NBA Stats API ({def_season} Regular Season) · updates hourly"
            if def_ranks else
            "⚠️ Live API unavailable — rank set manually in sidebar"
        )
        st.caption(f"{source_note}  |  Modifier **{def_mod:.2f}x** applied to PTS & AST projections")

    st.markdown("---")
    st.subheader("🔮 Predict the Next Game")
    st.caption(
        "Choose the game conditions below. The prediction is based on how this player has performed "
        "against this opponent before, their recent 5-game form, and how strong the opponent's defense is."
    )

    rolling_pts, rolling_reb, rolling_ast = baseline_pts, baseline_reb, baseline_ast
    rolling_fgp = baseline_fgp
    if not all_player_df.empty:
        last5 = all_player_df.sort_values('GAME_DATE').tail(5)
        rolling_pts = last5['PTS'].mean()
        rolling_reb = last5['REB'].mean()
        rolling_ast = last5['AST'].mean()
        rolling_fgp = last5['FG_PCT'].mean() if 'FG_PCT' in last5.columns else baseline_fgp
        st.caption(
            f"📊 Recent 5-game form (used in prediction): "
            f"**{rolling_pts:.1f} PTS** | **{rolling_reb:.1f} REB** | **{rolling_ast:.1f} AST** | **{rolling_fgp*100:.1f}% FG**"
        )

    col1, col2 = st.columns(2)
    with col1:
        venue = st.radio("Where is the game played?", ["🏟️ Home Game", "✈️ Away Game"])
        is_home_val = 1 if "Home" in venue else 0
    with col2:
        is_b2b_val = st.checkbox("Playing back-to-back games? (player may be more tired)")
        is_b2b_input = 1 if is_b2b_val else 0

    if st.button("🔮 Predict Performance", use_container_width=True):
        if model_bundle:
            feature_map = {
                'IS_HOME': is_home_val,
                'IS_B2B': is_b2b_input,
                'HIST_PTS_AVG': baseline_pts,
                'HIST_REB_AVG': baseline_reb,
                'HIST_AST_AVG': baseline_ast,
                'HIST_FG_PCT_AVG': baseline_fgp,
                'ROLLING_PTS_5': rolling_pts,
                'ROLLING_REB_5': rolling_reb,
                'ROLLING_AST_5': rolling_ast,
                'ROLLING_FG_PCT_5': rolling_fgp
            }
            prediction = run_model_prediction(model_bundle, feature_map, selected_player)

            p_pts = max(0.0, prediction[0] * def_mod)
            p_reb = max(0.0, prediction[1])
            p_ast = max(0.0, prediction[2] * def_mod)
            p_fgp = max(0.0, min(1.0, prediction[3])) if len(prediction) >= 4 else baseline_fgp

            st.session_state['last_scenario'] = {
                'player': selected_player,
                'opponent': selected_opponent,
                'is_home': is_home_val,
                'is_b2b': is_b2b_input,
                'p_pts': p_pts,
                'p_reb': p_reb,
                'p_ast': p_ast,
                'p_fgp': p_fgp,
                'baseline_pts': baseline_pts,
                'baseline_reb': baseline_reb,
                'baseline_ast': baseline_ast,
                'baseline_fgp': baseline_fgp,
                'def_rank': opponent_def_rank,
                'tier_label': tier_label,
                'def_mod': def_mod
            }

            st.markdown(f"### 🎯 Scenario Projections vs. {selected_opponent}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Projected Points", f"{p_pts:.1f}", delta=f"{p_pts - baseline_pts:+.1f} vs H2H Avg")
            c2.metric("Projected Rebounds", f"{p_reb:.1f}", delta=f"{p_reb - baseline_reb:+.1f} vs H2H Avg")
            c3.metric("Projected Assists", f"{p_ast:.1f}", delta=f"{p_ast - baseline_ast:+.1f} vs H2H Avg")
            c4.metric("Projected FG%", f"{p_fgp*100:.1f}%", delta=f"{(p_fgp - baseline_fgp)*100:+.1f}pp vs H2H Avg")
            st.caption(
                f"Baseline: H2H averages ({baseline_pts:.1f} PTS / {baseline_reb:.1f} REB / {baseline_ast:.1f} AST / {baseline_fgp*100:.1f}% FG)  "
                f"| Defense modifier: **{def_mod:.2f}x** (Rank #{opponent_def_rank} — {tier_label})"
            )
            st.success(
                "✅ Prediction saved. Open the **How Accurate Is It?** tab to see real 2025-26 game results "
                "for this matchup — your prediction is pinned at the top of that tab for easy comparison."
            )
        else:
            st.error("Model not loaded. Run `train_models.py` to generate 'nba_best_model.pkl'.")


# =========================================================
# TAB 2: 2025-2026 BACKTEST VALIDATION
# =========================================================
with tab2:
    st.info(
        "This tab checks the model against reality, in 3 steps: "
        "**① Your prediction** from Tab 1 · **② What actually happened** in real 2025-26 games · "
        "**③ Final comparison** — your prediction vs the real games that match your chosen conditions."
    )

    st.subheader("① Your Prediction (from Tab 1)")
    scenario = st.session_state.get('last_scenario', {})
    if scenario and scenario.get('player') == selected_player and scenario.get('opponent') == selected_opponent:
        with st.container(border=True):
            venue_label = "Home" if scenario['is_home'] else "Away"
            fatigue_label = "Back-to-Back" if scenario['is_b2b'] else "Normal Rest"
            sr1, sr2, sr3, sr4, sr5 = st.columns([2, 1, 1, 1, 1])
            sr1.markdown(
                f"**Conditions you chose:** {venue_label} · {fatigue_label}  \n"
                f"Opponent defense: #{scenario['def_rank']} ({scenario['tier_label']})"
            )
            sr2.metric("Predicted PTS", f"{scenario['p_pts']:.1f}")
            sr3.metric("Predicted REB", f"{scenario['p_reb']:.1f}")
            sr4.metric("Predicted AST", f"{scenario['p_ast']:.1f}")
            sr5.metric("Predicted FG%", f"{scenario.get('p_fgp', 0)*100:.1f}%")
            st.caption(
                f"For comparison, {scenario['player']}'s usual stats vs this opponent in past seasons: "
                f"**{scenario['baseline_pts']:.1f} PTS / {scenario['baseline_reb']:.1f} REB / {scenario['baseline_ast']:.1f} AST**"
            )
    else:
        st.caption(
            "💡 No prediction yet — go to **Tab 1**, choose the game conditions, and click "
            "**🔮 Predict Performance**. Your prediction will then appear here so you can "
            "compare it against real games."
        )

    st.markdown("---")
    st.subheader("② What Actually Happened — Real 2025-26 Games")

    if is_fallback or all_player_df.empty:
        st.error(
            f"No player data found for {selected_player}. "
            "Run `fetch_historical_data.py` to populate the dataset, then restart the app."
        )
    elif valid_split.empty:
        st.warning(
            f"No 2025-26 season games found for **{selected_player}** vs **{selected_opponent}**. "
            "They may not have faced each other in the 2025-26 Regular Season or Playoffs, "
            "or this player missed the season due to injury. "
            "Re-run `fetch_historical_data.py` to pull the full season including Playoffs."
        )
    else:
        val_df = valid_split.copy().sort_values('GAME_DATE')
        base_p = train_split['PTS'].mean() if not train_split.empty else baseline_pts
        base_r = train_split['REB'].mean() if not train_split.empty else baseline_reb
        base_a = train_split['AST'].mean() if not train_split.empty else baseline_ast
        base_fg = train_split['FG_PCT'].mean() if not train_split.empty and 'FG_PCT' in train_split.columns else baseline_fgp

        has_season_type = 'SEASON_TYPE' in val_df.columns
        reg_count = len(val_df[val_df['SEASON_TYPE'] == 'Regular Season']) if has_season_type else len(val_df)
        playoff_count = len(val_df[val_df['SEASON_TYPE'] == 'Playoffs']) if has_season_type else 0

        breakdown = f"Regular Season: **{reg_count}**"
        if playoff_count > 0:
            breakdown += f"  +  Playoffs: **{playoff_count}**"

        st.write(
            f"🔎 Found **{len(val_df)} H2H game(s)** — {selected_player} vs {selected_opponent} "
            f"in the 2025-26 season.  |  {breakdown}"
        )

        if reg_count == 0 and playoff_count > 0:
            st.caption(
                "⚠️ Regular Season games show 0 — the regular season data may have been missed during the last fetch "
                "(likely a temporary API timeout). Re-run `fetch_historical_data.py` to pull the missing games."
            )
        actual_rows = []
        for _, row in val_df.iterrows():
            matchup_str = str(row['MATCHUP'])
            actual_rows.append({
                "Date": row['GAME_DATE'].strftime('%Y-%m-%d'),
                "Season Type": str(row['SEASON_TYPE']) if has_season_type else 'Regular Season',
                "Matchup": matchup_str,
                "Home/Away": "Home" if "vs" in matchup_str else "Away",
                "B2B": "Yes" if int(row['IS_B2B']) else "No",
                "Result": str(row['WL']) if 'WL' in row.index else '—',
                "PTS": row['PTS'],
                "REB": row['REB'],
                "AST": row['AST'],
                "FG%": f"{row['FG_PCT']*100:.1f}%" if 'FG_PCT' in row.index else '—',
            })
        st.dataframe(pd.DataFrame(actual_rows), use_container_width=True, hide_index=True)

        records = []
        absolute_errors = []
        reb_absolute_errors = []
        ast_absolute_errors = []
        fgp_absolute_errors = []

        for _, row in val_df.iterrows():
            matchup_str = str(row['MATCHUP'])
            is_home = 1 if "vs" in matchup_str else 0
            is_b2b_actual = int(row['IS_B2B'])
            season_type_val = str(row['SEASON_TYPE']) if has_season_type else 'Regular Season'

            feature_map = {
                'IS_HOME': is_home, 'IS_B2B': is_b2b_actual,
                'HIST_PTS_AVG': base_p, 'HIST_REB_AVG': base_r, 'HIST_AST_AVG': base_a, 'HIST_FG_PCT_AVG': base_fg,
                'ROLLING_PTS_5': base_p, 'ROLLING_REB_5': base_r, 'ROLLING_AST_5': base_a, 'ROLLING_FG_PCT_5': base_fg
            }
            pred = run_model_prediction(model_bundle, feature_map, selected_player)
            pred_pts = (pred[0] * def_mod) if pred is not None else base_p
            pred_reb = pred[1] if pred is not None else base_r
            pred_ast = (pred[2] * def_mod) if pred is not None else base_a
            pred_fgp = max(0.0, min(1.0, pred[3])) if pred is not None and len(pred) >= 4 else base_fg

            actual_pts = row['PTS']
            actual_reb = row['REB']
            actual_ast = row['AST']
            actual_fgp = row['FG_PCT'] if 'FG_PCT' in row.index else base_fg
            error_val = abs(pred_pts - actual_pts)
            absolute_errors.append(error_val)
            reb_absolute_errors.append(abs(pred_reb - actual_reb))
            ast_absolute_errors.append(abs(pred_ast - actual_ast))
            fgp_absolute_errors.append(abs(pred_fgp - actual_fgp))
            raw_diff = actual_pts - pred_pts
            variance_str = f"+{raw_diff:.1f}" if raw_diff >= 0 else f"{raw_diff:.1f}"

            records.append({
                "Date": row['GAME_DATE'].strftime('%Y-%m-%d'),
                "Season Type": season_type_val,
                "Matchup": matchup_str,
                "Home/Away": "Home" if is_home else "Away",
                "B2B": "Yes" if is_b2b_actual else "No",
                "Pred PTS": round(pred_pts, 1),
                "Actual PTS": actual_pts,
                "PTS Error": variance_str,
                "Pred REB": round(pred_reb, 1),
                "Actual REB": actual_reb,
                "Pred AST": round(pred_ast, 1),
                "Actual AST": actual_ast,
                "Pred FG%": f"{pred_fgp*100:.1f}%",
                "Actual FG%": f"{actual_fgp*100:.1f}%"
            })

        verify_display_df = pd.DataFrame(records)

        mean_error = np.mean(absolute_errors)
        mean_reb_error = np.mean(reb_absolute_errors)
        mean_ast_error = np.mean(ast_absolute_errors)
        mean_fgp_error = np.mean(fgp_absolute_errors) if fgp_absolute_errors else 0.0
        accuracy_est = max(0.0, 100.0 - (mean_error / base_p * 100)) if base_p > 0 else 0.0

        st.markdown("---")
        st.subheader("③ Final Comparison — Your Prediction vs Same-Condition Games")
        if scenario and scenario.get('player') == selected_player and scenario.get('opponent') == selected_opponent:
            sc_home = scenario['is_home']
            sc_b2b = scenario['is_b2b']
            venue_label = "Home" if sc_home else "Away"
            fatigue_label = "Back-to-Back" if sc_b2b else "Normal Rest"
            is_home_series = val_df['MATCHUP'].astype(str).str.contains('vs').astype(int)
            matched_df = val_df[(is_home_series == sc_home) & (val_df['IS_B2B'].astype(int) == sc_b2b)]
            st.caption(
                f"Your Tab 1 prediction was for a **{venue_label} game · {fatigue_label}**. "
                f"Out of the **{len(val_df)}** real game(s) above, **{len(matched_df)}** matched these exact conditions."
            )
            if matched_df.empty:
                st.warning(
                    "No real 2025-26 games were played under these exact conditions. "
                    "Compare cautiously against the table above — or go back to **Tab 1** and change the "
                    "conditions (venue / rest) to match one of the real games."
                )
            else:
                if len(matched_df) == 1:
                    game_row = matched_df.iloc[0]
                    st.write(
                        f"Matching game: **{game_row['GAME_DATE'].strftime('%Y-%m-%d')}** ({game_row['MATCHUP']})"
                    )
                else:
                    match_dates = ", ".join(matched_df.sort_values('GAME_DATE')['GAME_DATE'].dt.strftime('%Y-%m-%d'))
                    st.write(f"Comparing against the **average of {len(matched_df)} matching games**: {match_dates}")

                act_pts = matched_df['PTS'].mean()
                act_reb = matched_df['REB'].mean()
                act_ast = matched_df['AST'].mean()
                act_fgp = matched_df['FG_PCT'].mean() if 'FG_PCT' in matched_df.columns else None
                final_rows = [
                    {"Stat": "Points",   "Your Prediction": f"{scenario['p_pts']:.1f}", "What Really Happened": f"{act_pts:.1f}", "Difference": f"{scenario['p_pts'] - act_pts:+.1f}"},
                    {"Stat": "Rebounds", "Your Prediction": f"{scenario['p_reb']:.1f}", "What Really Happened": f"{act_reb:.1f}", "Difference": f"{scenario['p_reb'] - act_reb:+.1f}"},
                    {"Stat": "Assists",  "Your Prediction": f"{scenario['p_ast']:.1f}", "What Really Happened": f"{act_ast:.1f}", "Difference": f"{scenario['p_ast'] - act_ast:+.1f}"},
                ]
                if act_fgp is not None:
                    final_rows.append({
                        "Stat": "Field Goal %",
                        "Your Prediction": f"{scenario.get('p_fgp', 0)*100:.1f}%",
                        "What Really Happened": f"{act_fgp*100:.1f}%",
                        "Difference": f"{(scenario.get('p_fgp', 0) - act_fgp)*100:+.1f}%"
                    })
                st.dataframe(pd.DataFrame(final_rows), use_container_width=True, hide_index=True)

                pts_gap = abs(scenario['p_pts'] - act_pts)
                pred_accuracy = max(0.0, 100.0 - (pts_gap / act_pts * 100)) if act_pts > 0 else 0.0
                st.progress(
                    min(1.0, pred_accuracy / 100),
                    text=f"🎯 This prediction's Points Accuracy: **{pred_accuracy:.1f}%**"
                )
                st.caption(
                    f"How close *your* prediction came to the real result. Compare it with the model's "
                    f"overall accuracy of **{accuracy_est:.1f}%** across all {len(val_df)} real game(s) "
                    "in the section below."
                )
                if pts_gap <= mean_error:
                    st.success(
                        f"✅ Your prediction was off by only **{pts_gap:.1f} points** — within the model's "
                        f"typical miss of ±{mean_error:.1f} pts for this matchup. A reliable prediction!"
                    )
                else:
                    st.info(
                        f"Your prediction was off by **{pts_gap:.1f} points** — a bit more than the typical "
                        f"miss of ±{mean_error:.1f} pts. Keep in mind only {len(matched_df)} game(s) matched "
                        "your conditions, and a single game can swing a lot."
                    )
        else:
            st.caption(
                "💡 Run a prediction in **Tab 1** first — this section will then automatically find the real "
                "games that match your chosen conditions (venue and rest) and compare them side by side."
            )

        st.markdown("---")
        st.subheader("📏 Model Accuracy for These Games")
        st.caption(
            "Behind the scenes, the model also re-predicted every real game above under its actual "
            "conditions. These numbers show how far those predictions were from the real results."
        )
        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("Points — Avg Miss", f"±{mean_error:.1f} pts",
                  help="On average, the predicted points were this far from the player's actual points.")
        v2.metric("Rebounds — Avg Miss", f"±{mean_reb_error:.1f}",
                  help="On average, the predicted rebounds were this far from the actual rebounds.")
        v3.metric("Assists — Avg Miss", f"±{mean_ast_error:.1f}",
                  help="On average, the predicted assists were this far from the actual assists.")
        v4.metric("FG% — Avg Miss", f"±{mean_fgp_error*100:.1f}%",
                  help="On average, the predicted field-goal percentage was this far from the actual one.")
        v5.metric("Points Accuracy", f"{accuracy_est:.1f}%",
                  help="How close the point predictions were overall — 100% would mean a perfect prediction every game.")
        st.caption(
            "💡 Smaller \"miss\" numbers mean better predictions. "
            "For reference, most fans surveyed said 80–90% accuracy is good enough to trust a prediction system."
        )

        if len(absolute_errors) >= 3:
            st.subheader("📊 Predicted vs Actual Points")
            cmp_data = pd.DataFrame({
                'Game': verify_display_df['Date'].str[-5:],
                'Predicted': verify_display_df['Pred PTS'].values,
                'Actual': verify_display_df['Actual PTS'].values
            })
            cmp_melted = cmp_data.melt(id_vars='Game', var_name='Type', value_name='PTS')
            fig_cmp, ax_cmp = plt.subplots(figsize=(8, 3))
            sns.barplot(data=cmp_melted, x='Game', y='PTS', hue='Type',
                        palette={'Predicted': 'steelblue', 'Actual': 'coral'}, ax=ax_cmp)
            ax_cmp.set_xlabel('Game Date (MM-DD)')
            ax_cmp.set_ylabel('Points')
            ax_cmp.set_title(f'Predicted vs Actual PTS — {selected_player} vs {selected_opponent}')
            ax_cmp.legend(title='', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_cmp)
            plt.close(fig_cmp)
            st.caption("Blue = predicted · Orange = what actually happened. When the two bars are close in height, the prediction was accurate.")

        with st.expander("🔍 See the model's game-by-game predictions (no data leakage)"):
            st.write(
                f"The model was trained **only on games before the 2025-26 season**, so it has never seen "
                f"the games above. For each game, it was given the real conditions (home/away, back-to-back) "
                f"plus what it knew about {selected_player} before the season — his overall averages of "
                f"**{base_p:.1f} PTS / {base_r:.1f} REB / {base_a:.1f} AST / {base_fg*100:.1f}% FG** — "
                "and its prediction (\"Pred\" columns) is compared against what he actually did "
                "(\"Actual\" columns)."
            )
            st.write(
                "**Why can these predictions differ slightly from your Tab 1 prediction?** "
                "Tab 1 uses the player's *current* recent form, while this test only allows *pre-season* "
                "knowledge — no information from the future is leaked into it. That keeps the accuracy "
                "test fair, and a small gap between the two predictions is expected and normal."
            )
            st.dataframe(verify_display_df, use_container_width=True, hide_index=True)


# =========================================================
# TAB 3: PIPELINE MAINTENANCE
# =========================================================
with tab3:
    st.info(
        "A quick summary of how accurate the prediction model is, "
        "plus an accuracy check across all 10 players."
    )

    if model_bundle:
        mae_vals  = model_bundle.get("mae", [])
        p_mae_all = model_bundle.get("player_mae", {})
        p_tg_all  = model_bundle.get("player_train_games", {})

        if mae_vals:
            st.subheader("📊 How Accurate Is the Model?")
            sel_mae = p_mae_all.get(selected_player, [])
            tg = p_tg_all.get(selected_player, '?')
            if sel_mae and all(v is not None for v in sel_mae):
                st.caption(
                    f"Showing **{selected_player}**'s personal prediction model, built from **{tg} past games**. "
                    "Each number shows how far off the predictions typically are — smaller is better."
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Points", f"±{sel_mae[0]:.1f} pts")
                m2.metric("Rebounds", f"±{sel_mae[1]:.1f}")
                m3.metric("Assists", f"±{sel_mae[2]:.1f}")
                m4.metric("Field Goal %", f"±{sel_mae[3]*100:.1f}%")
            else:
                st.caption(
                    f"No 2025-26 test games for **{selected_player}** yet — showing the average across all players. "
                    "Each number shows how far off the predictions typically are — smaller is better."
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Points", f"±{mae_vals[0]:.1f} pts")
                m2.metric("Rebounds", f"±{mae_vals[1]:.1f}")
                m3.metric("Assists", f"±{mae_vals[2]:.1f}")
                if len(mae_vals) >= 4 and mae_vals[3] is not None:
                    m4.metric("Field Goal %", f"±{mae_vals[3]*100:.1f}%")
            st.markdown("---")

    else:
        st.warning("No model bundle loaded. Run `train_models.py` to generate the model.")

    st.markdown("---")
    st.subheader("📋 Accuracy Check — All 10 Players (2025-26 Season)")
    st.write(
        "This tests the model against **every 2025-26 game** for all 10 players — "
        "games the model never saw during training, so it's a fair, real-world accuracy test."
    )
    st.caption(
        "The model was trained on 2023-24 and 2024-25 data only, so every 2025-26 game is a genuine test. "
        "If '2025-26 Games = 0' shows for any player, re-run `fetch_historical_data.py` to pull the latest data."
    )

    if st.button("▶ Run Accuracy Check (all 10 players)", use_container_width=True):
        with st.spinner("Computing predictions on all 2025-26 holdout games..."):
            audit_df = compute_overall_validation(model_bundle)
        if not audit_df.empty:
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
            numeric_rows = audit_df[audit_df['MAE PTS'] != '—'].copy()
            if not numeric_rows.empty:
                numeric_rows['MAE PTS'] = pd.to_numeric(numeric_rows['MAE PTS'])
                numeric_rows['MAE REB'] = pd.to_numeric(numeric_rows['MAE REB'])
                numeric_rows['MAE AST'] = pd.to_numeric(numeric_rows['MAE AST'])
                overall_mae = numeric_rows['MAE PTS'].mean()
                total_games = numeric_rows['2025-26 Games'].sum()
                overall_acc = numeric_rows['PTS Accuracy'].str.replace('%', '').astype(float).mean()
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Overall MAE (PTS)", f"{overall_mae:.2f} pts")
                a2.metric("Total Validation Games", int(total_games))
                a3.metric("Avg PTS Accuracy", f"{overall_acc:.1f}%")
                if 'MAE FG%' in numeric_rows.columns:
                    fg_numeric_rows = numeric_rows[numeric_rows['MAE FG%'] != '—'].copy()
                    if not fg_numeric_rows.empty:
                        fg_numeric_rows['MAE FG%'] = pd.to_numeric(fg_numeric_rows['MAE FG%'])
                        a4.metric("Avg FG% MAE", f"{fg_numeric_rows['MAE FG%'].mean():.2f}pp")
                st.markdown("---")
                st.write("**Prediction Error Comparison — All Players (MAE)**")
                chart_cols = ['MAE PTS', 'MAE REB', 'MAE AST']
                if 'MAE FG%' in numeric_rows.columns and (numeric_rows['MAE FG%'] != '—').any():
                    fg_chart = numeric_rows[numeric_rows['MAE FG%'] != '—'].copy()
                    fg_chart['MAE FG%'] = pd.to_numeric(fg_chart['MAE FG%'])
                    chart_data = numeric_rows.set_index('Player')[chart_cols].copy()
                    chart_data['MAE FG%'] = fg_chart.set_index('Player')['MAE FG%']
                else:
                    chart_data = numeric_rows.set_index('Player')[chart_cols]
                st.bar_chart(chart_data, height=300)
        else:
            st.warning("No validation data computed — ensure the model bundle is loaded and 2025-26 data exists in the CSV.")
