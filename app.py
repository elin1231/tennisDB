import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from collections import OrderedDict
import csv

# Constants
DEFAULT_ELO = 1500
K_FACTOR = 32
DATA_DIR = "atp_data"
YEARS = range(2020, 2024)

# Initialize dictionaries
surface_elo = {surface: {} for surface in ["hard", "clay", "grass", "unknown", "overall"]}
elo_timeseries = {surface: {} for surface in ["hard", "clay", "grass", "unknown", "overall"]}

# Load player data
with open(os.path.join(DATA_DIR, "atp_players.csv")) as pf:
    id_to_name_players = OrderedDict((row[0], row) for row in csv.reader(pf))

# Helper functions
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(elo_dict, winner, loser, surface, match_date):
    winner_rating = elo_dict.get(winner, DEFAULT_ELO)
    loser_rating = elo_dict.get(loser, DEFAULT_ELO)

    expected_winner = expected_score(winner_rating, loser_rating)
    expected_loser = 1 - expected_winner

    elo_dict[winner] = winner_rating + K_FACTOR * (1 - expected_winner)
    elo_dict[loser] = loser_rating - K_FACTOR * expected_loser

    for player, rating in [(winner, elo_dict[winner]), (loser, elo_dict[loser])]:
        elo_timeseries[surface].setdefault(player, []).append({"date": match_date, "elo": rating})

def load_data(years, data_dir):
    files = [os.path.join(data_dir, f"atp_matches_{year}.csv") for year in years]
    return pd.concat(
        (pd.read_csv(file).assign(tourney_date=lambda df: pd.to_datetime(df['tourney_date'], format='%Y%m%d'))
         for file in files if os.path.exists(file)),
        ignore_index=True,
    )

# Load and process match data
df = load_data(YEARS, DATA_DIR)
df['surface'] = df['surface'].fillna('unknown').str.lower()

for _, row in df.iterrows():
    update_elo(surface_elo[row['surface']], row['winner_id'], row['loser_id'], row['surface'], row['tourney_date'])
    update_elo(surface_elo['overall'], row['winner_id'], row['loser_id'], "overall", row['tourney_date'])

# Prepare Elo dataframes
elo_dataframes = {
    surface: pd.DataFrame(
        [{"player_id": player, **match} for player, matches in players.items() for match in matches]
    )
    for surface, players in elo_timeseries.items()
}

# Streamlit app
st.title("Tennis Player Elo Ratings Over Time")

player_ids = set(df['winner_id']).union(df['loser_id'])
player_name_to_id = {
    tuple(id_to_name_players.get(str(player_id))[1:3]): player_id for player_id in player_ids
}

player1 = st.selectbox("Select Player 1:", player_name_to_id.keys(), format_func=lambda x: ' '.join(x))
player2 = st.selectbox("Select Player 2:", player_name_to_id.keys(), format_func=lambda x: ' '.join(x))

if player1 and player2:
    st.write("### Elo Ratings Time Series")

    for surface, data in elo_dataframes.items():
        player1_data = data[data['player_id'] == player_name_to_id[player1]].sort_values(by="date")
        player2_data = data[data['player_id'] == player_name_to_id[player2]].sort_values(by="date")

        if player1_data.empty and player2_data.empty:
            st.write(f"No data available for {surface.capitalize()} Court Elo Ratings.")
            continue

        fig = go.Figure()

        for player_data, name in [(player1_data, player1), (player2_data, player2)]:
            if not player_data.empty:
                fig.add_trace(go.Scatter(
                    x=player_data['date'],
                    y=player_data['elo'],
                    mode='lines+markers',
                    name=' '.join(name),
                ))

        fig.update_layout(
            title=f"{surface.capitalize()} Court Elo Ratings Over Time",
            xaxis_title="Date",
            yaxis_title="Elo Rating",
            legend_title="Players",
            template="plotly_white"
        )
        st.plotly_chart(fig)
