import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from collections import OrderedDict
import csv

surface_elo = {
    "hard": {},
    "clay": {},
    "grass": {},
    "unknown": {},
    "overall": {}
}
default_elo = 1500
k_factor = 32

elo_timeseries = {
    "hard": {},
    "clay": {},
    "grass": {},
    "unknown": {},
    "overall": {}
}
pf = open("atp_data/atp_players.csv") 
id_to_name_players = OrderedDict((row[0], row) for row in csv.reader(pf))

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(elo_dict, winner, loser, surface, match_date):
    winner_rating = elo_dict.get(winner, default_elo)
    loser_rating = elo_dict.get(loser, default_elo)
    
    expected_winner = expected_score(winner_rating, loser_rating)
    expected_loser = 1 - expected_winner
    
    new_winner_rating = winner_rating + k_factor * (1 - expected_winner)
    new_loser_rating = loser_rating + k_factor * (0 - expected_loser)
    
    elo_dict[winner] = new_winner_rating
    elo_dict[loser] = new_loser_rating

    if winner not in elo_timeseries[surface]:
        elo_timeseries[surface][winner] = []
    if loser not in elo_timeseries[surface]:
        elo_timeseries[surface][loser] = []

    elo_timeseries[surface][winner].append({"date": match_date, "elo": new_winner_rating})
    elo_timeseries[surface][loser].append({"date": match_date, "elo": new_loser_rating})

def load_data(years, data_dir):
    dfs = []
    for year in years:
        file_path = os.path.join(data_dir, f"atp_matches_{year}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
            dfs.append(df)
        else:
            print(f"File not found: {file_path}")
    return pd.concat(dfs, ignore_index=True)

years = list(range(2020, 2024))
data_dir = "atp_data"
df = load_data(years, data_dir)
df['surface'] = df['surface'].fillna('unknown')

for _, row in df.iterrows():
    surface = row['surface'].lower()
    winner_id = row['winner_id']
    loser_id = row['loser_id']
    match_date = row['tourney_date']

    update_elo(surface_elo[surface], winner_id, loser_id, surface, match_date)
    update_elo(surface_elo['overall'], winner_id, loser_id, "overall", match_date)

elo_dataframes = {}
for surface, players in elo_timeseries.items():
    timeseries_data = []
    for player, matches in players.items():
        for match in matches:
            timeseries_data.append({"player_id": player, "date": match["date"], "elo": match["elo"]})
    elo_dataframes[surface] = pd.DataFrame(timeseries_data)

st.title("Tennis Player Elo Ratings Over Time")

player_ids = list(set(df['winner_id']).union(set(df['loser_id'])))
player_name_to_id = {tuple(id_to_name_players.get(str(player_id))[1:3]): player_id for player_id in player_ids}

player1 = st.selectbox("Select Player 1 ID:", player_name_to_id.keys(), key="player1", format_func=lambda x: ' '.join(x))
player2 = st.selectbox("Select Player 2 ID:", player_name_to_id.keys(), key="player2", format_func=lambda x: ' '.join(x))

if player1 and player2:
    st.write("### Elo Ratings Time Series")
    
    for surface in ["overall", "hard", "clay", "grass"]:
        player1_data = elo_dataframes[surface][elo_dataframes[surface]['player_id'] == player_name_to_id[player1]]
        player2_data = elo_dataframes[surface][elo_dataframes[surface]['player_id'] == player_name_to_id[player2]]

        player1_data = player1_data.sort_values(by="date") if not player1_data.empty else player1_data
        player2_data = player2_data.sort_values(by="date") if not player2_data.empty else player2_data

        if player1_data.empty and player2_data.empty:
            st.write(f"No data available for {surface.capitalize()} Court Elo Ratings.")
            continue

        fig = go.Figure()

        if not player1_data.empty:
            fig.add_trace(go.Scatter(
                x=player1_data['date'],
                y=player1_data['elo'],
                mode='lines+markers',
                name=f"{' '.join(player1)}"
            ))

        if not player2_data.empty:
            fig.add_trace(go.Scatter(
                x=player2_data['date'],
                y=player2_data['elo'],
                mode='lines+markers',
                name=f"{' '.join(player2)}"
            ))

        fig.update_layout(
            title=f"{surface.capitalize()} Court Elo Ratings Over Time",
            xaxis_title="Date",
            yaxis_title="Elo Rating",
            legend_title="Players",
            template="plotly_white"
        )

        st.plotly_chart(fig)
