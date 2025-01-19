import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from collections import OrderedDict
import csv

# Initialize Elo dictionaries and default settings
surface_elo = {
    "hard": {},
    "clay": {},
    "grass": {},
    "unknown": {},
    "overall": {}
}
default_elo = 1500
k_factor = 32

# Elo timeseries storage
elo_timeseries = {
    "hard": {},
    "clay": {},
    "grass": {},
    "unknown": {},
    "overall": {}
}
pf = open("atp_data/atp_players.csv") 
id_to_name_players = OrderedDict((row[0], row) for row in csv.reader(pf))
# Define Elo calculation functions
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

    # Update timeseries
    if winner not in elo_timeseries[surface]:
        elo_timeseries[surface][winner] = []
    if loser not in elo_timeseries[surface]:
        elo_timeseries[surface][loser] = []

    elo_timeseries[surface][winner].append({"date": match_date, "elo": new_winner_rating})
    elo_timeseries[surface][loser].append({"date": match_date, "elo": new_loser_rating})

# Load data
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

# Data directory and years
years = list(range(2020, 2024))
data_dir = "atp_data"
df = load_data(years, data_dir)
df['surface'] = df['surface'].fillna('unknown')

# Process matches and update Elo
for _, row in df.iterrows():
    surface = row['surface'].lower()
    winner_id = row['winner_id']
    loser_id = row['loser_id']
    match_date = row['tourney_date']  # Assuming 'tourney_date' is in the dataset

    update_elo(surface_elo[surface], winner_id, loser_id, surface, match_date)
    update_elo(surface_elo['overall'], winner_id, loser_id, "overall", match_date)

# Convert Elo timeseries into DataFrames
elo_dataframes = {}
for surface, players in elo_timeseries.items():
    timeseries_data = []
    for player, matches in players.items():
        for match in matches:
            timeseries_data.append({"player_id": player, "date": match["date"], "elo": match["elo"]})
    elo_dataframes[surface] = pd.DataFrame(timeseries_data)

# Streamlit application
st.title("Tennis Player Elo Ratings Over Time")

# Player selection
player_ids = list(set(df['winner_id']).union(set(df['loser_id'])))
player_name_to_id = {tuple(id_to_name_players.get(str(player_id))[1:3]): player_id for player_id in player_ids}

player1 = st.selectbox("Select Player 1 ID:", player_name_to_id.keys(), key="player1",format_func=lambda x: ' '.join(x))
player2 = st.selectbox("Select Player 2 ID:", player_name_to_id.keys(), key="player2",format_func=lambda x: ' '.join(x))

# Elo timeseries visualization with enhanced plots
if player1 and player2:
    st.write("### Elo Ratings Time Series")
    
    for surface in ["overall", "hard", "clay", "grass"]:
        # Filter data for selected players
        player1_data = elo_dataframes[surface][elo_dataframes[surface]['player_id'] == player_name_to_id[player1]]
        player2_data = elo_dataframes[surface][elo_dataframes[surface]['player_id'] == player_name_to_id[player2]]

        # Sort by date
        player1_data = player1_data.sort_values(by="date") if not player1_data.empty else player1_data
        player2_data = player2_data.sort_values(by="date") if not player2_data.empty else player2_data

        if player1_data.empty and player2_data.empty:
            st.write(f"No data available for {surface.capitalize()} Court Elo Ratings.")
            continue

        # Set up Matplotlib style
        plt.style.use('Solarize_Light2')  # Use a built-in style for clean visuals

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        if not player1_data.empty:
            ax.plot(
                player1_data['date'],
                player1_data['elo'],
                label=f"{' '.join(player1)}",
                marker='o',
                markersize=6,
                linewidth=2,
                alpha=0.8,
                color='blue'
            )
        if not player2_data.empty:
            ax.plot(
                player2_data['date'],
                player2_data['elo'],
                label=f"{' '.join(player2)}",
                marker='s',
                markersize=6,
                linewidth=2,
                alpha=0.8,
                color='green'
            )

        # Customize the plot
        ax.set_title(f"{surface.capitalize()} Court Elo Ratings Over Time", fontsize=16, weight='bold', color='darkblue')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Elo Rating", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True)

        # Render the plot in Streamlit
        st.pyplot(fig)
