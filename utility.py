import os
import pandas as pd
from collections import OrderedDict
import csv
import requests

DEFAULT_ELO = 1500
K_FACTOR = 32

def retrieve_betting_data(years) -> list:
    base_url_xlsx = "http://tennis-data.co.uk/{}/{}.xlsx"
    data_dir = "betting_data"
    os.makedirs(data_dir, exist_ok=True)
    
    for year in years:
        url_xlsx = base_url_xlsx.format(year, year)
        response_xlsx = requests.get(url_xlsx)
        if response_xlsx.status_code == 200:
            filename_xlsx = os.path.join(data_dir, f"{year}.xlsx")
            with open(filename_xlsx, "wb") as file:
                file.write(response_xlsx.content)
            print(f"Downloaded {filename_xlsx}")
        else:
            print(f"Failed to download {url_xlsx}")

def load_player_data(data_dir):
    with open(os.path.join(data_dir, "atp_players.csv")) as pf:
        return OrderedDict((row[0], row) for row in csv.reader(pf))
    
def head_to_head_wins(df, player1_id, player2_id):
    player1_wins = len(df[(df['winner_id'] == player1_id) & (df['loser_id'] == player2_id)])
    player2_wins = len(df[(df['winner_id'] == player2_id) & (df['loser_id'] == player1_id)])
    return {
        "player1_wins": player1_wins,
        "player2_wins": player2_wins
    }

def load_match_data(years, data_dir):
    dfs = []
    for year in years:
        file_path = os.path.join(data_dir, f"atp_matches_{year}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(elo_dict, winner, loser, surface, match_date, elo_timeseries):
    winner_rating = elo_dict.get(winner, DEFAULT_ELO)
    loser_rating = elo_dict.get(loser, DEFAULT_ELO)
    expected_winner = expected_score(winner_rating, loser_rating)
    expected_loser = 1 - expected_winner
    elo_dict[winner] = winner_rating + K_FACTOR * (1 - expected_winner)
    elo_dict[loser] = loser_rating - K_FACTOR * expected_loser
    for player, rating in [(winner, elo_dict[winner]), (loser, elo_dict[loser])]:
        elo_timeseries[surface].setdefault(player, []).append({"date": match_date, "elo": rating})

def calculate_elo(df):
    surface_elo = {surface: {} for surface in ["hard", "clay", "grass", "unknown", "overall"]}
    elo_timeseries = {surface: {} for surface in ["hard", "clay", "grass", "unknown", "overall"]}
    df['surface'] = df['surface'].fillna('unknown').str.lower()
    for _, row in df.iterrows():
        update_elo(surface_elo[row['surface']], row['winner_id'], row['loser_id'], row['surface'], row['tourney_date'], elo_timeseries)
        update_elo(surface_elo['overall'], row['winner_id'], row['loser_id'], "overall", row['tourney_date'], elo_timeseries)
    return surface_elo, elo_timeseries

def prepare_elo_dataframes(elo_timeseries):
    return {
        surface: pd.DataFrame(
            [{"player_id": player, **match} for player, matches in players.items() for match in matches]
        )
        for surface, players in elo_timeseries.items()
    }
