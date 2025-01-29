import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import csv
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from rapidfuzz import process

DEFAULT_ELO = 1500
K_FACTOR = 32

def retrieve_betting_data(years) -> list:
    base_url_xlsx = "http://tennis-data.co.uk/{}/{}.xlsx"
    data_dir = "betting_data"
    os.makedirs(data_dir, exist_ok=True)

    for year in years:
        filename_xlsx = os.path.join(data_dir, f"{year}.xlsx")

        # Check if the file already exists
        if os.path.exists(filename_xlsx):
            print(f"File already exists, skipping: {filename_xlsx}")
            continue

        url_xlsx = base_url_xlsx.format(year, year)
        response_xlsx = requests.get(url_xlsx)

        if response_xlsx.status_code == 200:
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
    surface_elo = {surface: {} for surface in ["overall","hard", "clay", "grass", "unknown" ]}
    elo_timeseries = {surface: {} for surface in ["overall", "hard", "clay", "grass", "unknown"]}
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

def calculate_serve_metrics(df, player_id):
    player_matches = df[(df['winner_id'] == player_id) | (df['loser_id'] == player_id)]
    
    total_matches = len(player_matches)
    if total_matches == 0:
        return {
            "First Serve Percentage": "N/A",
            "First Serve Points Won (%)": "N/A",
            "Second Serve Points Won (%)": "N/A",
            "Aces Per Match": "N/A",
            "Double Faults Per Match": "N/A"
        }
    
    first_serves = (
        player_matches['w_1stIn'].where(player_matches['winner_id'] == player_id, player_matches['l_1stIn'])
    ).sum()
    serve_points = (
        player_matches['w_svpt'].where(player_matches['winner_id'] == player_id, player_matches['l_svpt'])
    ).sum()
    first_serve_wins = (
        player_matches['w_1stWon'].where(player_matches['winner_id'] == player_id, player_matches['l_1stWon'])
    ).sum()
    second_serve_wins = (
        player_matches['w_2ndWon'].where(player_matches['winner_id'] == player_id, player_matches['l_2ndWon'])
    ).sum()
    aces = (
        player_matches['w_ace'].where(player_matches['winner_id'] == player_id, player_matches['l_ace'])
    ).sum()
    double_faults = (
        player_matches['w_df'].where(player_matches['winner_id'] == player_id, player_matches['l_df'])
    ).sum()

    first_serve_percentage = (first_serves / serve_points * 100) if serve_points > 0 else 0
    first_serve_points_won = (first_serve_wins / first_serves * 100) if first_serves > 0 else 0
    second_serve_points_won = (second_serve_wins / (serve_points - first_serves) * 100) if (serve_points - first_serves) > 0 else 0
    aces_per_match = aces / total_matches
    double_faults_per_match = double_faults / total_matches

    return {
        "First Serve Percentage": f"{first_serve_percentage:.2f}%",
        "First Serve Points Won (%)": f"{first_serve_points_won:.2f}%",
        "Second Serve Points Won (%)": f"{second_serve_points_won:.2f}%",
        "Aces Per Match": f"{aces_per_match:.2f}",
        "Double Faults Per Match": f"{double_faults_per_match:.2f}"
    }

def load_data_for_years(years, base_dir="betting_data"):
    """Load and combine data for the specified years."""
    all_data = []
    for year in years:
        file_path = os.path.join(base_dir, f"{year}.xlsx")
        if os.path.exists(file_path):
            yearly_data = pd.read_excel(file_path)
            all_data.append(yearly_data)
        else:
            print(f"File not found: {file_path}")

    combined_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    return combined_data

def calculate_betting_strategies(file_path, initial_cash=1000):
    data = pd.ExcelFile(file_path)
    df = data.parse(data.sheet_names[0]).sort_values(by='Date').reset_index(drop=True)

    bet_size = 10
    cash_random, cash_higher_odds, cash_higher_rank = initial_cash, initial_cash, initial_cash

    random_returns, higher_odds_returns, higher_rank_returns = [], [], []

    np.random.seed(42)
    for _, row in df.iterrows():
        if cash_random > 0:
            random_outcome = np.random.choice(['Winner', 'Loser'])
            random_return = bet_size * (row['AvgW'] - 1) if random_outcome == 'Winner' else -bet_size
            cash_random += random_return
            if cash_random < 0:
                cash_random = 0
        else:
            random_return = 0
        random_returns.append(random_return)

        if cash_higher_odds > 0:
            higher_odds_bet = 'Winner' if row['AvgW'] > row['AvgL'] else 'Loser'
            higher_odds_return = bet_size * (row['AvgW'] - 1) if higher_odds_bet == 'Winner' else -bet_size
            cash_higher_odds += higher_odds_return
            if cash_higher_odds < 0:
                cash_higher_odds = 0
        else:
            higher_odds_return = 0
        higher_odds_returns.append(higher_odds_return)

        if cash_higher_rank > 0:
            higher_rank_bet = (
                'Winner' if (row['WRank'] < row['LRank']) or pd.isna(row['LRank']) else
                'Loser' if (row['LRank'] < row['WRank']) or pd.isna(row['WRank']) else 'No Bet'
            )
            higher_rank_return = (
                bet_size * (row['AvgW'] - 1) if higher_rank_bet == 'Winner' else
                -bet_size if higher_rank_bet == 'Loser' else 0
            )
            cash_higher_rank += higher_rank_return
            if cash_higher_rank < 0:
                cash_higher_rank = 0
        else:
            higher_rank_return = 0
        higher_rank_returns.append(higher_rank_return)

    df['Cumulative_Random'] = pd.Series(random_returns).cumsum()
    df['Cumulative_HigherOdds'] = pd.Series(higher_odds_returns).cumsum()
    df['Cumulative_HigherRank'] = pd.Series(higher_rank_returns).cumsum()

    return df[['Date', 'Cumulative_Random', 'Cumulative_HigherOdds', 'Cumulative_HigherRank']]

def visualize_logistic_regression(model, X_test, y_test, results):
    plt.figure(figsize=(8, 6))
    plt.scatter(results['Actual'], results['Win_Probability'], alpha=0.6, c=results['Predicted'], cmap='coolwarm', edgecolor='k')
    plt.title("Predicted Win Probability vs Actual Outcome")
    plt.xlabel("Actual Outcome (0 = Loss, 1 = Win)")
    plt.ylabel("Predicted Win Probability")
    plt.colorbar(label="Predicted Outcome")
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.show()

# Function to add player_id tag to tennis-data.co.uk data
def get_betting_data_player_names():
    player_names = set()
    for file in os.listdir("betting_data"):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join("betting_data", file))
            player_names.update(df['Winner'].unique())
            player_names.update(df['Loser'].unique())
    
    print(f"Found {len(player_names)} player names in tennis-data.co.uk data")
    return player_names

def compute_favorite_and_underdog(df):
    def get_favorite(row):
        if row['B365W'] < row['B365L']:
            return row['Winner']
        else:
            return row['Loser']

    def get_underdog(row):
        if row['B365W'] < row['B365L']:
            return row['Loser']
        else:
            return row['Winner']

    # Create columns for Favorite and Underdog
    df['Favorite'] = df.apply(get_favorite, axis=1)
    df['Underdog'] = df.apply(get_underdog, axis=1)
    return df