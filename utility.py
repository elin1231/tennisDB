import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

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
    

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(elo_dict, winner, loser, default_elo=1500, k_factor=32):
    winner_rating = elo_dict.get(winner, default_elo)
    loser_rating = elo_dict.get(loser, default_elo)
    
    expected_winner = expected_score(winner_rating, loser_rating)
    expected_loser = 1 - expected_winner
    
    elo_dict[winner] = winner_rating + k_factor * (1 - expected_winner)
    elo_dict[loser] = loser_rating + k_factor * (0 - expected_loser)

def load_data(years, data_dir):
    dfs = []
    for year in years:
        file_path = os.path.join(data_dir, f"atp_matches_{year}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"File not found: {file_path}")
    return pd.concat(dfs, ignore_index=True)