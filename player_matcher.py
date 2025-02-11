#!/usr/bin/env python3

import re
import pandas as pd
from collections import defaultdict
from unidecode import unidecode
import os

def remove_accents_and_lower(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unidecode(s).lower().strip()


def parse_betting_name(name: str):
    parts = name.split()
    if len(parts) < 2:
        return "", ""
    initials_part_raw = parts[-1]
    last_name_part_raw = " ".join(parts[:-1])
    initials_part_raw = re.sub(r"\.", "", initials_part_raw)
    last_name_part = remove_accents_and_lower(last_name_part_raw)
    initials_part = remove_accents_and_lower(initials_part_raw)
    return last_name_part, initials_part


def compute_first_name_initials(name_first: str) -> str:
    if not isinstance(name_first, str):
        return ""
    name_first_norm = remove_accents_and_lower(name_first).replace("-", " ")
    tokens = name_first_norm.split()
    return "".join(token[0] for token in tokens if token)


def build_lookup_dict(df_players: pd.DataFrame):
    df_players["name_last"] = df_players["name_last"].astype(str).fillna("")
    df_players["name_first"] = df_players["name_first"].astype(str).fillna("")
    df_players["_last_name_norm"] = (
        df_players["name_last"]
        .apply(remove_accents_and_lower)
        .apply(lambda x: x.replace("-", " "))
    )
    df_players["_first_name_inits"] = df_players["name_first"].apply(compute_first_name_initials)
    lookup = defaultdict(list)
    for idx, row in df_players.iterrows():
        last_name_clean = row["_last_name_norm"].replace(" ", "")
        first_inits = row["_first_name_inits"]
        key = (last_name_clean, first_inits)
        lookup[key].append(row["player_id"])
    return lookup


def find_player_id_fast(betting_name: str, lookup_dict) -> int:
    last_name_part, initials_part = parse_betting_name(betting_name)
    if not last_name_part or not initials_part:
        return None
    key = (last_name_part.replace(" ", ""), initials_part)
    matches = lookup_dict.get(key, [])
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return matches[0]
    else:
        return None


def main():
    df_players = pd.read_csv("atp_data/atp_players.csv")
    betting_data_dir = "betting_data"

    for file_name in os.listdir(betting_data_dir):
        if file_name.endswith(".xlsx"):
            file_path = os.path.join(betting_data_dir, file_name)
            df_betting = pd.read_excel(file_path)
            lookup = build_lookup_dict(df_players)
            
            df_betting["winner_id"] = df_betting["Winner"].apply(lambda x: find_player_id_fast(x, lookup))
            df_betting["loser_id"] = df_betting["Loser"].apply(lambda x: find_player_id_fast(x, lookup))
            
            total_rows = len(df_betting)
            matched_winners = df_betting["winner_id"].notnull().sum()
            matched_losers = df_betting["loser_id"].notnull().sum()
            total_matches = matched_winners + matched_losers
            total_possible_matches = total_rows * 2  # Each row has a winner and a loser to match
            match_percentage = (total_matches / total_possible_matches) * 100
            
            print(f"File: {file_name}")
            print(f"Match percentage: {match_percentage:.2f}%")
            
            output_file_path = os.path.join("betting_data_w_id", file_name.replace(".xlsx", ".csv"))
            df_betting.to_csv(output_file_path, index=False)