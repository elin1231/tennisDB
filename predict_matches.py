import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

def prepare_data(matches_df, players_df):
    # Merge DOB for winner and loser
    matches_df = matches_df.merge(
        players_df[["player_id", "dob"]], 
        left_on="winner_id", 
        right_on="player_id", 
        how="left"
    ).rename(columns={"dob": "Winner_dob"})
    
    matches_df = matches_df.merge(
        players_df[["player_id", "dob"]], 
        left_on="loser_id", 
        right_on="player_id", 
        how="left"
    ).rename(columns={"dob": "Loser_dob"})
    
    # Calculate ages
    matches_df["Date"] = pd.to_datetime(matches_df["Date"])
    matches_df["Winner_age"] = (matches_df["Date"] - pd.to_datetime(matches_df["Winner_dob"])).dt.days / 365.25
    matches_df["Loser_age"] = (matches_df["Date"] - pd.to_datetime(matches_df["Loser_dob"])).dt.days / 365.25
    
    # Determine favorite (player with lower rank)
    matches_df["favorite_rank"] = matches_df[["WRank", "LRank"]].min(axis=1)
    matches_df["favorite_age"] = matches_df.apply(
        lambda row: row["Winner_age"] if row["WRank"] < row["LRank"] else row["Loser_age"], 
        axis=1
    )
    
    # Calculate Info variable (from bookmakers' odds for the favorite)
    matches_df["favorite_odd"] = matches_df.apply(
        lambda row: row["B365W"] if row["WRank"] < row["LRank"] else row["B365L"], 
        axis=1
    )
    matches_df["Info"] = matches_df["favorite_odd"].apply(lambda x: x if x > 2 else 0)
    
    # Define target: 1 if favorite wins, 0 otherwise
    matches_df["target"] = matches_df.apply(
        lambda row: 1 if (row["WRank"] < row["LRank"]) else 0, 
        axis=1
    )
    
    # Select features and drop missing values
    features = ["favorite_age", "Info", "favorite_rank"]
    df_clean = matches_df[features + ["target"]].dropna()
    
    return df_clean

def train_logistic_model(df):
    # Split data into features (X) and target (y)
    X = df[["favorite_age", "Info", "favorite_rank"]]
    y = df["target"]
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler

matches_df = pd.read_csv("betting_data_w_id/2023.csv")
players_df = pd.read_csv('atp_data/atp_players.csv')

df_clean = prepare_data(matches_df, players_df)

model, scaler = train_logistic_model(df_clean)