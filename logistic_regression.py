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

def preprocess_data(data):
    numerical_columns = ['WRank', 'LRank', 'WPts', 'LPts', 'B365W', 'B365L']
    for col in numerical_columns:
        data[col] = data[col].fillna(data[col].median())

    categorical_columns = ['Surface', 'Court']
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])

    winner_data = data.copy()
    loser_data = data.copy()

    winner_data['Winner_Binary'] = 1
    loser_data['Winner_Binary'] = 0

    loser_data['Rank_Diff'] = loser_data['LRank'] - loser_data['WRank']
    loser_data['Pts_Diff'] = loser_data['LPts'] - loser_data['WPts']
    loser_data['Win_Prob'] = 1 / loser_data['B365L'].replace(0, np.nan).fillna(1)
    loser_data['Lose_Prob'] = 1 / loser_data['B365W'].replace(0, np.nan).fillna(1)

    winner_data['Rank_Diff'] = winner_data['WRank'] - winner_data['LRank']
    winner_data['Pts_Diff'] = winner_data['WPts'] - winner_data['LPts']
    winner_data['Win_Prob'] = 1 / winner_data['B365W'].replace(0, np.nan).fillna(1)
    winner_data['Lose_Prob'] = 1 / winner_data['B365L'].replace(0, np.nan).fillna(1)

    combined_data = pd.concat([winner_data, loser_data], ignore_index=True)
    combined_data = combined_data.dropna()
    return combined_data

def feature_engineering(data):
    features = [
        'Rank_Diff', 'Pts_Diff', 'Win_Prob', 'Lose_Prob', 'Best of',
        'Surface', 'Court'
    ]

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        max_date = data['Date'].max()
        data['Time_Weight'] = data['Date'].apply(lambda x: np.exp(-(max_date - x).days / 365))
    else:
        data['Time_Weight'] = 1

    X = data[features]
    y = data['Winner_Binary']

    X = pd.get_dummies(X, columns=['Surface', 'Court'], drop_first=True)

    for col in X.columns:
        X[col] = X[col] * data['Time_Weight']

    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    return model, X_test, y_test

def predict_outcomes(model, X_test, y_test):
    predictions = model.predict(X_test)
    prediction_probs = model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        'Predicted': predictions,
        'Actual': y_test,
        'Win_Probability': prediction_probs
    })
    return results