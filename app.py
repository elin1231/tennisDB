import streamlit as st
import plotly.graph_objects as go
from utility import load_player_data, load_match_data, calculate_elo, prepare_elo_dataframes

DATA_DIR = "atp_data"
YEARS = range(2020, 2024)

id_to_name_players = load_player_data(DATA_DIR)
df = load_match_data(YEARS, DATA_DIR)
surface_elo, elo_timeseries = calculate_elo(df)
elo_dataframes = prepare_elo_dataframes(elo_timeseries)

st.title("Tennis Player Elo Ratings Over Time")

player_ids = set(df['winner_id']).union(df['loser_id'])

latest_elos = {
    player_id: max(
        (elo_timeseries[surface][player_id][-1]['elo'] if player_id in elo_timeseries[surface] else 0)
        for surface in elo_timeseries
    )
    for player_id in player_ids
}

sorted_players = sorted(
    player_ids,
    key=lambda player_id: latest_elos.get(player_id, 0),
    reverse=True
)

player_name_to_id = {
    tuple(id_to_name_players.get(str(player_id))[1:3]): player_id
    for player_id in sorted_players
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

        if not player1_data.empty:
            fig.add_trace(go.Scatter(
                x=player1_data['date'],
                y=player1_data['elo'],
                mode='lines+markers',
                name=' '.join(player1),
                line=dict(width=2),
                marker=dict(size=6)
            ))
            fig.add_trace(go.Scatter(
                x=[player1_data['date'].iloc[-1]],
                y=[player1_data['elo'].iloc[-1]],
                mode='markers+text',
                name=f"End: {' '.join(player1)}",
                marker=dict(size=10, symbol='circle'),
                text=[f"{player1_data['elo'].iloc[-1]:.0f}"],
                textposition="top center"
            ))

        if not player2_data.empty:
            fig.add_trace(go.Scatter(
                x=player2_data['date'],
                y=player2_data['elo'],
                mode='lines+markers',
                name=' '.join(player2),
                line=dict(width=2),
                marker=dict(size=6)
            ))
            fig.add_trace(go.Scatter(
                x=[player2_data['date'].iloc[-1]],
                y=[player2_data['elo'].iloc[-1]],
                mode='markers+text',
                name=f"End: {' '.join(player2)}",
                marker=dict(size=10, symbol='circle'),
                text=[f"{player2_data['elo'].iloc[-1]:.0f}"],
                textposition="top center"
            ))

        fig.update_layout(
            title=f"{surface.capitalize()} Court Elo Ratings Over Time",
            xaxis_title="Date",
            yaxis_title="Elo Rating",
            legend_title="Players",
            template="plotly_white"
        )
        st.plotly_chart(fig)
