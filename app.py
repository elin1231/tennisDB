import streamlit as st
import plotly.graph_objects as go
from utility import (
    load_player_data,
    load_match_data,
    calculate_elo,
    prepare_elo_dataframes,
    head_to_head_wins
)

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

def get_player_stats(player_id):
    matches_played = len(df[(df['winner_id'] == player_id) | (df['loser_id'] == player_id)])
    wins = len(df[df['winner_id'] == player_id])
    losses = matches_played - wins
    win_percentage = (wins / matches_played) * 100 if matches_played > 0 else 0
    return {
        "Matches Played": matches_played,
        "Wins": wins,
        "Losses": losses,
        "Win Percentage": f"{win_percentage:.2f}%"
    }

if player1 and player2:
    player1_id = player_name_to_id[player1]
    player2_id = player_name_to_id[player2]
    player1_stats = get_player_stats(player1_id)
    player2_stats = get_player_stats(player2_id)
    h2h = head_to_head_wins(df, player1_id, player2_id)

    st.markdown("---")
    st.write("### Player Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(' '.join(player1))
        st.write(f"**Wins vs {' '.join(player2)}:** {h2h['player1_wins']}")
        for key, value in player1_stats.items():
            st.write(f"**{key}:** {value}")

    with col2:
        st.subheader(' '.join(player2))
        st.write(f"**Wins vs {' '.join(player1)}:** {h2h['player2_wins']}")
        for key, value in player2_stats.items():
            st.write(f"**{key}:** {value}")
        

    st.markdown("---")
    st.write("### Elo Ratings Time Series")

    for surface, data in elo_dataframes.items():
        player1_data = data[data['player_id'] == player1_id].sort_values(by="date")
        player2_data = data[data['player_id'] == player2_id].sort_values(by="date")

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
