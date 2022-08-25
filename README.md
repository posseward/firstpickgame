# firstpickgame
A Board Game Recommendation Engine

First Pick Game is designed to be a recommendation system for board games.  The recommender system is available as a dashboard. The user can use interactive sliders to filter outputs based on desired board game parameters, such as game length, game complexity, and number of players. Data is downloaded from the website Board Game Geek, using their website API to download many XML files containing game and review information. The XML files are processed with Pandas and the recommendation system is created with the Surprise library. The dashboard is then generated using the Plotly library Dash.
