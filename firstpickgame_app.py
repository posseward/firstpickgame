import pandas as pd

#from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
#from surprise.model_selection import cross_validate

#from surprise import SVD

from collections import defaultdict

#from surprise.model_selection import GridSearchCV

#from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
#from surprise import SVD, SVDpp, NMF
#from surprise import SlopeOne, CoClustering
#from surprise import BaselineOnly

#import plotly.express as px
import plotly.graph_objects as go

#from jupyter_dash import JupyterDash
import dash
from dash import Dash, dcc, html, Input, Output, dash_table

from PIL import Image
import requests
from io import BytesIO

import dash_bootstrap_components as dbc

import joblib

from helper_functions import *


#%%
def get_top_x_recs(trainset, username, x_recs, players_min, players_max, time_min, time_max, complexity_min, complexity_max, fillValue):
    target_username = username

    targetUser = trainset.to_inner_uid(target_username)  # inner_id of the target user

    anti_testset_user = []

    user_item_ratings = trainset.ur[targetUser]
    user_items = [item for (item, _) in (user_item_ratings)]
    ratings = trainset.all_ratings()

    for iid in trainset.all_items():
        if (iid not in user_items):
            anti_testset_user.append((trainset.to_raw_uid(targetUser), trainset.to_raw_iid(iid), fillValue))

    predictions = algo.test(anti_testset_user)

    pred = pd.DataFrame(predictions)

    pred.sort_values(by=['est'], inplace=True, ascending=False)

    pred = pred.rename(columns={"uid": "user", "iid": "name", "r_ui": "dataset_average", "est": "predicted_rating"})

    pred = pred.drop(columns=["details"])

    pred = pred.merge(features_df, left_on="name", right_on="name")

    pred = pred[pred['players_max'] <= players_max]

    pred = pred[pred['players_max'] >= players_min]

    pred = pred[pred['playingtime'] >= time_min]

    pred = pred[pred['playingtime'] <= time_max]

    pred = pred[pred['complexity_average'] >= complexity_min]

    pred = pred[pred['complexity_average'] <= complexity_max]

    pred = pred.round({'predicted_rating': 1, 'rating_average': 1})

    top_x = pred[:x_recs]

    return top_x


#%%

#get game info
#folder = r"C:\Users\peter\Desktop\tdi_capstone"

#folder = r"assets"
#game_info_df = pd.read_csv(folder + "\\bgg_game_info.csv", index_col = 0)

folder = "assets"
#game_info_df = pd.read_csv(folder + "/bgg_game_info.csv", index_col = 0)

game_info_df = pd.read_pickle(folder + "/bgg_game_info.pkl")
#get ratings and comments
#folder = r"C:\Users\peter\Desktop\tdi_capstone\bgg_api_data"

#read in master game csv
#ratings_df = pd.read_csv(folder + "\\bgg_all_games.csv", index_col = 0)
#ratings_df = pd.read_csv(folder + "/bgg_all_games.csv", index_col = 0)

ratings_df = pd.read_pickle(folder + "/bgg_all_games.pkl")
#ratings_df = ratings_df.reset_index(drop=True)

#ratings_df = ratings_df.merge(game_info_df[['bgg_id', 'name']], left_on='Bggid', right_on='bgg_id')

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 10))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(ratings_df[['Username', 'name', 'Rating']], reader)

# First train an SVD algorithm on the movielens dataset.
data = data
trainset = data.build_full_trainset()

#%%

filename = folder + '\\recommender_model.sav'

algo = joblib.load(filename)


#algo =  NormalPredictor() #SVD()  
#algo.fit(trainset)


user_df = pd.DataFrame(ratings_df['Username'].unique())
user_df = user_df.rename(columns={0: "user"})

fillValue =  trainset.global_mean #None

features_of_interest = ['name','bgg_id' ,'players_min', 'players_max', 'playingtime','rating_average' ,'complexity_average','boardgamecategory','boardgamemechanic', 'thumbnail']

features_df = game_info_df[features_of_interest]

#MAKE DASHBOARD
#%%


app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])

#app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


controls = dbc.Card([

    html.H4("First Pick Game: A Board Game Recommender System"),

    dbc.Card([

        dbc.Label(["User for game recommendations"]),

        html.Div([
            dcc.Dropdown(user_df.user[:500], user_df.user[0], id='pandas-dropdown-2'),
            #     html.Div(id='pandas-output-container-2')
        ]),
    ], body=True, outline=True),

    dbc.Card([

        dbc.Label(["Number of players in game"]),

        dcc.RangeSlider(min=1, max=10, step=1, value=[1, 10], id='my-range-slider'),
    ], body=True, outline=True),

    dbc.Card([

        dbc.Label(["Average game time"]),

        dcc.RangeSlider(min=0, max=5,

                        marks={
                            0: {'label': '0 min'},
                            1: {'label': '30 min'},
                            2: {'label': '1hr'},
                            3: {'label': '2hr'},
                            4: {'label': '4hr'},
                            5: {'label': 'Unlimted'}
                        },

                        step=1, value=[0, 5], id='my-range-slider2'),
    ], body=True, outline=True),

    dbc.Card([
        dbc.Label(["Game complexity"]),

        dcc.RangeSlider(min=0, max=5, step=1, value=[0, 5], id='my-range-slider3'),
    ], body=True, outline=True),

], body=True, outline=True, style={'backgroundColor': 'white'})

pictures = dbc.Card([

    dbc.Label(["Top Five Recommended Games"]),
    dbc.Row([

        dbc.Col([
            dbc.Label(id="Game1"),
            dbc.Card([(html.Img(id="image1", width="162", height="162"))])], width=1.5),

        dbc.Col([
            dbc.Label(id="Game2"),
            dbc.Card([(html.Img(id="image2", width="162", height="162"))])], width=1.5),

        dbc.Col([
            dbc.Label(id="Game3"),
            dbc.Card([(html.Img(id="image3", width="162", height="162"))])], width=1.5),

        dbc.Col([
            dbc.Label(id="Game4"),
            dbc.Card([(html.Img(id="image4", width="162", height="162"))])], width=1.5),

        dbc.Col([
            dbc.Label(id="Game5"),
            dbc.Card([(html.Img(id="image5", width="162", height="162"))])], width=1.5),

    ])
], body=True, outline=False, style={'backgroundColor': 'white'})

table = dbc.Card([

    dcc.Graph(id='indicator-graphic2'),

], body=True)

app.layout = dbc.Card(
    [

        #   dbc.Row(
        #       dbc.Col(navbar),
        #          no_gutters=True),

        # dbc.Row(
        #     [
        #         dbc.Col(html.H1("First pick game"), width=4),
        # dbc.Col(pictures, width = 8),
        #     ]),

        # html.Hr(),

        dbc.Row(
            [
                dbc.Col(controls, width=3),

                dbc.Col([

                    dbc.Col(dbc.Row(pictures), lg={"size": 12, "offset": 2}),

                    dbc.Label(["Game Info"]),

                    dbc.Row(table), ], width=8)

            ],

        ),

        # dcc.Graph(id='indicator-graphic'),

    ]

)


@app.callback(
    #   Output('pandas-output-container-2', 'children'),
    Output('image1', 'src'),
    Output('image2', 'src'),
    Output('image3', 'src'),
    Output('image4', 'src'),
    Output('image5', 'src'),
    Output('Game1', 'children'),
    Output('Game2', 'children'),
    Output('Game3', 'children'),
    Output('Game4', 'children'),
    Output('Game5', 'children'),
    # Output('indicator-graphic', 'figure'),
    Output('indicator-graphic2', 'figure'),
    Input('pandas-dropdown-2', 'value'),
    Input('my-range-slider', 'value'),
    Input('my-range-slider2', 'value'),
    Input('my-range-slider3', 'value'))
def update_graph(value, range_values, range_values2, range_values3):
    # convert the time slider
    range_values2 = [range_slider_transform(v) for v in range_values2]

    # get data frame based off slider values
    dff = get_top_x_recs(trainset, value, 5, range_values[0], range_values[1], range_values2[0],
                         range_values2[1], range_values3[0], range_values3[1], fillValue)

    # get df for app table, by filtering on the columns that are displayed
    df_table = dff[['name', 'predicted_rating', 'rating_average', 'boardgamecategory', 'boardgamemechanic',
                    'players_min', 'players_max', 'playingtime', 'complexity_average']]

    table_list = df_table.values.tolist()
    table_list.insert(0, df_table.columns.tolist())

    table = go.Figure(data=[go.Table(
        # header=dict(values=list(df_table.columns),
        #            fill_color='white',
        #            align='left'),
        cells=dict(values=table_list,
                   # df_table, #cells=dict(values=df_table.transpose().values.tolist(), # cells=dict(values=df_table.values.tolist(),
                   fill_color='white',
                   align='left'))
    ])

    table.update_layout(height=440, margin={"r": 0, "t": 0, "l": 0, "b": 0})

    # get image thumbnails
    img1 = get_image(dff, 0)
    img2 = get_image(dff, 1)
    img3 = get_image(dff, 2)
    img4 = get_image(dff, 3)
    img5 = get_image(dff, 4)

    # get hyperlinks for games to board game geek
    game1 = make_link(dff, 0)
    game2 = make_link(dff, 1)
    game3 = make_link(dff, 2)
    game4 = make_link(dff, 3)
    game5 = make_link(dff, 4)
    
    print('test')

    return img1, img2, img3, img4, img5, game1, game2, game3, game4, game5, table


if __name__ == '__main__':
    app.run_server(debug=True)
    # app.run_server(mode='inline')

