import requests
from PIL import Image
from io import BytesIO
from dash import Dash, dcc, html, Input, Output, dash_table

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n





def get_image(dff, x_iloc):
    image_url = dff['thumbnail'].iloc[x_iloc]

    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    return img

def range_slider_transform(value):
    if value == 0:
        return 0
    if value < 5:
        return 2 ** (value -1) * 30
    if value ==5:
        return 99999


def make_link(dff, loc):
    web_address = 'https://boardgamegeek.com/boardgame/'

    game_name = dff['name'].iloc[loc]
    if len(game_name) > 20:
        game_name = game_name[:17] + '...'
    game_web = web_address + str(dff['bgg_id'].iloc[loc])

    return html.A(game_name, href=game_web)