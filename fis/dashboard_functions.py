import dash
import dash_bootstrap_components as dbc
import webbrowser
import time

def run_where(place = 'local'):
    if place == 'local':
        app = dash.Dash(__name__, suppress_callback_exceptions=True)
    else:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    return app


