#!/usr/bin/env python3
# -*- coding: utf-8 -*-

db_path = 'intrusiondet.db'
video_path = './captures/'
time_interval = 30 # how often the page reloads in seconds

intrusion_det_db_query = '''
    SELECT
    location as "Camera Zone",
    class_name as "Detection Class",
    datetime_utc as "Detection Time",
    notes as "Review Status",
    filename,
    id
    FROM detections d
    '''

import dash
from dash_extensions.enrich import Input, Output, Dash, State
from dash.exceptions import PreventUpdate
from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc
import webbrowser
import base64
import json
import os
import flask
import numpy as np
import pandas as pd
import sqlite3
import pytz
from datetime import datetime
####################################################################
# kpmg colors
kpmg_blue = 'rgb(0, 51, 141)'
light_blue = 'rgb(0, 145, 218)'
purple = 'rgb(109, 32, 119)'
medium_blue = 'rgb(0, 94, 184)'
green = 'rgb(0, 163, 161)'
yellow = 'rgb(234, 170, 0)'
light_green ='rgb(67, 176, 42)'
pink = 'rgb(198, 0, 126)'
dark_brown = 'rgb(117, 63, 25)'
light_brown = 'rgb(155, 100, 45)'
olive = 'rgb(157, 147, 117)'
beige = 'rgb(227, 188, 159)'
light_pink = 'rgb(227, 104, 119)'
dark_green = 'rgb(69, 117, 37)'
orange = 'rgb(222, 134, 38)'
bright_green = 'rgb(162, 234, 66)'
light_yellow = 'rgb(255, 218, 90)'
rose = 'rgb(226, 160, 197)'
red = 'rgb(188, 32, 75)'
tan = 'rgb(178, 162, 68)'
kpmg_color_palette = [kpmg_blue, light_blue, purple, medium_blue, green, yellow, light_green,
                      pink, dark_brown, light_brown, olive, beige, light_pink, dark_green,
                     orange, bright_green, light_yellow, rose, red, tan]
font_family = '"Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"'
####################################################################
# import kpmg logo
kpmg_logo = './assets/logos/kpmg-logo-png-transparent.png'
kpmg_encoded_image = base64.b64encode(open(kpmg_logo, 'rb').read())
####################################################################
server = flask.Flask(__name__)
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], server=server)

app.title = 'Physical Intrusion Review Board'

navbar = dbc.Navbar([
    html.A([
        html.Img(src='data:image/png;base64,{}'.format(kpmg_encoded_image.decode()),
               style={'height':'2em', 'padding-left':'1rem'}),
        ], id='reload-page', href='/', style={'margin-right':'1rem'}),
    html.Div([
        html.H1(
            'Physical Intrusion Review Board',
            style={
                'color':'white',
                'margin-bottom':'0rem',
                'width':'100%',
                'font-family':'KPMG',
                }
            ),
        ], className='ml-auto', style={'padding-right':'1rem'}),
    ], color=kpmg_blue, sticky='top')

page_content = html.Div([
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.Div(id='most-recent-alert'),
            dash_table.DataTable(
                id='cv-results-table',
                page_size=15,
                # filter_action='native',
                sort_action='native',
                row_selectable='single',
                # style cell styles whole table
                style_cell={'font-family':'Arial',
                            'font-size':'80%',
                            'textAlign':'left',
                            'height':'auto',
                            'whiteSpace':'normal',
                            },
                style_header={'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'state':'selected'}, 'backgroundColor':'inherit !important', 'border':'inherit !important'},
                    {'if': {'row_index':'odd'}, 'backgroundColor':'rgba(0, 0, 0, 0.03)'},
                    ],
                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                ) # end of table
            ], style={'padding-left':'2rem'}),
        dbc.Col([
            html.Div([
                html.Div(id='timer-display'),
                html.Video(
                    id='preview-video',
                    controls=True,
                    muted=True,
                    # src='/captures/yolo50_AAAAAAAAAAAA_capture_1674838207-1674838267_58_0_image_1280x720.mp4',
                    # id = 'movie_player',
                    #src = "https://www.youtube.com/watch?v=gPtn6hD7o8g",
                    # src = '/cv/cv-demo-1.mp4',
                    autoPlay=True,
                    className='vid-resize'
                    )
                ])
            ], style={'padding-right':'2rem'})
        ])
    ])

app.layout = html.Div([
    navbar,
    page_content,
    dcc.Location(id='url', refresh=False), # url management
    dcc.Interval(id='refresh-interval', n_intervals=0, interval=time_interval*1000), # refresh database
    dcc.Interval(id='display-interval', n_intervals=0, interval=1*1000), # countdown every second
    html.Div(id='write-to-database', style={'display':'None'}),
    ])
####################################################################

@app.callback(Output('timer-display', 'children'),
              Input('display-interval', 'n_intervals'))
def update_output(n):
    # database reloads every time_interval seconds
    #calculate remaining time in ms
    remaining = time_interval * 1000 - (n * 1000)

    # minute = (remaining // 60000)
    second = (remaining % 60000) // 1000
    millisecond = (remaining % 1000) // 10

    display = dbc.Alert([
        html.I(className="bi bi-info-circle-fill me-2"),
        'Checking for new and unreviewed intrusion candidates in {}.{} seconds...'.format(second, millisecond)
        ], color='info')

    return display

@app.callback(Output('display-interval', 'n_intervals'),
             Input('refresh-interval', 'n_intervals'))
def reset_display(refresh):
    return 0

@app.callback(Output('cv-results-table', 'data'),
              Output('cv-results-table', 'columns'),
              Output('cv-results-table', 'dropdown'),
              Output('cv-results-table', 'tooltip_header'),
              Output('cv-results-table', 'style_header_conditional'),
              Input('url', 'pathname'),
              Input('refresh-interval', 'n_intervals'))
def load_cv_table(path, refresh):
    if path:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(intrusion_det_db_query)
        column_names = [x[0] for x in cursor.description]
        extract = cursor.fetchall()
        connection.close()
        df = pd.DataFrame(extract, columns=column_names)

        # convert intrusion time to EST
        df['Detection Time'] = pd.to_datetime(df['Detection Time'])
        df['Detection Time'] = df['Detection Time'].dt.tz_localize('UTC')
        df['Detection Time'] = df['Detection Time'].dt.tz_convert('US/Eastern')

        # if null, set to requires review
        df.loc[(df['Review Status']=='nan') | (df['Review Status']==np.nan) | (df['Review Status']==''), 'Review Status'] = 'REQUIRES REVIEW'
        df['Review Status'] = df['Review Status'].fillna('REQUIRES REVIEW')
        df = df.sort_values(['Detection Time', 'Review Status'], ascending=False) # sort by most recent

        # columns is an element of the data table
        excluded_cols = ['filename', 'id', 'Review Status']
        columns = [{'name': i, 'id': i} for i in df.columns if i not in excluded_cols]
        columns = columns + [{'name': i, 'id': i, 'presentation': 'dropdown', 'editable':True} for i in ['Review Status']]

        table_data = df.to_dict('records') # required format for data table

        dropdown={
            'Review Status':
                {'options': [{'label':i,'value':i} for i in ['ALL CLEAR','REQUIRES REVIEW']], 'clearable':False}
            }

        tooltip_header = {
            'Review Status': 'Use the dropdowns in this column to track reviews of detections. Changes will be written to the database.',
            'Detection Class': 'This is the object most likely detected according to the algorithm.'
            }

         # Style headers with a dotted underline to indicate a tooltip
        style_header_conditional=[
            {'if': {'column_id': 'Review Status'},
                'textDecoration': 'underline',
                'textDecorationStyle': 'dotted',
            },
            {'if': {'column_id': 'Detection Class'},
                'textDecoration': 'underline',
                'textDecorationStyle': 'dotted',
            },
            ]

        return table_data, columns, dropdown, tooltip_header, style_header_conditional
    raise PreventUpdate

@app.callback(Output('preview-video', 'src'),
              Input('cv-results-table', 'selected_rows'),
              State('cv-results-table', 'data'))
def show_video_preview(selected_rows, data):
    if selected_rows:
        df = pd.DataFrame(data)
        df = df.iloc[selected_rows[0]]
        filename = df['filename']
        src = video_path + filename
        return src
    raise PreventUpdate

@app.callback(Output('cv-results-table', 'selected_rows'),
              Input('cv-results-table', 'data'),
              State('cv-results-table', 'selected_rows'))
def update_selection(load, row):
    if not row:
        return [0]
    return row

@app.callback(Output('most-recent-alert', 'children'),
              Input('cv-results-table', 'data'))
def most_recent_detection(data):
    if data:
        df = pd.DataFrame(data)
        # get most recent unreviewed detection
        review = df[(df['Review Status']=='REQUIRES REVIEW') & (df['Detection Time']==df['Detection Time'].max())]
        if not review.empty:
            detection = review.iloc[0]
            detection_location = detection['Camera Zone']
            detection_time = pd.to_datetime(detection['Detection Time']).strftime('%Y-%m-%d %H:%M:%S')
            alert = dbc.Alert([
                html.I(className="bi bi-x-octagon-fill me-2"),
                'There\'s been unreviewed, unscheduled activity at {} at {} Eastern Time. '\
                'Please review detections below and follow-up accordingly.'.format(detection_location, detection_time),
                ], color="danger", dismissable=True)
            return alert
        if review.empty:
            alert = dbc.Alert([
                html.I(className="bi bi-check-circle-fill me-2"),
                'All intruder candidate activity has been reviewed as normal.'
                ], color="success", dismissable=True)
            return alert
    return None

@app.callback(Output('write-to-database', 'children'),
              Input('cv-results-table', 'data'))
def update_database(data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'cv-results-table' in changed_id and data:
        for attempt in range(10): # try a few times if there's an error
            print('Attempt #{}'.format(attempt))
            try:
                # compare to sql database
                connection = sqlite3.connect(db_path)
                cursor = connection.cursor()
                cursor.execute(intrusion_det_db_query)
                column_names = [x[0] for x in cursor.description]
                extract = cursor.fetchall()
                connection.close()
                df = pd.DataFrame(extract, columns=column_names)

                df2 = pd.DataFrame(data) # updated data

                comparison = df[['Review Status']].compare(df2[['Review Status']]) # compare for changes
                # iterate through rows of comparison and commit to database
                for idx, row in comparison.iterrows():

                    # first, get id of sql row based on index; it's its own column
                    row_id = df.iloc[idx]['id']

                    # get updated review status from table
                    new_status = row['Review Status']['other']

                    print('Attempting to update id {} with review status == {}'.format(row_id, new_status))

                    # update database
                    connection = sqlite3.connect(db_path)
                    cursor = connection.cursor()
                    update_query = '''
                        UPDATE detections
                        SET notes = \"{}\"
                        WHERE id = {}
                        '''.format(new_status, row_id)
                    cursor.execute(update_query)
                    connection.commit()
                    print('Successfully updated id {}'.format(row_id))
                    connection.close()
            except:
                continue
            break

# need this function to pull videos in from the captures directory
@server.route('/captures/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'captures'), path)

###################################################################
if __name__ == '__main__':
    # webbrowser.open('http://127.0.0.1:8050/')
    # app.run_server(debug=False)
    app.run_server(port=8050, host="0.0.0.0", debug=True)
