#!/usr/bin/env python3

# -*- coding: utf-8 -*-

 
from datetime import datetime
import dash

from dash import Dash, html, Input, Output, State

from dash.exceptions import PreventUpdate

from dash import dash_table, dcc, html

import dash_bootstrap_components as dbc

import webbrowser

import base64

import json

import os

import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import flask

from flask import Flask, Response

import dash_player

import sqlite3

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

import subprocess

import random

import string

import re

import datetime

 

####################################################################

# Stand Alone Function Definitions

def time_diff_in_hours(start_time, end_time): 
    start = datetime.datetime.strptime(start_time, '%H%M') 
    end = datetime.datetime.strptime(end_time, '%H%M') 
    diff = end - start 
    
    diff_hours = diff.total_seconds() / 3600 # Convert the difference to hours return diff_hours

    return diff_hours

def extract_video_metadata(df):

    # directory/folder path

    dir_path = './static'

    new_df = pd.DataFrame()

 

    #these lists will be used to populate our dataframe

    start_dates = []

    end_dates = []

    time_spent = []

    total_activity_hours = []

    file_paths = []

    location = []

 

    # this is to make sure we summarize time information if we have more than one row with the same date

    duplicate_dates = []

 

    # pattern to get start and end time

    pattern = r'2023.*?\.'

 

    # also want to create an excel sheet with all the filenames to facilitate playing videos in AIMS.py

    new_files_df = pd.DataFrame()

 

    # # Iterate directory

    for file_path in os.listdir(dir_path):

        # extract date information from each video

        matches = re.findall(pattern, str(file_path))

 

        if len(matches) > 1:

            # add start date, end date, time_spent and total hours info to their respective lists

            start_dates.append(matches[0][:10])

            end_dates.append(matches[1][:10])

            start_time = (str(matches[0][11:16])).replace(":", "")
            end_time = (str(matches[1][11:16])).replace(":", "")

            time_spent.append(start_time + " - " + end_time)

            total_hours = time_diff_in_hours(str(start_time), str(end_time))

            total_activity_hours.append(total_hours)

            file_paths.append(file_path)

            location.append(str(file_path[-14:-4]))

 

    # this is to check if we have more than one row with the same date

    # basically we compare the starting and ending times of both rows so we have the actual total span of time

    #change the time spent and total hours info for the first row with that date

    # then we keep track of the index of the duplicate date row so we can drop it later

    for date in range(len(start_dates)):

        for duplicate in range(len(start_dates)):

            if start_dates[date] == start_dates[duplicate] and date != duplicate and date not in duplicate_dates:

                starting_time = min(int(time_spent[date][:4]), int(time_spent[duplicate][:4]))

                print(starting_time, flush=True)

                if len(str(starting_time)) < 4:

                    starting_time = "0" + str(starting_time)
                

                ending_time = max(int(time_spent[date][7:]), int(time_spent[duplicate][7:]))

                if len(str(ending_time)) < 4:

                    ending_time = "0" + str(ending_time)

                print(ending_time, flush=True)

                time_spent[date] = str(starting_time) + " - " + str(ending_time)

                print(time_spent, flush=True)

                reassigned_time = time_diff_in_hours(str(starting_time), str(ending_time))

                total_activity_hours[date] = reassigned_time

                duplicate_dates.append(duplicate)

 

    # create the rest of the cols in our df

    new_df["Date"] = start_dates

    new_df["Time Spent"] = time_spent

    new_df["Total Activity Hours"] = total_activity_hours

    new_df["Location"] = location

    new_df["Maximum Amount of Personnel"] = [random.randint(2, 5) for i in range(len(new_df))]

    new_df["JON #"] = [''.join(random.choices(string.ascii_uppercase + string.digits, k = 8)) for i in range(len(new_df))]

 

 

    vehicle_id = list(df["Vehicle ID"])

    new_df["Vehicle ID"] = [random.choice(["Unclassified", "LAV-25A2"]) for i in range(len(new_df))]

 

    if "" in vehicle_id:

        new_df["Vehicle ID"] = df["Vehicle ID"]

 

 

    new_df["Notes"] = df["Notes"]

    new_df["File Name"] = file_paths

 

    #drop duplicate date rows

    new_df = new_df.drop(duplicate_dates, axis="index")

    new_df['Date'] = pd.to_datetime(new_df['Date'])

 

    #sort df by date

    new_df = new_df.sort_values(by='Date')

 

    dates = [date.strftime('%Y-%m-%d') for date in list(new_df["Date"])]

 

    new_df["Date"] = dates

 

    new_files_df["File Name"] = new_df["File Name"]

    new_df = new_df.drop(columns=['File Name'])

 

    for i in range(len(start_dates)):

        for j in range(len(list(df["Date"]))):

            if start_dates[i] == list(df["Date"])[j]:

                new_df.iloc[i, 7] = df.iloc[j,7]

                new_df.iloc[i,6] = df.iloc[j,6]

 

 

    new_df = new_df.reset_index(drop=True)

    new_files_df = new_files_df.reset_index(drop=True)

    new_files_df.to_excel("files.xlsx")

 

    return new_files_df, new_df

 

 

def new_sql():

    #conn.close()

    print("Creating new sqlite file", flush=True)

    try:

        os.remove("./pi.db")

    except:

        pass

    conn = sqlite3.connect("pi.db", check_same_thread=False)

    c = conn.cursor()

    c.execute(

        """

        CREATE TABLE IF NOT EXISTS activity (Date TEXT, "Time Spent" TEXT, "Total Activity Hours" REAL,

        Location TEXT, "Maximum Amount of Personnel" INTEGER, "JON #" TEXT, "Vehicle ID" TEXT, Notes TEXT);

        """

    )

    return conn

 

####################################################################

 

 

####################################################################

# Global Variable Definitions

 

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

kpmg_logo = './kpmg-logo-png-transparent.png'

kpmg_encoded_image = base64.b64encode(open(kpmg_logo, 'rb').read())

####################################################################

app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO, dbc.icons.BOOTSTRAP])

 

app.title = 'Physical Intrusion Dashboard'

 

conn = new_sql()

 

# Creates button for notes and vehicle ID submission

button = dbc.Button("Submit")

 

# Create dataframe

df = pd.read_sql("SELECT * FROM activity", conn)

 

df_files, df = extract_video_metadata(df)

 

df.to_sql('activity', conn, if_exists='replace',index=False)

conn.commit()

 

# Creates graph on home page

fig = px.line(x=df["Date"], y=df["Maximum Amount of Personnel"])

fig.add_bar(x=df["Date"], y=df["Total Activity Hours"], name="Total Activity Hours")

fig.update_layout(height=800, width=800, colorway=kpmg_color_palette, xaxis_title="Date", yaxis_title="Maximum  Amount of Personnel")

####################################################################

 

 

#Navigation Bar

navbar = dbc.Navbar([

    html.A([

        html.Img(src='data:image/png;base64,{}'.format(kpmg_encoded_image.decode()),

               style={'height':'2em', 'padding-left':'1rem'}),

        ], id='reload-page', href='/', style={'margin-right':'1rem'}),

    html.Div([

        html.H1(

            'Code B AIMS: AI-Powered Activity Detection',

            style={

                'color':'white',

                'margin-bottom':'0rem',

                'width':'100%',

                'font-family':'KPMG',

                }

            ),

        ], className='ml-auto', style={'padding-right':'1rem'}),

    html.Div([

        dbc.NavLink('Activity Summary', href='/', className='page-link'),

        dbc.NavLink('Individual Detections', href='/individual-detections-page', className='page-link'),

        ], className='ml-auto', style={'width':'100%'})

    ], color=kpmg_blue, sticky='top')

 

 

 

# layout for home page

home_page_layout = html.Div([

    dbc.Row([

        dbc.Col([

            dbc.Row([

                dbc.Button("Welcome to the AIMS homepage.\n To view individual detections, use the Individual Detections link above."),

 

                dash_table.DataTable(

                                df.to_dict('records'),

                                [{"name":i, "id":i} for i in df.columns],

                                id='tbl',

                                page_size=10,

                                style_as_list_view=True, # removes vertical lines

                                #style cell styles whole table

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

                                ),

            dbc.Col([date_drop := dcc.Dropdown([x for x in sorted(df.Date.unique())], multi=True, placeholder="Select Date")], width=3),

            dbc.Col([personnel_drop := dcc.Dropdown([x for x in sorted((df["Maximum Amount of Personnel"]).unique())], multi=True, placeholder="Filter Maximum Amount of Personnel")], width=3),

            dbc.Col([hours_drop := dcc.Dropdown([x for x in sorted((df["Total Activity Hours"]).unique())], multi=True, placeholder="Filter Total Activity Hours")], width=3),

            ],

            justify = "center")

        ]),

        dbc.Col(

            dbc.Row(

                [

                    dcc.Graph(figure=fig, id="graph")

                ],

                justify = "center",

                ),

            ),

 

    ],

    justify = "center",

    )

])

 

 

#creates layout for individual detections page

individual_detections_layout = html.Div([

    dbc.Row([

        dbc.Col([

            dbc.Row([

                dbc.Button("Last activity detection: " + str((df.iloc[-1,0]))),

 

                dash_table.DataTable(

                                df.to_dict('records'),

                                [{"name":i, "id":i} for i in df.columns],

                                id='tbl2',

                                page_current=0,

                                page_size=10,

                                row_selectable="single",

                                cell_selectable=True,

                                style_as_list_view=True, # removes vertical lines

                                #style cell styles whole table

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

                                )

            ],

            justify = "center")

        ]),

        dbc.Col([

            dbc.Row(

            [html.Video(

             controls='CONTROLS',

             muted=True,

             id="pi-video-player",

             autoPlay=True,

             loop=True),

            dbc.FormFloating([dbc.Input(id="Notes"), dbc.Label("Enter Notes or Vehicle ID")]),

            button,

            ]),

                    ]),

            ]),

        ])

 

 

app.layout = dbc.Container([html.Div([

    dcc.Interval(id='interval-component', interval=1*60000, n_intervals=0),  # Poll every 1 minute

    navbar,

    dcc.Location(id='url', refresh=False),

    html.Div(id='content'),

    html.Div(id='hidden-content', style={'display':'None'}),

    html.Div(id='hidden-content1', style={'display':'None'}),

    ]),

    ],

    fluid=True)

####################################################################

 

 

####################################################################

# Callback Function Definitions

 

@app.callback(

        Output('hidden-content1', 'children'),

        Input('interval-component', 'n_intervals')

)

def ingest_new_blobs(n):

    global df, df_files, conn
    try:

        #AIMS Blob Storage Connection String:

        account_url = "https://aims.blob.rack01.usmc5gsmartwarehouse.com/"

        container_name = "videos"

        credential = {

                "account_name": 'aims',

                "account_key": 'FsOkNlgC493jkC1W2sUYzFZyQzn9nJurFTh+W+qua8KltfDOo05H05paGnnoJHG5FcMrtXKrpv0V3JQgvjthWQ=='

            }

        container_client = ContainerClient(account_url = account_url, container_name = container_name, credential = credential, api_version = '2019-02-02')

        blob_list = container_client.list_blobs()

        for blob_name in blob_list:

            blob_client = container_client.get_blob_client(blob=blob_name)

            save_path = "./static/" + str(blob_name.name)

            with open(file=save_path, mode='wb+') as file:

                blob_data = blob_client.download_blob()

                file.write(blob_data.readall())

            blob_client.delete_blob()

        path = "./static/"   

        for file in os.listdir(path):

            fullpath   = os.path.join(path,file)    

            timestamp  = os.stat(fullpath).st_ctime # get timestamp of file

            createtime = datetime.datetime.fromtimestamp(timestamp)

            now        = datetime.datetime.now()

            delta      = now - createtime

            if delta.days > 90:

                os.remove(fullpath)

        conn = new_sql()

        df_files, df = extract_video_metadata(df)

        df.to_sql('activity', conn, if_exists='replace', index=False)

        conn.commit()

    except Exception as e:

        print(str(e))

    return None

 

# callback for url management

@app.callback(Output('content', 'children'),

              Input('url', 'pathname'))

 

def display_page(path):

    if path == '/individual-detections-page':

        return individual_detections_layout

    else:

        return home_page_layout

 

@app.callback(Output('hidden-content', 'children'),

              Input('url', 'pathname'))

def load_hidden_pages(path):

    # load hidden pages so callbacks are available

    if path == '/individual-detections-page':

        return home_page_layout

    else:

        return individual_detections_layout

 

 

 

#callback to connect dropdown selection to graph output

@app.callback(

    Output('tbl', 'data'),

    Output('tbl', 'columns'),

    Output('graph', 'figure'),

    Input(date_drop, 'value'),

    Input(hours_drop, 'value'),

    Input(personnel_drop, 'value'),

)

def update_dropdown_options(date_v, hours_v, personnel_v):

    dff = df.copy()

 

    if date_v:

        dff = dff[dff.Date.isin(date_v)]

    if hours_v:

        dff = dff[(dff["Total Activity Hours"]).isin(hours_v)]

 

    if personnel_v:

        dff = dff[(dff["Maximum Amount of Personnel"]).isin(personnel_v)]

 

 

    fig = px.line(x=dff["Date"], y=dff["Maximum Amount of Personnel"])

 

    fig.add_bar(x=dff["Date"], y=dff["Total Activity Hours"], name="Total Activity Hours")

 

    fig.update_layout(height=800, width=800, colorway=kpmg_color_palette, xaxis_title="Date", yaxis_title="Maximum  Amount of Personnel")

 

    return dff.to_dict('records'), [{"name":i, "id":i} for i in dff.columns], fig

 

#shows active cell in table

@app.callback((

    Output('tbl2', 'selected_rows'),

    ), Input('tbl2', 'active_cell'),

    prevent_initial_call=True)

 

def update_graphs(active_cell):

    return (

        str(active_cell),

        [active_cell['row']],

    ) if active_cell else (str(active_cell),dash.no_update)

    #) if active_cell else (str(active_cell))

 

#adds value in form to notes or vehicle ID column

@app.callback(

    (

        Output('tbl2', 'data'),

        Output('tbl2', 'columns'),

        Output('tbl2','style_data_conditional'),

        Output('tbl2','active_cell'),

        Output('tbl2','selected_cells')

    ),

    Input(button, "n_clicks"),

    Input('tbl2', 'derived_viewport_selected_row_ids'),

    State("Notes", "value"),

    State('tbl2', 'derived_viewport_selected_rows'),

    State('tbl2', 'active_cell'),

    State('tbl2', 'selected_cells'),

    State('tbl2', 'page_current'),

    prevent_initial_call=True)

def update_graphs2(_,selected_row_ids,notes,selected_rows, active_cell, selected_cells, page_current):

        global conn

        if selected_row_ids:

            if active_cell:

 

                current_page = int(str(page_current) + "0")

                chosen_row = selected_rows[0] + current_page

                #active_cell['row'] = selected_rows[0]

                active_cell['row'] = chosen_row

                if active_cell['column_id'] == "Vehicle ID":

 

                    df.at[active_cell['row'], "Vehicle ID"] = notes

 

                else:

                    df.at[active_cell['row'], "Notes"] = notes

 

                df.to_sql('activity', conn, if_exists='replace',index=False)

                conn.commit()

 

 

 

        return [df.to_dict("records"), [{"name":i, "id":i} for i in df.columns],

            [

                {

                    "if": {

                        "filter_query": "{{id}} = '{}'".format(i)

                    },

                    "backgroundColor": medium_blue,

                    "border": "1px solid rgb(0, 51, 141)",

                } for i in selected_row_ids

            ] + [

                {

                    "if": {

                        "state": "active"  # 'active' | 'selected'

                    },

                    "backgroundColor": light_blue,

                    "border": "1px solid rgb(0, 51, 141)",

                }

            ],

            active_cell,

            []

        ]

 

#plays video associated to selected row

@app.callback(

    Output("pi-video-player", "src"),

    Input('tbl2', 'derived_viewport_selected_row_ids'),

    Input('tbl2', 'derived_viewport_selected_rows'),

    State('tbl2', 'active_cell'),

    State('tbl2', 'page_current'),

)

def play_video_after_user_selects_datatable_row(selected_row_ids,selected_rows,active_cell,page_current):

    """After user selects row in datatable, get relevant video for playback"""


    global df_files
 

    path = ""

 

    if selected_row_ids:

 

        current_page = int(str(page_current) + "0")

 

 

        #if active_cell:

        chosen_row = selected_rows[0] + current_page

        #active_cell['row'] = selected_rows[0] + current_page

        #path = "static/" + str(df_files["File Name"][int(active_cell['row'])])

        path = "static/" + str(df_files["File Name"][chosen_row])

        #print(path, flush=True)

        #print(df_files["File Name"][-1], flush=True)

 

    return path

 

@app.server.route('/static/<path:path>')

def serve_static(path) -> Response:

    root_dir = os.getcwd()

    return flask.send_from_directory(os.path.join(root_dir, 'static'), path)

 

###################################################################

if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:7060/')
    app.run_server(debug=False, use_reloader=True, host='0.0.0.0', port=7060)