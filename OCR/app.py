#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:50:46 2022

@author: kritz
"""
print("Sanity check", flush=True)
# set deploy to false if running locally; true if sending to deployment
deploy = True
speed_it_up = True
version = 'V4.1'
# set blobSwitch to false if using local files; true if using files in blob storage
blobSwitch = True
# set path for prepare database
ocr_db_path = './'
ocr_db_name = 'prepare.db'

import dash
from dash_extensions.enrich import Input, Output, Dash, State
from dash.exceptions import PreventUpdate
from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc
from dash_extensions import Download
from dash_extensions.snippets import send_bytes
import webbrowser
import base64
import json
import os
from datetime import datetime
import pytz
import cv2
import pytesseract
import re
import numpy as np
import imutils
import pandas as pd
from skimage.transform import rotate
import collections
from fuzzywuzzy import process
import math
import datetime as dt
import ast
import random
import string
import io
import glob
import sqlite3
import blobStorage
from pyzbar.pyzbar import decode
from createPrepareDatabase import create_ocr_database
from PyPDF2 import PdfFileMerger

####################################################################
# map all due in columns to 1348 columns
duein_column_map = {
    # due in v1
    'Standard Document Number':'Document Number',
    'Qty':'Quantity',
    'Nomenclature':'Item Nomenclature',
    # due in v2
    'DOC_NBR':'Document Number',
    'STATUS_QTY':'Quantity',
    'SHIP_FROM_ORG':'Ship From',
    'SHIP_TO_ORG':'Ship To',
    'NSN_ORDERED':'NSN',
    'PROJECT_CD':'Project',
    'SUPP_ADD':'Supplementary Address',
    'AS1_3_DT':'Date Received',
    # both
    'DIC':'Document Identity',
}

duein_1348_overlapping = [
    'Document Number', 'NSN', 'Quantity', 'Item Nomenclature', 'Advice Code',
    'Condition Code', 'Ship To', 'Ship From', 'Project',
    'Supplementary Address', 'Document Identity',
]

speed_it_up_columns = [
    'Document Number', 'NSN', 'Quantity', 'Item Nomenclature'
    ]

if deploy:
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

####################################################################
if blobSwitch:
    print("Before trying to get environment variables", flush=True)
    # blob storage container string
    connect_str = os.environ.get('BLOB_STORAGE_CONNECT_STR')
    sas_credential = os.environ.get('BLOB_STORAGE_CREDENTIALS')
    print("After trying to get environment variables", flush=True)
    # aligned photos
    alignedContainerName = 'upload-aligned-ocr-images'
    alignedContainer = blobStorage.getContainerClient(connect_str, alignedContainerName, credential = sas_credential)

    # template converted images
    templateimageContainerName = 'ocr-template-converted-images'
    templateimageContainer = blobStorage.getContainerClient(connect_str, templateimageContainerName, credential = sas_credential)

    # template json files
    templatejsonContainerName = 'ocr-template-json-files'
    templatejsonContainer = blobStorage.getContainerClient(connect_str, templatejsonContainerName, credential = sas_credential)

    # due in reports
    dueinContainerName = 'due-in-reports'
    dueinContainer = blobStorage.getContainerClient(connect_str, dueinContainerName, credential = sas_credential)

    # due out reports
    dueoutContainerName = 'due-out-reports'
    dueoutContainer = blobStorage.getContainerClient(connect_str, dueoutContainerName, credential = sas_credential)

    # supporting files
    supportingContainerName = 'ocr-app-supporting-files'
    supportingContainer = blobStorage.getContainerClient(connect_str, supportingContainerName, credential = sas_credential)

    # d365 data dumps
    d365ContainerName = 'ocr-data-for-d365'
    d365Container = blobStorage.getContainerClient(connect_str, d365ContainerName, credential = sas_credential)
else:
    None
####################################################################
# read supporting files from blob storage
kpmg_logo = 'kpmg-logo-png-transparent.png'
if blobSwitch:
    kpmg_blob = supportingContainer.download_blob(kpmg_logo).readall()
    kpmg_encoded_image = base64.b64encode(kpmg_blob)
else:
    kpmg_encoded_image = base64.b64encode(open(kpmg_logo, 'rb').read())
####################################################################
# create folder if it does not exist
# this is only relevant if blobswitch is false
upload_photos = 'uploads/photos/'
upload_deskewed = 'uploads/deskewed/'
aligned_photos = 'uploads/aligned/'
aligned_overlaid_photos = 'uploads/overlaid/'
upload_temp = 'uploads/temporary_files/'
template_parameters = 'prepare/templates/'
duein_path = 'gcss_reports/due_in/'
dueout_path = 'gcss_reports/due_out/'
d365_files = 'd365_files/'

parent_paths = [
    'uploads/',
    'prepare/'
    'gcss_reports/',
    'd365_files/'
    ]

paths = [
    upload_photos, upload_deskewed, aligned_photos, aligned_overlaid_photos,
    upload_temp,
    template_parameters,
    duein_path, dueout_path
    ]

if blobSwitch:
    None
else:
    for p in parent_paths:
        if not os.path.exists(p):
            os.mkdir(p)

    for p in paths:
        if not os.path.exists(p):
            os.mkdir(p)

####################################################################
# create ocr database in sqlite (if it exists, nothing will happen)
create_ocr_database(ocr_db_path + ocr_db_name, template_parameters, speed_it_up=True)

####################################################################
# read existing 1348 templates from local or blob storage
if blobSwitch:
    available_templates = [blob for blob in blobStorage.listBlobs(templateimageContainer) if blob.split('.')[0] + '.json' in blobStorage.listBlobs(templatejsonContainer) and '1348' in blob and 'SHORT' not in blob]
else:
    available_templates = [t for t in os.listdir(template_parameters) if t.split('.')[0] + '.json' in os.listdir(template_parameters) and '1348' in t and '.jpg' in t]
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

# hex colors
red_hex = '#BC204B'
blue_hex = '#00338D'
####################################################################
def cleanup_unixtime(path, create_or_mod_time):
    '''
    convert unix time to a nicely formatted timestamp
    '''
    try:
        if create_or_mod_time == 'create':
            unix_time = os.path.getctime(path)
        else:
            unix_time = os.path.getmtime(path)
        utc_time = datetime.utcfromtimestamp(unix_time)
        localized_time = pytz.utc.localize(utc_time)
        eastern_time = localized_time.astimezone(pytz.timezone('America/New_York'))
        formated_time = eastern_time.strftime('%Y-%m-%d %H:%M')
        return formated_time
    except:
        return 'No data available.'

def utc_timezone_cleanup(time):
    time = time.astimezone(pytz.timezone('US/Eastern'))
    time = time.strftime('%Y-%m-%d %H:%M')
    return time

def get_pixel_diff(img):
    # if diff in largest and smallest pixel (255*3 = white, 0*3 = black)
    # greater than 350, mark the image as empty
    max_pixel = 0
    min_pixel = 255*3
    for idx, y in enumerate(img):
        for idx2, x in enumerate(y):
            if sum(x) > max_pixel:
                max_pixel = sum(x)
            if sum(x) < min_pixel:
                min_pixel = sum(x)
    diff = max_pixel - min_pixel
    if diff > (255*3)/2:
        return True
    else:
        return False

def extract_text(img0, scale, barcode='False', field_type='Default'):
    # correct image
    img = cv2.resize(img0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    if percent_pixels(img) >= 0.4: # threshold for % pixels that are not white
        return ''

    # barcode removal if applicable
    if barcode == 'True':
        # Decode the barcode image
        detectedBarcodes = decode(img0)
        if not detectedBarcodes:
            # Remove horizontal lines
            gray = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
            remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
            cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = [c*scale for c in cnts]
            for c in cnts:
                cv2.drawContours(img, [c], -1, (255,255,255), 15)

            # Remove vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
            remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
            cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = [c*scale for c in cnts]
            for c in cnts:
                cv2.drawContours(img, [c], -1, (255,255,255), 15)

            # extract text
            text = pytesseract.image_to_string(img)
            if not any(letter.isalnum() for letter in text):
                text = pytesseract.image_to_string(img, config='--psm 6')
            return text
        else:
            for barcode in detectedBarcodes:
                return barcode.data.decode()

    else:
        # Remove horizontal lines
        gray = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = [c*scale for c in cnts]
        for c in cnts:
            cv2.drawContours(img, [c], -1, (255,255,255), 15)

        # Remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = [c*scale for c in cnts]
        for c in cnts:
            cv2.drawContours(img, [c], -1, (255,255,255), 15)

        # detect text if default field; otherwise, don't trim
        if field_type == 'Default':
            ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
            dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
                x, y, w, h = cv2.boundingRect(contour)
                img = img[y:y + h, x:x + w]

            # extract text
            text = pytesseract.image_to_string(img)
            if not any(letter.isalnum() for letter in text):
                text = pytesseract.image_to_string(img, config='--psm 6')

            return text

        if field_type == 'Short':
            edged = cv2.Canny(img, 30, 200)
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            letters = {}
            for idx, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area > 0:
                    x, y, w, h = cv2.boundingRect(c)
                    preview = img[y:y + h, x:x + w]
                    # add border
                    preview = cv2.copyMakeBorder(preview,10,10,10,10,cv2.BORDER_CONSTANT,value=[255, 255, 255])
                    text = pytesseract.image_to_string(preview, config='--psm 6')
                    letters[x] = cleanup_text(text)
            sorted_letters = dict(collections.OrderedDict(sorted(letters.items())))
            output = ''.join(v for v in sorted_letters.values())
            return output

def percent_pixels(img):
    # what % of pixels are black
    white = 0
    black = 0
    for col in img:
        for row in col:
            if row >= 200:
                white += 1
            else:
                black += 1
    percent = black/(white + black)
    return percent

def cleanup_text(text):
    text = ''.join([c if ord(c) < 128 else '' for c in text]).strip()
    text = text.replace('  ', ' ')
    text = text.rstrip()
    return text

# blur detection
def blur_detection(image):
    def variance_of_laplacian(image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    if fm < 100: # blur detection threshold
        return False
    return True

def resolution(image):
    height, width = image.shape[:2]
    if height < 1000 or width < 1000:
        return False
    return True

def generate_random_filename():
    x = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    return x
####################################################################
app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO, dbc.icons.BOOTSTRAP])
if deploy:
    server = app.server

app.title = 'PREParE'

navbar = dbc.Navbar([
    html.A([
        html.Img(src='data:image/png;base64,{}'.format(kpmg_encoded_image.decode()),
               style={'height':'2em', 'padding-left':'1rem'}),
        ], id='reload-page', href='/', style={'margin-right':'1rem'}),
    html.Div([
        html.H1(
            'PREParE',
            style={
                'color':'white',
                'margin-bottom':'0rem',
                'width':'100%',
                'font-family':'KPMG',
                }
            ),
        ], className='ml-auto', style={'padding-right':'1rem'}),
    html.Div([
        dcc.Markdown(
            '''
            ***P***re***R***eceipt ***E***xpedited ***Pa***pe***r***work ***E***xtraction
            ''',
            style={
                'color':'white',
                'font-family':'KPMG',
                'font-size':'1.65em',
                'margin-bottom':-20,
                'width':'100%'
                })
        ], style={'width':'75%'}),
    html.Div([
        dbc.NavLink('Upload GCSS Reports', href='/gcss-upload-page', className='page-link'),
        dbc.NavLink('Paperwork Database', href='/database-page', className='page-link'),
        dbc.NavLink('Item Paperwork Pre-Receipt', href='/', className='page-link'),
        ], className='ml-auto', style={'width':'100%'}) # reverse order bc of ml-auto
    ], color=kpmg_blue, style={'list-style-type':'none'}, sticky='top')

packet_page = html.Div([
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Button(
                    'Process a New Packet',
                    id='start-packet-btn',
                    className='kpmg-button-secondary'
                    )
                ], className='d-grid gap-2')
            ], width={'size': 3}),
        dbc.Col([
            html.Div([
                dbc.Button(
                    'Complete Packet',
                    id='end-packet-btn',
                    disabled=True,
                    className='kpmg-button-secondary'
                    )
                ], className='d-grid gap-2')

            ], width={'size': 3}),
        dbc.Col([
            html.Div([
                dbc.Button(
                    'Discard Packet',
                    id='discard-packet-btn',
                    disabled=True,
                    className='kpmg-button-secondary'
                    )
                ], className='d-grid gap-2')

            ], width={'size': 3}),
        dbc.Col([
            html.Div([
                dbc.Button(
                    'Preview Packet',
                    id='preview-packet-btn',
                    disabled=True,
                    className='kpmg-button-secondary'
                    )
                ], className='d-grid gap-2')

            ], width={'size': 3}),
        ], style={'textAlign':'center'}, justify='center'),
    ])

new_packet_page = html.Div([
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H3('Upload Packet Components', style={'margin-bottom':'0rem', 'font-family':'KPMG'})
            ], width=3, style={'border-right':'1px {} dotted'.format(kpmg_blue)}),
        dbc.Col([
            html.H3('Processing Status', style={'margin-bottom':'0rem', 'font-family':'KPMG'})
            ], style={'border-right':'1px {} dotted'.format(kpmg_blue)}),
        dbc.Col([
            html.H3('Processing Alerts', style={'margin-bottom':'0rem', 'font-family':'KPMG'})
            ], style={'border-right':'1px {} dotted'.format(kpmg_blue)}),
        ], style={'padding-bottom':'0.25rem', 'border-bottom':'1px {} dotted'.format(kpmg_blue)}),
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Upload([
                    html.Div([
                        dbc.Button(
                            'Process 1348 Form',
                            className='kpmg-button-tertiary'),
                        ], className='d-grid gap-2'),
                    ],
                    accept=['.jpg', '.png'],
                    id='upload-1348'),
                ], style={'padding-bottom':'0.25rem', 'padding-top':'0.25rem'}),
            html.Div([
                dcc.Upload([
                    html.Div([
                        dbc.Button('Process Additional Forms', id='process-additional-forms-btn', className='kpmg-button-tertiary'),
                        ], className='d-grid gap-2'),
                    ],
                    accept=['.jpg', '.png'],
                    id='upload-additional'),
                ], style={'padding-bottom':'0.25rem', 'padding-top':'0.25rem'}),

            ], width=3, style={'border-right':'1px {} dotted'.format(kpmg_blue)}),
        dbc.Col([
            html.Div(id='packet-contents'),
            html.Div(id='1348-spinner'),
            html.Div(id='additional-form-status'),
            ], style={'border-right':'1px {} dotted'.format(kpmg_blue)}),
        dbc.Col([
            html.Div(id='1348-begin-processing-check'),
            html.Div(id='1348-resolution-check'),
            html.Img(id='1348-resolution', style={'display':'None'}),
            html.Div(id='1348-blur-check'),
            html.Img(id='1348-blur', style={'display':'None'}),
            html.Div(id='1348-template-check'),
            html.Img(id='1348-template', style={'display':'None'}),
            html.Div(id='template-scale', style={'display':'None'}),
            html.Div(id='1348-template-to-use', style={'display':'None'}),
            html.Div(id='1348-extract-check'),
            html.Div(id='1348-extract', style={'display':'None'}),
            html.Div(id='1348-match-check'),
            html.Div(id='1348-match', style={'display':'None'}),
            html.Div(id='additional-form-begin-check'),
            html.Div(id='additional-form-resolution-check'),
            html.Img(id='additional-form-resolution', style={'display':'None'}),
            html.Div(id='additional-form-blur-check'),
            html.Img(id='additional-form-blur', style={'display':'None'}),
            html.Div(id='additional-form-extract-check'),
            html.Div(id='additional-form-extract', style={'display':'None'}),
            ], style={'border-right':'1px {} dotted'.format(kpmg_blue)}),
        ], style={'padding-bottom':'0.25rem', 'padding-top':'0.25rem'}),
    dbc.Modal([
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([
                    html.P('Rotate form if necessary, then close the window to proceed.', style={'padding-bottom':'0rem', 'margin-bottom':'0rem'}),
                    dbc.ButtonGroup([
                        dbc.Button(html.I(className="bbi bi-arrow-counterclockwise"), id='rotate-left', className='kpmg-button'),
                        dbc.Button(html.I(className="bbi bi-arrow-clockwise"), id='rotate-right', className='kpmg-button')
                        ]),
                    html.Br(),
                    html.Br(),
                    dbc.Spinner([
                        html.Div([
                            html.Img(id='preview-additional-form', style={'width':'100%'})
                            ])
                        ], spinner_style={'color':'{}'.format(kpmg_blue)}),
                    ])
                ], style={'textAlign':'center'}),
            ]),
        dbc.ModalFooter([
            dbc.Button('Close', className='kpmg-button-secondary', id='close-additional-form-modal')
            ])
        ], id='additional-form-modal', scrollable=True, backdrop="static"),
    dbc.Modal([
        dbc.ModalHeader([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Button(
                                'Download 1348 Data (.xslx)',
                                id='download-1348-xlsx-btn',
                                size='sm',
                                className='kpmg-button'
                                ),
                            ], className='d-grid gap-2'),
                        ]),
                    dbc.Col([
                        html.Div([
                            dbc.Button(
                                'Download Packet (Searchable PDF)',
                                id='download-packet-pdf-btn',
                                size='sm',
                                className='kpmg-button'
                                ),
                            ], className='d-grid gap-2'),
                        ]),
                    ], justify='center'),
                dbc.Row([
                    dbc.Col([
                        dbc.Spinner(Download(id='download-1348-xlsx'), fullscreen=True, spinner_style={"width": "4rem", "height": "4rem", 'margin-top':'0.5rem', 'color':'{}'.format(kpmg_blue)}),
                        dbc.Spinner(Download(id='download-packet-pdf'), fullscreen=True, spinner_style={"width": "4rem", "height": "4rem", 'margin-top':'0.5rem', 'color':'{}'.format(kpmg_blue)}),
                        ])
                    ], style={'fontAlign':'center'})
                ], className='d-grid gap-2'),
            ]),
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([
                    dbc.Spinner(html.Div(id='preview-content'), spinner_style={"width": "2rem", "height": "2rem", 'color':'{}'.format(kpmg_blue)})
                    ])
                ], style={'fontAlign':'center'})
            ]),
        dbc.ModalFooter([
            dbc.Button('Close', className='kpmg-button-secondary', id='close-preview-modal')
            ])
        ], id='preview-modal', scrollable=True),
    html.Div(id='additional-form-filename', style={'display':'None'}),
    html.Div(id='all-processed-forms', style={'display':'None'}),
    html.Div(id='new-1348-filename', style={'display':'None'}),
    ])

gcss_layout = dbc.Container([
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H2('Upload Due In/Due Out Reports',
                    style={
                        'margin-bottom':'0rem',
                        'font-family':'KPMG',
                        }),
            html.Div(id='last-uploaded-report', style={'padding-bottom':'0rem', 'margin-bottom':'0rem'}),
            html.Hr(),
            ])
        ], justify='center', align='center'),
    dbc.Form([
        dbc.Row([
            dbc.Label('What kind of report is this?', width=3),
            dbc.Col(
                dbc.RadioItems(
                    options=[{'label':i, 'value':i} for i in ['Due In', 'Due Out']],
                    value='Due In',
                    id='upload-report-category',
                    ),
                width=6
                )
            ], className="mb-3", align="center"),
        dbc.Row([
            dbc.Label('Upload the report', width=3),
            dbc.Col([
                dbc.Spinner(
                    dcc.Upload([
                        html.Div([
                            'Drag and Drop or ',
                            html.A('Select Document', href="#")
                            ], style={'margin-left':'40px', 'margin-right':'40px'}),
                        ], style={
                            'height': '40px',
                            'lineHeight': '40px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'verticalAlign':'top',
                            },
                        accept=['.csv', '.xlsx'],
                        id='upload-report'),
                    ),
                html.Div(id='uploaded-report-filename'),


                ], width=6)
            ], className="mb-3", align="center"),
        dbc.Row([
            dbc.Label('Confirm and Initiate Processing', width=3),
            dbc.Col([
                html.Div([
                    dbc.Button('Process Document', id='process-document-btn', className='kpmg-button'),# block=True),
                    ], className='d-grid gap-2'),
                dbc.Spinner(html.Div(id='process-upload-report')),
                ], width=6
                )
            ], className="mb-3", align="center"),
        ]),
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.H3('Preview Current Due In Report', style={'margin-bottom':'0rem', 'font-family':'KPMG', 'font-weight':'bold'}),
            html.Div([
                dash_table.DataTable(
                    id='preview-due-in',
                    page_size=10,
                    style_as_list_view=True, # removes vertical lines
                    filter_action='native',
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
                    ),
                ])
            ],  style={'textAlign':'center'})
        ])
    ], fluid=True)

homepage_layout = dbc.Container([
    html.Br(),
    dbc.Row([
        dbc.Col([

            html.Div([
                dbc.Button(
                    'Begin Pallet Pre-Receipt',
                    id='start-pallet-btn',
                    className='kpmg-button'
                    )
                ], className='d-grid gap-2')
            ], width={'size': 3}),
        dbc.Col([
            html.Div([
                dbc.Button(
                    'Complete Pallet Pre-Receipt',
                    id='end-pallet-btn',
                    disabled=True,
                    className='kpmg-button'
                    )
                ], className='d-grid gap-2')

            ], width={'size': 3})
        ], style={'textAlign':'center'}, justify='center'),
    dbc.Row([
        dbc.Col([
            html.Div(id='packet-section'),
            html.Div(id='new-packet-section'),
            ])
        ]),
    ], fluid=True)


database_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Br(),
            html.H3('Search PREParE Database', style={'margin-bottom':'0rem', 'font-family':'KPMG', 'font-weight':'bold'}),
            html.P('Search the PREParE database below. Use the radio buttons to filter your search query.'),
            html.Hr(),
            ])
        ]),
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Label('Specify packets to search', style={'padding-bottom':'0rem', 'font-weight':'bold'}),
                dbc.RadioItems(
                    id='forms-to-search',
                    options=[
                        {'label':'All', 'value':'All'},
                        {'label':'Only Packets with 1348 Forms', 'value':'1348 Only'}
                    ],
                    value='All',
                    ),
                ], style={'padding':'1rem'}),

            ], style={'border':'1px {} solid'.format(kpmg_blue)}),
        dbc.Col([
            html.Div([
                dbc.Label('Frustrated or all packets', style={'padding-bottom':'0rem', 'font-weight':'bold'}),
                dbc.RadioItems(
                    id='frustrated-filter',
                    options=[
                        {'label':'All', 'value':'All'},
                        {'label':'Frustrated Only', 'value':'Frustrated Only'},
                        {'label':'Processed Only', 'value':'Processed Only'},
                    ],
                    value='All',
                    ),
                ], style={'padding':'1rem'}),

            ], style={'border':'1px {} solid'.format(kpmg_blue)}),
        dbc.Col([
            html.Div([
                dbc.Label('Type to search all fields', style={'padding-bottom':'0rem', 'font-weight':'bold'}),
                dbc.Input(
                    placeholder='Type here...',
                    autoComplete='off',
                    id='record-search',
                    debounce=True,
                    ),
                ], style={'padding':'1rem'}),
            ], style={'border':'1px {} solid'.format(kpmg_blue)}, width=4),
        dbc.Col([
            html.Div([
                html.Div([
                    dbc.Button(
                        'Reset Search Criteria',
                        id='reset-search-btn',
                        className='kpmg-button-tertiary',
                        ),
                    dbc.Button(
                        'Preview Selected Packet',
                        id='preview-packet-db-btn',
                        className='kpmg-button-tertiary',
                        ),
                    ], className='d-grid gap-2')
                ], style={'padding':'1rem'}),
            ], style={'border':'1px {} solid'.format(kpmg_blue)}),
        ], className="g-0", style={'font-size':'80%'}),
    dbc.Row([
        dbc.Col([
            html.Br(),
            dash_table.DataTable(
                id='prepare-database',
                page_size=10,
                style_as_list_view=True, # removes vertical lines
                # filter_action='native',
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
                ),
            dbc.Modal([
                dbc.ModalHeader([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    dbc.Button(
                                        'Download 1348 Data (.xslx)',
                                        id='download-1348-xlsx-btn-db',
                                        size='sm',
                                        className='kpmg-button'
                                        ),
                                    ], className='d-grid gap-2'),
                                ]),
                            dbc.Col([
                                html.Div([
                                    dbc.Button(
                                        'Download Packet (Searchable PDF)',
                                        id='download-packet-pdf-btn-db',
                                        size='sm',
                                        className='kpmg-button'
                                        ),
                                    ], className='d-grid gap-2'),
                                ]),
                            ], justify='center'),
                        dbc.Row([
                            dbc.Col([
                                dbc.Spinner(Download(id='download-1348-xlsx-db'), fullscreen=True, spinner_style={"width": "4rem", "height": "4rem", 'margin-top':'0.5rem', 'color':'{}'.format(kpmg_blue)}),
                                dbc.Spinner(Download(id='download-packet-pdf-db'), fullscreen=True, spinner_style={"width": "4rem", "height": "4rem", 'margin-top':'0.5rem', 'color':'{}'.format(kpmg_blue)}),
                                ])
                            ], style={'fontAlign':'center'})
                        ], className='d-grid gap-2'),
                    ]),
                dbc.ModalBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Spinner(html.Div(id='preview-content-db'), spinner_style={"width": "2rem", "height": "2rem", 'color':'{}'.format(kpmg_blue)})
                            ])
                        ], style={'fontAlign':'center'})
                    ]),
                dbc.ModalFooter([
                    dbc.Button('Close', className='kpmg-button-secondary', id='close-preview-modal-db')
                    ])
                ], id='preview-modal-db', scrollable=True),
            ])

        ])
    ], fluid=True)

app.layout = html.Div([
    navbar,
    dcc.Location(id='url', refresh=False),
    html.Div(id='content'),
    html.Div(id='hidden-content', style={'display':'None'}),
    dcc.Store(id='duein-report'),
    html.Div(id='pallet-id', style={'display':'None'}),
    html.Div(id='packet-id', style={'display':'None'}),
    html.Div(id='delete-discarded-rows', style={'display':'None'}),

    ])
####################################################################
@app.callback(Output('content', 'children'),
              Input('url', 'pathname'))
def display_page(path):
    if path == '/database-page':
        return database_layout
    elif path == '/gcss-upload-page':
        return gcss_layout
    else:
        return homepage_layout

@app.callback(Output('hidden-content', 'children'),
              Input('url', 'pathname'))
def load_hidden_pages(path):
    # load hidden pages so callbacks are available
    if path == '/database-page':
        return [gcss_layout, homepage_layout]
    elif path == '/gcss-upload-page':
        return [database_layout, homepage_layout]
    else:
        return [database_layout, gcss_layout]


@app.callback(Output('packet-section', 'children'),
              Input('start-pallet-btn', 'n_clicks'),
              Input('end-pallet-btn', 'n_clicks'))
def show_packet_section(btn1, btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'start-pallet-btn' in changed_id:
        return packet_page
    if 'end-pallet-btn' in changed_id:
        return None
    raise PreventUpdate

@app.callback(Output('new-packet-section', 'children'),
              Input('start-packet-btn', 'n_clicks'),
              Input('end-packet-btn', 'n_clicks'),
              Input('discard-packet-btn', 'n_clicks'))
def show_new_packet_section(btn1, btn2, btn3):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'start-packet-btn' in changed_id and value_id:
        return new_packet_page
    if 'end-packet-btn' in changed_id and value_id:
        return None
    if 'discard-packet-btn' in changed_id and value_id:
        return None
    raise PreventUpdate

###################################################################
@app.callback(Output('pallet-id', 'children'),
              Input('start-pallet-btn', 'n_clicks'))
def create_pallet_info(btn1):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'start-pallet-btn' in changed_id:
        # get max pallet id from sql database
        connection = sqlite3.connect(ocr_db_name)
        cursor = connection.cursor()

        # get max row ID
        query = '''
            SELECT MAX(p."Pallet ID") AS "Next ID" FROM CompletePackets p
            '''
        cursor.execute(query)
        # column_names = [x[0] for x in cursor.description]
        extract = cursor.fetchall()
        # extract_df = pd.DataFrame(extract, columns=column_names)
        # next_id = extract_df['Packet ID'].max()
        next_id = extract[0][0]
        if next_id is np.nan:
            next_id = 0
        elif next_id == None:
            next_id = 0
        else:
            next_id = next_id + 1
        connection.close()

        return next_id
    raise PreventUpdate

@app.callback([Output('packet-id', 'children')],
              [Input('start-packet-btn', 'n_clicks')])
def create_packet_info(btn1):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'start-packet-btn' in changed_id and value_id:
        # get max packet id from sql database
        connection = sqlite3.connect(ocr_db_name)
        cursor = connection.cursor()

        # get max row ID
        query = '''
            SELECT MAX(p."Packet ID") AS "Next ID" FROM CompletePackets p
            '''
        cursor.execute(query)
        # column_names = [x[0] for x in cursor.description]
        extract = cursor.fetchall()
        # extract_df = pd.DataFrame(extract, columns=column_names)
        # next_id = extract_df['Packet ID'].max()
        next_id = extract[0][0]
        if next_id is np.nan:
            next_id = 0
        elif next_id == None:
            next_id = 0
        else:
            next_id = next_id + 1
        connection.close()

        return next_id
    raise PreventUpdate

@app.callback(Output('start-pallet-btn', 'disabled'),
              Input('start-pallet-btn', 'n_clicks'),
              Input('end-pallet-btn', 'n_clicks'))
def disable_start_pallet_btn(btn1, btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'start-pallet-btn' in changed_id:
        return True
    if 'end-pallet-btn' in changed_id:
        return False
    raise PreventUpdate

@app.callback(Output('end-pallet-btn', 'disabled'),
              Input('end-pallet-btn', 'n_clicks'),
              Input('start-packet-btn', 'disabled'))
def disable_end_pallet_btn(btn1, packet_complete):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'end-pallet-btn' in changed_id:
        return True
    if packet_complete == True:
        return True
    return False

@app.callback(Output('start-packet-btn', 'disabled'),
              Input('start-packet-btn', 'n_clicks'),
              Input('end-packet-btn', 'n_clicks'),
              Input('discard-packet-btn', 'n_clicks'))
def disable_start_packet_btn(btn1, btn2, btn3):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'start-packet-btn' in changed_id and value_id:
        return True
    if 'end-packet-btn' in changed_id and value_id:
        return False
    if 'discard-packet-btn' in changed_id and value_id:
        return False
    raise PreventUpdate

@app.callback(Output('end-packet-btn', 'disabled'),
              Input('start-packet-btn', 'n_clicks'),
              Input('end-packet-btn', 'n_clicks'),
              Input('discard-packet-btn', 'n_clicks'))
def disable_end_packet_btn(btn1, btn2, btn3):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'start-packet-btn' in changed_id and value_id:
        return False
    if 'end-packet-btn' in changed_id and value_id:
        return True
    if 'discard-packet-btn' in changed_id and value_id:
        return True
    raise PreventUpdate

@app.callback(Output('discard-packet-btn', 'disabled'),
              Input('start-packet-btn', 'n_clicks'),
              Input('end-packet-btn', 'n_clicks'),
              Input('discard-packet-btn', 'n_clicks'))
def disable_discard_packet_btn(btn1, btn2, btn3):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'start-packet-btn' in changed_id and value_id:
        return False
    if 'end-packet-btn' in changed_id and value_id:
        return True
    if 'discard-packet-btn' in changed_id and value_id:
        return True
    raise PreventUpdate

###################################################################

@app.callback(Output('1348-resolution', 'src'),
              Output('1348-resolution-check', 'children'),
              Output('new-1348-filename', 'children'),
              Output('1348-begin-processing-check', 'children'),
              Input('upload-1348', 'contents'),
              Input('upload-1348', 'filename'))
def upload_1348_resolution_check(contents, filename):
    if filename:
        # filetype = filename.split('.')[-1]
        # generate a new filename so no duplicates; store original filename in var for expediting ocr
        new_filename = generate_random_filename() + '.jpg'
    if contents:
        # decode image to numpy array
        decoded_image = contents.split(',')[1]
        decoded_image = base64.decodebytes(decoded_image.encode('ascii'))
        nparr = np.frombuffer(decoded_image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        begin_check = html.P('Beginning 1348 form processing...', style={'padding-bottom':'0rem', 'margin-top':'0rem', 'margin-bottom':'0rem', 'font-weight':'bold'})
        if not resolution(img_np):
            resolution_check = dbc.Alert([
                html.I(className="bi bi-x-octagon-fill me-2"),
                'This image is too low resolution for text processing. Please retake it.',
                ],
                className="d-flex align-items-center",
                color='danger',
                style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
                )
            return None, resolution_check, None, begin_check
        else:
            resolution_check = html.P('Resolution check passed, checking for blur...', style={'padding-bottom':'0rem', 'margin-top':'0rem', 'margin-bottom':'0rem'})
            return contents, resolution_check, new_filename, begin_check
    raise PreventUpdate

@app.callback(Output('1348-blur', 'src'),
              Output('1348-blur-check', 'children'),
              Input('1348-resolution', 'src'))
def upload_1348_blur_detection(contents):
    if contents:
        # decode image to numpy array
        decoded_image = contents.split(',')[1]
        decoded_image = base64.decodebytes(decoded_image.encode('ascii'))
        nparr = np.frombuffer(decoded_image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if not blur_detection(img_np):
            blur_alert = dbc.Alert([
                html.I(className="bi bi-x-octagon-fill me-2"),
                'This image is too blurry for text processing. Please retake it.',
                ],
                className="d-flex align-items-center",
                color='danger',
                style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
                )
            return None, blur_alert
        else:
            blur_alert = html.P('Blur check passed, looking for form template...', style={'padding-bottom':'0rem', 'margin-top':'0rem', 'margin-bottom':'0rem'})
            return contents, blur_alert
    raise PreventUpdate

@app.callback(Output('1348-template', 'src'),
              Output('1348-template-check', 'children'),
              Output('1348-template-to-use', 'children'),
              Output('template-scale', 'children'),
              Input('1348-blur', 'src'),
              State('new-1348-filename', 'children'))
def upload_1348_template_alignment(contents, filename):
    if contents:
        # decode image to numpy array
        decoded_image = contents.split(',')[1]
        decoded_image = base64.decodebytes(decoded_image.encode('ascii'))
        nparr = np.frombuffer(decoded_image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        for t in available_templates:
            # align image to template
            if blobSwitch:
                template = blobStorage.downloadSpecificImageBlob(t, templateimageContainer)
            else:
                template = cv2.imread(template_parameters + t)
            template_name = t.split('.')[0]

            scale = 1500
            scale_template = template.shape[0]/scale

            image = imutils.resize(image, height = scale)
            template = imutils.resize(template, height = scale)

            orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
            (kpsA, descsA) = orb.detectAndCompute(image, None)
            (kpsB, descsB) = orb.detectAndCompute(template, None)

            method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
            matcher = cv2.DescriptorMatcher_create(method)
            matches = matcher.match(descsA, descsB, None)

            # sort the matches by their distance (the smaller the distance,
            # the "more similar" the features are)
            matches = sorted(matches, key=lambda x:x.distance)

            # keep only the top matches
            keepPercent = 0.2
            keep = int(len(matches) * keepPercent)
            matches = matches[:keep]

            # check to see if we should visualize the matched keypoints
            debug = True
            if debug:
                matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
                matches, None)
                matchedVis = imutils.resize(matchedVis, width=1000)

            ptsA = np.zeros((len(matches), 2), dtype="float")
            ptsB = np.zeros((len(matches), 2), dtype="float")

            # loop over the top matches
            for (i, m) in enumerate(matches):
                # indicate that the two keypoints in the respective images
                # map to each other
                ptsA[i] = kpsA[m.queryIdx].pt
                ptsB[i] = kpsB[m.trainIdx].pt

            # compute the homography matrix between the two sets of matched points
            (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

            (h, w) = template.shape[:2]
            aligned = cv2.warpPerspective(image, H, (w, h))

            overlay = template.copy()
            output = aligned.copy()
            cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

            has_text = pytesseract.image_to_string(aligned)
            if has_text:
                break

        if not has_text:
            alignment_error = dbc.Alert([
                html.I(className="bi bi-x-octagon-fill me-2"),
                'We cannot align this image to any of our 1348 templates. Please retake the image.',
                ],
                className="d-flex align-items-center",
                color='danger',
                style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
                )
            return None, alignment_error, None, None
        else:
            alignment_error = html.P('Matching 1348 template detected, extracting text...', style={'padding-bottom':'0rem', 'margin-top':'0rem', 'margin-bottom':'0rem'})

            # pass aligned image into src and save
            if blobSwitch:
                # upload aligned image to blob storage
                success, im_arr = cv2.imencode('.jpg', aligned)
                im_bytes = im_arr.tobytes()
                blobStorage.addImageBlobs(alignedContainer, filename, im_bytes)
                aligned_img_encoded = base64.b64encode(im_bytes)

            else:
                # write aligned image
                cv2.imwrite(aligned_photos + filename, aligned)
                aligned_img_encoded = base64.b64encode(open(aligned_photos + filename, 'rb').read())
            return 'data:image/png;base64,{}'.format(aligned_img_encoded.decode()), alignment_error, template_name, scale_template
    raise PreventUpdate

@app.callback(Output('1348-extract', 'children'),
              Output('1348-extract-check', 'children'),
              Input('1348-template', 'src'),
              Input('1348-template-to-use', 'children'),
              Input('template-scale', 'children'),
              State('packet-id', 'children'),
              State('pallet-id', 'children'),
              State('new-1348-filename', 'children'))
def complete_1348_text_extraction(contents, template_name, scale_template, packet_id, pallet_id, filename):
    if contents and template_name:
        # convert contents src to aligned img
        # decode image to numpy array
        decoded_image = contents.split(',')[1]
        decoded_image = base64.decodebytes(decoded_image.encode('ascii'))
        nparr = np.frombuffer(decoded_image, np.uint8)
        aligned = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # get template parameters
        if blobSwitch:
            template_filename = template_name + '.json'
            print(f'Downloading {template_filename} from {templatejsonContainer.container_name} to memory...')
            parameters = blobStorage.downloadSpecificDictBlob(template_filename, templatejsonContainer)
        else:
            file_path = template_parameters + template_name + '.json'
            with open(file_path) as file:
                data = file.read()
                parameters = json.loads(data)
                
        # begin text extraction
        output = {}
        for loc in parameters:
            if speed_it_up: # speed it up by only extracting labeled text as needed
                if loc.get('Label') not in speed_it_up_columns:
                    continue
                
            x0 = int(loc.get('x0') / scale_template)
            y0 = int(loc.get('y0') / scale_template)
            x1 = int(loc.get('x1') / scale_template)
            y1 = int(loc.get('y1') / scale_template)

            img = aligned[y0:y1, x0:x1]
            has_text = get_pixel_diff(img)
            if has_text:
                text = extract_text(img, 5, barcode=loc.get('Barcode'), field_type=loc.get('Field Category'))

                # 1348 rules
                if loc.get('Label') == 'Document Identity':
                    # remove lowercase letters
                    text = re.sub('[a-z]', '', text)
                    if text == 'ASJ':
                        text = 'A5J'
                    if text == 'ASA':
                        text = 'A5A'
                output[loc.get('Label')] = cleanup_text(text)

        if speed_it_up: # extract all text as a block for searchability
            # black and white color 
            img1 = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            # add blur
            img1 = cv2.GaussianBlur(img1, (5, 5), 0)
            # extract all text as a block
            all_text = pytesseract.image_to_string(img1)

        timestamp = dt.datetime.now().strftime('%m%d%Y_%H%M')
        # save to database if not there already - complete packets
        connection = sqlite3.connect(ocr_db_name)
        cursor = connection.cursor()
        query = '''
            INSERT OR IGNORE INTO CompletePackets ("Pallet ID", "Packet ID", "Packet Processing Time")
            VALUES (?, ?, ?)
        '''
        cursor.execute(query, [pallet_id, packet_id, timestamp])
        connection.commit()
        connection.close()

        # save to database - 1348 table
        cols = ['Pallet ID', 'Packet ID', 'Filename', 'Extraction Time']
        timestamp = dt.datetime.now().strftime('%m%d%Y_%H%M')
        values = [pallet_id, packet_id, filename, timestamp]

        # reformat the output as needed 
        for k, v in output.items():
            cols.append(k)
            values.append(v)
            
        if speed_it_up:
            cols.append('All 1348 Fields')
            values.append(all_text)
        
        # required for the query
        questions = '(?'
        for i in range(len(values)-1):
            questions = questions +', ?'
        questions = questions + ')'

        connection = sqlite3.connect(ocr_db_name)
        cursor = connection.cursor()
        query = '''
            INSERT INTO Data1348 {}
            VALUES {}
        '''.format(tuple(cols), questions)
        cursor.execute(query, values)
        connection.commit()
        connection.close()

        # save ocr to json for d365
        dOut = {}
        dOut['Pallet ID'] = pallet_id
        dOut['Packet ID'] = packet_id
        dOut['Filename'] = filename
        dOut['Form Save Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dOut['OCR Results'] = output
        blobName = '{}_{}_{}.json'.format(pallet_id, packet_id, '1348-A')

        if blobSwitch:
            dOut = json.dumps(dOut).encode('utf-8')
            d365Container.upload_blob(blobName, dOut, overwrite=True)
        else:
            with open(d365_files + blobName, 'w') as file:
                json.dump(dOut, file)

        extraction_complete = html.P('Text extraction complete, matching to due-in report...', style={'padding-bottom':'0rem', 'margin-top':'0rem', 'margin-bottom':'0rem'})
        return str(output), extraction_complete
    raise PreventUpdate

@app.callback(Output('1348-match', 'children'),
              Output('1348-match-check', 'children'),
              Input('1348-extract', 'children'),
              Input('duein-report', 'data'),
              State('pallet-id', 'children'),
              State('packet-id', 'children'))
def find_duein_match(output, duein, pallet_id, packet_id):
    # print(duein)
    if output:
        duein = pd.read_json(duein, orient='split')
        output = ast.literal_eval(output)
        overlapping_cols = [c for c in duein_1348_overlapping if c in duein.columns and c in output.keys()]
        # compare values for each col
        for col in overlapping_cols:
            # fuzzy matching
            top_matches = process.extract(str(output.get(col)), duein[col].unique().tolist(), limit=None)
            for m, v in top_matches:
                duein.loc[duein[col]==m, '{} Match Confidence'.format(col)] = v
                # if there are no matches, set confidence to 0
                if type(m)==float and math.isnan(m):
                    duein['{} Match Confidence'.format(col)] = 0.0

        # weighted average for matching columns
        # number of cols to evaluate could range from 8-11; rank and weight columns
        weights = {'Document Number':10, 'NSN':4, 'Quantity':3, 'Item Nomenclature':2}
        weights = {k:v for k, v in weights.items() if k in overlapping_cols} # remove weight if not in overlapping cols
        weights = {k:v for k, v in weights.items() if k in output.keys()} # remove weights if they're not in the output
        remaining_cols = [c for c in overlapping_cols if c not in weights.keys()]
        # remaining_weight = (1 - sum(weights.values()))/len(remaining_cols)
        for c in remaining_cols:
            weights[c] = 1

        def weighted_match_score(row):
            match_score = sum([row['{} Match Confidence'.format(c)]*weights[c] for c in overlapping_cols])/sum(weights.values())
            return match_score


        duein['Weighted Match Likelihood'] = duein.apply(weighted_match_score, axis=1)
        duein = duein[[c for c in duein.columns if 'Match Confidence' not in c]]
        # duein_output = duein[overlapping_cols + ['Weighted Match Likelihood']].sort_values('Weighted Match Likelihood', ascending=False)

        if duein['Weighted Match Likelihood'].max() > 90:

            match = duein.loc[duein['Weighted Match Likelihood'] == duein['Weighted Match Likelihood'].max()].to_dict('records')[0]

            # save as json file for d365
            filename = '{}_{}_{}.json'.format(pallet_id, packet_id, 'due-in-match')
            if blobSwitch:
                dOut = json.dumps(match).encode('utf-8')
                d365Container.upload_blob(filename, dOut, overwrite=True)
            else:
                with open(d365_files + filename, 'w') as file:
                    json.dump(match, file)

            match_level = dbc.Alert([
                html.I(className="bi bi-check-circle-fill me-2"),
                'Matching due-in record identified! Data sent to D365.'
                ],
                className="d-flex align-items-center",
                color='success',
                style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
                )
        else:
            # add item to frustrated items sql database
            connection = sqlite3.connect(ocr_db_name)
            cursor = connection.cursor()
            query = '''
                INSERT INTO FrustratedItems ("Pallet ID", "Packet ID")
                VALUES (?, ?)
            '''
            cursor.execute(query, [pallet_id, packet_id])
            connection.commit()
            connection.close()

            match = {} # create empty match object

            match_level = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                'This item is not on the due-in report. We\'ll store the packet in our database, but you cannot receive it into D365.'
                ],
                className="d-flex align-items-center",
                color='warning',
                style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
                )
        return str(match), match_level
    raise PreventUpdate

@app.callback(Output('1348-spinner', 'children'),
              Input('upload-1348', 'contents'),
              Input('1348-match-check', 'children'))
def show_1348_spinner(start, stop):
    if stop:
        return [
            dbc.Alert([
                html.I(className="bi bi-check-circle-fill me-2"),
                '1348 processing complete.'
                ],
                className="d-flex align-items-center",
                color='success',
                style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
                )
            ]
    if start:
        return dbc.Row([
            dbc.Col([
                dbc.Spinner(spinner_style={"width": "2rem", "height": "2rem", 'margin-top':'0.5rem', 'color':'{}'.format(kpmg_blue)})
                ], className='col-sm-auto'),
            dbc.Col([
                html.P('Please wait while this 1348 is processed. You can view step-by-step alerts to the right.'),
                ])
            ])
    return None

@app.callback(Output('delete-discarded-rows', 'children'),
              Input('discard-packet-btn', 'n_clicks'),
              State('packet-id', 'children'))
def delete_discarded_packets(btn, packet_id):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'discard-packet-btn' in changed_id and value_id:
        # delete existing rows for that packet
        connection = sqlite3.connect(ocr_db_name)

        query = '''
            DELETE FROM CompletePackets
            WHERE "Packet ID" = {}
        '''.format(packet_id)
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()

        query = '''
            DELETE FROM Data1348
            WHERE "Packet ID" = {}
        '''.format(packet_id)
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()

        query = '''
            DELETE FROM AdditionalForms
            WHERE "Packet ID" = {}
        '''.format(packet_id)
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()

        query = '''
            DELETE FROM FrustratedItems
            WHERE "Packet ID" = {}
        '''.format(packet_id)
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()

        connection.close()

###################################################################
# allow user to upload additional forms; continue processing and adding as searchable until done
# upload form > assign form uid > make searchable

@app.callback(Output('additional-form-modal', 'is_open'),
              Input('upload-additional', 'contents'),
              Input('close-additional-form-modal', 'n_clicks'))
def open_additional_forms_modal(btn1, btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'upload-additional' in changed_id and value_id:
        return True
    return False

@app.callback(Output('preview-additional-form', 'src'),
              Input('upload-additional', 'contents'),
              Input('rotate-left', 'n_clicks'),
              Input('rotate-right', 'n_clicks'),
              State('preview-additional-form', 'src'))
def show_additional_form_modal_content(content, rt_left, rt_right, state):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'upload-additional' in changed_id and value_id:
        return content
    if 'rotate-left' in changed_id and value_id:
        if state:
            img = state
        else:
            img = content
        decoded_image = img.split(',')[1]
        decoded_image = base64.decodebytes(decoded_image.encode('ascii'))
        nparr = np.frombuffer(decoded_image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rotated = rotate(image, 90, resize=True) * 255

        # convert photo to bytes
        success, im_arr = cv2.imencode('.jpg', rotated)
        im_bytes = im_arr.tobytes()
        encoded_img = base64.b64encode(im_bytes)
        return 'data:image/png;base64,{}'.format(encoded_img.decode())
    if 'rotate-right' in changed_id and value_id:
        if state:
            img = state
        else:
            img = content
        decoded_image = img.split(',')[1]
        decoded_image = base64.decodebytes(decoded_image.encode('ascii'))
        nparr = np.frombuffer(decoded_image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rotated = rotate(image, -90, resize=True) * 255

        # convert photo to bytes
        success, im_arr = cv2.imencode('.jpg', rotated)
        im_bytes = im_arr.tobytes()
        encoded_img = base64.b64encode(im_bytes)
        return 'data:image/png;base64,{}'.format(encoded_img.decode())
    return None

@app.callback(Output('additional-form-resolution', 'src'),
              Output('additional-form-resolution-check', 'children'),
              Output('additional-form-filename', 'children'),
              Output('additional-form-begin-check', 'children'),
              Input('preview-additional-form', 'src'),
              Input('upload-additional', 'filename'),
              Input('additional-form-modal', 'is_open'))
def additional_form_resolution(contents, filename, modal):
    if filename:
        # filetype = filename.split('.')[-1]
        # generate a new filename so no duplicates; store original filename in var for expediting ocr
        new_filename = generate_random_filename() + '.jpg'
    if modal == False and contents:
        # decode image to numpy array
        decoded_image = contents.split(',')[1]
        decoded_image = base64.decodebytes(decoded_image.encode('ascii'))
        nparr = np.frombuffer(decoded_image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        begin_check = html.P('Beginning additional form processing...', style={'padding-bottom':'0rem', 'margin-top':'0rem', 'margin-bottom':'0rem', 'font-weight':'bold'})
        if not resolution(img_np):
            resolution_check = dbc.Alert([
                html.I(className="bi bi-x-octagon-fill me-2"),
                'This image is too low resolution for text processing. Please retake it.',
                ],
                className="d-flex align-items-center",
                color='danger',
                style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
                )
            return None, resolution_check, None, begin_check
        else:
            resolution_check = html.P('Resolution check passed, checking for blur...', style={'padding-bottom':'0rem', 'margin-top':'0rem', 'margin-bottom':'0rem'})
            return contents, resolution_check, new_filename, begin_check
    raise PreventUpdate

@app.callback(Output('additional-form-blur', 'src'),
              Output('additional-form-blur-check', 'children'),
              Input('additional-form-resolution', 'src'))
def additional_form_blur(contents):
    if contents:
        # decode image to numpy array
        decoded_image = contents.split(',')[1]
        decoded_image = base64.decodebytes(decoded_image.encode('ascii'))
        nparr = np.frombuffer(decoded_image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if not blur_detection(img_np):
            blur_alert = dbc.Alert([
                html.I(className="bi bi-x-octagon-fill me-2"),
                'This image is too blurry for text processing. Please retake it.',
                ],
                className="d-flex align-items-center",
                color='danger',
                style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
                )
            return None, blur_alert
        else:
            blur_alert = html.P('Blur check passed, extracting text...', style={'padding-bottom':'0rem', 'margin-top':'0rem', 'margin-bottom':'0rem'})
            return contents, blur_alert
    raise PreventUpdate

@app.callback(Output('additional-form-extract', 'children'),
              Output('additional-form-extract-check', 'children'),
              Input('additional-form-blur', 'src'),
              State('packet-id', 'children'),
              State('pallet-id', 'children'),
              State('additional-form-filename', 'children'))
def additional_form_extraction(contents, packet_id, pallet_id, filename):
    if contents:
        # decode image to numpy array
        decoded_image = contents.split(',')[1]
        decoded_image = base64.decodebytes(decoded_image.encode('ascii'))
        nparr = np.frombuffer(decoded_image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # black and white color
        img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        # add blur
        img = cv2.GaussianBlur(img, (5, 5), 0)
        # extract all text as block
        all_text = pytesseract.image_to_string(img)

        if all_text:
            # add image of form to blob storage/save file
            if blobSwitch:
                # upload image to blob storage
                blobStorage.addImageBlobs(alignedContainer, filename, decoded_image)
            else:
                # write image
               with open(aligned_photos + filename , 'wb') as f:
                   f.write(decoded_image)

            timestamp = dt.datetime.now().strftime('%m%d%Y_%H%M')
            # save to database if not there already - complete packets
            connection = sqlite3.connect(ocr_db_name)
            cursor = connection.cursor()
            query = '''
                INSERT OR IGNORE INTO CompletePackets ("Pallet ID", "Packet ID", "Packet Processing Time")
                VALUES (?, ?, ?)
            '''
            cursor.execute(query, [pallet_id, packet_id, timestamp])
            connection.commit()
            connection.close()

            timestamp = dt.datetime.now().strftime('%m%d%Y_%H%M')
            # insert extracted data into adtnl forms database
            connection = sqlite3.connect(ocr_db_name)
            cursor = connection.cursor()
            query = '''
                INSERT INTO AdditionalForms ('Pallet ID', 'Packet ID', 'Extraction Time', 'Filename', 'Text')
                VALUES (?, ?, ?, ?, ?)
            '''
            cursor.execute(query, [pallet_id, packet_id, timestamp, filename, all_text])
            connection.commit()
            connection.close()

            # save searchable pdf for download


            extract_alert = dbc.Alert([
                html.I(className="bi bi-check-circle-fill me-2"),
                'Text extraction complete. Form added to the searchable database. You can process additional forms the same way this form was processed.'
                ],
                className="d-flex align-items-center",
                color='success',
                style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
                )
            return all_text, extract_alert
        else:
            extract_alert = dbc.Alert([
                html.I(className="bi bi-x-octagon-fill me-2"),
                'No text was detected for this image. Consider retaking the image or processing another form.',
                ],
                className="d-flex align-items-center",
                color='danger',
                style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
                )
            return None, extract_alert

@app.callback(Output('all-processed-forms', 'children'),
              Output('packet-contents', 'children'),
              Input('additional-form-extract', 'children'),
              Input('1348-extract', 'children'),
              State('all-processed-forms', 'children'))
def all_processed_forms(additional_forms, f1348, state):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    # print('Starting state: {}'.format(state))
    if not state:
        state = {
            '1348':0,
            'Additional Forms':0
            }
    if '1348-extract' in changed_id and value_id and f1348:
        # print('Adding 1348')
        state['1348'] += 1

    if 'additional-form-extract' in changed_id and value_id and additional_forms:
        # print('Adding additional form')
        state['Additional Forms'] += 1

    if state['1348'] > 0 and state['Additional Forms'] > 0:
        contents = dbc.Alert([
            html.I(className="bi bi-info-circle-fill me-2"),
            'This packet a processed 1348 form and {} additional processed forms.'.format(state['Additional Forms'])
            ],
            className="d-flex align-items-center",
            color='info',
            style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
            )
    elif state['1348'] > 0 and state['Additional Forms'] == 0:
        contents = dbc.Alert([
            html.I(className="bi bi-info-circle-fill me-2"),
            'This packet has a processed 1348 form but no additional processed forms.'
            ],
            className="d-flex align-items-center",
            color='info',
            style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
            )
    elif state['1348'] == 0 and state['Additional Forms'] > 0:
        contents = dbc.Alert([
            html.I(className="bi bi-info-circle-fill me-2"),
            'This packet does not have a processed 1348 form and has {} additional processed forms.'.format(state['Additional Forms'])
            ],
            className="d-flex align-items-center",
            color='info',
            style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
            )
    else:
        contents = dbc.Alert([
            html.I(className="bi bi-info-circle-fill me-2"),
            'This packet has no extracted content yet.'
            ],
            className="d-flex align-items-center",
            color='info',
            style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
            )
    # print('Ending state: {}'.format(state))
    return state, contents


@app.callback(Output('additional-form-status', 'children'),
              Input('upload-additional', 'contents'),
              Input('additional-form-extract', 'children'))
def show_additional_form_spinner(start, stop):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'additional-form-extract' in changed_id and value_id:
        return [
            dbc.Alert([
                html.I(className="bi bi-check-circle-fill me-2"),
                'Additional form processing complete.'
                ],
                className="d-flex align-items-center",
                color='success',
                style={'padding-bottom':'0.5rem', 'padding-top':'0.5rem'}
                )
            ]
    if 'upload-additional' in changed_id and value_id:
        return dbc.Row([
            dbc.Col([
                dbc.Spinner(spinner_style={"width": "2rem", "height": "2rem", 'margin-top':'0.5rem', 'color':'{}'.format(kpmg_blue)})
                ], className='col-sm-auto'),
            dbc.Col([
                html.P('Please wait while this form is processed. You can view step-by-step alerts to the right.'),
                ])
            ])
    return None

@app.callback(Output('additional-form-resolution-check', 'children'),
              Input('upload-additional', 'contents'),
              State('additional-form-resolution-check', 'children'))
def reset_additional_form_alerts_1(content, state):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'upload-additional' in changed_id and value_id:
        return None
    return state

@app.callback(Output('additional-form-blur-check', 'children'),
              Input('upload-additional', 'contents'),
              State('additional-form-blur-check', 'children'))
def reset_additional_form_alerts_2(content, state):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'upload-additional' in changed_id and value_id:
        return None
    return state

@app.callback(Output('additional-form-extract-check', 'children'),
              Input('upload-additional', 'contents'),
              State('additional-form-extract-check', 'children'))
def reset_additional_form_alerts_3(content, state):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'upload-additional' in changed_id and value_id:
        return None
    return state

@app.callback(Output('preview-packet-btn', 'disabled'),
              Input('all-processed-forms', 'children'))
def enable_packet_preview(packet):
    if packet['1348'] > 0 or packet['Additional Forms'] > 0:
        return False
    return True

@app.callback(Output('preview-modal', 'is_open'),
              Input('preview-packet-btn', 'n_clicks'),
              Input('close-preview-modal', 'n_clicks'),)
def open_preview_modal(btn1, btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'preview-packet-btn' in changed_id and value_id:
        return True
    return False

@app.callback(Output('preview-content', 'children'),
              Input('preview-modal', 'is_open'),
              State('packet-id', 'children'))
def show_packet_images(modal, packet_id):
    if modal == True:

        # get all files for a given packet
        connection = sqlite3.connect(ocr_db_name)
        cursor = connection.cursor()
        query = '''
            SELECT
            CASE WHEN d.Filename is not NULL THEN "1348" ELSE "" END AS "Type",
            d.Filename
            FROM CompletePackets p
            LEFT JOIN Data1348 d
            ON d."Packet ID" = p."Packet ID"
            WHERE p."Packet ID" = {}
            UNION
            SELECT
            CASE WHEN a.Filename is not NULL THEN "Additional Form" ELSE "" END AS "Type",
            a.Filename
            FROM CompletePackets p
            LEFT JOIN AdditionalForms a
            ON a."Packet ID" = p."Packet ID"
            WHERE p."Packet ID" = {}
            '''.format(packet_id, packet_id)
        cursor.execute(query)
        column_names = [x[0] for x in cursor.description]
        extract = cursor.fetchall()
        df = pd.DataFrame(extract, columns=column_names).dropna()
        connection.close()

        items = []
        for idx, row in df.iterrows():
            if blobSwitch:
                print('TRYING TO LOAD IN FILE')
                print('Downloading {} from {} to memory...'.format(row['Filename'], alignedContainer.container_name))
                print(row['Filename'])
                image = alignedContainer.download_blob(row['Filename']).readall()
                encoded_image = base64.b64encode(image)
                decoded_image = 'data:image/png;base64,{}'.format(encoded_image.decode())
            else:
                encoded_image = base64.b64encode(open(aligned_photos + row['Filename'], 'rb').read())
                decoded_image = 'data:image/png;base64,{}'.format(encoded_image.decode())
            items.append({
                'key':idx,
                'src':decoded_image,
                'header':row['Type']
                })
        carousel = dbc.Carousel(
           controls=True,
           indicators=True,
           items=items,
           interval=False,
           )
        return carousel

@app.callback(Output('download-1348-xlsx-btn', 'disabled'),
              Input('all-processed-forms', 'children'))
def enable_1348_xlsx_btn(content):
    if content['1348'] > 0:
        return False
    return True

@app.callback(Output('download-packet-pdf', 'disabled'),
              Input('all-processed-forms', 'children'))
def enable_download_pdf_btn(content):
    if content['1348'] > 0 or content['Additional Forms'] > 0:
        return False
    return True

@app.callback(Output('download-1348-xlsx', 'data'),
              Input('download-1348-xlsx-btn', 'n_clicks'),
              State('packet-id', 'children'))
def download_1348_xlsx(btn, packet_id):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'download-1348-xlsx-btn' in changed_id and value_id:
        # query data from sql database

        connection = sqlite3.connect(ocr_db_name)
        cursor = connection.cursor()
        query = '''
            SELECT
            *
            FROM Data1348 d
            WHERE d."Packet ID" = {}
            '''.format(packet_id)
        cursor.execute(query)
        column_names = [x[0] for x in cursor.description]
        extract = cursor.fetchall()
        df = pd.DataFrame(extract, columns=column_names)
        connection.close()

        df = df.drop(columns=['Pallet ID', 'Packet ID', 'Filename'])

        def to_xlsx(bytes_io):
            xslx_writer = pd.ExcelWriter(bytes_io, engine='xlsxwriter')
            df.to_excel(xslx_writer, index=False, sheet_name='1348_extract')
            xslx_writer.save()

        timestamp = dt.datetime.now().strftime('%m%d%Y_%H%M')
        return send_bytes(to_xlsx, '1348_PREParE_{}.xlsx'.format(timestamp))

@app.callback(Output('download-packet-pdf', 'data'),
              Input('download-packet-pdf-btn', 'n_clicks'),
              State('packet-id', 'children'))
def download_packet_pdf(btn, packet_id):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'download-packet-pdf-btn' in changed_id and value_id:

        # use sql database to get filenames for each img in packet
        connection = sqlite3.connect(ocr_db_name)
        cursor = connection.cursor()
        query = '''
            SELECT
            d.Filename
            FROM CompletePackets p
            LEFT JOIN Data1348 d
            ON d."Packet ID" = p."Packet ID"
            WHERE p."Packet ID" = {}
            UNION
            SELECT
            a.Filename
            FROM CompletePackets p
            LEFT JOIN AdditionalForms a
            ON a."Packet ID" = p."Packet ID"
            WHERE p."Packet ID" = {}
            '''.format(packet_id, packet_id)
        cursor.execute(query)
        column_names = [x[0] for x in cursor.description]
        extract = cursor.fetchall()
        df = pd.DataFrame(extract, columns=column_names)
        connection.close()
        df = df[df['Filename'].notna()]

        filenames = df['Filename'].unique().tolist()

        # create a pdf of all documents
        merger = PdfFileMerger()

        # pull aligned photo in and convert to searchable pdf
        for idx, f in enumerate(filenames):
            if blobSwitch:
                image = blobStorage.downloadSpecificImageBlob(f, alignedContainer)
            else:
                image = cv2.imread(aligned_photos + f)
            PDF = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')

            pdf_buffer = io.BytesIO()
            pdf_buffer.write(PDF)
            merger.append(pdf_buffer)

        # Write merged data into memory file
        temp_file = io.BytesIO()
        merger.write(temp_file)

        # for export
        def to_pdf(bytes_io):
            bytes_io.write(temp_file.getbuffer())

        timestamp = dt.datetime.now().strftime('%m%d%Y_%H%M')
        return send_bytes(to_pdf, 'PREParE_{}.pdf'.format(timestamp))

###################################################################

@app.callback(Output('last-uploaded-report', 'children'),
              Input('url', 'pathname'),
              Input('process-document-btn', 'n_clicks'))
def find_last_uploaded_report(url, btn):
    error_in = False
    error_out = False
    try:
        if blobSwitch:
            # get blob modification time
            blob_list = dueinContainer.list_blobs()
            blob_props = []
            blob_names = []
            for blob in blob_list:
                blob_names.append(blob.name)
                blob_props.append(blob.last_modified.astimezone(pytz.timezone('US/Eastern')))
            latest_filedate_duein = max(blob_props)
            latest_filedate_duein = latest_filedate_duein.strftime('%Y-%m-%d %H:%M')
            # latest_report = blob_names[blob_props.index(max(blob_props))]
        else:
            list_of_files = glob.glob(duein_path + '*')
            latest_file_path = max(list_of_files, key=os.path.getctime)
            latest_filedate_duein = cleanup_unixtime(latest_file_path, 'create')
    except:
        error_in = True

    try:
        if blobSwitch:
            # get blob modification time
            blob_list = dueoutContainer.list_blobs()
            blob_props = []
            blob_names = []
            for blob in blob_list:
                blob_names.append(blob.name)
                blob_props.append(blob.last_modified.astimezone(pytz.timezone('US/Eastern')))
            latest_filedate_dueout = max(blob_props)
            latest_filedate_dueout = latest_filedate_dueout.strftime('%Y-%m-%d %H:%M')
            # latest_report = blob_names[blob_props.index(max(blob_props))]
        else:
            list_of_files = glob.glob(dueout_path + '*')
            latest_file_path = max(list_of_files, key=os.path.getctime)
            latest_filedate_dueout = cleanup_unixtime(latest_file_path, 'create')
    except:
        error_out = True
    if error_in and error_out:
        return dcc.Markdown(
            '''
            **There are no due in or due out reports on file.**
            Use the form below to upload a new report.
            ''',
            style={'padding':'0rem', 'margin':'0rem', 'color':kpmg_blue})
    elif error_in and not error_out:
        return dcc.Markdown(
            '''
            **There are no Due In reports on file.**
            The most recent Due Out report on file was uploaded at **{}**.
            Use the form below to upload a new report.
            '''.format(latest_filedate_dueout),
            style={'padding':'0rem', 'margin':'0rem', 'color':kpmg_blue})
    elif not error_in and error_out:
        return dcc.Markdown(
            '''
            The most recent Due In report on file was uploaded at **{}**.
            **There are no Due Out reports on file.**
            Use the form below to upload a new report.
            '''.format(latest_filedate_duein),
            style={'padding':'0rem', 'margin':'0rem', 'color':kpmg_blue})
    else:
        return dcc.Markdown(
            '''
            The most recent Due In report on file was uploaded at **{}**.
            The most recent Due Out report on file was uploaded at **{}**.
            Use the form below to upload a new report.
            '''.format(latest_filedate_duein, latest_filedate_dueout),
            style={'padding':'0rem', 'margin':'0rem', 'color':kpmg_blue})

@app.callback(Output('process-upload-report', 'children'),
              Input('process-document-btn', 'n_clicks'),
              State('upload-report', 'filename'),
              State('upload-report', 'contents'),
              State('upload-report-category', 'value'))
def process_uploaded_report(btn, filename, contents, category):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'process-document-btn' in changed_id:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        timestamp = dt.datetime.now().strftime('%m%d%Y_%H%M')
        if category == 'Due In':
            if blobSwitch:
                blobStorage.addDataFrameBlobs(df, '{}_{}.csv'.format(category, timestamp), dueinContainer)
            else:
                df.to_csv(duein_path + '{}_{}.csv'.format(category, timestamp), index=False)
        elif category == 'Due Out':
            if blobSwitch:
                blobStorage.addDataFrameBlobs(df, '{}_{}.csv'.format(category, timestamp), dueoutContainer)
            else:
                df.to_csv(dueout_path + '{}_{}.csv'.format(category, timestamp), index=False)
        return html.P('Document processing complete.')
    return None

@app.callback(Output('uploaded-report-filename', 'children'),
              Input('upload-report', 'filename'))
def preview_report_filename(filename):
    if filename:
        return 'Uploaded file: {}'.format(filename)
    return None

@app.callback(Output('duein-report', 'data'),
              Input('process-document-btn', 'n_clicks'),
              Input('reload-page', 'n_clicks'),
              State('upload-report', 'filename'),
              State('upload-report', 'contents'),
              State('upload-report-category', 'value'))
def cleanup_duein_report(btn, pg, filename, contents, upload_category):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'process-document-btn' in changed_id and upload_category == 'Due In' and contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    else:
        if blobSwitch:
            # get blob modification time
            blob_list = dueinContainer.list_blobs()
            blob_props = []
            blob_names = []
            for blob in blob_list:
                blob_names.append(blob.name)
                blob_props.append(blob.last_modified.astimezone(pytz.timezone('US/Eastern')))
            latest_report = blob_names[blob_props.index(max(blob_props))]
            print(f'Downloading {latest_report} from {dueinContainer.container_name} to memory...')
            df_encoded = dueinContainer.download_blob(latest_report).readall()
            data = io.StringIO(df_encoded.decode('utf-8'))
            df = pd.read_csv(data)
        else:
            list_of_files = glob.glob(duein_path + '*')
            latest_file_path = max(list_of_files, key=os.path.getctime)
            df = pd.read_csv(latest_file_path)

    # df = df[due_in_columns]
    for c in df.columns:
        df[c] = df[c].astype(str)
    df = df.rename(columns=duein_column_map)
    df['NSN'] = df['NSN'].apply(lambda x: '{0:0>13}'.format(x))
    df['id'] = df.index
    return df.to_json(date_format='iso', orient='split')

@app.callback(Output('preview-due-in', 'data'),
              Output('preview-due-in', 'columns'),
              Input('duein-report', 'data'))
def preview_duein(df):
    df = pd.read_json(df, orient='split')
    return df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns if i in duein_1348_overlapping]

###################################################################
@app.callback(Output('prepare-database', 'data'),
              Output('prepare-database', 'columns'),
              Output('prepare-database', 'row_selectable'),
              Output('prepare-database', 'selected_rows'),
              Input('forms-to-search', 'value'),
              Input('frustrated-filter', 'value'),
              Input('record-search', 'value'))
def prepare_database(forms_to_search, frustrated, search_value):
    # pull all data into database
    connection = sqlite3.connect(ocr_db_name)
    cursor = connection.cursor()
    query = '''
        SELECT
        p.*,
        CASE WHEN SUM(CASE WHEN d."Filename" IS NOT NULL THEN 1 ELSE 0 END) > 0 THEN "True" ELSE "False" END AS "1348 Included",
        SUM(CASE WHEN a."Filename" IS NOT NULL THEN 1 ELSE 0 END) AS "Additional Form Count",
        CASE WHEN d."NSN" IS NOT NULL THEN d."NSN" ELSE "N/A" END AS "NSN",
        CASE WHEN d."Document Number" IS NOT NULL THEN d."Document Number" ELSE "N/A" END AS "Document Number",
        CASE WHEN d."Quantity" IS NOT NULL THEN d."Quantity" ELSE "N/A" END AS "Quantity",
        CASE WHEN f."Packet ID" IS NOT NULL THEN "True" ELSE "False" END AS "Frustrated"
        FROM CompletePackets p
        LEFT JOIN Data1348 d
        ON p."Packet ID" = d."Packet ID"
        LEFT JOIN AdditionalForms a
        ON P."Packet ID" = a."Packet ID"
        LEFT JOIN FrustratedItems f
        ON P."Packet ID" = f."Packet ID"
        GROUP BY p."Packet ID"
        '''
    cursor.execute(query)
    column_names = [x[0] for x in cursor.description]
    extract = cursor.fetchall()
    df = pd.DataFrame(extract, columns=column_names)
    connection.close()

    df['Packet Processing Time'] = pd.to_datetime(df['Packet Processing Time'], format='%m%d%Y_%H%M')
    df['Packet Processing Time'] = df['Packet Processing Time'].astype(str)

    if forms_to_search == '1348 Only':
        df = df[df['1348 Included']=='True']

    if frustrated == 'Frustrated Only':
        df = df[df['Frustrated']=='True']

    if frustrated == "Processed Only":
        df = df[df['Frustrated']=='False']

    if search_value:
        search_value = search_value.lower()
        search_result_packets = []

        # check 1348 database
        connection = sqlite3.connect(ocr_db_name)
        cursor = connection.cursor()
        query = '''
            SELECT
            *
            FROM Data1348 d
            '''
        cursor.execute(query)
        column_names = [x[0] for x in cursor.description]
        extract = cursor.fetchall()
        df1 = pd.DataFrame(extract, columns=column_names)
        connection.close()

        mask = np.column_stack([df1[col].astype(str).str.lower().str.contains(search_value, na=False) for col in df1])
        search_results = df1.loc[mask.any(axis=1)]
        search_result_packets = search_result_packets + search_results['Packet ID'].unique().tolist()

        # check additional forms
        connection = sqlite3.connect(ocr_db_name)
        cursor = connection.cursor()
        query = '''
            SELECT
            a."Packet ID",
            a."Text"
            FROM AdditionalForms a
            '''
        cursor.execute(query)
        column_names = [x[0] for x in cursor.description]
        extract = cursor.fetchall()
        df2 = pd.DataFrame(extract, columns=column_names)
        connection.close()

        mask = np.column_stack([df2[col].astype(str).str.lower().str.contains(search_value, na=False) for col in df2])
        search_results = df2.loc[mask.any(axis=1)]
        search_result_packets = search_result_packets + search_results['Packet ID'].unique().tolist()

        # remove duplicates
        search_result_packets = list(dict.fromkeys(search_result_packets))
        df = df[df['Packet ID'].isin(search_result_packets)]

    cols = [{'name': i, 'id': i} for i in df.columns]
    return df.to_dict('records'), cols, 'single', []

@app.callback(Output('preview-packet-db-btn', 'disabled'),
              Input('prepare-database', 'selected_rows'))
def disable_preview_btn_db(selected_rows):
    if selected_rows:
        return False
    return True

@app.callback(Output('preview-modal-db', 'is_open'),
              Input('preview-packet-db-btn', 'n_clicks'),
              Input('close-preview-modal-db', 'n_clicks'),)
def open_preview_db_modal(btn1, btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'preview-packet-db-btn' in changed_id and value_id:
        return True
    return False

@app.callback(Output('preview-content-db', 'children'),
              Input('preview-modal-db', 'is_open'),
              State('prepare-database', 'selected_rows'),
              State('prepare-database', 'data'))
def show_db_packet_images(modal, selected_rows, data):
    if modal == True:

        df = pd.DataFrame(data)
        df = df.iloc[selected_rows[0]]
        packet_id = df['Packet ID']

        # get all files for a given packet
        connection = sqlite3.connect(ocr_db_name)
        cursor = connection.cursor()
        query = '''
            SELECT
            CASE WHEN d.Filename is not NULL THEN "1348" ELSE "" END AS "Type",
            d.Filename
            FROM CompletePackets p
            LEFT JOIN Data1348 d
            ON d."Packet ID" = p."Packet ID"
            WHERE p."Packet ID" = {}
            UNION
            SELECT
            CASE WHEN a.Filename is not NULL THEN "Additional Form" ELSE "" END AS "Type",
            a.Filename
            FROM CompletePackets p
            LEFT JOIN AdditionalForms a
            ON a."Packet ID" = p."Packet ID"
            WHERE p."Packet ID" = {}
            '''.format(packet_id, packet_id)
        cursor.execute(query)
        column_names = [x[0] for x in cursor.description]
        extract = cursor.fetchall()
        df = pd.DataFrame(extract, columns=column_names).dropna()
        connection.close()

        items = []
        for idx, row in df.iterrows():
            if blobSwitch:
                print('Downloading {} from {} to memory...'.format(row['Filename'], alignedContainer.container_name))
                image = alignedContainer.download_blob(row['Filename']).readall()
                encoded_image = base64.b64encode(image)
                decoded_image = 'data:image/png;base64,{}'.format(encoded_image.decode())
            else:
                encoded_image = base64.b64encode(open(aligned_photos + row['Filename'], 'rb').read())
                decoded_image = 'data:image/png;base64,{}'.format(encoded_image.decode())
            items.append({
                'key':idx,
                'src':decoded_image,
                'header':row['Type']
                })
        carousel = dbc.Carousel(
            controls=True,
            indicators=True,
            items=items,
            interval=False,
            )
        return carousel

@app.callback(Output('download-1348-xlsx-btn-db', 'disabled'),
              Input('prepare-database', 'selected_rows'),
              State('prepare-database', 'data'))
def enable_1348_xlsx_db_btn(selected_rows, data):
    if selected_rows:
        df = pd.DataFrame(data)
        df = df.iloc[selected_rows[0]]
        if df['1348 Included'] == 'True':
            return False
        return True

@app.callback(Output('download-1348-xlsx-db', 'data'),
              Input('download-1348-xlsx-btn-db', 'n_clicks'),
              State('prepare-database', 'selected_rows'),
              State('prepare-database', 'data'))
def download_1348_xlsx_db(btn, selected_rows, data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'download-1348-xlsx-btn-db' in changed_id and value_id:
        # query data from sql database
        df = pd.DataFrame(data)
        df = df.iloc[selected_rows[0]]
        packet_id = df['Packet ID']

        connection = sqlite3.connect(ocr_db_name)
        cursor = connection.cursor()
        query = '''
            SELECT
            *
            FROM Data1348 d
            WHERE d."Packet ID" = {}
            '''.format(packet_id)
        cursor.execute(query)
        column_names = [x[0] for x in cursor.description]
        extract = cursor.fetchall()
        df = pd.DataFrame(extract, columns=column_names)
        connection.close()

        df = df.drop(columns=['Pallet ID', 'Packet ID', 'Filename'])

        def to_xlsx(bytes_io):
            xslx_writer = pd.ExcelWriter(bytes_io, engine='xlsxwriter')
            df.to_excel(xslx_writer, index=False, sheet_name='1348_extract')
            xslx_writer.save()

        timestamp = dt.datetime.now().strftime('%m%d%Y_%H%M')
        return send_bytes(to_xlsx, '1348_PREParE_{}.xlsx'.format(timestamp))

@app.callback(Output('download-packet-pdf-db', 'data'),
              Input('download-packet-pdf-btn-db', 'n_clicks'),
              State('prepare-database', 'selected_rows'),
              State('prepare-database', 'data'))
def download_packet_pdf_db(btn, selected_rows, data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'download-packet-pdf-btn-db' in changed_id and value_id:
        df = pd.DataFrame(data)
        df = df.iloc[selected_rows[0]]
        packet_id = df['Packet ID']

        # use sql database to get filenames for each img in packet
        connection = sqlite3.connect(ocr_db_name)
        cursor = connection.cursor()
        query = '''
            SELECT
            d.Filename
            FROM CompletePackets p
            LEFT JOIN Data1348 d
            ON d."Packet ID" = p."Packet ID"
            WHERE p."Packet ID" = {}
            UNION
            SELECT
            a.Filename
            FROM CompletePackets p
            LEFT JOIN AdditionalForms a
            ON a."Packet ID" = p."Packet ID"
            WHERE p."Packet ID" = {}
            '''.format(packet_id, packet_id)
        cursor.execute(query)
        column_names = [x[0] for x in cursor.description]
        extract = cursor.fetchall()
        df = pd.DataFrame(extract, columns=column_names)
        connection.close()
        df = df[df['Filename'].notna()]

        filenames = df['Filename'].unique().tolist()

        # create a pdf of all documents
        merger = PdfFileMerger()

        # pull aligned photo in and convert to searchable pdf
        for idx, f in enumerate(filenames):
            if blobSwitch:
                image = blobStorage.downloadSpecificImageBlob(f, alignedContainer)
            else:
                image = cv2.imread(aligned_photos + f)
            PDF = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')

            pdf_buffer = io.BytesIO()
            pdf_buffer.write(PDF)
            merger.append(pdf_buffer)

        # Write merged data into memory file
        temp_file = io.BytesIO()
        merger.write(temp_file)

        # for export
        def to_pdf(bytes_io):
            bytes_io.write(temp_file.getbuffer())

        timestamp = dt.datetime.now().strftime('%m%d%Y_%H%M')
        return send_bytes(to_pdf, 'PREParE_{}.pdf'.format(timestamp))

@app.callback(Output('forms-to-search', 'value'),
              Output('frustrated-filter', 'value'),
              Output('record-search', 'value'),
              Input('reset-search-btn', 'n_clicks'))
def reset_search_criteria(btn):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    value_id = [v['value'] for v in dash.callback_context.triggered][0]
    if 'reset-search-btn' in changed_id and value_id:
        return 'All', 'All', ''
    return 'All', 'All', ''
###################################################################
if __name__ == '__main__':
    if deploy:
        app.run_server(port=5000, host="0.0.0.0", debug=True)
    else:
        webbrowser.open('http://127.0.0.1:8050/')
        app.run_server(debug=False)
