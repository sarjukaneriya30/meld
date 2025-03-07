# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:59:22 2022

@author: kritz
"""
import sqlite3
import json
import os


def create_ocr_database(db_path, template_path, speed_it_up):

    '''
    CREATE 1348 TABLE (Data1348)
    '''
    # loop through all 1348 templates

    available_templates = [t for t in os.listdir(template_path) if t.split('.')[0] + '.json' in os.listdir(template_path) and '1348' in t and '.jpg' in t]

    cols = {
        'Pallet ID':'INT',
        'Packet ID':'INT',
        'Filename':'TEXT',
        'Extraction Time':'TEXT',
    }

    for t in available_templates:
        template_name = t.split('.')[0]
        with open('prepare/templates/{}.json'.format(template_name)) as file:
            parameters = file.read()
            parameters = json.loads(parameters)
        for p in [p['Label'] for p in parameters]:
            if p.lower() not in [c.lower() for c in cols.keys()]:
                cols[p] = 'TEXT'

    if speed_it_up:
        cols['All 1348 Fields'] = 'TEXT'
    # create main table with columns
    query = 'CREATE TABLE IF NOT EXISTS Data1348 ('
    for col, ctype in cols.items():
        col = "\"{}\"".format(col)
        query = query + col + ' ' + ctype + ', '
    query = query.strip(', ') + ')'
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    # create table
    cursor.execute(query)
    connection.commit()

    '''
    CREATE CompletePackets TABLE
    '''
    cols = {
     'Pallet ID':'INT',
     'Packet ID':'INT',
     'Packet Processing Time':'TEXT'
    }
    # create main table with columns
    query = 'CREATE TABLE IF NOT EXISTS CompletePackets ('
    for col, ctype in cols.items():
        col = "\"{}\"".format(col)
        query = query + col + ' ' + ctype + ', '
    query = query + ' UNIQUE ('
    for col in ['Pallet ID', 'Packet ID']:
        query = query + "\"{}\"".format(col) + ', '
    query = query.strip(', ') + '))'
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    # create table
    cursor.execute(query)
    connection.commit()

    '''
    CREATE AdditionalForms TABLE
    '''
    cols = {
     'Pallet ID':'INT',
     'Packet ID':'INT',
     'Extraction Time':'TEXT',
     'Filename':'TEXT',
     'Text':'TEXT'
    }
    # create main table with columns
    query = 'CREATE TABLE IF NOT EXISTS AdditionalForms ('
    for col, ctype in cols.items():
        col = "\"{}\"".format(col)
        query = query + col + ' ' + ctype + ', '
    query = query.strip(', ') + ')'
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    # create table
    cursor.execute(query)
    connection.commit()

    '''
    CREATE FrustratedItems TABLE
    '''
    cols = {
     'Pallet ID':'INT',
     'Packet ID':'INT',
    }
    # create main table with columns
    query = 'CREATE TABLE IF NOT EXISTS FrustratedItems ('
    for col, ctype in cols.items():
        col = "\"{}\"".format(col)
        query = query + col + ' ' + ctype + ', '
    query = query.strip(', ') + ')'
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    # create table
    cursor.execute(query)
    connection.commit()

    connection.close()

    print('Database setup complete.')
    return None
