# version 1.2 updated 3 mar 2023

from datetime import datetime, timedelta
import json
import numpy as np
import os
from pprint import pprint
import pandas as pd
import pulp
import random
import sqlite3
from storage.containers import Bin, Tote, BulkContainer
from storage.item import Item
from storage.graphCreator8_Libs import *  # custom classes
from storage.SB_App import *  # custom classes
from storage.warehouse_input_parser import WarehouseInputParser
import time
import pyodbc
import pika
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

##############################
# Connect to RabbitMQ server.
# Note: BlockingConnection blocks the execution thread until the called function returns.

connection = pika.BlockingConnection(pika.ConnectionParameters(host='10.13.0.34', port=5672))
# Connect to 'storage opt' queue. D365 will send messages to an HTTP endpoint
channel = connection.channel()
channel.queue_declare(queue='StorageOptimization', durable=True)

def create_db():
    # create transaction history db
    db_path = './thistory.db' # change path as needed

    cols = {
        'LoadIdentifier':'TEXT', # license plate number
        'AssignmentTime':'TEXT', # last time our code touched it
        'StagingLoc':'TEXT', # staging location
        'FinalLoc':'TEXT', # final location
        'ContainerType': 'TEXT'
    }

    # create main table with columns
    query = 'CREATE TABLE IF NOT EXISTS TransactionHistory ('
    for col, ctype in cols.items():
        if col == 'LoadIdentifier':
            col = "\"{}\"".format(col)
            query = query + col + ' ' + ctype + ' ' + 'UNIQUE' + ', '
        col = "\"{}\"".format(col)
        query = query + col + ' ' + ctype + ', '
    query = query.strip(', ') + ')'
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    # create table
    cursor.execute(query)
    connection.commit()

def get_units(n_units=3):

    container_types = (Bin, Tote, BulkContainer)
    containers = []
    items = []


    for i in range(n_units):
        container = np.random.choice(container_types, p=[0.8,0.1,0.1])()
        container.priority = np.random.choice(('A','B','C'), p=[0.1,0.8,0.1])

        # btw 1 and 6 items per container
        n_items = random.randint(1,6)
        for i in range(n_items):
            item = Item(priority = container.priority)
            items.append(item)
            c = container
            c.d = c.dimensions['outer']
            c.l, c.w, c.h = c.d['length'], c.d['width'], c.d['height']
            c.add_item_to_container(item)


        containers.append(c)


    units = [
            {'ID':c.uuid.hex,
             'Name': c.name,
             'Full Name': c.full_name,
             'Length':c.dimensions['outer']['length'],
             'Width':c.dimensions['outer']['width'],
             'Height':c.dimensions['outer']['height'],
             'Volume': c.l*c.w*c.h,
             'Priority':c.priority,
             'Weight': min(c.weight, 2000)} for c in containers ]

    return items, containers, units


def get_warehouse(warehouse_path='warehouse-layout-input-standard-only.xlsx'):

    warehouse = WarehouseInputParser(warehouse_path).process_warehouse_input()
    all_shelf_ids = list(warehouse['ShelfID'])
    shelves = warehouse.to_dict('records')
    for shelf in shelves:
        shelf['Volume'] = shelf.get('ShelfLength')*shelf.get('ShelfWidth')*shelf.get('ShelfHeight')
    shelf_count = len(all_shelf_ids)

    return warehouse, all_shelf_ids, shelves, shelf_count

def get_shelf_by_id(ShelfID):
    for shelf in shelves:
        if shelf['ShelfID'] == ShelfID:
            return shelf


def optimize_function(  x=dict(), # old soln x
                        # y=dict(), # old soln y
                        n_units=3, # number of units to add
                        old_units=[],
                        new_units=[],
                        shelves=None,
                        all_shelf_ids=None):


    # store solution for x
    solutionX = dict()
    for k, v in x.items():
        solutionX[k] = v.value()


    units = old_units + new_units

    print(f'There are now {len(units)} units.')




    '''
    Variables
    '''

    shelf_loc = [(u.get('ID'), shelfID) for u in units for shelfID in all_shelf_ids]


    # # x is options for item + bin combinations
    x = pulp.LpVariable.dicts(
        'OnShelf', shelf_loc,
        lowBound = 0, # item is not on the shelf
        upBound = 1, # item is on the shelf
        cat = 'Integer' # solver uses integers, i think?
    )

    # z is a dummy variable for shelf waste; minimizing it then using it in a constraint acts like taking the max would
    z = pulp.LpVariable.dicts(
        'ShelfWaste', [(unit.get('ID'), shelf.get('ShelfID')) for unit in units for shelf in shelves],
        0,
        cat = 'Integer'
    )

    # add bounds unique to each shelf
    for i in z:
        for shelf in shelves:
            if shelf.get('ShelfID') == i:
                z[i].bounds(0, shelf.get('Volume'))


    # initialize problem
    prob = pulp.LpProblem('Shelf_Stocking', pulp.LpMinimize)

    '''
    Objective Function
    '''
    prob += pulp.lpSum([z[(unit.get('ID'), shelf.get('ShelfID'))] for unit in units for shelf in shelves]), 'Objective: Minimize Excess Shelf Space'

    '''
    constraints
    '''

    # enforce old solution
    for k,v in solutionX.items():
        for i in range(shelf_count):
            if int(v) == 1:
                try:
                    prob += x[k] == 1.0, f'Enforce old solution {k}'
                except:
                    pass

    for unit in units:
        prob += pulp.lpSum([x[(unit.get('ID'), shelfID)] for shelfID in all_shelf_ids]) == 1, 'Item {} can only be on one shelf'.format(unit.get('ID'))


    # # sum of item widths should not exceed shelf width ("length")
    for i, shelf in enumerate(shelves):
        prob += pulp.lpSum([unit.get('Length') * x[(unit.get('ID'), shelf['ShelfID'])] for unit in units]) <= shelf['ShelfLength'], 'Sum of widths cannot exceed shelf {} width'.format(i)


        # the sum of item weights should not exceed shelf capacity
        prob += pulp.lpSum([unit.get('Weight') * x[(unit.get('ID'), shelf['ShelfID'])] for unit in units]) <= shelf['WeightCapacity'], 'Sum of weights cannot exceed shelf {} weight cap'.format(i)


        for unit in units:
            # minimize shelf waste
            prob += ( shelf.get('Volume') - unit.get('Volume')) * x[(unit.get('ID'), shelf.get('ShelfID'))] <= z[unit.get('ID'), shelf.get('ShelfID')], 'Minimize wasted space created by inserting item item {} onto shelf {}'.format(unit.get('ID'), shelf.get('ShelfID'))


    # the difference in priority btw item and shelf should be 0
    prob+= pulp.lpSum([abs(ord(unit.get('Priority')) - ord(shelf.get('ShelfPriority'))) * x[(unit.get('ID'), shelf['ShelfID'])]\
                        for unit in units for shelf in shelves ]) == 0, 'All items are matched to priority'


    # set initial values
    for k, v in solutionX.items():
        if int(v) == 1:
            # print(k,'-->',v)
            x[k].setInitialValue(v)



    # '''
    # Solve
    # '''
    start_time = time.time()
    solver = pulp.PULP_CBC_CMD(msg=True)
    prob.solve(solver)
    print('Solved in {:.3f} seconds.'.format(time.time() - start_time))
    return x, z, units

def get_stocked_shelves(x):
    stocked_shelves = {}
    for item in x.keys():
        if x[item].value()==1:
            itemNum = item[0]
            shelfNum = item[1]
            if shelfNum in stocked_shelves:
                stocked_shelves[shelfNum].append(itemNum)
            else:
                stocked_shelves[shelfNum] = [itemNum]
    return stocked_shelves

def put_away(units=[],
             x=dict(),
             used_shelf_ids=[],
             warehouse_path='warehouse-layout-input-standard-only.xlsx'):

    warehouse, all_shelf_ids, shelves, shelf_count = get_warehouse(warehouse_path=warehouse_path)

    all_shelf_ids = list(set(all_shelf_ids)-set(used_shelf_ids))
    shelves = [ shelf for shelf in shelves if shelf['ShelfID'] in all_shelf_ids ]
    shelf_count = len(shelves)

    xold = x
    x, z, units = optimize_function(x=xold,
                                    new_units=units,
                                    all_shelf_ids=all_shelf_ids,
                                    shelves=shelves)

    return get_stocked_shelves(x)

def get_d365_inventory_df(dbcon=None):

    if dbcon is None:
        print('Establishing connection to inventory DB')
        driver='{ODBC Driver 18 for SQL Server}'
        server='''10.13.0.68\SQLHOSTINGSERVE,1433'''
        database='BYODB'
        uid='d365admin'
        pwd='!!USMC5G!!'
        connect_str = 'Driver={};Server={};Database={};Uid={};Pwd={};Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;'.format(driver,server,database,uid,pwd)
        dbcon = pyodbc.connect(connect_str)

    query = '''
        SELECT
        s.LICENSEPLATEID AS "LPN",
        s.AVAILPHYSICAL AS "Avail",
        l.LICENSEPLATENUMBER AS "LPN2",
        l.CONTAINERTYPEID AS "ContainerType",
        l.PARENTLICENSEPLATENUMBER as "ParentLPN",
        p.CONTAINERTYPEID AS "ParentContainerType",
        s.WMSLOCATIONID AS "WAREHOUSELOCATIONID"
        FROM "KPMGInventSumStaging" s
        LEFT JOIN "WHSLicensePlateStaging" l on s.LICENSEPLATEID = l.LICENSEPLATENUMBER
        LEFT JOIN "WHSLicensePlateStaging" p on l.PARENTLICENSEPLATENUMBER = p.LICENSEPLATENUMBER
        '''

    df = pd.read_sql(query, dbcon)

    df = df[df['Avail']>0]
    df = df[df['WAREHOUSELOCATIONID'].notna()]
    df = df[df['WAREHOUSELOCATIONID'].str.match('.2.\d{4}\w{2}')]
    df = df.drop(columns=['LPN2']) # duplicate
    # if an item has a parent lpn, use that instead
    df.loc[df['ParentLPN'].notna(), 'LPN'] = df['ParentLPN']
    # if an item has a parent lpn, use that container type instead
    df.loc[df['ParentContainerType'].notna(), 'ContainerType'] = df['ParentContainerType']
    # drop parent cols now
    df = df.drop(columns=['ParentLPN', 'ParentContainerType'])
    # drop duplicates created from multiple rows having the same parent info
    df = df[~df.duplicated()]

    return df


def get_tx_history_df(search_back_minutes=10000):
    # get relevant transaction history data
    db_path = './thistory.db'
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    query = '''SELECT * FROM TransactionHistory'''
    cursor.execute(query)
    column_names = [x[0] for x in cursor.description]
    extract = cursor.fetchall()
    dd = pd.DataFrame(extract, columns=column_names)
    connection.close()

    # update storage availability with any updates that we've processed in the last 30 min
    dd['AssignmentTime'] = pd.to_datetime(dd['AssignmentTime'])

    dd = dd[dd.AssignmentTime >= datetime.now() - timedelta(minutes=search_back_minutes)]

    return dd

def assign_loc_and_return_modified_signal(signal):
     # parse additional data as needed
    data = signal['KPMGAdditionalData']
    # container type

    container_type = data['ContainerType']
    # container priority
    try:
        container_priority = data['priority']
    except:
        container_priority = 'A'

    # container dimensions - useful for pallets with unknown heights
    container_height = data['Height']
    container_width = data['width']
    container_depth = data['depth']

    try:
        container_weight = data['weight']
    except:
        container_weight = 1

    container_id = signal['LoadIdentifier']

    unit = [{'ID': container_id,
              'Name': container_type,
              'Length': container_depth,
              'Width': container_width,
              'Height': container_height,
              'Volume': container_width*container_depth*container_height,
              'Priority': container_priority,
              'Weight': container_weight}]

    '''
    INSERT CODE TO DETERMINE CURRENT STORAGE AVAILABILITY (AZURE SQL)
    '''

    print('Calling get_d365_inventory_df within assign_loc_and_return_modified_signal')
    df = get_d365_inventory_df()


    container_ids = [
        'MODULAR NESTING CONT',
        'BULK CONT LENGTHDOOR',
        'BULK CONT WIDTHDOOR',
        'WOOD SHIPPING PALLET'
    ]

    # df
    def loc_id_to_tuple(locid):
        l = locid
        return (int(l[5:7]), int(l[3:5]),l[7])

    mod_nest_counter = {loc_id_to_tuple(locid):0 for locid in df.WAREHOUSELOCATIONID}
    MOD_NEST_MAX = 8
    used_shelf_ids = []


    for idx, row in df.iterrows():
        locid = loc_id_to_tuple(row.WAREHOUSELOCATIONID)

        if row.ContainerType == 'MODULAR NESTING CONT':
            mod_nest_counter[locid]+=1

        else:
        # elif row.ContainerType in ['BULK CONT LENGTHDOOR','BULK CONT WIDTHDOOR','WOOD SHIPPING PALLET']:
            used_shelf_ids.append(locid)



    '''
    READ TRANSACTION HISTORY + UPDATE STORAGE AVAILABILITY
    '''

    dd = get_tx_history_df()

    for idx, row in dd.iterrows():
        locid = loc_id_to_tuple(row.FinalLoc)

        if locid not in mod_nest_counter.keys():
            mod_nest_counter[locid] = 0
        if row.ContainerType == 'MODULAR NESTING CONT':
            mod_nest_counter[locid]+=1
        else:
        # elif row.ContainerType in ['BULK CONT LENGTHDOOR','BULK CONT WIDTHDOOR','WOOD SHIPPING PALLET']:
            used_shelf_ids.append(locid)

    for locid, num_mod_nest in mod_nest_counter.items():
        if container_type == 'MODULAR NESTING CONT':
            if num_mod_nest >= MOD_NEST_MAX:
                used_shelf_ids.append(locid)
        else:
            if num_mod_nest >=1:
                used_shelf_ids.append(locid)

    '''
    INSERT STORAGE OPTIMIZATION CODE WHERE OUTPUT == final_location
    '''

    def get_mod_nest_loc():

        mn_loc = None
        if container_type != 'MODULAR NESTING CONT':
            return mn_loc

        wh = WarehouseInputParser(infile='warehouse-layout-input-standard-only.xlsx').input_data
        wh=wh[wh.ShelfPriority==container_priority]

        for idx, row in wh.iterrows():
            locidA = ( int(row.ShelfColumn), int(row.ShelfAisle), 'A')
            if locidA in mod_nest_counter.keys():
                v = mod_nest_counter[locidA]
                if v >= 1 and v < MOD_NEST_MAX:
                    mn_loc = locidA
                    break

            locidB = ( int(row.ShelfColumn), int(row.ShelfAisle), 'B')
            if locidB in mod_nest_counter.keys():
                v = mod_nest_counter[locidB]
                if v >= 1 and v < MOD_NEST_MAX:
                    mn_loc = locidB
                    break
            
            locidC = ( int(row.ShelfColumn), int(row.ShelfAisle), 'C')
            if locidC in mod_nest_counter.keys():
                v = mod_nest_counter[locidC]
                if v >= 1 and v < MOD_NEST_MAX:
                    mn_loc = locidC
                    break

            locidD = ( int(row.ShelfColumn), int(row.ShelfAisle), 'D')
            if locidD in mod_nest_counter.keys():
                v = mod_nest_counter[locidD]
                if v >= 1 and v < MOD_NEST_MAX:
                    mn_loc = locidD
                    break

        return mn_loc

    new = get_mod_nest_loc()

    # print(unit)
    if new is None:
        # print('Finding new loc')
        new = put_away(units=unit,
                       used_shelf_ids=used_shelf_ids)
        new = list(new.keys())[0]
    else:
        print(f'Using mod nest location: {new}')

    def to_loc_fmt(key_fmt):
        return 'I21' + str(key_fmt[1]).zfill(2) + str(key_fmt[0]).zfill(2) + key_fmt[2] + 'A'


    final_location = to_loc_fmt(new) # just a sample output that storage may have, but output must be in this format
    print('Final loc: {}'.format(final_location))
    idx = [signal['Locations'].index(s) for s in signal['Locations'] if s['Operation']=='drop'][0]

    signal['Locations'][idx]['Name'] = final_location

    '''
    WRITE DECISION TO TRANSACTION HISTORY
    '''
    db_path = './thistory.db'
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    query = '''
        INSERT OR REPLACE INTO TransactionHistory ("LoadIdentifier", "AssignmentTime", "FinalLoc", "ContainerType")
        VALUES (?, ?, ?, ?)
    '''
    cursor.execute(query, [signal['LoadIdentifier'], datetime.now().strftime('%Y-%m-%d %H:%M:%S'), final_location, container_type])
    connection.commit()
    connection.close()

    # drop KPMGAdditionalData from output - TMO doesn't need it
    signal.pop('KPMGAdditionalData', None)

    return (final_location, signal)


'''
Staging bin helpers from optimal path app.
'''


def get_nearest_staging_bin_to_location(locid, signal):
    '''
    Takes as only argument a Location ID looking like

    I216318AA

    and return the Location ID of the staging bin physically
    nearest this location in bay2.
    '''
    print('Calculating staging bin...', flush=True)
    picklefile = open('BAY2Obj6', 'rb')
    BAY2Obj_loaded = pickle.load(picklefile)
    picklefile.close()

    StagingBinLocs = {'1A':'(79, 10)'} # demo and sit4 only
    # StagingBinLocs = {'1A':'(79, 10)','2A':'(79, 34)'} # new from JG 11/3, updated feb
    AppObj = SB_App(AppID='App One',BAY=BAY2Obj_loaded,StagingBinsLocs=StagingBinLocs)
    StagingBins = AppObj.Dist_StagingBins3(StagingBinLocs=StagingBinLocs)


    def loc_id_to_tuple_without_shelf(locid):
        l = locid
        return f'({int(l[5:7])}, {int(l[3:5])})'


    def get_key_for_smallest_val(dictionary):
        d = dictionary
        smallest = min(d.values())
        try:
            return [ k for k,v in d.items() if v == smallest ][0]
        except:
             return [ k for k,v in d.items() if np.isclose( v ,smallest) ][0]

    location_tup = loc_id_to_tuple_without_shelf(locid)
    distances_to_staging_bins = StagingBins[location_tup][0]

    staging_bin_name = get_key_for_smallest_val(distances_to_staging_bins)
    staging_bin_loc = StagingBinLocs[staging_bin_name]

    col = staging_bin_loc[1:3]
    aisle = staging_bin_loc[-3:-1]
    stg_bin = 'I21'+aisle+col+'AA'
    print('Identified a nearby staging bin: {}'.format(stg_bin), flush=True)

    db_path = './thistory.db'
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    query = '''
        UPDATE TransactionHistory
        SET "StagingLoc" = "{}"
        WHERE "LoadIdentifier" = "{}"
    '''.format(stg_bin, signal['LoadIdentifier'])
    cursor.execute(query)
    connection.commit()
    connection.close()

    return stg_bin

def storage_opt(signal, dbcon=None):
    '''
    Main function.
    '''

    if not os.path.isfile('thistory.db'):
        print('Creating new history database...', flush=True)
        create_db()

    action = [s for s in signal['Locations'] if s['Operation']=='drop'][0]['Name']
    # if signal loc for 'drop' is named 'stage', return signal as-is
    try:
        if action == 'Stage':
            # we will modify this later
            final_location, signal = assign_loc_and_return_modified_signal(signal)

            stage_bin = get_nearest_staging_bin_to_location(final_location, signal)

            idx = [signal['Locations'].index(s) for s in signal['Locations'] if s['Operation']=='drop'][0]

            signal['Locations'][idx]['Name'] = stage_bin

            return signal

        elif action == 'Final':
            lpn = signal['LoadIdentifier']

            print('Looking for a local history for LPN {}'.format(lpn), flush=True)
            # get tx history for this lpn
            db_path = './thistory.db'
            connection = sqlite3.connect(db_path)
            cursor = connection.cursor()
            query = '''SELECT * FROM TransactionHistory WHERE "LoadIdentifier" = "{}"'''.format(lpn)
            cursor.execute(query)
            column_names = [x[0] for x in cursor.description]
            extract = cursor.fetchall()
            dd = pd.DataFrame(extract, columns=column_names)
            connection.close()
            lpn_hist = dd.to_dict('records')
            if not lpn_hist:
                print('Never calculated a final location for LPN {}, calculating location...'.format(lpn), flush=True)
                final, stg = assign_loc_and_return_modified_signal(signal)
                print('Assigned a final location: {}'.format(final), flush=True)
                idx = [signal['Locations'].index(s) for s in signal['Locations'] if s['Operation']=='drop'][0]
                signal['Locations'][idx]['Name'] = final
                return signal
            else:
                lpn_hist = lpn_hist[0] # get the first dictionary in the list
                
            print('Previously calculated a final location for LPN {} at {}'.format(lpn, lpn_hist['FinalLoc']), flush=True)

            # determine if final loc is still available from d365 database
            print('Checking D365 for {} availability...'.format(lpn_hist['FinalLoc']), flush=True)

            print('Establishing connection to D365 database...', flush=True)
            driver='{ODBC Driver 18 for SQL Server}'
            server='''10.13.0.68\SQLHOSTINGSERVE,1433'''
            database='BYODB'
            uid='d365admin'
            pwd='!!USMC5G!!'
            connect_str = 'Driver={};Server={};Database={};Uid={};Pwd={};Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;'.format(driver,server,database,uid,pwd)
            dbcon = pyodbc.connect(connect_str)

            query = '''
                SELECT
                s.LICENSEPLATEID AS "LPN",
                s.AVAILPHYSICAL AS "Avail",
                l.LICENSEPLATENUMBER AS "LPN2",
                l.CONTAINERTYPEID AS "ContainerType",
                l.PARENTLICENSEPLATENUMBER as "ParentLPN",
                p.CONTAINERTYPEID AS "ParentContainerType",
                s.WMSLOCATIONID AS "WAREHOUSELOCATIONID"
                FROM "KPMGInventSumStaging" s
                LEFT JOIN "WHSLicensePlateStaging" l on s.LICENSEPLATEID = l.LICENSEPLATENUMBER
                LEFT JOIN "WHSLicensePlateStaging" p on l.PARENTLICENSEPLATENUMBER = p.LICENSEPLATENUMBER
                WHERE WAREHOUSELOCATIONID = "{}"
                '''.format(lpn_hist['FinalLoc'])



            query = '''
                    SELECT
                    s.LICENSEPLATEID AS "LPN",
                    s.AVAILPHYSICAL AS "Avail",
                    l.LICENSEPLATENUMBER AS "LPN2",
                    l.CONTAINERTYPEID AS "ContainerType",
                    l.PARENTLICENSEPLATENUMBER as "ParentLPN",
                    p.CONTAINERTYPEID AS "ParentContainerType",
                    s.WMSLOCATIONID AS "WAREHOUSELOCATIONID"
                    FROM "OnHandInventory" s
                    LEFT JOIN "LicensePlates" l on s.LICENSEPLATEID = l.LICENSEPLATENUMBER
                    LEFT JOIN "LicensePlates" p on l.PARENTLICENSEPLATENUMBER = p.LICENSEPLATENUMBER
                    WHERE s.WMSLOCATIONID = (?) AND s.AVAILPHYSICAL > 0
                    '''

            cursor = dbcon.cursor()
            cursor.execute(query, [lpn_hist['FinalLoc']])
            extract = cursor.fetchall()
            dbcon.close()
            # filter by available
            # d365_df = d365_df[d365_df['Avail']>0]
            #d365_df = pd.read_sql(query, dbcon)
            #loc_avail = d365_df.to_dict('records')

            if extract:
                print('This location is now taken. Recalculate location...', flush=True)
                final, stg = assign_loc_and_return_modified_signal(signal)
                print('Using a new location: {}'.format(final), flush=True)
                idx = [signal['Locations'].index(s) for s in signal['Locations'] if s['Operation']=='drop'][0]
                signal['Locations'][idx]['Name'] = final
                return signal
            else:
                print('This location is still available. Using {}'.format(lpn_hist['FinalLoc']), flush=True)
                idx = [signal['Locations'].index(s) for s in signal['Locations'] if s['Operation']=='drop'][0]
                signal['Locations'][idx]['Name'] = lpn_hist['FinalLoc']
                return signal
        else:
            print('Signal not identified as staging or final destination. Returning signal as-is.', flush=True)
            return signal

    except Exception as e:
        print(e, flush=True)
        raise Exception('Error parsing input data.')

def callback(ch, method, properties, body):
    result = storage_opt(json.loads(body))
    print("Result: " + str(result), flush=True)
    channel.basic_ack(delivery_tag = method.delivery_tag)

    '''New output code'''
    # D365 Blob Storage Connection String:
    account_url = "https://sharedaccount.blob.rack01.usmc5gsmartwarehouse.com/"
    container_name = "d365"
    credential = {
            "account_name": 'sharedaccount',
            "account_key": 'ecqdYYx8tlBmKJbmjrBKSroK/oeZZz5WbSiM1NjO56yr8SI8TzfdVd9d32kB+X158lMzwicfN4SEB6RSd8q0ZQ=='
        }
    container_client = ContainerClient(account_url = account_url, container_name = container_name, credential = credential, api_version = '2019-02-02')
    blob_client = container_client.get_blob_client(blob='StorageOptimization/incoming' + str(datetime.now()) + '.txt')

    blob_client.upload_blob(str(result))
    print("Uploading to blob storage...", flush=True)

    ''' Previous output Code'''
    # Publish the result to the AMQP queue "StorageOptimizationOutput"
    #publish_channel = connection.channel()
    #publish_channel.queue_declare(queue='TMO', durable=True)
    #publish_channel.basic_publish(exchange='', routing_key='TMO', body=json.dumps(result))


print("Before consuming")
# Start consuming messages from the queue.
channel.basic_consume(
    queue='StorageOptimization', on_message_callback=callback, auto_ack=False)

# Start consuming messages from the queue.
print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
