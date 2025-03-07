import numpy as np
import os
import pickle
from pprint import pprint
import pandas as pd
import pulp
import random
import storage
from storage.warehouse_input_parser import WarehouseInputParser
from storage.containers import Bin, Tote, BulkContainer
from storage.item import Item
import sys
import time




def get_units(n_units=3):
    # create Item objects
   
    container_types = (Bin, Tote, BulkContainer)
    containers = []
    items = []


    for i in range(n_units):
        container = np.random.choice(container_types, p=[0.8,0.1,0.1])()
        container.priority = np.random.choice(tuple('ABC'), p = [0.1, 0.8, 0.1])
        
        # btw 1 and 6 items per container
        n_items = random.randint(1,6)
        for i in range(n_items):
            item = Item(priority = container.priority)
            items.append(item)
            container.add_item_to_container(item)

        containers.append(container)

    units = [ 
            {'ID':c.uuid.hex,
             'Name': c.name,
             'Full Name': c.full_name,
             'Length':c.dimensions['outer']['length'], 
             'Width':c.dimensions['outer']['width'], 
             'Height':c.dimensions['outer']['height'], 
             'Priority':c.priority,
             'Weight': min(c.weight, 2000)} for c in containers ]


    return items, containers, units


def get_unit_by_id(unitID, units):
    for unit in units:
        if unitID == unit['ID']:
            return unit

# return the item by ID
def get_item_by_id(unitID, items):
    for item in items:
        if unitID == item.uuid:
            return item

def get_container_by_id(containerID, containers):
    for container in containers:
        if containerID == container.uuid.hex:
            return container



def get_shelf_by_id(ShelfID, shelves):
    for shelf in shelves:
        if shelf['ShelfID'] == ShelfID:
            return shelf
        
        
def dump_state(x, y, items, containers, units, path='.'):
    
    outfile = os.path.join(path, 'x.pkl')
    with open(outfile, 'wb') as handle:
        pickle.dump(x, handle)

    outfile = os.path.join(path, 'y.pkl')
    with open(outfile, 'wb') as handle:
        pickle.dump(y, handle)

#     outfile = os.path.join(path, 'items.pkl')
#     with open(outfile, 'wb') as handle:
#         pickle.dump(items, handle)

#     outfile = os.path.join(path, 'containers.pkl')
#     with open(outfile, 'wb') as handle:
        # pickle.dump(containers, handle)

    outfile = os.path.join(path, 'units.pkl')
    with open(outfile, 'wb') as handle:
        pickle.dump(units, handle)


def load_state(x_filename, 
               y_filename, 
               units_filename):
    
    with open(x_filename, 'rb') as handle:
        x = pickle.load(handle)
        
    with open(y_filename, 'rb') as handle:
        y = pickle.load(handle)
    
    with open(units_filename, 'rb') as handle:
        units = pickle.load(handle)
        
    return x, y, units

##########################       
#  CONSTAINTS
##########################
warehouse = WarehouseInputParser('warehouse-layout-input.xlsx').process_warehouse_input()
all_shelf_ids = list(warehouse['ShelfID'])
shelves = warehouse.to_dict('records')
shelf_count = len(all_shelf_ids)
# N_UNITS = 4 # number of units to add per cycle



def optimize_function(  x=dict(), # old soln x
                        y=dict(), # old soln y
                        n_units=3, # number of units to add
                        items=[],
                        containers=[],
                        units=[] ):


    # store solution for x
    solutionX = dict()
    for k, v in x.items():
        solutionX[k] = v.value()

    # store solution for y
    solutionY = dict()
    for k, v in y.items():
        solutionY[k] = v.value()


    new_items,new_containers,new_units = get_units(n_units=n_units)

    items.extend(new_items)
    containers.extend(new_containers)
    units.extend(new_units)

    print(f'There are now {len(units)} units.')



    '''
    Variables
    '''
    # y is shelves
    y = pulp.LpVariable.dicts(
        'ShelfUsed', range(shelf_count),
        lowBound = 0, # there's nothing on the shelf
        upBound = 1, # there's something on the shelf
        cat = 'Integer')

    # # all possible combinations of items and shelves
    # shelf_loc = [(i.get('ID'), shelf) for i in units for shelf in range(shelf_count)]
    shelf_loc = [(u.get('ID'), shelf) for u in units for shelf in all_shelf_ids]
    # pprint(shelf_loc)
    # [('3e0618710a7c4253a7f21bbbd406e218', ('91', '3', 'A')),
    #  ('3e0618710a7c4253a7f21bbbd406e218', ('91', '3', 'B')),
    #  ('3e0618710a7c4253a7f21bbbd406e218', ('91', '3', 'C')),
    #  ('3e0618710a7c4253a7f21bbbd406e218', ('90', '3', 'A')),
    #  ('3e0618710a7c4253a7f21bbbd406e218', ('90', '3', 'B'))
    # ...


    # # x is options for item + bin combinations
    x = pulp.LpVariable.dicts(
        'OnShelf', shelf_loc,
        lowBound = 0, # item is not on the shelf
        upBound = 1, # item is on the shelf
        cat = 'Integer' # solver uses integers, i think?
    )

    # # initialize the problem
    prob = pulp.LpProblem('Shelf Stocking', pulp.LpMinimize)

    # # objective function
    prob += pulp.lpSum([y[i] for i in range(shelf_count)]), "Objective: Minimize Shelves Used" # I think?



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


    # # sum of item widths should not exceed shelf width
    for i, shelf in enumerate(shelves):
        prob += pulp.lpSum([unit.get('Length') * x[(unit.get('ID'), shelf['ShelfID'])] for unit in units]) <= shelf['ShelfLength']*y[i], 'Sum of widths cannot exceed shelf {} width'.format(i)


    # the sum of item weights should not exceed shelf capacity
        prob += pulp.lpSum([unit.get('Weight') * x[(unit.get('ID'), shelf['ShelfID'])] for unit in units]) <= shelf['WeightCapacity']*y[i], 'Sum of weights cannot exceed shelf {} weight cap'.format(i)


    # the difference in priority btw item and shelf should be 0
    prob+= pulp.lpSum([abs(ord(unit.get('Priority')) - ord(shelf.get('ShelfPriority'))) * x[(unit.get('ID'), shelf['ShelfID'])]\
                        for unit in units for shelf in shelves ]) == 0, 'All items are matched to priority'

    # set initial values
    for k, v in solutionX.items():
        if int(v) == 1:
            #print(k,'-->',v)
            x[k].setInitialValue(v)


    for k, v in solutionY.items():
        if int(v) == 1:
            #print(k,'-->',v)
            y[k].setInitialValue(v)

    # '''
    # Solve
    # '''
    start_time = time.time()
    solver = pulp.PULP_CBC_CMD(msg=True)
    prob.solve(solver)
    print('Solved in {:.3f} seconds.'.format(time.time() - start_time))

    # report output
    # shelves used
    print('Shelves used: {}\n'.format(sum(([y[i].value() for i in range(shelf_count)]))))
    print()

    return x, y, items, containers, units



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

# pprint(get_stocked_shelves(x))

def detail_report(stocked_shelves, shelves, units):
 
    for shelfID, unitIDs in stocked_shelves.items():
        print(f'\nShelf {shelfID}:')
        shelfdata = get_shelf_by_id(shelfID, shelves)
        print(f"\t{shelfdata['ShelfType']}\tpriotiry: {shelfdata['ShelfPriority']}\tcontains {unitIDs}\n")

        # for unitID in unitIDs:
        #     item = get_item_by_id(unitID)
        #     item.put_item(shelfID=shelfID)
        #     pprint(item)

        for unitID in unitIDs:
            unit = get_unit_by_id(unitID, units)
            pprint(unit)


def main():
    if len(sys.argv) > 1:
        N_UNITS = int(sys.argv[1])
        TARGET = int(sys.argv[2])
    else:
        N_UNITS=3
        TARGET=50
        
    ##########################       
    #  CONSTAINTS
    ##########################
    warehouse = WarehouseInputParser('warehouse-layout-input.xlsx').process_warehouse_input()
    all_shelf_ids = list(warehouse['ShelfID'])
    shelves = warehouse.to_dict('records')
    shelf_count = len(all_shelf_ids)
    # N_UNITS = 4 # number of units to add per cycle

    
    x = dict()
    y = dict()
    items = []
    containers = []
    units = []

    target = TARGET # number of units for demo to place on default warehouse racks


    while len(units) < target :
        x, y, items, containers, units = optimize_function(x=x,
                                                           y=y,
                                                           n_units=N_UNITS,
                                                           items=items,
                                                           containers=containers,
                                                           units=units)

        if len(units) == N_UNITS or len(units)%(N_UNITS*10)==0:
            stocked_shelves = get_stocked_shelves(x)
            pprint(stocked_shelves)
            detail_report(stocked_shelves, shelves, units)
            dump_state(x, y, items, containers, units, path=os.getcwd())
            
            
if __name__ == '__main__':
    main()
