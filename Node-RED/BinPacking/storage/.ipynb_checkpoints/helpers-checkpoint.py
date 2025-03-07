from abc import ABC
from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, Iterable, List, Tuple
import uuid
from storage.warehouse import  front, back, middle
from storage.racks import StandardRack, AdjustableRack
# from storage.locations import Locations

# locations   = Locations()
# adjustables = [ loc for loc in locations.locs if Bay2locs.its_adjustable(loc) ]
# standards   = [ loc for loc in locations.locs if loc not in adjustables ]
################################################################################
################################################################################
##############################     helpers     #################################
################################################################################
################################################################################
################################################################################


@dataclass
class HelperFunctions:
    
    err_msg1 = 'Inital inventory must be a dictionary with the only {keys} contained in {[A-D]}, and values of lists of items in each compartment!'
    compartment_names = tuple(list('ABCD'))
    
    @classmethod
    def validate_initial_inventory(cls, initial_inventory):
        
        if not(isinstance(initial_inventory, dict)):
            raise TypeError(cls.err_msg1)
        
        for key, val in initial_inventory.items():
            
            if not key in cls.compartment_names:
                print(f'key: {key} not in {cls.compartment_names}!')
                raise ValueError(cls.err_msg1)
            
            if not isinstance(val, list):
                if not isinstance(val, str):
                    raise TypeError(cls.err_msg1)
            # here may be logic to determine if the items on the shelf are not going to fit
        
    
    @staticmethod
    def get_empty_rack(kind='s'):
        output = dict(  A=[],
                        B=[],
                        C=[],
                        D=[])
        if kind=='s' or kind.lower().startswith('st'):
            return output
        if kind=='a' or kind.lower().startswith('ad'):
            return { k:v for k,v in output.items() if k not in ('C','D') }
    
    @staticmethod
    def get_packed_rack(kind='s'):
        random = np.random
        output = HelperFunctions.get_empty_rack(kind=kind)
        num_cycles = np.random.randint(1,5)
        fact = faker.Factory().create()
        
        i=0
        while i<= num_cycles:
            for key in output.keys():
                num_items = np.random.randint(0,2)
                if num_items > 0:
                    output[key] = output[key] + [fact.catch_phrase()]
            i+=1
        
        outp2 = dict()
        for k, item in output.items():
            for name in item:
                outp2[k] = {'Name': name,
                            'volume': random.randint(2,77)*random.randint(2,56)*random.randint(3,66)
                           }
        return outp2

    
    @staticmethod
    def item_goes_to_adjustable(item=None):
        '''
        Returns True if the item passed as an argument 
        is too large for standard racking
        '''
        
        for dimension in item.dimensions:
            if dimension > 60:
                return True
            
        # more conditions ... return True if we must use adjustable rack
            
        return False
    
    
    @staticmethod
    def parse_location_id(location_id):
        
        def compartment_to_description(compartment_id='A'):
            if compartment_id == 'A':
                return 'bulk'
            return 'unknown'
        
        lid = str(location_id)
        assert(len(lid)==9)
        return {
            'building': lid[0],
            'bay': lid[1],
            'bay_2': lid[2],
            'aisle': lid[3:5],
            'column': lid[5:7],
            'height': lid[7],
            'compartment': lid[8],
            'compartment desc.': compartment_to_description(lid[8])
        }
        

        
        
    @staticmethod
    def loc_is_adjustable(location):
        aisle = int(location[1])
        if aisle < 30 and aisle > 16:
            return True
        return False
    
    @staticmethod
    def get_uuid():
        return uuid.uuid4()


    @staticmethod
    def list_sum( *args: Iterable[List]):
        outp = []
        for alist in args:
            outp += alist
        return outp
    
    
    @staticmethod
    def stringfy(iterator):
        return tuple(map(str, iterator))
    
    
helper = HelperFunctions()