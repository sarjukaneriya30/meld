from abc import ABC
from dataclasses import dataclass
#from storage.helpers import HelperFunctions, helper
from storage.locations import locations, Bay2locs, adjustables, standards
import json
import logging
import numpy as np
import random
from typing import Any, Dict, Iterable, List, Tuple
import warnings

################################################################################
################################################################################
###################    Racks (standard and adjustable)     #####################
################################################################################
################################################################################
################################################################################  


class Rack(ABC):
    
    @classmethod
    def get_weight(cls, *items)->dict:
        '''
        should take an item and return its weight
        '''
        
    class RackLocationError(Exception):
        pass
    
    def __init__(self):

        def location_is_adjustable(self,loc):
            return (loc in Bay2locs().adjustables)


        def location_is_standard(self,loc):
            return (loc in Bay2locs().standards)
        
        
        def set_location(self, location):
            if isinstance(self, AdjustanbleRack):
                if not location in adjustables:
                    raise RackLocationError(f'Expected an adjustable rack location; got {location}')
            self.location = location
            


    
    
class AdjustableRack(Rack):
    '''
    Adjustable Rack Class
    
    ::A stateful class for the shelf positon and contents of the two compartments in the adjustable rack.::
    
    Constructor 
    def __init__(self, 
                 shelf_position=None, 
                 shelf_name=None, 
                 items_A=[], 
                 items_B=[]):
    -------------
    :param shelf_position=None: the initial height of the shelf,
    where None shall default to the middle of the rack (a heaight of 15 feet * 12 inches/foot / 2).
    :param shelf_name: this will be the name of the shelf in (column, aisle) format
    :param items_A=[]: this is the list of items on shelf A
    :param items_B=[]: same as above
    
    This is not a dataclass since,
    when the shelf positon is moved,
    the sizes of compartment A and B are impacted at once.
    '''

    
   
        
    
    def __init__(self,
                 location=None,
                 gap_description='6-5-4', 
                 shelf_name=None,
                 initial_inventory=dict(
                                     A=[],
                                     B=[],
                                     C=[])):
        
        super().__init__()
        
        if location is None:    
            location = random.choice(adjustables)
            
        self.location = location
        self.gap_description = gap_description
        self.shelf_name = shelf_name
        self.initial_inventory = initial_inventory
        self.inventory = initial_inventory

        self.adj_rack_spacings_feet = self.parse_gap_description(self.gap_description)
        
        
        self.rack_dims = self.get_rackdims()
        #'fix this' #self.get_rackdims_in_inches(self.adj_rack_spacings_feet)

        self.D365_name = 'Vidmar'
        self.initial_inventory = initial_inventory
        self.inventory = self.initial_inventory
        self.total_height = 15*12
        self.name = shelf_name
        
        self.lower_shelf_position = self.rack_dims['A']
        self.upper_shelf_position = self.lower_shelf_position+self.rack_dims['B']
        
        self.h_A = self.lower_shelf_position
        self.h_B = self.upper_shelf_position - self.lower_shelf_position
        self.h_C = 15*12 - self.upper_shelf_position
        
        
        
        self.length = 56
        self.width  = 59
        
        
        

        # self.volume = self.get_volumes(container=None)
        # self.volume_A = self.get_volumes(container='A')
        # self.volume_B =self.get_volumes(container='B')
        # self.volume_C = self.get_volumes(container='C')
        self.total_volume = 15*12*self.length*self.width


        self.capacity = self.get_capacity() # 2000 lbs, regardless of comparement

        self.location = location
        self.compartment_names = ('A','B','C')
        self.location_is_adjustable = self.location_is_adjustable()
        self.location_is_standard = self.location_is_standard()
        self.location_is_valid = self.location_is_adjustable



    def sanity_test(self):
        print(self.h_A, self.rack_dims['A'])
        print(self.h_B, self.rack_dims['B'])
        print(self.h_C, self.rack_dims['C'])
        
    def get_inventory(self):
        return self.inventory

    def get_weights(self):
        weights = [ float(self.get_total_weight(*items)) for items in list(self.inventory.keys()) ]
        wA,wB,wC = weights[0],weights[1],weights[2]
        return dict(A=wA,
                    B=wB,
                    C=wC)


    def parse_gap_description(self,gap_description):
        '''
        parse_gap_description('9(8,3')
        (9,8,3)
        parse_gap_description('9-8,3')
        (9,8,3)
        '''

        return tuple(
                    map(
                        int, filter(str.isdigit, gap_description)
                    )
        )


    def get_parsed_gap_description(self, gapdesc=None):
        if gapdesc is None:
            gapdesc = self.gap_description

        return self.parse_gap_description(str(gapdesc))

    @property
    def gaps_tuple(self):
        '''
        returns like (6,5,4
        '''

        return tuple(
                    map( 
                        int, self.get_parsed_gap_description()
            )
        )


    def get_rackdims_by_spacing(self, parsed_gap_description: tuple)-> dict:
        tup = parsed_gap_description # e.g. (8, 4, 3) all in feet
        return {'A': tup[0]*12, # feet to inches 
                'B': tup[1]*12,
                'C': tup[2]*12}

    # @property
    def get_rackdims(self):
        return self.get_rackdims_by_spacing(self.gaps_tuple)


    @property
    def compartment_sizes(self):
        return self.get_rackdims_by_spacing(self.parse_gap_description(self.gap_description))


    def get_rackdims_by_loc(self, loc):
        if self.location_is_standard:
            return StandardRack().dimensions


    def get_capacity(self, compartment=None):
        return 2000



    def location_is_adjustable(self):
        return (self.location in adjustables or self.location is None)


    def location_is_standard(self):
        return (self.location in standards or self.location is None)


    def set_location(self, location):
        if isinstance(self, AdjustanbleRack):
            if not location in adjustables:
                raise RackLocationError(f'Expected an adjustable rack location; got {location}')
        self.location = location
               

#     def get_volumes(self, container=None):
#         if not container in ('A','B',None):
#             raise ValueError('Param container must be either A, B, or None!')
#         area = self.width*self.length
#         volumeA = self.h_A*area
#         volumeB = self.h_B*area

#         if container is None:
#             return {'A': volumeA,
#                     'B': volumeB}
#         if container == 'A':
#             return volumeA
#         if container == 'B':
#             return volumeB
#         return False


#     def set_shelf_position(self, new_position):

#         if self.items_A or self.items_B:
#             raise ValueError('Shelves must be empty to adjust shelf position!')

#         self.shelf_position = new_position
#         self.h_A = self.shelf_position
#         self.h_B = 15*12 - self.h_A

#         self.renew_volumes()



    def add_item(self, item, target):
        '''
        add an items ok 
        '''
        
        if target not in 'ABC':
            raise ValueError('Try again!')
        self.inventory[target].append(item)
        item.put_item(destloc=self.location, target=target)

        
    def remove_item(self, item, target):
        '''
        Just dont use a key other than ABC, or you will be cursed
        with bad luck for a random number of years that is not smaller
        than the number of hair follicles in all of 1970's film,
        and bounded above by infinity.
        '''
        if target not in 'ABC':
            raise ValueError('Try again!')

        self.inventory[target].remove(item)
    
            
    def get_items(self):
        # return {'A':self.items_A,
        #         'B':self.items_B,
        #         'C': self.items_C}
    
        return { k: self.inventory[k] for k in 'ABC' }
   


    
    
    @property
    def items(self):
        return self.get_items()

    @property
    def is_empty(self):
        return any(self.get_items().keys())

    def __str__(self):
        return f"<AdjustableRack, containing\n {self.inventory}>"
    def __repr__(self):
        return str(self)



        
   
    
@dataclass
class StandardRack(Rack):
    '''
    A stateful class for the standard rack
    that can manipulate items on the shelf.
    :param initial_inventory: the initial shelf state, specified as
        a dictionary with the keys being the compartment names 
        and the values lists of items on the shelf.
    '''
    # class D365_props:
    #     NAME = 'Husky'
    
    
    def __init__(self,   location = None,
                         initial_inventory = 
                                     dict(   
                                        A=[],
                                        B=[],
                                        C=[],
                                        D=[])):
        super().__init__()
        if location is None:
            location = random.choice(standards)
        self.D365_name = 'Husky'
        self.location = location
        self.initial_inventory = initial_inventory
        self.inventory = self.initial_inventory
        
        
        self.refresh_item_data()
 

        

        self.err_msg1 = 'Inital inventory must be a dictionary with the only keys A-D, and values of lists of items in each compartment!'
        
        self.compartment_names = tuple("ABCD")
        self.capacity = 5060 #lbs
        self.width = 42
        self.length = 9*12//2
        self.height = 60
        
        self.dimensions = {
            'width': 42, # inches
            #'length': 9*12, # 9 ft
            'length': 9*12//2, # 9 ft
            'height': 60}
        
        for k,v in self.dimensions.items():
            setattr(self, k, v)
            
            
            
            
            
            
            
        self.compartment_volume = self.width*self.length*self.height
        self.total_volume = 3*self.compartment_volume

        self.location_is_adjustable = self.location_is_adjustable()
        self.location_is_standard = self.location_is_standard()
        self.location_is_valid = self.location_is_standard
    
        

    def location_is_adjustable(self):
        return (self.location in adjustables or self.location is None)


    def location_is_standard(self):
        return (self.location in standards or self.location is None)


    def set_location(self, location):
        if isinstance(self, StandardRack):
            if not location in standards:
                raise RackLocationError(f'Expected an adjustable rack location; got {location}')
        self.location = location
    
    
    def refresh_item_data(self):
        self.items_A = self.inventory['A']
        self.items_B = self.inventory['B']
        self.items_C = self.inventory['C']
        self.items_D = self.inventory['D']
            
       
    def get_items(self):
        return { key: self.inventory[key] for key in tuple("ABCD") }
  
    @property
    def items(self):
        return self.get_items()

        
    def get_empty_compartment_volumes(self):
        dims = self.dimensions
        l,w,h = dims['length'],dims['width'],dims['height']
        vol = l*w*h
        vols = { k:vol for k in self.compartment_names if k != 'D' }
        vols['D'] = np.nan
        return vols
        
    def refresh_data(self):
        self.items = self.get_items()
        
        
    def add_item(self, item: Any=None, target: str=None):
        '''
        Adds an items to the specified target compartment.
        '''
        if target not in self.compartment_names:
            # raise ValueError(f'''Set target equal to one of {self.compartment_names}''')
            target = random.choice('ABCD')
            print(f'Specify a target compartment! randomly chose: {target}')
        self.inventory[target].append(item)
        item.put_item(destloc=self.location)
        self.refresh_item_data()
     
            
    def remove_item(self, item, target):
        
        class ItemRemovalError(Exception):
            pass
     
                    
        if target not in self.compartment_names:
            # raise ValueError(f'''Set target equal to one of {self.compartment_names}''')
            target = random.choice('ABCD')
            print(f'Specify a target compartment! randomly chose: {target}')
        
        try:
            self.inventory[target].remove(item)
            self.refresh_item_data()
        except:
            if item not in self.inventory[target]:
                raise ValueError(f'Item {item} isnt in compartment {target}')
            else:
                raise ItemRemovalError('Something has gone horribly wrong!')
                
                
    def validate_location(self, location=None):
        if location is None:
            test_location = self.location
        if test_location in standards:
            return True
        elif test_location is None:
                      # if self.location_has_never_been_set:
            #     return True
            # else:
            # raise self.RackLocationError('To be honest, this rack is homeless!')
            print('This rack has no location')
            return True
        return False
    
    def rack_is_empty(self):
        return any(self.get_items().keys())
    
    @property
    def is_empty(self):
        return self.rack_is_empty()
   

    def __str__(self):
        return  f"""{self.inventory}>"""

    def __repr__(self):
        return str(self)
    
    
standard_shelf = StandardRack()
adjustable_shelf = AdjustableRack()
    