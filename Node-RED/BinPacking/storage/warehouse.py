import copy
from storage.container_data import Struct, ContainerData
from dataclasses import dataclass
from itertools import filterfalse
# from storage.helpers import helper
from os.path import isfile
import pandas as pd
from storage.racks import standard_shelf, adjustable_shelf
import sys
from typing import List, Tuple



###############################################################################
###############################################################################
###################     warehouse & locations     #############################
###############################################################################
###############################################################################
###############################################################################


@dataclass
class Locations:
    
    locs = [('91', '3'), ('90', '3'), ('89', '3'), ('88', '3'), ('87', '3'), ('86', '3'), ('85', '3'), ('84', '3'), ('80', '3'), ('79', '3'), ('78', '3'), ('77', '3'), ('76', '3'), ('75', '3'), ('74', '3'), ('73', '3'), ('68', '3'), ('67', '3'), ('66', '3'), ('65', '3'), ('64', '3'), ('63', '3'), ('62', '3'), ('61', '3'), ('56', '3'), ('55', '3'), ('54', '3'), ('53', '3'), ('52', '3'), ('51', '3'), ('91', '4'), ('91', '6'), ('90', '4'), ('90', '6'), ('89', '4'), ('89', '6'), ('88', '4'), ('88', '6'), ('87', '4'), ('87', '6'), ('86', '4'), ('86', '6'), ('85', '4'), ('85', '6'), ('84', '4'), ('84', '6'), ('80', '4'), ('80', '6'), ('79', '4'), ('79', '6'), ('78', '4'), ('78', '6'), ('77', '4'), ('77', '6'), ('76', '4'), ('76', '6'), ('75', '4'), ('75', '6'), ('74', '4'), ('74', '6'), ('73', '4'), ('73', '6'), ('68', '4'), ('68', '6'), ('67', '4'), ('67', '6'), ('66', '4'), ('66', '6'), ('65', '4'), ('65', '6'), ('64', '4'), ('64', '6'), ('63', '4'), ('63', '6'), ('62', '4'), ('62', '6'), ('61', '4'), ('61', '6'), ('56', '4'), ('56', '6'), ('55', '4'), ('55', '6'), ('54', '4'), ('54', '6'), ('53', '4'), ('53', '6'), ('52', '4'), ('52', '6'), ('51', '4'), ('51', '6'), ('91', '7'), ('91', '9'), ('90', '7'), ('90', '9'), ('89', '7'), ('89', '9'), ('88', '7'), ('88', '9'), ('87', '7'), ('87', '9'), ('86', '7'), ('86', '9'), ('85', '7'), ('85', '9'), ('84', '7'), ('84', '9'), ('80', '7'), ('80', '9'), ('79', '7'), ('79', '9'), ('78', '7'), ('78', '9'), ('77', '7'), ('77', '9'), ('76', '7'), ('76', '9'), ('75', '7'), ('75', '9'), ('74', '7'), ('74', '9'), ('73', '7'), ('73', '9'), ('68', '7'), ('68', '9'), ('67', '7'), ('67', '9'), ('66', '7'), ('66', '9'), ('65', '7'), ('65', '9'), ('64', '7'), ('64', '9'), ('63', '7'), ('63', '9'), ('62', '7'), ('62', '9'), ('61', '7'), ('61', '9'), ('56', '7'), ('56', '9'), ('55', '7'), ('55', '9'), ('54', '7'), ('54', '9'), ('53', '7'), ('53', '9'), ('52', '7'), ('52', '9'), ('51', '7'), ('51', '9'), ('91', '10'), ('90', '10'), ('89', '10'), ('88', '10'), ('87', '10'), ('86', '10'), ('85', '10'), ('84', '10'), ('91', '17'), ('90', '17'), ('89', '17'), ('88', '17'), ('87', '17'), ('86', '17'), ('80', '10'), ('79', '10'), ('78', '10'), ('77', '10'), ('76', '10'), ('75', '10'), ('74', '10'), ('73', '10'), ('79', '17'), ('78', '17'), ('77', '17'), ('76', '17'), ('75', '17'), ('74', '17'), ('73', '17'), ('72', '17'), ('71', '17'), ('70', '17'), ('68', '10'), ('67', '10'), ('66', '10'), ('65', '10'), ('64', '10'), ('63', '10'), ('62', '10'), ('61', '10'), ('69', '17'), ('68', '17'), ('67', '17'), ('66', '17'), ('65', '17'), ('64', '17'), ('63', '17'), ('62', '17'), ('61', '17'), ('56', '10'), ('55', '10'), ('54', '10'), ('53', '10'), ('52', '10'), ('51', '10'), ('56', '17'), ('55', '17'), ('54', '17'), ('53', '17'), ('52', '17'), ('51', '17'), ('91', '18'), ('91', '20'), ('90', '18'), ('90', '20'), ('89', '18'), ('89', '20'), ('88', '18'), ('88', '20'), ('87', '18'), ('87', '20'), ('86', '18'), ('86', '20'), ('79', '18'), ('79', '20'), ('78', '18'), ('78', '20'), ('77', '18'), ('77', '20'), ('76', '18'), ('76', '20'), ('75', '18'), ('75', '20'), ('74', '18'), ('74', '20'), ('73', '18'), ('73', '20'), ('72', '18'), ('72', '20'), ('71', '18'), ('71', '20'), ('70', '18'), ('70', '20'), ('69', '18'), ('69', '20'), ('68', '18'), ('68', '20'), ('67', '18'), ('67', '20'), ('66', '18'), ('66', '20'), ('65', '18'), ('65', '20'), ('64', '18'), ('64', '20'), ('63', '18'), ('63', '20'), ('62', '18'), ('62', '20'), ('61', '18'), ('61', '20'), ('56', '18'), ('56', '20'), ('55', '18'), ('55', '20'), ('54', '18'), ('54', '20'), ('53', '18'), ('53', '20'), ('52', '18'), ('52', '20'), ('51', '18'), ('51', '20'), ('91', '21'), ('91', '23'), ('90', '21'), ('90', '23'), ('89', '21'), ('89', '23'), ('88', '21'), ('88', '23'), ('87', '21'), ('87', '23'), ('86', '21'), ('86', '23'), ('79', '21'), ('79', '23'), ('78', '21'), ('78', '23'), ('77', '21'), ('77', '23'), ('76', '21'), ('76', '23'), ('75', '21'), ('75', '23'), ('74', '21'), ('74', '23'), ('73', '21'), ('73', '23'), ('72', '21'), ('72', '23'), ('71', '21'), ('71', '23'), ('70', '21'), ('70', '23'), ('69', '21'), ('69', '23'), ('68', '21'), ('68', '23'), ('67', '21'), ('67', '23'), ('66', '21'), ('66', '23'), ('65', '21'), ('65', '23'), ('64', '21'), ('64', '23'), ('63', '21'), ('63', '23'), ('62', '21'), ('62', '23'), ('61', '21'), ('61', '23'), ('56', '21'), ('56', '23'), ('55', '21'), ('55', '23'), ('54', '21'), ('54', '23'), ('53', '21'), ('53', '23'), ('52', '21'), ('52', '23'), ('51', '21'), ('51', '23'), ('91', '24'), ('90', '24'), ('89', '24'), ('88', '24'), ('87', '24'), ('86', '24'), ('79', '24'), ('79', '26'), ('78', '24'), ('78', '26'), ('77', '24'), ('77', '26'), ('76', '24'), ('76', '26'), ('75', '24'), ('75', '26'), ('74', '24'), ('74', '26'), ('73', '24'), ('73', '26'), ('72', '24'), ('72', '26'), ('71', '24'), ('71', '26'), ('70', '24'), ('70', '26'), ('69', '24'), ('69', '26'), ('68', '24'), ('68', '26'), ('67', '24'), ('67', '26'), ('66', '24'), ('66', '26'), ('65', '24'), ('65', '26'), ('64', '24'), ('64', '26'), ('63', '24'), ('63', '26'), ('62', '24'), ('62', '26'), ('61', '24'), ('61', '26'), ('56', '24'), ('56', '26'), ('55', '24'), ('55', '26'), ('54', '24'), ('54', '26'), ('53', '24'), ('53', '26'), ('52', '24'), ('52', '26'), ('51', '24'), ('51', '26'), ('79', '27'), ('79', '29'), ('78', '27'), ('78', '29'), ('77', '27'), ('77', '29'), ('76', '27'), ('76', '29'), ('75', '27'), ('75', '29'), ('74', '27'), ('74', '29'), ('73', '27'), ('73', '29'), ('72', '27'), ('72', '29'), ('71', '27'), ('71', '29'), ('70', '27'), ('70', '29'), ('69', '27'), ('69', '29'), ('68', '27'), ('68', '29'), ('67', '27'), ('67', '29'), ('66', '27'), ('66', '29'), ('65', '27'), ('65', '29'), ('64', '27'), ('64', '29'), ('63', '27'), ('63', '29'), ('62', '27'), ('62', '29'), ('61', '27'), ('61', '29'), ('56', '27'), ('56', '29'), ('55', '27'), ('55', '29'), ('54', '27'), ('54', '29'), ('53', '27'), ('53', '29'), ('52', '27'), ('52', '29'), ('51', '27'), ('51', '29'), ('91', '34'), ('90', '34'), ('89', '34'), ('88', '34'), ('87', '34'), ('86', '34'), ('85', '34'), ('84', '34'), ('80', '34'), ('79', '34'), ('78', '34'), ('77', '34'), ('76', '34'), ('75', '34'), ('74', '34'), ('73', '34'), ('68', '34'), ('67', '34'), ('66', '34'), ('65', '34'), ('64', '34'), ('63', '34'), ('62', '34'), ('61', '34'), ('56', '34'), ('55', '34'), ('54', '34'), ('53', '34'), ('52', '34'), ('51', '34'), ('91', '35'), ('91', '37'), ('90', '35'), ('90', '37'), ('89', '35'), ('89', '37'), ('88', '35'), ('88', '37'), ('87', '35'), ('87', '37'), ('86', '35'), ('86', '37'), ('85', '35'), ('85', '37'), ('84', '35'), ('84', '37'), ('80', '35'), ('80', '37'), ('79', '35'), ('79', '37'), ('78', '35'), ('78', '37'), ('77', '35'), ('77', '37'), ('76', '35'), ('76', '37'), ('75', '35'), ('75', '37'), ('74', '35'), ('74', '37'), ('73', '35'), ('73', '37'), ('68', '35'), ('68', '37'), ('67', '35'), ('67', '37'), ('66', '35'), ('66', '37'), ('65', '35'), ('65', '37'), ('64', '35'), ('64', '37'), ('63', '35'), ('63', '37'), ('62', '35'), ('62', '37'), ('61', '35'), ('61', '37'), ('56', '35'), ('56', '37'), ('55', '35'), ('55', '37'), ('54', '35'), ('54', '37'), ('53', '35'), ('53', '37'), ('52', '35'), ('52', '37'), ('51', '35'), ('51', '37'), ('91', '38'), ('91', '40'), ('90', '38'), ('90', '40'), ('89', '38'), ('89', '40'), ('88', '38'), ('88', '40'), ('87', '38'), ('87', '40'), ('86', '38'), ('86', '40'), ('85', '38'), ('85', '40'), ('84', '38'), ('84', '40'), ('80', '38'), ('80', '40'), ('79', '38'), ('79', '40'), ('78', '38'), ('78', '40'), ('77', '38'), ('77', '40'), ('76', '38'), ('76', '40'), ('75', '38'), ('75', '40'), ('74', '38'), ('74', '40'), ('73', '38'), ('73', '40'), ('68', '38'), ('68', '40'), ('67', '38'), ('67', '40'), ('66', '38'), ('66', '40'), ('65', '38'), ('65', '40'), ('64', '38'), ('64', '40'), ('63', '38'), ('63', '40'), ('62', '38'), ('62', '40'), ('61', '38'), ('61', '40'), ('56', '38'), ('56', '40'), ('55', '38'), ('55', '40'), ('54', '38'), ('54', '40'), ('53', '38'), ('53', '40'), ('52', '38'), ('52', '40'), ('51', '38'), ('51', '40'), ('91', '41'), ('90', '41'), ('89', '41'), ('88', '41'), ('87', '41'), ('86', '41'), ('85', '41'), ('84', '41'), ('80', '41'), ('79', '41'), ('78', '41'), ('77', '41'), ('76', '41'), ('75', '41'), ('74', '41'), ('73', '41'), ('68', '41'), ('67', '41'), ('66', '41'), ('65', '41'), ('64', '41'), ('63', '41'), ('62', '41'), ('61', '41'), ('56', '41'), ('55', '41'), ('54', '41'), ('53', '41'), ('52', '41'), ('51', '41')]
    
    def __init__(self):
        self.locs = Locations.locs
        
    @classmethod
    def warehouse_layout(cls, priority_mapper=None):
        
        
        if priority_mapper is None:

            locs_dicts = []
            for loc in cls.locs:
                if Bay2locs.its_adjustable(loc):
                    shelftype = 'Adjustable'
                    shelfname = 'Vidmar'
                    shelfpriority = random.choice('ABC')
                else:
                    shelftype = 'Standard'
                    shelfname = 'Husky'

                aisle = loc[1]
                column = loc[0]
                ldict = {
                    'ShelfAisle': aisle,
                    'ShelfColumn': column,
                    'ShelfType': racktype,
                    'ShelfPriotity': rackname
                }
                locs_dicts.append(ldict)
            
        return locs_dicts
    

        



@dataclass
class Bay2locs(object):
    '''
    Here we will encode the properties of bay two
    including which shelves are which and what aisle,columns are allowed
    and of which type.
    '''

    
    class BrutalError(Exception):
        pass

    @classmethod
    def its_adjustable(cls,location):
        aisle = int(location[1])
        if aisle < 30 and aisle > 16:
            return True
        return False
    
    
        
    def __init__(self):
        self.locations = locations
        self.nlocs = [ tuple(map(int, thing)) for thing in self.locations ]
        self.adjustables = list(filter(self.its_adjustable, self.locations))
        self.standards = list(filterfalse(self.its_adjustable, self.locations))
        self.front, self.middle, self.back = self.differentiate_adjustables()
        

    @classmethod
    def differentiate_adjustables(cls, default=True):
            
        front,middle,back = [],[],[]
        for loc in [tuple(map(int, thing)) for thing in adjustables]:
                if loc[0]>85:
                    front.append(loc)
                elif loc[0]<57:
                    back.append(loc)
                else:
                    middle.append(loc)
        return front, middle, back
    


    @staticmethod
    def test():
        passed = (len(nlocs) == len(adjustables) + len(standards))
        if passed:
            print('YAY!')
        raise BrutalError('This should never have happened!')

        
        
locations   = Locations()
adjustables = [ loc for loc in locations.locs if Bay2locs.its_adjustable(loc) ]
standards   = [ loc for loc in locations.locs if loc not in adjustables ]
front,middle,back = Bay2locs.differentiate_adjustables()


class Adjustable_locs:
    
    def __init__(self):
        self.all_locs = Bay2locs.differentiate_adjustables()
        self.front = self.all_locs[0]
        self.middle = self.all_locs[1]
        self.back = self.all_locs[2]
        self.sfront = self.stringify(self.front)
        self.smiddle = self.stringify(self.middle)
        self.sback = self.stringify(self.sback)
    
    def stringify(locs: List[Tuple]):
        return [ tuple(map(str, _ )) for _ in locs ]
    
    def __repr__(self):
        return f'Adjustable shelf locs, boii'
    

    


# @metaclass
class Warehouse(object):

    def __init__(self, infile=None):
        
        if infile is None:

            self.locations   = Locations()
            self.adjustables = [ loc for loc in locations.locs if Bay2locs.its_adjustable(loc) ]
            self.standards   = [ loc for loc in locations.locs if loc not in adjustables ]



            
            self.front = []
            self.back = []
            self.middle = []
            

            for loc in [tuple(map(int, thing)) for thing in adjustables]:
                if loc[0]>85:
                    self.front.append(loc)
                elif loc[0]<57:
                    self.back.append(loc)
                else:
                    self.middle.append(loc)
        
        else:
            
            raise InvalidWarehouseInputError(infile=infile, message = 'We are not prepared for any input file quitre yet.')
            
            @property
            def compartment_keys(self):
                return self.get_shelf_ids().keys()
            
            
    def get_locs(self, 
                 verbose=None, 
                 asstruct: bool=False, 
                 include_compartment: bool=True) -> object:
        '''
        
            def get_locs(self, 
                 verbose=None, 
                 asstruct=False, 
                 with_comparement=True):
                 
                 
        Returns a struct that has as data, the locations of the various shelf types in
        column, aisle format.
        
        Set asstruct to false to get a dict back.
        :param include_compartments: if this is set to true, the result will be 3x the size, having keys 3 
        keys for location 57,3 e.g. (57,3, x) for x in 'ABC'
        '''

        self.all_locs = {
            'all': self.locations.locs,
            'standard': self.standards,
            'adjustable_all': self.adjustables,
            'adjustable_front': self.front, #8-4-3
            'adjustable_middle': self.middle, #6-5-4
            'adjustable_back': self.back
            }

        if verbose:
            print('Standard have are 4.5 ft legnth, 46 inches depth, 60inches height')

            print(f'8 – 4 – 3   Section on Front wall of Bay 2\n{front}')
            print(f'6 – 5 - 4      Middle Section of Bay 2\n{middle}')
            print(f'7 – 4 – 4    Section on back wall of Bay 2\n{back}')
            # assert(len(adjustable_locs)+len(standard_locs)==len(locations))

        return self.all_locs

    
    def get_shelf_ids(self,
                     verbose=None,
                     id_mapper=None):
        

        
        def add_comparements_to_loc(input_tuples):
            '''
            Takes a loc (x,y) and gets [(x,y,c) for cin 'ABC']
            '''
            
            if verbose:
                print('Standard have are 4.5 ft legnth, 46 inches depth, 60inches height')

                print(f'8 – 4 – 3   Section on Front wall of Bay 2\n{front}')
                print(f'6 – 5 - 4      Middle Section of Bay 2\n{middle}')
                print(f'7 – 4 – 4    Section on back wall of Bay 2\n{back}')
                # assert(len(adjustable_locs)+len(standard_locs)==len(locations))


            outp = []
            for loc in input_tuples:
                col,aisle = int(loc[0]), int(loc[1])
                outp = outp + [ (col,aisle,compartment) for compartment in 'ABC' ]
            return outp
        
        if id_mapper is None: # assume it is bay 2
            return {
            'all': add_comparements_to_loc(self.locations.locs),
            'standard': add_comparements_to_loc(self.standards),
            'adjustable_all': add_comparements_to_loc(self.adjustables),
            'adjustable_front': add_comparements_to_loc(self.front), #8-4-3
            'adjustable_middle': add_comparements_to_loc(self.middle), #6-5-4
            'adjustable_back': add_comparements_to_loc(self.back)
            }
        else:

            pass

       
     

        

        
        
        
        
        
    
                                           
                                           
                                         
 