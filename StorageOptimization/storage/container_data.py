from abc import ABC
from dataclasses import dataclass
import json
import logging
import random
from typing import Any, Dict, Iterable, List, Tuple



class Struct:
    '''
    Class for turning a dict into an object.
    '''
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
    def __str__(self):
        return self.dimensions
    
    def __repr__(self):
        return self.__str__()
        

    
    @staticmethod
    def dict_to_object(dict_item):
        class _:
            pass
        for key, value in dict_item.items():
            setattr(_, key, value)
        return _


@dataclass
class ContainerData(ABC):
    '''
    Contains the dimensions of the storage containers.
    '''
    
 
    bin_json = {
        'full_name' : 'modular nesting container',
        'name': 'bin',
        'model_no': 'MA24110989',
        'weight': 25,
        'dimensions': {
            'units' : 'imperial',
            'inner': {
                'width': 20.2,
                'length': 8.4,
                'height': 7.1
            },
            'outer': {
                'width': 23.5,
                'length': 10,
                'height': 9
            }
            
        }
    }
    bin_object = Struct(**bin_json)


    container_json = {
        'full_name' : 'collapsable bulk container',
        'name': 'pallet',
        'model_no': 'BG48404602',
        'weight': 25,
        'dimensions': {
            'units' : 'imperial',
            'inner': {
                'length': 45,
                'width': 37,
                'height': 40
            },
            'outer': {
                'length': 48,
                'width': 40,
                'height': 46
            }
        }
    }
    container_object = Struct(**container_json)
    
    tote_json = {
        'full_name' : 'extra-duty bulk box',
        'name': 'tote',
        'model_no': 'BN48452520',
        'weight': 25,
        'dimensions': {
            'units' : 'imperial',
            'inner': {
                'length': 44.5,
                'width': 31.5,
                'height': 18.3,
                'volume': 44.5*31.5*18.3
            },
            'outer': {
                'length': 48,
                'width': 45,
                'height': 25,
                'volume': 48*45*25
            }
        }
    }
    tote_object = Struct(**tote_json)
    
    all_jsons = [bin_json,container_json,tote_json]
    
    
    
    
class ContainerGen(ContainerData):
    
    def __init__(self, kind=None):
        super().__init__()
        self.kind = kind

    def get_struct_by_name(self,
                           name):
        
        name = name.lower()
        
        if name == 'bin':
            return self.bin_object
        elif name == 'pallet' or 'bulk' in name.lower():
            return self.container_object
        elif name == 'tote':
            return self.tote_object
    

    def get_json_by_name(self,
                         name):
        
        name = name.lower()
        
        if name == 'bin':
            return self.bin_json
        elif name == 'pallet' or 'bulk' in name.lower():
            return self.container_json
        elif name == 'tote':
            return self.tote_json
    
  
    def get_data_by_key(self,
                        name):
        
        name = name.lower()
        
        return {
            'bin': self.bin_json,
            'pallet': self.container_json,
            'tote': self.tote_json
        }
    
    @staticmethod
    def get_obj_by_key( name):
        
        name = name.lower()
        
        return { k: Struct(**val) for k, val in cls.get_data_by_key(name) }
    
    
    
    
#     def get_json_by_name(self, short_name):
#         return [ d for d in all_jsons if d['name'] == short_name ][0]
    
    
#     def get_struct_by_name(self, short_name):
#         jsondata = self.get_json_by_name(short_name)
#         return Struct(**jsondata)
    
    
CD = ContainerData()
CG = ContainerGen()