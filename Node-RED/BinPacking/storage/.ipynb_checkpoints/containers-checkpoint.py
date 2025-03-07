from abc import ABC, abstractmethod
from dataclasses import dataclass
# from storage.container_data import ContainerData, Struct
from storage.storable import Storable #, Item
#from storage.item import Item
# from storage.racks import *
# from .racks import Rack, StandardRack, AjustableRack
# from abc import ABC
# from dataclasses import dataclass
import json
import logging
import random
from typing import Any, Dict, Iterable, List, Tuple
import uuid




class Struct:
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
 
    bin_json = {
        'full_name' : 'modular nesting container',
        'name': 'bin',
        'model_no': 'MA24110989',
        'tare_weight': 6, # 5.8lbs
        'dimensions': {
            'units' : 'imperial',
            
            'inner': {
                'width': 20.2,
                'length': 8.4,
                'height': 7.1
            },
            'outer': {
                'width': 23.5,
                'length': 11,
                'height': 9
            }
        }
    }
    bin_object = Struct(**bin_json)


    container_json = {
        'full_name' : 'collapsable bulk container',
        'name': 'pallet',
        'model_no': 'BG48404602',
        'tare_weight': 105, # 100lbs
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
        'tare_weight': 145, #142 lbs
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
    all_strcuts = [Struct(**jsondata) for jsondata in all_jsons]
    
    
    def get_json_by_name(self, short_name):
        return [ d for d in all_jsons if d['name'] == short_name ][0]
    
    
    def get_struct_by_name(self, short_name):
        jsondata = self.get_json_by_name(short_name)
        return Struct(**jsondata)
    
    
CD = ContainerData

class Container(Storable):
    
    def __init__(self, data_file):
  
        # super().__init__()
        # self.uuid = uuid.uuid4()
        self.contents = []
        self.dataf = data_file
        self.object = Struct(**data_file)
        self.data = self.get_data()
        
        self.full_name = data_file.get('full_name',-1)
        self.shortname = data_file.get('name',-1)
        self.model_no = data_file.get('model_no',-1)
        self.tare_weight = self.data['tare_weight']
        
        if any([ self.full_name == 'extra-duty bulk box',
                 self.shortname == 'tote',
                 self.model_no == 'BG48404602']):
                 self.object = Tote()
            
        elif any([  self.full_name == 'collapsable bulk container',
                    self.shortname == 'pallet',
                    # self.shortname == 'bulkcontainer',
                    self.model_no =='BN48452520']):
            self.object = BulkContainer()
            
        elif any([   self.full_name == 'modular nesting container',
                     self.shortname == 'bin',
                     self.model_no == 'MA24110989']):
            self.object = Bin()
        
        if self.name  == 'bin':
            self = Bin()
            return self.object
        elif self.name == 'tote':
            self = Tote()
            return self.object
        else:
            self.name = 'pallet'
            self = BulkContainer()
            return self.object
        
    def refresh_data(self):
        self.weight = self.get_weight() + self.tare_weight
        self.data   = self.get_data()
            
    def get_object(self):
        return self.object

    def get_data(self):
        return self.dataf

    @abstractmethod
    def get_contents(self):
        return self.contents
        

    @abstractmethod
    def __repr__(self):
        return str(self.data)
    
    @staticmethod
    def gen_uuid():
        return uuid.uuid4()

        

class Tote(Container):
    
    def __init__(self):
        # super().__init__(ContainerData.tote_json)
        self.uuid          = self.gen_uuid()
        self.data_object   = ContainerData.tote_object
        self.data          = ContainerData.tote_json
        self.full_name     = self.data['full_name']
        self.name          = self.data['name']
        self.model_no      = self.data['model_no']
        self.dimensions    = self.data['dimensions']
        self.units         = self.dimensions['units']
        self.inner_dimensions = self.dimensions['inner']
        self.outer_dimensions = self.dimensions['outer']
        self.tare_weight = self.data['tare_weight']
        self.location      = None
        self.uuid = uuid.uuid4()
        self.data['uuid'] = self.uuid
        
        self.contents = []
        self.weight   = self.get_weight()
        
    # @staticmethod
    # def gen_uuid():
    #     return uuid.uuid4()
        
    def get_weight(self):
        weight = self.tare_weight
        for item in self.contents:
            weight += item.weight
        self.weight = weight
        return weight
        # reutrn sum([_.weight for _ in self.contents])
        
    #@abstractmethod
    def get_datasheet_as_json(self):
        return self.data
        
    #@abstractmethod
    def get_internal_dimensions(self):
        return self.inner_dimensions
    
    #@abstractmethod
    def get_external_dimensions(self):
        return self.outer_dimensions
    
    #@abstractmethod from Storable
    def get_contents(self):
        return self.contents
    
    def add_item_to_container(self, item, **kwargs):
        '''
        Adds an item to the container.
        :param item: the item to be added
        '''
        
        if isinstance(item, str):
            item = Item(name=item, **kwargs)

        self.contents.append(item)
        self.weight = self.weight + item.weight
  
        
        
    def put_item(self, location, target = None):
        '''
        def put_item(self, location, target=None):
        --------------------------------------------
        Put the tote onto a rack at location.
        :param location: the destination of the tote.
        location can be provided as a tuple or a Rack object.
        
        If the location is provided as a Rack, the items
        location will be reflected as the racks, and the racks
        inventory will be modified to show that the item belongs
        to the rack and compartment (if specified).
        '''
        if isinstance(location, tuple):
            self.location = location
            
        elif isinstance(location, Rack):
            # the containers location becomes the destination rack's
            self.location = location.location
            
            if target is None:
                if isinstance(location, StandardRack):
                    target = random.choice("ABCD")
                else:
                    target = random.choice("AB")
                    
            location.add_item(item, target=target)
 
            
        
    
               
    
    
    
    

class BulkContainer(Container):

    def __init__(self):
        self.uuid = self.gen_uuid()
        self.data_object   = ContainerData.container_object
        self.data          = ContainerData.container_json
        self.full_name     = self.data['full_name']
        self.name          = self.data['name']
        self.model_no      = self.data['model_no']
        self.dimensions    = self.data['dimensions']
        self.units         = self.dimensions['units']
        self.inner_dimensions = self.dimensions['inner']
        self.outer_dimensions = self.dimensions['outer']
        self.location      = None
        self.tare_weight = self.data['tare_weight']
        #self.uuid = self.data['uuid']
        self.data['uuid'] = self.uuid

        self.contents = []
        self.weight   = self.get_weight()
        
    def get_weight(self):
        weight = self.tare_weight
        for item in self.contents:
            weight += item.weight
        self.weight = weight
        return weight


    #@abstractmethod
    def get_datasheet_as_json(self):
        return self.data

    #@abstractmethod
    def get_internal_dimensions(self):
        return self.inner_dimensions

    #@abstractmethod
    def get_external_dimensions(self):
        return self.outer_dimensions

    #@abstractmethod from Storable
    def get_contents(self):
        return self.contents

    def add_item_to_container(self, item, **kwargs):
        '''
        Adds an item to the container.
        :param item: the item to be added
        '''
        
        if isinstance(item, str):
            item = Item(name=item, **kwargs)

        self.contents.append(item)
        self.weight = self.weight + item.weight
        
    def put_item(self, location, target = None):
        '''
        def put_item(self, location, target=None):
        --------------------------------------------
        Put the container onto a rack at location.
        :param location: the destination of the tote.
        location can be provided as a tuple or a Rack object.
        
        If the location is provided as a Rack, the items
        location will be reflected as the racks, and the racks
        inventory will be modified to show that the item belongs
        to the rack and compartment (if specified).
        '''
        if isinstance(location, tuple):
            self.location = location
            
        elif isinstance(location, Rack):
            # the containers location becomes the destination rack's
            self.location = location.location
            
            if target is None:
                if isinstance(location, StandardRack):
                    target = random.choice("ABCD")
                else:
                    target = random.choice("AB")
                    
            location.add_item(item, target=target)
            
        # Warehouse.rack.atloc(location).add_item(self)
        


class Bin(Container):

    def __init__(self):
        self.uuid = self.gen_uuid()
        self.data_object   = ContainerData.bin_object
        self.data          = ContainerData.bin_json
        self.full_name     = self.data['full_name']
        self.name          = self.data['name']
        self.model_no      = self.data['model_no']
        self.dimensions    = self.data['dimensions']
        self.units         = self.dimensions['units']
        self.inner_dimensions = self.dimensions['inner']
        self.outer_dimensions = self.dimensions['outer']
        self.location      = None
        self.tare_weight = self.data['tare_weight']
        #self.uuid = self.gen_uuid()
        self.data['uuid'] = self.uuid

        self.contents = []
        self.weight   = self.get_weight()
        
    def get_weight(self):
        weight = self.tare_weight
        for item in self.contents:
            weight += item.weight
        self.weight = weight
        return weight


    #@abstractmethod
    def get_datasheet_as_json(self):
        return self.data

    #@abstractmethod
    def get_internal_dimensions(self):
        return self.inner_dimensions

    #@abstractmethod
    def get_external_dimensions(self):
        return self.outer_dimensions

    #@abstractmethod from Storable
    def get_contents(self):
        return self.contents

    def add_item_to_container(self, item, **kwargs):
        '''
        Adds an item to the container.
        :param item: the item to be added
        '''
        
        if isinstance(item, str):
            item = Item(name=item, **kwargs)

        self.contents.append(item)
        self.weight = self.weight + item.weight
  
        
        
    def put_item(self, location, target=None):
        '''
        
        def put_item(self, location, target=None):
        --------------------------------------------
        Put the bin onto a rack at location.
        :param location: the destination of the tote.
        location can be provided as a tuple or a Rack object.
        
        If the location is provided as a Rack, the items
        location will be reflected as the racks, and the racks
        inventory will be modified to show that the item belongs
        to the rack and compartment (if specified).
        '''
        if isinstance(location, tuple):
            self.location = location
            
        elif isinstance(location, Rack):
            # the containers location becomes the destination rack's
            self.location = location.location
            
            if target is None:
                if isinstance(location, StandardRack):
                    target = random.choice("ABCD")
                else:
                    target = random.choice("AB")
                    
            location.add_item(item, target=target)
            
        
        




# class Box(ABC):
    
#     bin_json = ContainerData.bin_json
#     bin_object = Struct(**self.bin_json)
#     bin_object.data = self.bin_json

#     container_json = ContainerData.container_json
#     container_object = Struct(**self.container_json)
#     container_object.data = self.container_json

#     tote_json = ContainerData.tote_json
#     tote_object = Struct(**self.tote_json)
#     tote_object.data = self.tote_json

#     all_jsons = [self.bin_json,
#                       self.container_json,
#                       self.tote_json]
#     all_strcuts = [Struct(**jsondata) for jsondata in self.all_jsons]

    
#     def __init__(self, kind=None):
#         if not kind:
#             return
        
#         this = cls.get_by_shortname(kind=kind)
#         this.json = cls.get_json_by_name(short_name=kind)
#         this.modelno = this.json['model_no']
#         this.fullname = this.json['full_name']
        
#         return this
        
    

#     @classmethod
#     def get_by_shortname(cls, kind):
#         if kind.lower() == 'bin':
#             return self.bin_object
#         if kind.lower() == 'container':
#             return self.bin_object
#         if kind.lower() == 'tote':
#             return self.tote_object
        
#     @classmethod
#     def get_box_json_by_name(cls, short_name):
#         return [ d for d in self.all_jsons if d['name'] == short_name ][0]
    
    
#     def get_boxobj_by_name(self, short_name):
#         jsondata = self.get_json_by_name(short_name)
#         return Struct(**jsondata)
    

