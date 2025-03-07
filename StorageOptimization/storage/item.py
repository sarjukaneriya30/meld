from abc import ABC, abstractmethod
import datetime
import numpy as np
import math
import faker
from storage.helpers import HelperFunctions
import json
import jsbeautifier
from storage.locations import locations, adjustables, standards
from storage.storable import Storable
import random
import uuid
from storage.containers import ContainerData, Tote, Bin, BulkContainer

    
    
# class Storable(object):
#     '''
#     The abstract base class for anything which can be stored in a location, on a shelf, in bay 2.
#     This includes both items for inventory and containers to put items in.
#     '''
    
#     @abstractmethod
#     def put_item(self):
#         pass
    
#     @abstractmethod
#     def get_item_data(self):
#         pass
    
#     @abstractmethod
#     def get_dimensions(self):
#         pass


#     @abstractmethod
#     def get_contents(self):
#         pass
    
#     @abstractmethod
#     def get_location(self):
#         pass
    
#     @abstractmethod
#     def gen_uuid(self):
#         '''
#         on init, every item, container, anything storage opt
#         cares about gets a UUID that is eternal.
#         '''
#         return uuid.uuid4() #HelperFunctions.get_uuid()
   
#     @abstractmethod
#     def __str__(self):
#         return str(repr(self))
   
#     @abstractmethod
#     def __repr__(self):
#         return f"< Instance of {self.__class__} >"

#     @property
#     def uuid(self):
#         return self.gen_uuid()
    

    
    

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
    def __str__(self):
        return self.dimensions
    
    def __repr__(self):
        return self.__str__()
    
    




class Item(Storable):
    '''
    Class for creating an inventory item.
    '''
    
    keys = [ 'ID',
             'uuid',
             'priority',
             'destloc',
             'name',
             'length',
             'width',
             'height',
             'volume',
             'weight',
             'is_on_shelf',
             'on_shelf_since',
             'location',
             'put_date_iso',
             'put_date',
             'time_on_shelf',
             'dimensions',
             'NSN']
    
    @staticmethod
    def get_random_dim(lower=4, upper=28):
        return np.random.randint(lower,upper)
    
    @staticmethod
    def impute_dimensions(lower=4, upper=28):
        return [np.random.randint(lower,upper) for x in (1,2,3)]
        
    @staticmethod
    def generate_NSN():
        return np.random.randint(1e7,11e7)
    
    def __init__(self,
                 name=None,
                 NSN=None,
                 l=None, 
                 w=None, 
                 h=None,
                 weight=None,
                 destloc=None,
                 priority=None,
                 impute_all=True,
                 time_on_shelf=0,
                 data_file = None):
        
        super().__init__()
        
        self.is_container = False
        
        self.ikeys = Item.keys
        
        self.fpriority = random.random()
        self.priorities = list('ABC')
        self.int_priority = math.floor(self.fpriority*3)
        self.priority = priority
        if priority is None:
            self.priority = self.priorities[self.int_priority]
     
        if l is None:
            l = Item.get_random_dim(3,40)
        if w is None:
            w = Item.get_random_dim(3,24)
        if h is None:
            h = Item.get_random_dim(3,24)
        
        
        if NSN is None:
            NSN = Item.generate_NSN()
        self.NSN = NSN
  
        self.location          = None 
        self.destloc           = destloc
        self.faker             = faker.Factory.create()
        self.name              = name
        
        if name is None:
            name               = self.NSN

        self.dimensions        = [l,w,h]
        if impute_all:
            self.dimensions    = Item.impute_dimensions()
            
        self.idims             = tuple(self.dimensions)
        
        self.length            = self.dimensions[0]
        self.width             = self.dimensions[1]
        self.height            = self.dimensions[2]
        self.l                 = self.length
        self.w                 = self.width
        self.h                 = self.height
        
        if weight is None:
            weight = Item.get_random_dim(5,200)
        self.weight = weight
        self.volume            = self.get_volume()
        
        self.put_datetime_obj  = None
        self.put_date_iso      = None
        self.put_date          = None
        self.time_on_shelf     = 0
        self.is_on_shelf       = False
        self.data              = self.get_data()
        self.opts              = jsbeautifier.default_options()
        self.opts.indent_size  = 1
        self.fstring           = self.build_fstring()
        self.goes_adj          = HelperFunctions.item_goes_to_adjustable
        self.racktype          = self.get_racktype(self)
        self.put_item_function = random.choice
        # self.uuid              = self.gen_uuid().hex
  
    
    def gen_uuid(self):
        return uuid.uuid4()
    
    #@abstractmethod
    def put_item(self, 
                 shelfID=None):
        # self.target           = target
        self.now              = datetime.datetime.now()
        self.location         = shelfID
        self.new_destloc      = shelfID
        self.destloc          = self.new_destloc
        self.is_on_shelf      = True
        
        self.put_datetime_obj = datetime.datetime.now()
        self.put_date_iso     = datetime.datetime.now().isoformat()
        self.put_date         = datetime.datetime.now().ctime()
        self.time_on_shelf    = 0
        self.fstring          = self.build_fstring()
    
    #@abstractmethod
    def get_item_data(self):
        return self.get_data()
    
    #@abstractmethod
    def get_location(self):
        return self.location
    
    #@abstractmethod
    def get_dimensions(self):
        dims = dict()
        data = self.data
        for k in  ('length','width','height','volume', 'weight'):
            dims[k] = data.get(k, None)
        return { k:v for k,v in dims.items() if v is not None }
    
    #@abstractmethod
    def get_contents(self):
        return "Item is not a container!"
    
    def get_racktype(self, item):
        if self.goes_adj(item):
            return 'Adjustable'
        return 'Standard'
    
    def get_volume(self):
        try:
            return self.length*self.width*self.height
        except:
            return -1
    
    def get_priority(self, priority=None):
        return self.priority
   
 
    def build_fstring(self):
        # self.data['uuid'] = self.data['uuid'].hex
        return jsbeautifier.beautify(json.dumps(self.data), self.opts)
    
    def get_data(self):
        outdata = dict()
        for key in self.ikeys:
            try:
                outdata[key] = self.__dict__[key]
            except:
                pass
        return outdata

    
    def get_time_on_shelf(self):
        if self.is_on_shelf:
            return (datetime.datetime.now() - self.put_datetime_obj).seconds
        return -1
 

    def refresh_data(self):
        self.data = self.get_data
        self.time_on_shelf = self.get_time_on_shelf()
        self.fstring = self.build_fstring()
        
        
    def get_strdims(self):
        return str(self.dimensions).replace('[','').replace(']','').replace(',',' x')
        
        
    def __str__(self):
        self.refresh_data()
        return self.fstring
    
    
    def __repr__(self):
        return jsbeautifier.beautify(json.dumps(self.data), self.opts)
    
    @property
    def d(self):
        print(self.build_fstring)

        
class UUIDifier:
    
    def __init__(self):
        def get_uuid():
            return uuid.uuid4()
    
    
