from abc import ABC, abstractmethod
import uuid


class Storable(object):
    '''
    The abstract base class for anything which can be stored in a location, on a shelf, in bay 2.
    This includes both items for inventory and containers to put items in.
    '''
    
    @abstractmethod
    def put_item(self):
        pass
    
    @abstractmethod
    def get_item_data(self):
        pass
    
    @abstractmethod
    def get_dimensions(self):
        pass

    @abstractmethod
    def get_contents(self):
        pass
    
    @abstractmethod
    def get_location(self):
        pass
    
    @abstractmethod
    def gen_uuid(self):
        '''
        on init, every item, container, anything storage opt
        cares about gets a UUID that is eternal.
        '''
        return uuid.uuid4() #HelperFunctions.get_uuid()
   
    @abstractmethod
    def __str__(self):
        return str(repr(self))
   
    @abstractmethod
    def __repr__(self):
        return f"< Instance of {self.__class__} >"

    def set_uuid(self):
        self.uuid4 = self.gen_uuid()
        self.uuid  = self.uuid4.hex
    
    def __init__(self):
        self.set_uuid()
    