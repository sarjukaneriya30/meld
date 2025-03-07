from storage.locations import Locations, Bay2locs
from storage.warehouse import front,back,middle
from storage.racks import StandardRack, AdjustableRack
from storage.helpers import HelperFunctions


class LocationParser:
    
    @classmethod
    def parse_location_id(cls, location_id):

        def compartment_to_description(compartment_id='A'):
            if compartment_id == 'A':
                return 'bulk'
            return 'unknown'

        lid = str(location_id)
        #assert(len(lid)==9)
        return {
            'building': lid[0],
            'bay': lid[1],
            'bay_2': lid[2],
            'aisle': lid[3:5],
            'column': lid[5:7],
            'height': lid[7],
            'compartment': lid[8],
            'compartment desc.': compartment_to_description(lid[8])}


    @classmethod
    def location_classifier(cls, loc_tuple, parse_locid=False):
        '''
        print(f'8 – 4 – 3   Section on Front wall of Bay 2\n{front}')
        print(f'6 – 5 - 4      Middle Section of Bay 2\n{middle}')
        print(f'7 – 4 – 4    Section on back wall of Bay 2\n{back}')
        '''

        locations   = Locations()
        adjustables = [ loc for loc in locations.locs if Bay2locs.its_adjustable(loc) ]
        standards   = [ loc for loc in locations.locs if loc not in adjustables ]


        output_class =  []

        loc_tuple = tuple(map(str, loc_tuple))

        if loc_tuple in [ loc for loc in locations.locs if loc not in adjustables ]:
            config = '-'.join([str(int(60/12))]*3)
            output_class = {'racktype':'Standard'}
            output_class['section'] = None
            output_class['rack_config_str'] = config
            output_class['rack_spacing'] = tuple(map(int, config.split('-')))
            output_class['sample_obj'] = StandardRack()
            # if parse_locid:
            #     output_class['parsed_id'] = HelperFunctions.parse_location_id(loc_tuple)
            return output_class


        else:
            output_class = {'racktype': 'Adjustable'}

            if loc_tuple in front:
                config = '8-4-3'
                output_class['section'] = 'front'
                output_class['rack_config_str'] = config
            elif loc_tuple in back:
                config = '7-4-4'
                output_class['section'] = 'back'
                output_class['rack_config_str'] = config
            else:
                config = '6-5-4' 
                output_class['section'] = 'middle'
                output_class['rack_config_str'] = config

            output_class['rack_spacing'] = tuple(map(int, config.split('-')))
            output_class['sample_obj'] = AdjustableRack()
                # if parse_locid:
                #     output_class['parsed_id'] = HelperFunctions.parse_location_id(loc_tuple)
            return output_class
        return output_class
    
Aux = LocationParser