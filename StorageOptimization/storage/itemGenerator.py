#!/usr/bin/env python3
import storage
import random
import copy
from pprint import pprint
from dataclasses import dataclass
import time
from pprint import pprint



# N_CONTAINERS = 10
# N_ITEMS = 1000

@dataclass
class Params:
    n_containers = 10
    n_items = 1000
    state = [1,2,3,4,5,6,7,8,9]


class ItemGenerator:

    @staticmethod
    def get_empty_containers(n_containers:int = Params.n_containers):
        
        boxes = [ storage.Tote(), storage.Bin(), storage.BulkContainer() ]
        containers = []
        for i in range(n_containers):
            containers.append(copy.deepcopy((random.choice(boxes))))
        return containers

    @staticmethod
    def get_items_list(n_items=Params.n_items):
        
        items_list = []
        for i in range(n_items):
            items_list.append(storage.Item())
        return items_list

    @staticmethod
    def fill_containers(containers: list, 
                        items_list: list):
        for container in containers:
            anitem = random.choice(items_list)
            container.add_item_to_container(anitem)
        filled_containers = copy.deepcopy(containers)
        return fill_containers
    
    @staticmethod
    def get_containers_filled_with_items(n_containers=Params.n_containers, 
                                         n_items=Params.n_items):
        
        containers = get_empty_containers(n_containers=n_containers)
        items      = get_items_list(n_items=n_items)
        return fill_containers(containers, items)



def main(init_state = None):
    '''
    def main(state = [4,2,1,0]):
        < code >
    This will also depend on data stored in
    the Params dataclass
    '''
    
    state = init_state or Params.state
    
    @dataclass
    class helperinfo:
       
        def __init__(self):
            self.msg1=f'state given is: {state}'
            self.msg2=f'''Params are:\n\tn_containers = {Params.n_containers}, n_items={Params.n_items}'''
            self.params = Params()
            print(msg1,msg2,params)
            print(self.__dict__())
    
    # now we get n_containers empties (can be anything)
    
    containers = ItemGenerator.get_empty_containers()
    
    print(f'''
Below is the state of the {Params.n_containers} container objects that were generated, 
represented as the number of items contained in each position\n''')
    print([ len(x.contents) for x in containers ])
    
    print('\n\n...One could imagine an idiom to get us the weights of each container...\n')
    
    print('''How about [ _.weight for _ in containers ]?\n''')
    print([ _.weight for _ in containers ])
    
    print('''\n\nOr, How about [ _.name for _ in containers ]?\n''')
    print([ _.name for _ in containers ])
          
    print('''
\n...Or is it? And it would be different if there we more than one item in a container.
Try this instead - sum the weight of the container itself with the sum of the item weights within each! \n
[ _.weight + sum(item.weight for item in _.contents) for _ in containers ]\nyields\n
            ''')
    
    print([ _.weight + sum(item.weight for item in _.contents) for _ in containers ])
    
    # get random index
    index = random.choice(range(len(containers)))
    
    print(f'Now\n\nLets now choose a random container to add a random item to... {index} was chosen.')
    
    print('... and drop an item into it\n')
    containers[-1].add_item_to_container(storage.Item())
    print('\nso that the new state becomes:')
    print([ len(x.contents) for x in containers ])
    
    print('... or how about...')
    time.sleep(3)
    print('We add items to a random slot a lot of times, say, 20x')
    
    
    for _ in range(20):
        chosen_one = random.choice(containers)
        chosen_one.add_item_to_container(storage.Item(weight=10+_**3//2))
        
    print('done')
        
    print('new state:\n\n')
    print('counts of items;', [ len(x.contents) for x in containers ])
    print('weights of containers;',  [ _.weight for _ in containers ])
    print('their names and IDs...')
    pprint([ (_.name,_.uuid) for _ in containers] )
    print('\n\n...15 second pause.')
    
    
    
    
    time.sleep(10)
    
    m = "Now we examine the inside of a container"
    
    index = random.choice(range(len(containers)))
                          
    chosen_one = containers[index]
                          
    print('The number of items is:' , len(chosen_one.contents), 'and the items themselves are here in fact.')
    
    items = chosen_one.contents
    
    chosen_item = random.choice(items)
    
    print(chosen_item.data)
    
    print('\n\n...15 second pause.')
    
    
    
    
    time.sleep(10)
    
    
    
    print('\n\n')
    print('='*80)
    print(f'\t\t\t\tThere shall now be {len(state)} rounds of modifications.')
    print('='*80)
    print('='*80)
    round_number = 1
    while state:
        
        '''
        we pluck the last value in the state list
        this value shall be the index where a new 
        items is placed.
        '''
        index = state.pop()
        print('='*30+f'\tround {round_number}\t'+'='*30)
        print(f'Initial state: {state}')
        print(f'Chosen indes: {index}!')
        
        containers[index].add_item_to_container(storage.Item())
        print([ len(x.contents) for x in containers ])
        print('='*80)
        print('='*80)

    print([ len(x.contents) for x in containers ])
    
    
    
if __name__ == '__main__':
    main()

