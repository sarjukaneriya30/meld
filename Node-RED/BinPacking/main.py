import copy
import json
import kpacker
import numpy as np
from pprint import pprint
import py3dbp
import storage
import sys
from typing import List, Tuple, Any
from collections import Counter


MAX_NUM_BINS = 3

def get_kitem():
    '''
    Using Storage.Item() to get random attributed, gets
    a random py3dbp.Item.
    '''
    si = storage.Item()
    dims = si.dimensions
    return kpacker.Item('item', dims[0], dims[1], dims[2], 10)


def get_kbc():
    '''
    Returns a py3dbp.Bin of the size of a
    storage.BulkContainer (BULK CONTAINER ____DOOR)
    '''
    bc = storage.Struct(**storage.BulkContainer().dimensions['inner'])
    return py3dbp.Bin('bc', bc.length, bc.width, bc.height, 5000)


def get_kbin():
    '''
    Returns a py3dbp.Bin of the size of a
    storage.Bin (MODULAR NESTING CONT)
    '''
    binn = storage.Struct(**storage.Bin().dimensions['inner'])
    return py3dbp.Bin('bin', binn.length, binn.width, binn.height, 5000)


def get_ktote():
    '''
    Returns a py3dbp.Bin of the size of a
    storage.Tote (BULK CONTAINER ____DOOR)
    '''
    tote = storage.Struct(**storage.Tote().dimensions['inner'])
    return py3dbp.Bin('tote', tote.length, tote.width, tote.height, 5000)





def get_bin(btype=None):
    '''
    Returns a py3dbp.Bin of one of the following fixed dimensiosns:

    MODULAR NESTING CONT == storage.Bin;
    BULK CONTAINER WIDTHDOOR == storage.(Tote, BulkContainer)
    BULK CONTAINER LENGTHDOOR ==  storage.(Tote, BulkContainer)

    :param btype=None:
    Determines with container type is returned.

    =====================
    Btype Value | Returns
    =====================
    'small'     | get_kbin()
    'medium'    | get_ktote()
     any        | get_kbc()
    '''

    if btype in ('small','s'):
        return get_kbin()
    elif btype in ('medium','m'):
        return get_ktote()
    else:
        return get_kbc()





def get_bins(bin_combo):
    '''
    get_bins(bin_combo: Tuple[int]) -> bins: Tuple[py3dbp.Bin]:

    Takes an argument of the form (m,n,l) and gets
    n,m,l number of small, medium, and large containers
    as a list.

    For example get_bins((2,1,1)) returns 2 smalls, 1 medium,
    and 1 large container as a list.

    :param bin_combo: Tuple[int]
    The tuple (n,m,l) specifying the numbers of small, medium,
    and large containers to return.
    :returns bin_combo: List[py3dbp.Bin]


    [
        py3dbp.Bin, # modular nesting
        py3dbp.Bin, # modular nesting
        py3dbp.Bin, # tote
        py3dbp.Bin, # bulk container
    ]
    '''
    bins_output = []

    num_binn = bin_combo[0]
    num_tote = bin_combo[1]
    num_bc   = bin_combo[2]

    for n in range(num_binn):
        b = get_kbin()
        b.name = 'MODULAR NESTING CONT'
        bins_output.append(b)

    for n in range(num_tote):
        b = get_ktote()
        b.name = 'BULK CONT WIDTHDOOR'
        bins_output.append(b)

    for n in range(num_bc):
        b = get_kbc()
        b.name = 'BULK CONT LENGTHDOOR'
        bins_output.append(b)

    return bins_output



def enrich(packer):
    '''
    f: packer -> packer

    This function takes a py3dbp.

    Set properites volume, item_volume,
    filling_ratio, and empty for each bin.

    set wasted volume for packer
    '''

    for b in packer.bins:
        # bin_vol = float(b.get_volume())
        b.volume = float(b.get_volume())
        b.item_vol = 0

        for item in b.items:
            item.volume = float(item.get_volume())
            b.item_vol += item.volume

        b.filling_ratio = b.item_vol/b.volume
        b.empty = b.volume - b.item_vol



    packer.wasted_volume = sum([b.empty for b in packer.bins])
    packer.n_unfit = len(packer.items)
    packer.packed = packer.n_unfit == 0
    packer.success = False or packer.packed

    return packer



def drop_empty_bins(packer):
    good_bins = []

    for idx, b in enumerate(packer.bins):
        if not(len(b.items)) == 0:
            #print(idx, len(b.items))
            good_bins.append(b)

    packer.bins = good_bins
    packer = enrich(packer)
    return packer



def report(packer):
    '''
    Prints a verbose output of the result of the
    packing.
    '''
    try:  print(f"PACKER BIN COMBO: {packer.combo}")
    except: pass
    print(f"PACKER WASTED VOLUME:\t{packer.wasted_volume}")
    print(f"SUCCESS: {packer.packed}")
    print(f"PACKER UNFIT ITEMS: \t{len(packer.items)}\n")

    for b in packer.bins:
        print("\t:::::::::::", b.string())

        print(f"\tFITTED ITEMS:\t{len(b.items)}")
        # for item in b.items:
        #     print("====> ", item.string())

        print(f"\tUNFITTED ITEMS:\t{len(b.unfitted_items)}")
        # for item in b.unfitted_items:
        #     print("====> ", item.string())

        print(f"\tFILLING RATIO:\t{b.filling_ratio}")

        print("\t***************************************************")
        # print("***************************************************")



def get_combos(N):
    '''
    get_combos(N: int) -> List[Tuple[int]]

    Gets all the combos of three integers n,m,l where each
    n,m,l < N.
    '''
    return  [ (x,y,z) for x in range(N) for y in range(N) for z in range(N) ]




def parse_item(item):
        '''
        {
          "ItemsToPack": [
            {
              "NIIN:": "YYYYYY",
              "LENGTH": 5,
              "HEIGHT": 10,
              "WIDTH": 4,
              "WEIGHT": 5,
              "PRIORITY": "A"
            }
          ]
        }
        '''
        return kpacker.Item(str(item['NIIN']),
                            item['LENGTH'],
                            item['HEIGHT'],
                            item['WIDTH'],
                            item['WEIGHT'])


def get_packing(items=None):
    '''
    get_packing(items: List[Any]) -> int

    Uses get_combos to pack the items into all possible combinations
    of MAX_NUM_BINS bins.  The default value of MAX_NUM_BINS = 3.
    '''

    if items is None:
        items = []
        for i in range(10):
            items.append(get_kitem())

    else:
        items = [ parse_item(item) for item in items['ItemsToPack'] ]


    all_results = dict()
    wasted_volumes = dict()


    for bin_combo in get_combos(MAX_NUM_BINS - 1):

        packer = py3dbp.Packer()
        packer.combo = bin_combo


        for b in get_bins(bin_combo):
            packer.add_bin(b)

        for item in items:
            packer.add_item(item)


        packer.pack(distribute_items=True)
        packer = enrich(packer)

        all_results[bin_combo] = packer
        wasted_volumes[bin_combo] = packer.wasted_volume



    def find_best_result(results):

        best_vol = np.inf
        best_k = None

        for k,v in results.items():
            if v.packed == True:
                if v.wasted_volume < best_vol:
                    best_vol = v.wasted_volume
                    best_k = k

        return best_k


    return find_best_result(all_results)


def main(items=None):
    string_output = '' # build a string to return

    packer = py3dbp.Packer()
    add_items = []
    # add duplicate NIIN as new items to pack
    for i in items['ItemsToPack']:
        if 'QTY' in i.keys():
            if i['QTY']>1:
                for q in range(int(i['QTY'])-1):
                    add_items.append(i)
    items['ItemsToPack'] = items['ItemsToPack'] + add_items

    bin_combo = get_packing(items=items)
    bins = get_bins(bin_combo)

    for b in bins:
        packer.add_bin(b)

    n_items = 0
    for item in items['ItemsToPack']:
        packer.add_item(parse_item(item))
        n_items += 1

    packer.pack(distribute_items=1)

    packer = enrich(packer)
    all_bins_content=[]
    all_bin_names = []
    n_bins = 0


    for b in bins:
        n_bins += 1
        all_bin_names.append(b.name)

        this_bin = {
                'ContainerType': b.name,
                'items': [],
                'n_items': 0
                }


        for i in b.items:
            this_bin['items'].append(i.name)
            this_bin['n_items'] += 1

        all_bins_content.append(this_bin)

    string_output = string_output + f'Number of items: {n_items}\n'
    string_output = string_output + f'Number of bins used: {n_bins}\n'
    string_output = string_output + f'Bins used: {", ".join(all_bin_names)}\n'
    string_output = string_output + 'Result:\n'
    string_output = string_output + ('='*25) + '\n'

    for idx, this_bin in enumerate(all_bins_content):
        string_output = string_output + this_bin['ContainerType']+':\n'
        string_output = string_output + f'Number of items: {this_bin["n_items"]}\n'
        string_output = string_output + 'Items to pack:\n'
        item_counts = dict(Counter(this_bin['items']))
        for key, value in item_counts.items():
            string_output = string_output + '- ' + key + ' (x{})'.format(value) + '\n'
        #
        # for item in this_bin['items']:
        #     string_output = string_output + '- ' + item + '\n'

        string_output = string_output + ('='*25) + '\n'

    # print('#'*25)
    # print('printing output...')
    # print('#'*25)
    # print(string_output)
    # print('#'*25)
    # print('print complete.')
    # print('#'*25)
    return string_output



# if __name__ == '__main__':
#     with open('./tests/qty.json') as f:
#         data = json.load(f)
#     print(main(items=data))

if __name__ == '__main__':
    with open('/data/temp.json') as f:
        data = json.load(f)
    print(main(items=data), file=sys.stdout)
