from storage.warehouse_input_parser import WarehouseInputParser

warehouse_path='warehouse-layout-input.xlsx'
warehouse = WarehouseInputParser(warehouse_path).process_warehouse_input()
all_shelf_ids = list(warehouse['ShelfID'])
def to_loc_fmt(key_fmt):
    return 'I21'+''.join(list(map(str,key_fmt)))+'A'
for shelf in all_shelf_ids:
    print(to_loc_fmt(shelf))
