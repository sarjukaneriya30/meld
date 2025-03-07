import pyodbc
import pandas as pd

driver='{ODBC Driver 18 for SQL Server}'
server='''10.13.0.68\SQLHOSTINGSERVE,1433'''
database='BYODB'
uid='d365admin'
pwd='!!USMC5G!!'
connect_str = 'Driver={};Server={};Database={};Uid={};Pwd={};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'.format(driver,server,database,uid,pwd)
print('I think this is our connection string: {}'.format(connect_str))

print('Establishing connection to D365 DB...')
dbcon = pyodbc.connect(connect_str)
print('Connection successful, I think...')


query = '''SELECT * from "WHSWarehouseWorkLineStaging"'''

df = pd.read_sql(query, dbcon)
print(df.head(5).to_dict('records'))
