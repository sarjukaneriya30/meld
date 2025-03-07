import pandas as pd
import sqlite3

df = pd.read_excel("Activity_Summary.xlsx")

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(df, flush=True)

db_conn = sqlite3.connect("pi.db")

c = db_conn.cursor()


c.execute(

    """
    CREATE TABLE IF NOT EXISTS activity (Date TEXT, "Time Spent" TEXT, "Total Activity Hours" REAL,
    Location TEXT, "Maximum Amount of Personnel" INTEGER, "JON #" TEXT, "Vehicle ID" TEXT, Notes TEXT);
    """
)

df.to_sql('activity', db_conn, if_exists='append', index=False)


print(pd.read_sql("SELECT * FROM activity LIMIT 10", db_conn))
