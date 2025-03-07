import pandas as pd
import os
import re
import numpy as np
import random
import string

# directory/folder path
dir_path = './static'


df = pd.DataFrame()

#these lists will be used to populate our dataframe
start_dates = []

end_dates = []

time_spent = []

total_activity_hours = []

file_paths = []

# this is to make sure we summarize time information if we have more than one row with the same date
duplicate_dates = []

# pattern to get start and end time
pattern = r'2023.*?\.'

# also want to create an excel sheet with all the filenames to facilitate playing videos in AIMS.py
files_df = pd.DataFrame()

# files_df["File Name"] = os.listdir(dir_path)
#
# index = files_df[files_df["File Name"] == ".DS_Store"].index


#files_df = files_df.drop(index)

# # Iterate directory
for file_path in os.listdir(dir_path):
    # extract date information from each video

    matches = re.findall(pattern, str(file_path))

    if len(matches) > 1:

        # add start date, end date, time_spent and total hours info to their respective lists
        start_dates.append(matches[0][:10])
        end_dates.append(matches[1][:10])
        time_spent.append(str(matches[0][11:15]) + " - " + str(matches[1][11:15]))

        total_hours = str(abs(int(matches[0][11:15]) - int(matches[1][11:15])))

        # make sure to convert minutes
        minute_conversion = int(total_hours[-2:])/60

        total_hours = float(total_hours[0]) + minute_conversion

        total_activity_hours.append(total_hours)

        file_paths.append(file_path)
# this is to check if we have more than one row with the same date
# basically we compare the starting and ending times of both rows so we have the actual total span of time
#change the time spent and total hours info for the first row with that date
# then we keep track of the index of the duplicate date row so we can drop it later
for date in range(len(start_dates)):

    for duplicate in range(len(start_dates)):

        if start_dates[date] == start_dates[duplicate] and date != duplicate and date not in duplicate_dates:

            starting_time = min(int(time_spent[date][:4]), int(time_spent[duplicate][:4]))

            ending_time = max(int(time_spent[date][7:]), int(time_spent[duplicate][7:]))

            time_spent[date] = str(starting_time) + " - " + str(ending_time)

            total_time = str(abs(starting_time - ending_time))

            minute_conversion = int(total_time[-2:])/60

            reassigned_time = float(total_time[0]) + minute_conversion

            #print(total_activity_hours[date])
            total_activity_hours[date] = reassigned_time

            duplicate_dates.append(duplicate)

# create the rest of the cols in our df
df["Date"] = start_dates
df["Time Spent"] = time_spent
df["Total Activity Hours"] = total_activity_hours
df["Location"] = "220-PROD-B23"
df["Maximum Amount of Personnel"] = [random.randint(0, 10) for i in range(len(df))]
df["JON #"] = [''.join(random.choices(string.ascii_uppercase + string.digits, k = 8)) for i in range(len(df))]
df["Vehicle ID"] = [random.choice(["Unclassified", "LAV-25A2"]) for i in range(len(df))]
df["Notes"] = ""
df["File Name"] = file_paths

#drop duplicate date rows
df = df.drop(duplicate_dates, axis="index")

df['Date'] = pd.to_datetime(df['Date'])

#sort df by date
df = df.sort_values(by='Date')

files_df["File Name"] = df["File Name"]

df = df.drop(columns=['File Name'])

df.to_excel("Activity_Summary.xlsx", index=False)

files_df.to_excel("filepaths.xlsx", index=False)
