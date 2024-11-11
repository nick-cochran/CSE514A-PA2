import numpy as np
import pandas as pd


df = pd.read_csv('Absenteeism_at_work.csv', delimiter=';')
print("Original dataframe:")
print(df.head())
print(df.shape)
print("Columns in dataframe:")
print(df.columns)

absent_hours = df['Absenteeism time in hours']
print(absent_hours)
mode_hours = absent_hours.mode()
print("Mode of Absenteeism time in hours:", mode_hours)

# setting a threshold for >8 as excessive absence hours
threshold = 8
df['Absenteeism time in hours'] = df['Absenteeism time in hours'].apply(lambda x: 1 if x > threshold else 0)
print(df.head())

# save modified dataframe to CSV file
# df.to_csv('Modified_Absenteeism_at_work.csv', index=False)

# separate 'Month' feature into 3 populations
def categorical_month(month):
    # 1 group by January-April
    if 1 <= month <= 4:
        return '1'
    # 2 group by May-August
    elif 5 <= month <= 8:
        return '2'
    # 3 group by September-December
    elif 9 <= month <= 12:
        return '3'
    else:
        return 'Unknown month'

df['Month Groups'] = df['Month of absence'].apply(categorical_month)
print("Modified dataframe with Month Groups:")
print(df.head())

# save modified Month Group dataframe to CSV file
df.to_csv('Modified_Populations.csv', index=False)

