import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
# remove rows with "Unknown month"
df = df[df['Month Groups'] != "Unknown month"]


print("Modified dataframe with Month Groups:")
print(df.head())

# save modified Month Group dataframe to CSV file
df.to_csv('Modified_Populations.csv', index=False)

features = df.drop(['Absenteeism time in hours', 'Month of absence'], axis=1)
target = df['Absenteeism time in hours']

# identify and clean problematic columns
for col in features.columns:
    if features[col].dtype == 'object':
        features[col] = features[col].str.replace(',', '').astype(float)

# Verify all features are numeric
print("Features after cleaning:")
print(features.head())

X = features.to_numpy()
y = target.to_numpy()

print("Features (X):", X[:5])  # show first 5 rows
print("Target (y):", y[:5])  # show first 5 rows

# standardize features
# scaler = MinMaxScaler()
# X_normalized = scaler.fit_transform(X)

# print p-values for data set
X_all = sm.add_constant(features)
model = sm.OLS(y, X_all).fit()
print(model.summary())
print(f"\np-values: {model.pvalues}")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=580, test_size=148, random_state=0)

# add constant for bias
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model_train = sm.OLS(y_train, X_train).fit()
print(model_train.summary())
