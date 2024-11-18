from numpy.random import sample
from sklearn import svm
# from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

# Load the data
populations = pd.read_csv('Modified_Populations.csv')
populations = populations[populations['Month Groups'] != "Unknown month"]

# population_A = populations[populations['Month Groups'] == '1']


# Split the data into features and target
feature_labels = ['ID','Reason for absence','Month of absence','Day of the week','Seasons',
                  'Transportation expense', 'Distance from Residence to Work','Service time',
                  'Age','Work load Average/day ','Hit target','Disciplinary failure','Education',
                  'Son','Social drinker','Social smoker','Pet','Weight','Height','Body mass index','Month Groups']
features = populations[feature_labels]
target = populations['Absenteeism time in hours']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

# Create the model
model = svm.SVC()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

kf = KFold(n_splits=5)
for train_index, test_index in kf.split(features): # idk what this does but copilot suggested it
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

# Are we allowed to use scikit learn for CV given this in the instructions:
# "Write code that will run 5- or 10-fold cross-validation for testing hyperparameter values."

# I'm just not sure what we are supposed to do with the cross validation/ what the takeaway is
# I do see that we need to graph it and visualize it