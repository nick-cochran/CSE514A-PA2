from numpy.random import sample
from sklearn import svm
# from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load the data
populations = pd.read_csv('Modified_Populations.csv')
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