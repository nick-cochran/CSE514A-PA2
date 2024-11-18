from numpy.random import sample
from sklearn import svm
# from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load the data
populations = pd.read_csv('Modified_Populations.csv')
populations = populations[populations['Month Groups'] != "Unknown month"]

population_A = populations[populations['Month Groups'] == '1']
population_B = populations[populations['Month Groups'] == '2']
population_C = populations[populations['Month Groups'] == '3']

A_train, A_test = train_test_split(population_A, test_size=0.1, random_state=42)
B_train, B_test = train_test_split(population_B, test_size=0.1, random_state=42)
C_train, C_test = train_test_split(population_C, test_size=0.1, random_state=42)

# A_train.to_csv('population_A_train.csv')
# A_test.to_csv('population_A_test.csv')
# B_train.to_csv('population_B_train.csv')
# B_test.to_csv('population_B_test.csv')
# C_train.to_csv('population_C_train.csv')
# C_test.to_csv('population_C_test.csv')

# Split the data into features and target
feature_labels = ['ID','Reason for absence','Month of absence','Day of the week','Seasons',
                  'Transportation expense', 'Distance from Residence to Work','Service time',
                  'Age','Work load Average/day ','Hit target','Disciplinary failure','Education',
                  'Son','Social drinker','Social smoker','Pet','Weight','Height','Body mass index','Month Groups']
features = populations[feature_labels]
A_features = population_A[feature_labels]
B_features = population_B[feature_labels]
C_features = population_C[feature_labels]

target = populations['Absenteeism time in hours']
A_target = population_A['Absenteeism time in hours']
B_target = population_B['Absenteeism time in hours']
C_target = population_C['Absenteeism time in hours']

# Split the data into training and testing sets
X_training, X_testing, y_training, y_testing = train_test_split(features, target, test_size=0.1, random_state=42)
AX_train, AX_test, Ay_train, Ay_test = train_test_split(A_features, A_target, test_size=0.1, random_state=42)
BX_train, BX_test, By_train, By_test = train_test_split(B_features, B_target, test_size=0.1, random_state=42)
CX_train, CX_test, Cy_train, Cy_test = train_test_split(C_features, C_target, test_size=0.1, random_state=42)


# Create the model
SVM_rbf = svm.SVC(kernel='rbf') # SVM hyperparameter value can be the kernel type
SVM_lin = svm.SVC(kernel='linear')
SVM_sig = svm.SVC(kernel='sigmoid')
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))

kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X_training): # idk what this does but copilot suggested it
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    SVM_rbf.fit(X_train, y_train)
    y_prediction_rbf = SVM_rbf.predict(X_test)
    # for i in range(len(y_prediction)):
    #     print(y_prediction[i], y_test.iloc[i])
    # print('\n')
    print("SVM rbf f1 score:", f1_score(y_test, y_prediction_rbf))
    print("SVM rbf accuracy score:", accuracy_score(y_test, y_prediction_rbf))
    # print('\n')
    # print(SVM_rbf.score(X_test, y_test))

    SVM_lin.fit(X_train, y_train)
    y_prediction_lin = SVM_lin.predict(X_test)
    print("SVM linear f1 score:", f1_score(y_test, y_prediction_lin))
    print("SVM linear accuracy score:", accuracy_score(y_test, y_prediction_lin))

    SVM_sig.fit(X_train, y_train)
    y_prediction_sig = SVM_sig.predict(X_test)
    print("SVM sigmoid f1 score:", f1_score(y_test, y_prediction_sig))
    print("SVM sigmoid accuracy score:", accuracy_score(y_test, y_prediction_sig))
    print('\n')

# Are we allowed to use scikit learn for CV given this in the instructions:
# "Write code that will run 5- or 10-fold cross-validation for testing hyperparameter values."

# I'm just not sure what we are supposed to do with the cross validation/ what the takeaway is
# I do see that we need to graph it and visualize it