from numpy.random import sample
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import time


# set reduce bool
if len(sys.argv) > 1 and sys.argv[1] == 'reduce':
    reduce = True
else:
    reduce = False

# load data
populations = pd.read_csv('Modified_Populations.csv')
populations = populations[populations['Month Groups'] != "Unknown month"]

population_A = populations[populations['Month Groups'] == 1]
population_B = populations[populations['Month Groups'] == 2]
population_C = populations[populations['Month Groups'] == 3]

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
feature_labels = ['ID', 'Reason for absence', 'Day of the week', 'Seasons', 'Transportation expense',
                  'Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ',
                  'Hit target', 'Disciplinary failure', 'Education', 'Son', 'Social drinker',
                  'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index', 'Month Groups']
reduced_feature_labels = ['Reason for absence', 'Day of the week', 'Seasons', 'Work load Average/day ',
                          'Disciplinary failure', 'Son', 'Social drinker', 'Pet', 'Body mass index', 'Month Groups']

target = populations['Absenteeism time in hours']
A_target = population_A['Absenteeism time in hours']
B_target = population_B['Absenteeism time in hours']
C_target = population_C['Absenteeism time in hours']

if reduce:
    features = populations[reduced_feature_labels]
    A_features = population_A[reduced_feature_labels]
    B_features = population_B[reduced_feature_labels]
    C_features = population_C[reduced_feature_labels]

    # population_A = population_A[reduced_feature_labels + ['Absenteeism time in hours']]
    # population_B = population_B[reduced_feature_labels + ['Absenteeism time in hours']]
    # population_C = population_C[reduced_feature_labels + ['Absenteeism time in hours']]
    #
    # A_train, A_test = train_test_split(population_A, test_size=0.1, random_state=42)
    # B_train, B_test = train_test_split(population_B, test_size=0.1, random_state=42)
    # C_train, C_test = train_test_split(population_C, test_size=0.1, random_state=42)
    #
    # A_train.to_csv('population_A_train_reduced.csv')
    # A_test.to_csv('population_A_test_reduced.csv')
    # B_train.to_csv('population_B_train_reduced.csv')
    # B_test.to_csv('population_B_test_reduced.csv')
    # C_train.to_csv('population_C_train_reduced.csv')
    # C_test.to_csv('population_C_test_reduced.csv')
else:
    features = populations[feature_labels]
    A_features = population_A[feature_labels]
    B_features = population_B[feature_labels]
    C_features = population_C[feature_labels]

# Split the data into training and testing sets
X_training, X_testing, y_training, y_testing = train_test_split(features, target, test_size=0.1, random_state=42)
AX_train, AX_test, Ay_train, Ay_test = train_test_split(A_features, A_target, test_size=0.1, random_state=42)
BX_train, BX_test, By_train, By_test = train_test_split(B_features, B_target, test_size=0.1, random_state=42)
CX_train, CX_test, Cy_train, Cy_test = train_test_split(C_features, C_target, test_size=0.1, random_state=42)

def svm_cross_validation(X_training, y_training, population):
    SVMs = [svm.SVC(kernel='linear'), svm.SVC(kernel='poly'), svm.SVC(kernel='rbf')]
    accuracy_results = []

    for SVM in SVMs:
        # perform 5-fold cross validation and calculate Accuracy score
        cv_scores = cross_val_score(SVM, X_training, y_training, cv=5, scoring='accuracy')
        print("SVM with kernel:  ", SVM.kernel, " has cross validation score: ", cv_scores)
        mean_accuracy = cv_scores.mean()
        accuracy_results.append(mean_accuracy)
        print(f"kernel = {SVM.kernel}: Cross-Validation Accuracy = {mean_accuracy:.4f}")

    best_kernel = SVMs[accuracy_results.index(max(accuracy_results))].kernel
    print(f"Best kernel: {best_kernel}")

    # train best model on test set
    start_training = time.time()
    final_model = svm.SVC(kernel=best_kernel)
    final_model.fit(X_training, y_training)
    end_training = time.time()
    print(f"\033[1mTraining time: {end_training - start_training} for population {population}\033[0m")
    start_prediction = time.time()
    y_test_pred = final_model.predict(X_testing)
    end_prediction = time.time()
    print(f"\033[1mPrediction time: {end_prediction - start_prediction} for population {population}\033[0m")

    # performance of test set
    test_accuracy = accuracy_score(y_testing, y_test_pred)
    print(f"Test Accuracy with k={best_kernel}: {test_accuracy:.4f}")

    kernels = ['linear', 'poly', 'rbf']
    colors = ['skyblue', 'lightgreen', 'salmon']
    plt.figure(figsize=(10, 6))
    plt.bar(kernels, accuracy_results, color=colors)
    plt.xlabel('Kernels')
    plt.ylabel('Cross-Validation Accuracy')
    if reduce:
        plt.title('5-Fold Cross-Validation Results for SVM Kernels on reduced population ' + population)
    else:
        plt.title('5-Fold Cross-Validation Results for SVM Kernels on population ' + population)
    plt.show()

def knn_cross_validation(X_training, y_training, population):
    kNN_values = [3, 7, 11, 13, 17]
    accuracy_results = []

    for k in kNN_values:
        kNN = KNeighborsClassifier(n_neighbors=k)

        # perform 5-fold cross validation and calculate Accuracy score
        cv_scores = cross_val_score(kNN, X_training, y_training, cv=5, scoring='accuracy')
        print("kNN ", k, " cross validation score: ", cv_scores)
        mean_accuracy = cv_scores.mean()
        accuracy_results.append(mean_accuracy)
        print(f"k = {k}: Cross-Validation Accuracy = {mean_accuracy:.4f}")

    best_k = kNN_values[accuracy_results.index(max(accuracy_results))]
    print(f"Best k: {best_k}")

    # train best model on test set
    start_training = time.time()
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_training, y_training)
    end_training = time.time()
    print(f"\033[1mTraining time: {end_training - start_training} for population {population}\033[0m")
    start_prediction = time.time()
    y_test_pred = final_model.predict(X_testing)
    end_prediction = time.time()
    print(f"\033[1mPrediction time: {end_prediction - start_prediction} for population {population}\033[0m")

    # performance of test set
    test_accuracy = accuracy_score(y_testing, y_test_pred)
    print(f"Test Accuracy with k={best_k}: {test_accuracy:.4f}")

    # plot scores per k neighbor
    plt.figure(figsize=(10, 6))
    plt.plot(kNN_values, accuracy_results, marker='o', label='Accuracy')
    # plt.plot(kNN_values, f1_results, marker='s', label='F1 Score')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validation Accuracy')
    if reduce:
        plt.title('5-Fold Cross-Validation Results for kNN on reduced population ' + population)
    else:
        plt.title('5-Fold Cross-Validation Results for kNN on population ' + population)
    plt.legend()
    plt.show()







# print("\033[1mKNN cross validation on Full Data\033[0m")
# knn_cross_validation(X_training, y_training)

print("\033[1mKNN cross validation on Population A\033[0m")
knn_cross_validation(AX_train, Ay_train, 'A')

print("\033[1mKNN cross validation on Population B\033[0m")
knn_cross_validation(BX_train, By_train, 'B')

print("\033[1mKNN cross validation on Population C\033[0m")
knn_cross_validation(CX_train, Cy_train, 'C')


# print("\033[1mSVM cross validation on Full Data\033[0m")
# svm_cross_validation(X_training, y_training)

print("\033[1mSVM cross validation on Population A\033[0m")
svm_cross_validation(AX_train, Ay_train, 'A')

print("\033[1mSVM cross validation on Population B\033[0m")
svm_cross_validation(BX_train, By_train, 'B')

print("\033[1mSVM cross validation on Population C\033[0m")
svm_cross_validation(CX_train, Cy_train, 'C')






