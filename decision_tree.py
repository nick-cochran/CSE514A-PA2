from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt


# load data
populations = pd.read_csv('Modified_Populations.csv')
populations = populations[populations['Month Groups'] != "Unknown month"]

# population_A = populations[populations['Month Groups'] == 1]
# population_B = populations[populations['Month Groups'] == 2]
# population_C = populations[populations['Month Groups'] == 3]


# A_train, A_test = train_test_split(population_A, test_size=0.1, random_state=42)
# B_train, B_test = train_test_split(population_B, test_size=0.1, random_state=42)
# C_train, C_test = train_test_split(population_C, test_size=0.1, random_state=42)

# split data into features and target
feature_labels = ['ID', 'Reason for absence', 'Month of absence', 'Day of the week', 'Seasons',
                  'Transportation expense', 'Distance from Residence to Work', 'Service time',
                  'Age', 'Work load Average/day ', 'Hit target', 'Disciplinary failure', 'Education',
                  'Son', 'Social drinker', 'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
                  'Month Groups']

features = populations[feature_labels]
# A_features = population_A[feature_labels]
# B_features = population_B[feature_labels]
# C_features = population_C[feature_labels]

target = populations['Absenteeism time in hours']
# A_target = population_A['Absenteeism time in hours']
# B_target = population_B['Absenteeism time in hours']
# C_target = population_C['Absenteeism time in hours']

# split the data into training and testing sets
X_training, X_testing, y_training, y_testing = train_test_split(features, target, test_size=0.1, random_state=42)
# AX_train, AX_test, Ay_train, Ay_test = train_test_split(A_features, A_target, test_size=0.1, random_state=42)
# BX_train, BX_test, By_train, By_test = train_test_split(B_features, B_target, test_size=0.1, random_state=42)
# CX_train, CX_test, Cy_train, Cy_test = train_test_split(C_features, C_target, test_size=0.1, random_state=42)

# initialize variables for hyperparameters and cross validation scores
dc_values = ['gini', 'entropy', 'log_loss']
accuracy_results = []

# run model on all hyperparameters with cross validation
for value in dc_values:
    decision_tree = DecisionTreeClassifier(criterion=value)

    # perform 5-fold cross validation and calculate Accuracy score
    cv_scores = cross_val_score(decision_tree, X_training, y_training, cv=5, scoring='accuracy')
    mean_accuracy = cv_scores.mean()
    accuracy_results.append(mean_accuracy)
    print(f"Criterion = {value}: Cross-Validation Accuracy = {mean_accuracy:.4f}")

# pick k that has highest accuracy score
best_value = dc_values[accuracy_results.index(max(accuracy_results))]
print(f"Best Criterion: {best_value}")

# train best model on test set
final_model = DecisionTreeClassifier(criterion=best_value)
final_model.fit(X_training, y_training)
y_test_pred = final_model.predict(X_testing)

# performance of test set
test_accuracy = accuracy_score(y_testing, y_test_pred)
print(f"Test Accuracy with Criterion = {best_value}: {test_accuracy:.4f}")

# plot scores per k neighbor
colors = ['skyblue', 'lightgreen', 'salmon']
plt.figure(figsize=(10, 6))
plt.bar(dc_values, accuracy_results, color=colors)
# plt.plot(kNN_values, f1_results, marker='s', label='F1 Score')
plt.xlabel('Criterion')
plt.ylabel('Cross-Validation Accuracy')
plt.title('5-Fold Cross-Validation Results for Decision Tree Criteria')
plt.show()

print()
# initialize variables for hyperparameters and cross validation scores
rf_values = [5, 10, 15, 20, 25]
accuracy_results = []

# run model on all hyperparameters with cross validation
for value in rf_values:
    rf = RandomForestClassifier(n_estimators=value)

    # perform 5-fold cross validation and calculate Accuracy score
    cv_scores = cross_val_score(rf, X_training, y_training, cv=5, scoring='accuracy')
    mean_accuracy = cv_scores.mean()
    accuracy_results.append(mean_accuracy)
    print(f"trees = {value}: Cross-Validation Accuracy = {mean_accuracy:.4f}")

# pick k that has highest accuracy score
best_value = rf_values[accuracy_results.index(max(accuracy_results))]
print(f"Best number of trees: {best_value}")

# train best model on test set
final_model = RandomForestClassifier(n_estimators=best_value)
final_model.fit(X_training, y_training)
y_test_pred = final_model.predict(X_testing)

# performance of test set
test_accuracy = accuracy_score(y_testing, y_test_pred)
print(f"Test Accuracy with number of trees = {best_value}: {test_accuracy:.4f}")

# plot scores per k neighbor
plt.figure(figsize=(10, 6))
plt.plot(rf_values, accuracy_results, marker='o', label='Accuracy')
# plt.plot(kNN_values, f1_results, marker='s', label='F1 Score')
plt.xlabel('Number of Trees')
plt.ylabel('Cross-Validation Accuracy')
plt.title('5-Fold Cross-Validation Results for Random Forest')
plt.legend()
plt.show()