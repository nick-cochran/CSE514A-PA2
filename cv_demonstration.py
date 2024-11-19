from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# load data
populations = pd.read_csv('Modified_Populations.csv')

# filter out unknown values
populations = populations[populations['Month Groups'] != "Unknown month"]

# split data into features and target
feature_labels = [
    'ID', 'Reason for absence', 'Month of absence', 'Day of the week', 'Seasons',
    'Transportation expense', 'Distance from Residence to Work', 'Service time',
    'Age', 'Work load Average/day ', 'Hit target', 'Disciplinary failure', 'Education',
    'Son', 'Social drinker', 'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
    'Month Groups'
]
features = populations[feature_labels]
target = populations['Absenteeism time in hours']

# initialize variables for hyperparameter tuning and cross validation scores
kNN_values = [3, 5, 7, 9, 11]
accuracy_results = []
f1_results = []

# K-Fold Cross Validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# run model on all hyperparameters with cross validation
for k in kNN_values:
    kNN = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = []
    fold_f1_scores = []

    for train_index, val_index in k_fold.split(features):
        X_train, X_val = features.iloc[train_index], features.iloc[val_index]
        y_train, y_val = target.iloc[train_index], target.iloc[val_index]

        kNN.fit(X_train, y_train)
        y_pred = kNN.predict(X_val)

        # calculate accuracy and F1 score
        fold_accuracies.append(accuracy_score(y_val, y_pred))
        fold_f1_scores.append(f1_score(y_val, y_pred, average='weighted'))

    # save mean accuracy and F1 score
    accuracy_results.append(sum(fold_accuracies) / len(fold_accuracies))
    f1_results.append(sum(fold_f1_scores) / len(fold_f1_scores))

# plot scores per k neighbor
plt.figure(figsize=(10, 6))
plt.plot(kNN_values, accuracy_results, marker='o', label='Accuracy')
plt.plot(kNN_values, f1_results, marker='s', label='F1 Score')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title('5-Fold Cross-Validation Results for kNN')
plt.legend()
plt.show()
