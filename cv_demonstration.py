from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# save 10% of data for testing
X_training, X_testing, y_training, y_testing = train_test_split(features, target, test_size=0.1, random_state=42)

# initialize variables for hyperparameters and cross validation scores
kNN_values = [2, 3, 5, 7, 11]
score_results = []

# run model on all hyperparameters with cross validation
for value in kNN_values:
    kNN = KNeighborsClassifier(n_neighbors=value)

    # perform 5-fold cross validation and calculate F-1 score
    cv_scores = cross_val_score(kNN, X_training, y_training, cv=5, scoring=make_scorer(f1_score))
    mean_score = cv_scores.mean()
    score_results.append(mean_score)
    print(f"k = {value}: Cross-Validation Accuracy = {mean_score:.4f}")

# pick k that has highest score
best_value = kNN_values[score_results.index(max(score_results))]
print(f"Best k: {best_value}")

# train best model on test set
final_model = KNeighborsClassifier(n_neighbors=best_value)
final_model.fit(X_training, y_training)
y_test_pred = final_model.predict(X_testing)

# performance of test set
test_score = f1_score(y_testing, y_test_pred)
print(f"Test Accuracy with k={best_value}: {test_score:.4f}")

# plot scores per k neighbor
plt.figure(figsize=(10, 6))
plt.plot(kNN_values, score_results, marker='o', label='F1-Score')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation F1-Score')
plt.title('5-Fold Cross-Validation Results for kNN')
plt.legend()
plt.show()

# confusion matrix to check balance of dataset
cm = confusion_matrix(y_testing, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
