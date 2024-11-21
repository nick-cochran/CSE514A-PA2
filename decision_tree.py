from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
populations = pd.read_csv('Modified_Populations.csv')
populations = populations[populations['Month Groups'] != "Unknown month"]

# split data into features and target
feature_labels = ['ID', 'Reason for absence', 'Month of absence', 'Day of the week', 'Seasons',
                  'Transportation expense', 'Distance from Residence to Work', 'Service time',
                  'Age', 'Work load Average/day ', 'Hit target', 'Disciplinary failure', 'Education',
                  'Son', 'Social drinker', 'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
                  'Month Groups']

features = populations[feature_labels]
target = populations['Absenteeism time in hours']

# split the data into training and testing sets
X_training, X_testing, y_training, y_testing = train_test_split(features, target, test_size=0.1, random_state=42)


# initialize variables for hyperparameters and cross validation scores
dc_values = ['gini', 'entropy', 'log_loss']
score_results = []

# run model on all hyperparameters with cross validation
for value in dc_values:
    decision_tree = DecisionTreeClassifier(criterion=value)

    # perform 5-fold cross validation and calculate Accuracy score
    cv_scores = cross_val_score(decision_tree, X_training, y_training, cv=5, scoring=make_scorer(f1_score))
    mean_score = cv_scores.mean()
    score_results.append(mean_score)
    print(f"Criterion = {value}: Cross-Validation F1-Score = {mean_score:.4f}")

# pick k that has highest accuracy score
best_value = dc_values[score_results.index(max(score_results))]
print(f"Best Criterion: {best_value}")

# train best model on test set
final_model = DecisionTreeClassifier(criterion=best_value)
final_model.fit(X_training, y_training)
y_test_pred = final_model.predict(X_testing)

# performance of test set
test_score = f1_score(y_testing, y_test_pred)
print(f"Test Accuracy with Criterion = {best_value}: {test_score:.4f}")

# plot scores
colors = ['c', 'm', 'y']
plt.figure(figsize=(10, 6))
plt.bar(dc_values, score_results, color=colors)
# plt.plot(kNN_values, f1_results, marker='s', label='F1 Score')
plt.xlabel('Criterion')
plt.ylabel('Cross-Validation Accuracy')
plt.title('5-Fold Cross-Validation Results for Decision Tree Criteria')
plt.show()

# confusion matrix to check balance of dataset
cm = confusion_matrix(y_testing, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


print()
# initialize variables for hyperparameters and cross validation scores
rf_values = [5, 10, 15, 20, 25]
score_results = []

# run model on all hyperparameters with cross validation
for value in rf_values:
    rf = RandomForestClassifier(n_estimators=value)

    # perform 5-fold cross validation and calculate Accuracy score
    cv_scores = cross_val_score(rf, X_training, y_training, cv=5, scoring=make_scorer(f1_score))
    mean_score = cv_scores.mean()
    score_results.append(mean_score)
    print(f"trees = {value}: Cross-Validation Accuracy = {mean_score:.4f}")

# pick k that has highest accuracy score
best_value = rf_values[score_results.index(max(score_results))]
print(f"Best number of trees: {best_value}")

# train best model on test set
final_model = RandomForestClassifier(n_estimators=best_value)
final_model.fit(X_training, y_training)
y_test_pred = final_model.predict(X_testing)

# performance of test set
test_score = f1_score(y_testing, y_test_pred)
print(f"Test Accuracy with number of trees = {best_value}: {test_score:.4f}")

# plot scores
plt.figure(figsize=(10, 6))
plt.plot(rf_values, score_results, marker='o', label='F1-Score')
plt.xlabel('Number of Trees')
plt.ylabel('Cross-Validation F1-Score')
plt.title('5-Fold Cross-Validation Results for Random Forest')
plt.legend()
plt.show()

# confusion matrix to check balance of dataset
cm = confusion_matrix(y_testing, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

