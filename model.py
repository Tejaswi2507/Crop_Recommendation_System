import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Split the dataset into input features (X) and target variable (y)
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the classifiers
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
decision_tree_clf = DecisionTreeClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

# Train the classifiers
random_forest_clf.fit(X_train, y_train)
decision_tree_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)

# Make predictions on the test set
rf_y_pred = random_forest_clf.predict(X_test)
dt_y_pred = decision_tree_clf.predict(X_test)
svm_y_pred = svm_clf.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_y_pred)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
svm_accuracy = accuracy_score(y_test, svm_y_pred)

# Save the models and their accuracies
with open('random_forest_model.pkl', 'wb') as rf_model_file:
    pickle.dump(random_forest_clf, rf_model_file)

with open('decision_tree_model.pkl', 'wb') as dt_model_file:
    pickle.dump(decision_tree_clf, dt_model_file)

with open('svm_model.pkl', 'wb') as svm_model_file:
    pickle.dump(svm_clf, svm_model_file)

with open('model_accuracies.pkl', 'wb') as acc_file:
    pickle.dump({
        'random_forest': rf_accuracy, 
        'decision_tree': dt_accuracy, 
        'svm': svm_accuracy
    }, acc_file)