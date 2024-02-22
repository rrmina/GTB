from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load Sample Data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the weights
n_samples = X_train.shape[0]
weights = np.ones(n_samples) / n_samples

# On the spot gradient tree boosting
n_estimators =  100
learning_rate =  0.1
trees = []

for _ in range(n_estimators):
    # Train a Decision Tree on the weighted data
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train, sample_weight=weights)
    trees.append(tree)
    
    # Make predictions and calculate the error
    predictions = tree.predict(X_train)
    errors = np.not_equal(predictions, y_train).astype(int)
    
    # Calculate the new weights
    weights = weights * np.exp(learning_rate * errors)
    weights = weights / np.sum(weights)  # Normalize the weights
    
# Make predictions and evaluate
y_pred = np.zeros_like(y_test)

for tree in trees:
    y_pred += tree.predict(X_test) * learning_rate

# Take the majority vote for classification
y_pred = np.argmax(np.bincount(y_pred, weights=y_pred), axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))