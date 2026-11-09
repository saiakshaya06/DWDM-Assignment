
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset
df = pd.read_csv("dataset/mushrooms.csv")

# Step 2: Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

# Step 3: Convert categorical data into numerical format
X = pd.get_dummies(X)

# Step 4: Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 5: Create CART decision tree
# Gini Index is used as the splitting criterion
cart_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

# Step 6: Train the model
print("Training CART Decision Tree...")
cart_model.fit(X_train, y_train)

# Step 7: Make predictions
predictions = cart_model.predict(X_test)

# Step 8: Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nCART DECISION TREE RESULTS")
print("Accuracy:", accuracy * 100, "%")

# Step 9: Display classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Step 10: Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Step 11: Display decision tree rules
print("\nCART TREE RULES:")
print(export_text(cart_model, feature_names=list(X.columns)))

print("\nCART algorithm completed successfully!")
