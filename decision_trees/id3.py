
import pandas as pd
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
df = pd.read_csv("dataset/mushrooms.csv")

X = df.drop("class", axis=1)
y = df["class"]

# Step 2: Calculate entropy
def entropy(target):
    counts = Counter(target)
    total = len(target)

    result = 0
    for count in counts.values():
        probability = count / total
        result -= probability * math.log2(probability)

    return result

# Step 3: Calculate information gain
def information_gain(data, feature, target):
    parent_entropy = entropy(data[target])
    total = len(data)

    weighted_entropy = 0

    for value in data[feature].unique():
        subset = data[data[feature] == value]
        weighted_entropy += (
            len(subset) / total
        ) * entropy(subset[target])

    return parent_entropy - weighted_entropy

# Step 4: Build ID3 decision tree
def build_tree(data, features, target, depth=0):

    majority_class = data[target].mode()[0]

    # Stop conditions
    if data[target].nunique() == 1:
        return majority_class

    if len(features) == 0 or depth == 5 or len(data) < 20:
        return majority_class

    # Find best feature
    best_feature = max(
        features,
        key=lambda feature: information_gain(data, feature, target)
    )

    tree = {
        "feature": best_feature,
        "majority": majority_class,
        "children": {}
    }

    # Create branches
    for value in data[best_feature].unique():

        subset = data[data[best_feature] == value]

        remaining_features = [
            f for f in features if f != best_feature
        ]

        tree["children"][value] = build_tree(
            subset,
            remaining_features,
            target,
            depth + 1
        )

    return tree

# Step 5: Predict using decision tree
def predict(tree, row):

    if not isinstance(tree, dict):
        return tree

    feature = tree["feature"]
    value = row[feature]

    if value in tree["children"]:
        return predict(tree["children"][value], row)

    return tree["majority"]

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_data = X_train.copy()
train_data["class"] = y_train

# Step 7: Train ID3 model
print("Training ID3 Decision Tree...")

id3_tree = build_tree(
    train_data,
    list(X.columns),
    "class"
)

# Step 8: Make predictions
predictions = [
    predict(id3_tree, row)
    for _, row in X_test.iterrows()
]

# Step 9: Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\nID3 DECISION TREE RESULTS")
print("Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("Root Attribute:", id3_tree["feature"])

print("\nID3 algorithm completed successfully!")
