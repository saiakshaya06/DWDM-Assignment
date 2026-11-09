
import pandas as pd
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
        p = count / total
        result -= p * math.log2(p)

    return result


# Step 3: Calculate Information Gain
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


# Step 4: Calculate Split Information
def split_info(data, feature):
    total = len(data)
    result = 0

    for value in data[feature].unique():
        probability = len(data[data[feature] == value]) / total

        if probability > 0:
            result -= probability * math.log2(probability)

    return result


# Step 5: Calculate Gain Ratio
def gain_ratio(data, feature, target):
    gain = information_gain(data, feature, target)
    split = split_info(data, feature)

    if split == 0:
        return 0

    return gain / split


# Step 6: Build C4.5 Decision Tree
def build_tree(data, features, target, depth=0):

    majority_class = data[target].mode()[0]

    # Stop conditions
    if data[target].nunique() == 1:
        return majority_class

    if len(features) == 0 or depth >= 5 or len(data) < 20:
        return majority_class

    # Calculate Information Gain for all features
    gains = {
        feature: information_gain(data, feature, target)
        for feature in features
    }

    average_gain = sum(gains.values()) / len(gains)

    # C4.5: Consider features with at least average Information Gain
    candidates = [
        feature for feature in features
        if gains[feature] >= average_gain
    ]

    if not candidates:
        candidates = features

    # Select feature with highest Gain Ratio
    best_feature = max(
        candidates,
        key=lambda feature: gain_ratio(data, feature, target)
    )

    tree = {
        "feature": best_feature,
        "majority": majority_class,
        "children": {}
    }

    remaining_features = [
        feature for feature in features
        if feature != best_feature
    ]

    # Create branches
    for value in data[best_feature].unique():

        subset = data[data[best_feature] == value]

        tree["children"][value] = build_tree(
            subset,
            remaining_features,
            target,
            depth + 1
        )

    return tree


# Step 7: Prediction
def predict(tree, row):

    if not isinstance(tree, dict):
        return tree

    feature = tree["feature"]
    value = row[feature]

    if value in tree["children"]:
        return predict(tree["children"][value], row)

    return tree["majority"]


# Step 8: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_data = X_train.copy()
train_data["class"] = y_train

# Step 9: Train C4.5 model
print("Training C4.5 Decision Tree...")

c45_tree = build_tree(
    train_data,
    list(X.columns),
    "class"
)

# Step 10: Make predictions
predictions = [
    predict(c45_tree, row)
    for _, row in X_test.iterrows()
]

# Step 11: Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\nC4.5 DECISION TREE RESULTS")
print("Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nRoot Attribute:", c45_tree["feature"])

print("\nC4.5 algorithm completed successfully!")
