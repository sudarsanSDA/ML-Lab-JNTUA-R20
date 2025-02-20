import pandas as pd
from math import log2

# Function to calculate entropy
def calculate_entropy(data):
    labels = data.iloc[:, -1]  # Target attribute
    label_counts = labels.value_counts()
    total = len(labels)
    entropy = -sum((count / total) * log2(count / total) for count in label_counts)
    return entropy

# Function to calculate information gain
def calculate_information_gain(data, attribute):
    total_entropy = calculate_entropy(data)
    values = data[attribute].unique()
    subset_entropy = sum(
        (len(subset) / len(data)) * calculate_entropy(subset)
        for value in values if (subset := data[data[attribute] == value]).size > 0
    )
    return total_entropy - subset_entropy

# ID3 algorithm to build the decision tree
def id3(data, features):
    # If all labels are the same, return the label
    if len(data.iloc[:, -1].unique()) == 1:
        return data.iloc[0, -1]
    
    # If no features left to split, return the majority label
    if len(features) == 0:
        return data.iloc[:, -1].mode()[0]
    
    # Select feature with highest information gain
    best_feature = max(features, key=lambda feature: calculate_information_gain(data, feature))
    tree = {best_feature: {}}
    
    # Build subtrees for each value of the best feature
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = id3(subset, [f for f in features if f != best_feature])
    
    return tree

# Load data
data = pd.read_csv("training_data_task3.csv")

# Features for ID3
features = list(data.columns[:-1])  # Convert Index to a Python list

# Build decision tree
decision_tree = id3(data, features)

print("Decision Tree:")
print(decision_tree)
