
import numpy as np
import pandas as pd
import pickle

# ---------- ID3 core (based on your train_model_ID3.py) ----------

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def information_gain(X, y, feature):
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) * entropy(y[X[feature] == values[i]])
        for i in range(len(values))
    ])
    return entropy(y) - weighted_entropy

def id3(X, y, features):
    if len(np.unique(y)) == 1:
        return y.iloc[0]
    if len(features) == 0:
        return y.mode()[0]

    gains = [information_gain(X, y, f) for f in features]
    best_feature = features[np.argmax(gains)]

    tree = {best_feature: {}}
    for value in np.unique(X[best_feature]):
        sub_X = X[X[best_feature] == value]
        sub_y = y[X[best_feature] == value]
        remaining_features = [f for f in features if f != best_feature]
        tree[best_feature][value] = id3(sub_X, sub_y, remaining_features)
    return tree

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    value = sample.get(feature, None)
    if value in tree[feature]:
        return predict(tree[feature][value], sample)
    else:
        # fallback to "e" (edible) if missing branch
        return "e"

def save_model(tree, path):
    with open(path, "wb") as f:
        pickle.dump(tree, f)

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
