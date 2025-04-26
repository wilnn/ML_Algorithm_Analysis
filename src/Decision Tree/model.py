from sklearn.tree import DecisionTreeClassifier


'''#model before tunning hyperparameters
def get_model():
    model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    return model

if __name__ == "__main__":
    model = get_model()
    print(model)'''

#model after tuning hyperparameters
def get_model():
    model = DecisionTreeClassifier(
        criterion='entropy',        # use 'entropy' instead of 'gini' for better splits
        max_depth=10,               # allow deeper trees to capture more patterns
        min_samples_split=5,        # a node must have at least 5 samples to split
        min_samples_leaf=2,         # minimum samples at a leaf node
        max_features='sqrt',        # use sqrt(number of features) when looking for best split
        random_state=42
    )
    return model

if __name__ == "__main__":
    model = get_model()
    print(model)


