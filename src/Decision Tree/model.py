from sklearn.tree import DecisionTreeClassifier

def get_model():
    model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    return model

if __name__ == "__main__":
    model = get_model()
    print(model)