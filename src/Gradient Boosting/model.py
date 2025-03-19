import xgboost as xgb

def gradient_boost_tree(n_estimators=100, max_depth=3, learning_rate=0.1):
    return xgb.XGBClassifier(
    n_estimators=n_estimators,  # Number of boosting rounds or the number of trees in the model
    max_depth=max_depth,       # Maximum depth of each tree
    learning_rate=learning_rate, # Step size at each iteration
    objective='binary:logistic', # For regression tasks
    eval_metric='logloss', # Evaluation metric
    verbosity=1,  # Verbosity level for the model
    #tree_method="hist", device = "cuda"
)

