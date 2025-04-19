import numpy as np

def collect_attack_data(shadow_model, train_data, test_data, feature_cols, target_variable):
    # shadow model outputs
    train_preds = shadow_model.predict_proba(train_data[feature_cols])
    test_preds = shadow_model.predict_proba(test_data[feature_cols])
    # lables for member & nonmember
    train_labels = [1] * len(train_preds)
    test_labels = [0] * len(test_preds)
    # real values
    train_true = train_data[target_variable].values
    test_true = test_data[target_variable].values
    # stack into 1 dataset 
    x_attack = np.vstack([train_preds, test_preds])
    y_attack = np.hstack([train_labels, test_labels])
    return x_attack, y_attack
