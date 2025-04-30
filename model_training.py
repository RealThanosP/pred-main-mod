# Processing Data Imports
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
import os
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

# Model Training Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Data Processing
from data_processing import import_reference_data, import_target_data

# Helper functions

# Get score of model
def get_score(model, X_train, X_test ,y_train, y_test):
    assert len(X_train) == len(y_train), "Train X and y mismatch"
    assert len(X_test) == len(y_test), "Test X and y mismatch"
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# Gets average
def get_av(score_list):
    av_score = np.median(score_list)
    return av_score

# Trains models for failure_mode
def train_failure_mode_classifiers(training_table:pd.DataFrame, model_dict:dict):
    """
    Trains a separate RandomForestClassifier for each binary failure mode.

    Args:
        training_table (pd.DataFrame): A DataFrame containing sensor/statistical features and failure mode columns.

    Returns:
        dict: A dictionary with trained models for each failure mode.
    """
    # Define the target columns for failure modes
    failure_modes = ["cooler_failure", "valve_failure", "pump_failure", "hydraulic_failure"]

    # Define input features by excluding the failure mode columns
    X = training_table.drop(columns=failure_modes)

    for mode in failure_modes:
        y = training_table[mode]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        model_dict[mode] = clf
        print(f"✅ Trained model for '{mode}' with test accuracy: {clf.score(X_test, y_test):.4f}")

    return 

def train_model(training_table, model_dict, feature_name, model_type, drop_columns: list[str]):
    """
    Trains the given model_type for the specified target feature and stores it in model_dict.

    Args:
        training_table (pd.DataFrame): Full dataset including the target feature.
        model_dict (dict): Dictionary to store the trained model.
        feature_name (str): Name of the target feature to predict.
        model_type: sklearn-compatible model instance (e.g., RandomForestClassifier()).
        drop_columns (list[str]): List of columns to drop for training.

    Returns:
        None
    """
    # Avoid data leakage
    drop_cols = list(set(drop_columns + [feature_name]))
    X = training_table.drop(columns=drop_cols)
    y = training_table[feature_name]

    # Determine stratify based on classification task
    stratify = y if isinstance(model_type, ClassifierMixin) and len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model = model_type.fit(X_train, y_train)
    model_dict[feature_name] = model

    # Evaluation
    y_pred = model.predict(X_test)
    if isinstance(model_type, ClassifierMixin):
        score = accuracy_score(y_test, y_pred)
        print(f"✅ Trained classifier for '{feature_name}' | Accuracy: {score:.4f}")
    elif isinstance(model_type, RegressorMixin):
        score = mean_squared_error(y_test, y_pred, squared=False)
        print(f"✅ Trained regressor for '{feature_name}' | RMSE: {score:.4f}")
    else:
        print("⚠️ Unsupported model type for scoring.")

# Main train_model function returns a dictionary with all the models as values and their respective str(target_names) as keys
def train_models(training_table:pd.DataFrame) -> dict:
    model_dict = {}

    # Regressing model
    # Splitting Data
    X = training_table.drop(columns=['maintenance_score'])
    y = training_table['maintenance_score']
    print(y)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train1, y_train1)

    # Finding top features
    importances = model.feature_importances_
    feature_names = X.columns
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    top_features = feat_imp_df['Feature'].head(20) # Top features from feature importance

    X_top = training_table[top_features]
    y = training_table['maintenance_score']

    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)

    # Models
    rf_model = RandomForestRegressor(n_estimators=100)
    svm_model = SVR()
    knn_model = KNeighborsRegressor()

    # Printing the scores of the models
    print("rf score:" + str(get_score(rf_model, X_train, X_test, y_train, y_test)))
    print("svm score:" + str(get_score(svm_model, X_train, X_test, y_train, y_test)))
    print("kneighbohrs score:" + str(get_score(knn_model, X_train, X_test, y_train, y_test)))

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)   
    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=kfold, scoring='r2')
    svm_scores = cross_val_score(svm_model, X_train, y_train, cv=kfold, scoring='r2')
    knn_scores = cross_val_score(knn_model, X_train, y_train, cv=kfold, scoring='r2')

    av_rf_score = get_av(rf_scores) # average of rf
    av_svm_score = get_av(svm_scores) # average of svm
    av_knn_score = get_av(knn_scores) # average of knn

    # Deciding the model
    score_dict = {
        "rf": av_rf_score,
        "svm": av_svm_score,
        "knn": av_knn_score
    }
    
    regression_model = max(score_dict, key=score_dict.get)

    model_dict["maintenance_score"] = regression_model

    print(f"Regression Model decided...")
    print(" RF score: ", av_rf_score)
    print("SVM score: ", av_svm_score)
    print("KNN score: ", av_knn_score)

    # Other targets
    drop_columns = ['failure_flag', 'rul', 'health_state']
    train_model(training_table=training_table, model_dict=model_dict, feature_name="failure_flag", model_type=RandomForestClassifier(), drop_columns=drop_columns)

    # # Splitting data

    # X_ff = training_table.drop(columns=['failure_flag','rul','health_state']) #removing 'failure_flag','rul','health_state' so we will not have data leakage
    # y_ff = training_table['failure_flag']

    # X_train_ff, X_test_ff, y_train_ff, y_test_ff = train_test_split(
    #     X_ff, y_ff, test_size=0.2, random_state=42, stratify=y_ff
    # )

    # # Failure Flag
    # rf_clas_ff = rf_clas.fit(X_train_ff, y_train_ff)
    # model_dict["failure_flag"] = rf_clas_ff # Save to the dictionary
    # rf_clas_score_ff = get_score(rf_clas, X_train_ff, X_test_ff, y_train_ff, y_test_ff)
    # print("RF CLASS: ", rf_clas_score_ff)

    # # Health State
    # X_hs = training_table.drop(columns=['failure_flag','rul','health_state']) #removing 'failure_flag','rul','health_state' so we will not have data leakage
    # y_hs = training_table['health_state']

    # X_train_hs, X_test_hs, y_train_hs, y_test_hs = train_test_split(
    #     X_hs, y_hs, test_size=0.2, random_state=42, stratify=y_hs
    # )

    # rf_clas_hs = rf_clas.fit(X_train_hs, y_train_hs)
    # model_dict["health_state"] = rf_clas_hs # Save to the dictionary
    # rf_clas_score_hs = get_score(rf_clas, X_train_hs, X_test_hs, y_train_hs, y_test_hs)
    # print("RF CLASS HS: ", rf_clas_score_hs)

    # # Remaining Useful Life
    # X_rul = training_table.drop(columns=['failure_flag','rul','health_state'])
    # y_rul = training_table['rul']

    # X_train_rul, X_test_rul, y_train_rul, y_test_rul = train_test_split(
    #     X_rul, y_rul, test_size=0.2, random_state=42, stratify=y_rul
    # )
    
    # rf_clas_rul = rf_clas.fit(X_train_rul, y_train_rul)
    # model_dict["rul"] = rf_clas_score_rul # Save to the dictionary
    # rf_clas_score_rul = get_score(rf_model, X_train_rul, X_test_rul, y_train_rul, y_test_rul)
    # print("RF CLASS RUL: ", rf_clas_score_rul)

    # # Failure mode 
    # fm_models_dict = train_failure_mode_classifiers(training_table)


if __name__ == '__main__':
    reference_table = import_reference_data()
    target_table = import_target_data()
    
    training_table = pd.concat([reference_table, target_table], axis=1)

    train_models(training_table)