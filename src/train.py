"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib
### Import MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, StackingClassifier, VotingClassifier, BaggingClassifier
)

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(col_transf, "artifacts/column_transformer.pkl")
    mlflow.log_artifact("artifacts/column_transformer.pkl", artifact_path="preprocessing")

    return col_transf, X_train, X_test, y_train, y_test

def train_with_grid_search(X_train, y_train,model_name):
    """
    Train multiple models using GridSearchCV and return the best one.
    """
    models_params = {
        "LogisticRegression1": {
            "model": LogisticRegression(max_iter=1000),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["liblinear"]
            }
        },
        "RandomForest1": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20, None]
            }
        },
        "SVM1": {
            "model": SVC(),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        },
        "KNN1": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7, 11],
                "weights": ["uniform", "distance"]
            }
        },
        "XGBoost1": {
            "model": xgb.XGBClassifier(eval_metric='logloss'),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        },
       
        "GradientBoosting1": {
            "model": GradientBoostingClassifier(),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }
        },
        "AdaBoost1": {
            "model": AdaBoostClassifier(),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1, 1]
            }
        },
        "ExtraTrees1": {
            "model": ExtraTreesClassifier(),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20]
            }
        }
    }
    mp = models_params[model_name]
    grid = GridSearchCV(mp["model"], mp["params"], cv=5, scoring='f1', return_train_score=True)
    grid.fit(X_train, y_train)
    for i, params in enumerate(grid.cv_results_["params"]):
        f1_mean = grid.cv_results_["mean_test_score"][i]
        mlflow.log_metric(f"{model_name}_f1_score_{i}", f1_mean)
        for param_key, param_value in params.items():
            mlflow.log_param(f"{model_name}_{param_key}_{i}", param_value)

    print(f"Model: {model_name}, Best F1 Score: {grid.best_score_:.4f}")
    return model_name, grid.best_estimator_, grid

def train_and_uplode(experiment_name):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        df = pd.read_csv("D:/stعdy/New folder/OneDrive_1_12-26-2024/work_in_iti/mlops/MLOps-Course-Labs/dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)
        model_name, model, grid = train_with_grid_search(X_train, y_train,experiment_name)
        mlflow.log_param("best_model", model_name)
        mlflow.log_params(grid.best_params_)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        conf_mat_disp.plot()
        os.makedirs("artifacts", exist_ok=True)
        plot_path = f"artifacts/confusion_matrix_{experiment_name}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")
        plt.show()
def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = infer_signature(X_train, log_reg.predict(X_train))
    # Log model
    mlflow.sklearn.log_model(
        sk_model=log_reg,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5],
        registered_model_name=None  
    )
    ### Log the data
    X_train_sample = X_train.head(100)
    X_train_sample.to_csv("sample_X_train.csv", index=False)
    mlflow.log_artifact("sample_X_train.csv")
    return log_reg


def main():
    os.environ["LOGNAME"] = "joe"
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    ### Set the experiment name
    train_and_uplode("LogisticRegression1")
    train_and_uplode("RandomForest1")
    train_and_uplode("SVM1")
    train_and_uplode("KNN1")
    train_and_uplode("XGBoost1")
    train_and_uplode("GradientBoosting1")
    train_and_uplode("AdaBoost1")
    train_and_uplode("ExtraTrees1")


if __name__ == "__main__":
    main()
