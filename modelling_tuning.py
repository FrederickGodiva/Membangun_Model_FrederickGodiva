import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import mlflow
import dagshub
import time
from datetime import datetime


def save_confusion_matrix(y_true, y_pred, labels, title, filename):
    os.makedirs("artifacts", exist_ok=True)
    path = os.path.join("artifacts", f"{filename}.png")

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


def plot_feature_importance(model, feature_names, title, filename):
    os.makedirs("artifacts", exist_ok=True)
    path = os.path.join("artifacts", f"{filename}.png")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title(title, fontsize=16, fontweight='bold')
        plt.bar(range(min(15, len(importances))), importances[indices[:15]])
        plt.xticks(range(min(15, len(importances))),
                   [feature_names[i] for i in indices[:15]], rotation=45, ha='right')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    return path


# Used when monitoring in Dagshub
dagshub.init(repo_owner='frederickgodiva',
             repo_name='Membangun_Model_FrederickGodiva', mlflow=True)
mlflow.set_tracking_uri(
    "https://dagshub.com/FrederickGodiva/Membangun_Model_FrederickGodiva.mlflow")

# Used when monitoring in localhost
# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("BreastCancer_Experiment")

df = pd.read_csv("processed_data.csv")

features = df.drop(columns='diagnosis', axis=1)
target = df['diagnosis']

le = LabelEncoder()
target_encoded = le.fit_transform(target)

features_train, features_test, target_train, target_test = train_test_split(
    features, target_encoded, test_size=0.3, random_state=42)

with mlflow.start_run(run_name="XGBoost_FineTune_Manual"):
    start_time = time.time()

    mlflow.xgboost.autolog(disable=True)

    print("Starting XGBoost fine-tuning ...")

    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_random_search = GridSearchCV(
        estimator=XGBClassifier(random_state=42, n_jobs=-1),
        param_grid=xgb_param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )

    xgb_random_search.fit(features_train, target_train)
    best_xgb_model = xgb_random_search.best_estimator_

    xgb_pred = best_xgb_model.predict(features_test)
    xgb_pred_proba = best_xgb_model.predict_proba(features_test)
    xgb_pred_labels = le.inverse_transform(xgb_pred)
    true_labels = le.inverse_transform(target_test)

    xgb_accuracy = accuracy_score(true_labels, xgb_pred_labels)
    xgb_f1 = f1_score(true_labels, xgb_pred_labels, average="weighted")
    xgb_precision = precision_score(
        true_labels, xgb_pred_labels, average="weighted")
    xgb_recall = recall_score(true_labels, xgb_pred_labels, average="weighted")
    xgb_auc = roc_auc_score(target_test, xgb_pred_proba[:, 1])

    mlflow.log_params(xgb_random_search.best_params_)
    mlflow.log_param("cv_best_score", xgb_random_search.best_score_)
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("feature_scaling", "StandardScaler")

    mlflow.log_metrics({
        "test_accuracy": xgb_accuracy,
        "test_f1_score": xgb_f1,
        "test_precision": xgb_precision,
        "test_recall": xgb_recall,
        "test_auc_roc": xgb_auc,
        "cross_val_score": xgb_random_search.best_score_
    })

    training_time = time.time() - start_time
    mlflow.log_metric("training_time_seconds", training_time)
    mlflow.log_metric("n_features", features_train.shape[1])
    mlflow.log_metric("n_training_samples", features_train.shape[0])
    mlflow.log_metric("n_test_samples", features_test.shape[0])

    mlflow.xgboost.log_model(best_xgb_model, "xgboost_model")

    cm_path = save_confusion_matrix(
        true_labels,
        xgb_pred_labels,
        labels=le.classes_,
        title="XGBoost FineTune - Confusion Matrix",
        filename="xgb_finetune_cm",
    )
    mlflow.log_artifact(cm_path)

    fi_path = plot_feature_importance(
        best_xgb_model,
        features.columns,
        "XGBoost FineTune - Feature Importance",
        "xgb_finetune_fi"
    )
    mlflow.log_artifact(fi_path)

    report = classification_report(true_labels, xgb_pred_labels)
    report_path = os.path.join("artifacts", "xgb_finetune_report.txt")
    with open(report_path, "w") as f:
        f.write("XGBoost FineTune - Classification Report\n")
        f.write("="*50 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(report)
        f.write(f"\nBest CV Score: {xgb_random_search.best_score_:.4f}")
        f.write(f"\nTraining Time: {training_time:.2f} seconds")
        f.write(f"\nBest Parameters: {xgb_random_search.best_params_}")

    mlflow.log_artifact(report_path)

    print("XGBoost FineTune Logging Completed!")
    print(f"Best CV Score: {xgb_random_search.best_score_:.4f}")
    print(f"Test Accuracy: {xgb_accuracy:.4f}")

# with mlflow.start_run(run_name="AdaBoost_FineTune"):
#     start_time = time.time()

#     mlflow.sklearn.autolog(disable=True)

#     print("Starting AdaBoost fine-tuning ...")

#     ada_param_grid = {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 0.5, 1.0]
#     }

#     ada_grid_search = GridSearchCV(
#         estimator=AdaBoostClassifier(random_state=42),
#         param_grid=ada_param_grid,
#         cv=5,
#         scoring='f1_weighted',
#         n_jobs=-1,
#         verbose=1
#     )

#     ada_grid_search.fit(features_train, target_train)
#     best_ada_model = ada_grid_search.best_estimator_

#     ada_pred = best_ada_model.predict(features_test)
#     ada_pred_proba = best_ada_model.predict_proba(features_test)
#     ada_pred_labels = le.inverse_transform(ada_pred)
#     true_labels = le.inverse_transform(target_test)

#     ada_accuracy = accuracy_score(true_labels, ada_pred_labels)
#     ada_f1 = f1_score(true_labels, ada_pred_labels, average="weighted")
#     ada_precision = precision_score(
#         true_labels, ada_pred_labels, average="weighted")
#     ada_recall = recall_score(true_labels, ada_pred_labels, average="weighted")
#     ada_auc = roc_auc_score(target_test, ada_pred_proba[:, 1])

#     mlflow.log_params(ada_grid_search.best_params_)
#     mlflow.log_param("cv_best_score", ada_grid_search.best_score_)
#     mlflow.log_param("model_type", "AdaBoost")
#     mlflow.log_param("feature_scaling", "StandardScaler")

#     mlflow.log_metrics({
#         "test_accuracy": ada_accuracy,
#         "test_f1_score": ada_f1,
#         "test_precision": ada_precision,
#         "test_recall": ada_recall,
#         "test_auc_roc": ada_auc,
#         "cross_val_score": ada_grid_search.best_score_
#     })

#     training_time = time.time() - start_time
#     mlflow.log_metric("training_time_seconds", training_time)
#     mlflow.log_metric("n_features", features_train.shape[1])
#     mlflow.log_metric("n_training_samples", features_train.shape[0])
#     mlflow.log_metric("n_test_samples", features_test.shape[0])

#     mlflow.sklearn.log_model(best_ada_model, "adaboost_model")

#     cm_path = save_confusion_matrix(
#         true_labels,
#         ada_pred_labels,
#         labels=le.classes_,
#         title="AdaBoost Finetune - Confusion Matrix",
#         filename="ada_finetune_cm",
#     )
#     mlflow.log_artifact(cm_path)

#     fi_path = plot_feature_importance(
#         best_ada_model,
#         features.columns,
#         "AdaBoost FineTune - Feature Importance",
#         "ada_finetune_fi"
#     )
#     mlflow.log_artifact(fi_path)

#     report = classification_report(true_labels, ada_pred_labels)
#     report_path = os.path.join("artifacts", "ada_finetune_report.txt")
#     with open(report_path, "w") as f:
#         f.write("AdaBoost FineTune - Classification Report\n")
#         f.write("="*50 + "\n")
#         f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(report)
#         f.write(f"\nBest CV Score: {ada_grid_search.best_score_:.4f}")
#         f.write(f"\nTraining Time: {training_time:.2f} seconds")
#         f.write(f"\nBest Parameters: {ada_grid_search.best_params_}")

#     mlflow.log_artifact(report_path)

#     print("AdaBoost FineTune Logging Completed!")
#     print(f"Best CV Score: {ada_grid_search.best_score_:.4f}")
#     print(f"Test Accuracy: {ada_accuracy:.4f}")
