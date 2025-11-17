import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import mlflow
import dagshub


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


with mlflow.start_run(run_name="XGBoost_Baseline"):
    mlflow.xgboost.autolog()

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(features_train, target_train)

    xgb_model_pred = xgb_model.predict(features_test)
    xgb_model_pred = le.inverse_transform(xgb_model_pred)
    labels = le.inverse_transform(target_test)

    cm_path = save_confusion_matrix(
        labels,
        xgb_model_pred,
        labels=le.classes_,
        title="XGB Confusion Matrix",
        filename="xgb_confusion_matrix",
    )

    mlflow.log_artifact(cm_path)

    report = classification_report(labels, xgb_model_pred)

    report_path = os.path.join(
        "artifacts", "xgb_classification_report.txt")

    with open(report_path, "w") as f:
        f.write(report)

    mlflow.log_artifact(report_path)

    xgb_model_acc = accuracy_score(labels, xgb_model_pred)
    xgb_model_f1 = f1_score(labels, xgb_model_pred, average="weighted")

    mlflow.log_metrics({
        "test_accuracy": xgb_model_acc,
        "test_f1_score": xgb_model_f1
    })

    print("Accuracy XGBoost Model: ", xgb_model_acc)
    print("F1 Score XGBoost Model: ", xgb_model_f1)

# with mlflow.start_run(run_name="AdaBoost_Baseline"):
#     mlflow.sklearn.autolog()

#     ada_model = AdaBoostClassifier(
#         n_estimators=300,
#         learning_rate=0.1,
#         random_state=42
#     )
#     ada_model.fit(features_train, target_train)

#     ada_model_pred = ada_model.predict(features_test)
#     ada_model_pred = le.inverse_transform(ada_model_pred)
#     labels = le.inverse_transform(target_test)

#     cm_path = save_confusion_matrix(
#         labels,
#         ada_model_pred,
#         labels=le.classes_,
#         title="AdaBoost Confusion Matrix",
#         filename="adaboost_confusion_matrix",
#     )

#     mlflow.log_artifact(cm_path)

#     report = classification_report(labels, ada_model_pred)

#     report_path = os.path.join(
#         "artifacts", "adaboost_classification_report.txt")

#     with open(report_path, "w") as f:
#         f.write(report)

#     mlflow.log_artifact(report_path)

#     ada_model_acc = accuracy_score(labels, ada_model_pred)
#     ada_model_f1 = f1_score(labels, ada_model_pred, average="weighted")

#     mlflow.log_metrics({
#         "test_accuracy": ada_model_acc,
#         "test_f1_score": ada_model_f1
#     })

#     print("Accuracy AdaBoost Model: ", ada_model_acc)
#     print("F1 Score AdaBoost Model: ", ada_model_f1)
