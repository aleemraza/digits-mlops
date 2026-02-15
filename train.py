import mlflow
import mlflow.sklearn
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Digits_Classifier")

# Load dataset
digits = datasets.load_digits()
X = digits.images.reshape(len(digits.images), -1)
y = digits.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run():
    # Model parameters
    params = {
        "C": 1.0,
        "gamma": 0.001,
        "kernel": "rbf",
        "random_state": 42
    }
    
    model = svm.SVC(**params)
    model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Log parameters
    mlflow.log_params(params)
    
    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("train_samples", len(X_train))
    mlflow.log_metric("test_samples", len(X_test))
    
    # Log classification report as artifact
    report = classification_report(y_test, y_pred, output_dict=True)
    for label in report:
        if label in [str(i) for i in range(10)]:  # Digit labels 0-9
            mlflow.log_metric(f"precision_{label}", report[label]["precision"])
            mlflow.log_metric(f"recall_{label}", report[label]["recall"])
            mlflow.log_metric(f"f1_{label}", report[label]["f1-score"])
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="DigitsClassifier"
    )
    
    print(f"Model registered with accuracy: {acc:.4f}")
    print(f"Model URI: models:/DigitsClassifier/Production")
