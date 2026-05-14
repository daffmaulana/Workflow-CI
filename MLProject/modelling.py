import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart Disease Experiment")

mlflow.sklearn.autolog()

df = pd.read_csv("heartdisease_preprocessing.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2026
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=2026
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

mlflow.log_metric("accuracy", accuracy)

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model"
)

joblib.dump(model, "random_forest_model.pkl")

# Ambil active run yang dibuat MLflow Project
run = mlflow.active_run()

if run:
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)