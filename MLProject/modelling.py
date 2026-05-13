import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart Disease Experiment")

mlflow.sklearn.autolog(log_models=False)

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    pip_requirements=[
        "mlflow==2.14.3",
        "scikit-learn==1.5.1",
        "pandas==2.2.2",
        "numpy==1.26.4"
    ]
)

df = pd.read_csv("heartdisease_preprocessing.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2026)


with mlflow.start_run() as run:

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=2026
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    run_id = run.info.run_id

    with open("run_id.txt", "w") as f:
        f.write(run_id)

    joblib.dump(model, "random_forest_model.pkl")