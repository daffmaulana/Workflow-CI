import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart Disease Experiment")

mlflow.sklearn.autolog()

df = pd.read_csv("heartdisease_preprocessing.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2026)


model = RandomForestClassifier(n_estimators=100, random_state=2026)  

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

joblib.dump(model, "random_forest_model.pkl")