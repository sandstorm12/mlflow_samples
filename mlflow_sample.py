import time
import mlflow
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def save_model():
    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    mlflow.log_param("gamma", "auto")
    mlflow.log_param("scaler", True)

    dataset = load_iris()

    indices = np.arange(len(dataset.data))
    np.random.shuffle(indices)

    labels = dataset.target[indices]
    features = dataset.data[indices]

    train_length = int(.8 * len(dataset.data))

    features_train = features[:train_length]
    labels_train = labels[:train_length]
    features_val = features[train_length:]
    labels_val = labels[train_length:]

    model.fit(features_train, labels_train)

    score = model.score(features_val, labels_val)
    
    mlflow.log_metric("score", score)

    mlflow.sklearn.save_model(model, "model_{}".format(time.time()))

    validation_predictions = (features_val, model.predict(features_val))
    with open("validation_predictions.pkl", "wb") as file:
        pickle.dump(validation_predictions, file)

    mlflow.log_artifact("validation_predictions.pkl")


if __name__ == "__main__":
    with mlflow.start_run():
        save_model()
