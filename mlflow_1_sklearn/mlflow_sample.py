import mlflow
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model():
    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    # Optional, just to show how to record a param
    mlflow.log_param("gamma", "auto")
    mlflow.log_param("scaler", True)

    features_train, labels_train, features_val, labels_val = load_features()

    model.fit(features_train, labels_train)
    score = model.score(features_val, labels_val)
    
    # Optional, just to show how to record a metric
    mlflow.log_metric("score", score) 

    store_results(model, features_val)

    # Optional, just to show how to log random files
    mlflow.log_artifact("validation_predictions.pkl")


def load_features():
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

    return features_train, labels_train, features_val, labels_val


def store_results(model, features_val):
    file_name = "validation_predictions.pkl"

    validation_predictions = (features_val, model.predict(features_val))
    with open(file_name, "wb") as file:
        pickle.dump(validation_predictions, file)

    return file_name


if __name__ == "__main__":
    # Optional, set expriment name, to categorize experiments
    mlflow.set_experiment("sample_experiment")
    
    with mlflow.start_run():
        # Logs model, parameters, and metrics 
        # of all supported frameworks' models
        mlflow.autolog()
        train_model()
