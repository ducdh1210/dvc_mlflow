import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Get url from DVC
import dvc.api

path = "data/winequality.csv"
repo = "/Users/ddo/Personal/Tutorials/dvc_mlfow"  # to change later
version = "v2"

data_url = dvc.api.get_url(path=path, repo=repo, rev=version)

mlflow.set_experiment("demo")


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file form the remote repo
    data = pd.read_csv(data_url, sep=";")

    # Log data params
    mlflow.log_param("data_url", data_url)
    mlflow.log_param("data_version", version)
    mlflow.log_param("input_rows", data.shape[0])
    mlflow.log_param("input_cols", data.shape[1])

    # Split the data input trianing and test sets (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality", which is a scalar from [3,9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Log artifacts: columns used for modeling
    cols_x = pd.DataFrame(list(train_x.columns))
    cols_x.to_csv("features.csv", header=False, index=False)
    mlflow.log_artifact("features.csv")

    cols_y = pd.DataFrame(list(train_y.columns))
    cols_y.to_csv("targets.csv", header=False, index=False)

    mlflow.log_artifact("targets.csv")

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    # Log params
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print("tracking_url_type_store: ", tracking_url_type_store)
    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(
            lr, "model", registered_model_name="ElasticnetWineModel"
        )
    else:
        mlflow.sklearn.log_model(lr, "model")

