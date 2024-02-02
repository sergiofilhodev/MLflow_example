import mlflow_version
import ner

ner = ner.NER() # necessário instalar o pytorch e transformers.

# Define the model hyperparameters
params = {"solver": "newton-cholesky", "max_iter": 1000, "multi_class": "auto", "random_state": None}

ver = mlflow_version.Version_model(
        MLFLOW_TRACKING_URI="https://dagshub.com/sergiofilhodev/MLflow_example.mlflow",
        MLFLOW_TRACKING_USERNAME="sergiofilhodev",
        remote_tracking_password = "0ac2c772d3ca23e6af0e35e04b44ac67d53a30bf",
        model_params = params,
        model_metrics = {"accuracy": 1.0},
        tags = {"Dataset": "SALE train NLP 1", "Team": "NLP"},
        model_name = "NER_NLP_complex",
        experiment_name = "NER_experiment_SALE",
        description = "Versionamento do modelo NER do projeto SALE NUVEN.",
        run_name = "NER NLP NUVEN SALE RUN",
        model = ner
    )

ver.versioning_model()