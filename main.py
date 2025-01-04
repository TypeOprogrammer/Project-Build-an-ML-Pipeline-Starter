import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute> mlflow run . -P steps=download,basic_cleaning
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),  # Path to the basic_cleaning folder
                "main",  
                parameters={
            "input_artifact": "sample.csv:latest",  # Input artifact to clean
            "output_artifact": "clean_sample.csv",  # Output cleaned artifact
            "output_type": "clean_sample",  # Type of output artifact
            "output_description": "Data with outliers and null values removed",  # Description of output data
            "min_price": config['etl']['min_price'],  # Minimum price filter
            "max_price": config['etl']['max_price'],  # Maximum price filter
        },
    )
            ##################
            # Implement here #
            ##################
            

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                 parameters={
            "csv": "clean_sample.csv:latest",  # Input artifact (cleaned data)
            "ref": "clean_sample.csv:latest",  # Reference dataset (same cleaned data in this case)
            "kl_threshold": config["data_check"]["kl_threshold"],  # Threshold for KL divergence
            "min_price": config['etl']['min_price'],  # Minimum price filter
            "max_price": config['etl']['max_price']  # Maximum price filter
        },
    )

            ##################
            # Implement here #
            ##################
            #pass

        if "data_split" in active_steps:
            ##################
            # Implement here #
            ##################
            pass

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################

            pass

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            pass


if __name__ == "__main__":
    go()
  #### save memlflow run . -P steps=basic_cleaning

