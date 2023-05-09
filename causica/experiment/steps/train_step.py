from logging import Logger
from typing import Any, Dict, Optional, Union

import mlflow
import nni

from ...datasets.dataset import Dataset, SparseDataset
from ...datasets.variables import Variables
from ...models.imodel import IModel
from ...models.torch_model import TorchModel
from ...models_factory import create_model, set_model_constraint, set_model_prior


def run_train_main(
    logger: Logger,
    model_type: str,
    output_dir: str,
    variables: Variables,
    dataset: Union[Dataset, SparseDataset],
    device: str,
    model_config: Dict[str, Any],
    train_hypers: Dict[str, Any],
    infer_config: Dict[str, Any],
    prior_path: Optional[str] = None,
    constraint_path: Optional[str] = None,
) -> IModel:
    
    # Create model
    logger.info("Creating new model")
    model = create_model(model_type, output_dir, variables, device, model_config)

    # set the prior
    model = set_model_prior(model, prior_path)

    # set the constraint
    model = set_model_constraint(model, constraint_path)

    if isinstance(model, TorchModel):
        num_trainable_parameters = sum(p.numel() for p in model.parameters())
        mlflow.set_tags({"num_trainable_parameters": num_trainable_parameters})
    dataset.save_data_split(save_dir=output_dir)

    logger.info(f"Created model with ID {model.model_id}.")

    # Train model
    logger.info("Training model.")

    # TODO fix typing. mypy rightly complains that we may pass SparseDataset to a model that can only handle (dense) Dataset here.
    best_mse = model.run_train(dataset=dataset, train_config_dict=train_hypers, infer_config_dict=infer_config)  # type: ignore

    nni.report_final_result(best_mse)

    return model
