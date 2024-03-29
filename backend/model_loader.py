import os
import torch
from modelLinkedBlocks import Model

# Global variable for the models directory
models_dir = "models"

# Retain models in memory for each user
user_models = {}


def save_model(model: Model):
    """
    Save the model's state dictionary to a file.

    Args:
        model: The model to save.
    """
    os.makedirs(models_dir, exist_ok=True)
    filename = os.path.join(models_dir, f"{model.id}.pt")
    model_dict = model.to_dict_state()
    torch.save(model_dict, filename)


def load_model(name):
    """
    Checks if model is loaded in memory otherwise load model from a file.

    Args:
        name: The name of the model to load.

    Returns:
        The loaded model, or None if the model file doesn't exist.
    """
    if name in user_models:
        return user_models[name]


    filename = os.path.join(models_dir, f"{name}.pt")
    model = Model(name)

    if not os.path.isfile(filename):
        user_models[name] = model
        return model

    model_dict = torch.load(filename)
    name = model_dict["name"]
    model = Model(name)
    model.from_dict(model_dict)
    user_models[name] = model
    return model


def create_model(name):
    """
    Create a new model and save it to a file.

    Args:
        name: The name of the model.

    Returns:
        The created model.
    """
    model = Model(name)
    save_model(model)
    return model #model.to_dict()


def load_models():
    """
    Load all models from the models directory.

    Returns:
        A list of loaded models.
    """
    models = []
    os.makedirs(models_dir, exist_ok=True)
    for filename in os.listdir(models_dir):
        if filename.endswith(".pt"):
            model_name = os.path.splitext(filename)[0]
            model = load_model(model_name)
            if model is not None:
                models.append(dict(name=model.name, id=model.id))
    return models

