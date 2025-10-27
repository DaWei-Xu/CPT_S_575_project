import os
import torch
import torch.nn as nn
from src.neural_networks import function_neural_network_main
from src.plotting import function_plotting_main
import numpy as np
from src.sub_functions import function_get_project_root

project_root = function_get_project_root()
os.chdir(project_root)

np.random.seed(42)
torch.manual_seed(42)

# Get case number according to the filename
filename = os.path.basename(__file__)
case_number = os.path.splitext(filename)[0]

print(os.getcwd())

processed_data_path = \
    os.path.join("data", "processed_data", "processed_for_training.csv")
best_model_path = \
    os.path.join("data", "training_results", case_number, "best_model.pth")
history_path = \
    os.path.join("data", "training_results", case_number, "training_history.npz")


config = {
        "hidden_layers": [256, 128, 64],
        "activation": nn.LeakyReLU(),
        "dropout": 0.25
    }

function_neural_network_main(processed_data_path=processed_data_path,
    test_size=0.2,
    random_state=42,
    config=config,
    num_epochs=1000,
    lr=1e-3,
    scheduler_factor=0.7,
    scheduler_patience=3,
    best_model_path=best_model_path,
    history_path=history_path
)

function_plotting_main(
    training_history_path=history_path,
    best_model_path=best_model_path,
    processed_data_path=processed_data_path,
    config=config,
    training_curves_save_path=\
        os.path.join("data", "training_results", case_number, "training_curves.png"),
    accuracy_bars_save_path=\
        os.path.join("data", "training_results", case_number, "accuracy_bars.png")
)
