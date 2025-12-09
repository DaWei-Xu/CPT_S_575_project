import os
import torch
import torch.nn as nn
import sys
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)


import numpy as np
from src.sub_functions import function_get_project_root


from src.neural_networks import function_neural_network_main, knn_baseline_main,fill_unknown_labels_with_model,evaluate_fill_accuracy_by_masking
from src.plotting import function_plotting_main, function_plot_accuracy_bars
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
    history_path=history_path,
    skip_if_history_exists = True,
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

filled_data_path = os.path.join(
    "data", "processed_data", "processed_for_training_filled.csv"
)

print("\n===== Filling Unknown / Unassigned labels with Stage1 FNN =====")
fill_unknown_labels_with_model(
    processed_data_path=processed_data_path,
    best_model_path=best_model_path,
    output_path=filled_data_path,
    config=config,
)
fill_eval_df =  evaluate_fill_accuracy_by_masking(
    processed_data_path=processed_data_path,
    best_model_path=best_model_path,
    config=config,
    mask_frac=0.3, 
    random_state=42,
)

print(fill_eval_df)
fill_eval_df.to_csv(
    os.path.join("result", f"{case_number}_fill_accuracy_by_rank.csv"),
    index=False,
)

case2 = case_number + "_stage2"
best_model_path_stage2 = os.path.join(
    "data", "training_results", case2, "best_model.pth"
)
history_path_stage2 = os.path.join(
    "data", "training_results", case2, "training_history.npz"
)

print("\n===== Stage 2: train FNN on filled labels (pseudo-labeled Unknowns) =====")
# function_neural_network_main(
#     processed_data_path=filled_data_path,
#     test_size=0.2,
#     random_state=42,
#     config=config,
#     num_epochs=700,
#     lr=1e-3,
#     scheduler_factor=0.7,
#     scheduler_patience=3,
#     best_model_path=best_model_path_stage2,
#     history_path=history_path_stage2,
#     skip_if_history_exists = False,
# )

# function_plotting_main(
#     training_history_path=history_path_stage2,
#     best_model_path=best_model_path_stage2,
#     processed_data_path=filled_data_path,
#     config=config,
#     training_curves_save_path=os.path.join(
#         "data", "training_results", case2, "training_curves.png"
#     ),
#     accuracy_bars_save_path=os.path.join(
#         "data", "training_results", case2, "accuracy_bars.png"
#     ),
# )

knn_csv_path = os.path.join("result", f"{case_number}_knn_accuracy_by_rank.csv")

knn_acc_df = knn_baseline_main(
    processed_data_path=processed_data_path,
    test_size=0.2,
    random_state=42,
    n_neighbors=5,
    output_path=knn_csv_path,
)

os.makedirs("result", exist_ok=True)
function_plot_accuracy_bars(
    accs=knn_acc_df["accuracy"].values,
    target_cols=knn_acc_df["rank"].values,
    save_path=os.path.join("result", f"{case_number}_knn_accuracy_bars.png"),
)
