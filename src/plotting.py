import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from src.neural_networks import MultiTaxonomyNN, load_processed_data



def function_load_training_history(history_path):
    """
    Load the saved training history from .npz file.
    Returns a dictionary with arrays for lr, train_loss, val_loss.
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history file not found: {history_path}")
    history = np.load(history_path)
    print(f"Loaded training history: {list(history.keys())}")
    return history


def function_plot_training_curves(history, save_path):
    """
    Plot training and validation loss over epochs, plus learning rate.
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history["train_loss"], label="Training Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Secondary y-axis for learning rate
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["lr"], color="orange", linestyle="--",
             label="Learning Rate")
    ax2.set_ylabel("Learning Rate", fontsize=12, color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    plt.title("Training & Validation Loss with Learning Rate", fontsize=13)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Saved training curve plot to {save_path}")


def function_evaluate_best_model(model_path,
                                data_path=None,
                                config=None):
    """
    Load best model and evaluate on the full dataset (or a test split).
    """
    # Load processed data
    X, y, target_cols = load_processed_data(data_path)

    if data_path is None:
        data_path = os.path.join(
            "data", "processed_data", "processed_for_training.csv")

    # config template
    #     config = {
    #         "hidden_layers": [256, 128, 64],
    #         "activation": nn.LeakyReLU(),
    #         "dropout": 0.25
    #     }

    num_classes_per_task = [len(torch.unique(y[:, i])) for i in range(y.shape[1])]
    model = MultiTaxonomyNN(input_dim=X.shape[1],
                            num_classes_per_task=num_classes_per_task,
                            **config)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(X)
        accs = []
        for i, col in enumerate(target_cols):
            pred_labels = preds[i].argmax(dim=1)
            acc = (pred_labels == y[:, i]).float().mean().item()
            accs.append(acc)
            print(f"{col:<20}: Accuracy = {acc:.3f}")
        avg_acc = sum(accs) / len(accs)
        print(f"Average accuracy across all taxonomy levels: {avg_acc:.3f}")

    return accs, target_cols


def function_plot_accuracy_bars(
        accs, target_cols, save_path, figsize=(8, 5), dpi=300):
    """
    Plot bar chart showing accuracy per taxonomy level.
    """
    plt.figure(figsize=figsize)
    plt.bar(target_cols, accs, color="steelblue", alpha=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy by Taxonomy Level")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)


def function_plotting_main(
        training_history_path,
        best_model_path,
        processed_data_path,
        config,
        training_curves_save_path,
        accuracy_bars_save_path
):
    history = function_load_training_history(training_history_path)
    function_plot_training_curves(history, save_path=training_curves_save_path)

    accs, target_cols = function_evaluate_best_model(
        model_path=best_model_path,
        data_path=processed_data_path,
        config=config
    )

    function_plot_accuracy_bars(accs, target_cols,
                                save_path=accuracy_bars_save_path)

