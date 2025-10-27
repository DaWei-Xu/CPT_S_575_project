import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np


def load_processed_data(processed_data_path=None):
    """
    Load the fully processed dataset produced by data_process.py.
    """
    if processed_data_path is None:
        processed_data_path = os.path.join(
            "data", "processed_data", "processed_for_training.csv"
        )

    df = pd.read_csv(processed_data_path)
    print(f"Loaded processed dataset: {df.shape[0]} samples, {df.shape[1]} columns.")

    feature_cols = ["pore.size", "GenomeName", "RefSeqID", "Proteins", "Size (Kb)"]
    target_cols = [
        "realm (Reference)", "phylum (Reference)", "class (Reference)",
        "order (Reference)", "family (Reference)", "subfamily (Reference)",
        "genus (Reference)"
    ]

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[target_cols].values, dtype=torch.long)
    return X, y, target_cols

class MultiTaxonomyNN(nn.Module):
    """Configurable shared-feature multi-output neural network."""
    def __init__(self, input_dim, num_classes_per_task,
                 hidden_layers=[128, 64],
                 activation=nn.ReLU(),
                 dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.shared = nn.Sequential(*layers)
        self.heads = nn.ModuleList([
            nn.Linear(prev_dim, n_classes) for n_classes in num_classes_per_task
        ])

    def forward(self, x):
        shared_out = self.shared(x)
        outputs = [head(shared_out) for head in self.heads]
        return outputs


def train_model(model, train_loader, X_val, y_val, target_cols,
                num_epochs, lr, scheduler_factor, scheduler_patience, device,
                best_model_path, history_path):
    """
    Train and evaluate the model after each epoch.
    Records learning rate, training loss, and validation loss history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor,
        patience=scheduler_patience
    )

    model.to(device)
    best_val_loss = float("inf")

    # --- History containers ---
    lr_history = []
    train_loss_history = []
    val_loss_history = []

    if os.path.exists(history_path):
        print(f"Training history file already exists. Skipping training.")
        return None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = sum(criterion(preds[i], y_batch[:, i]) for i in range(len(preds)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ---- Validation after each epoch ----
        val_loss, val_accs = evaluate_model(
            model, X_val, y_val, target_cols, device, return_loss=True
        )

        scheduler.step(val_loss)
        current_lr = scheduler.get_last_lr()[0]

        # ---- Record history ----
        lr_history.append(current_lr)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # ---- Save best model ----
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch [{epoch+1:02d}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f}"
        )

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    np.savez(
        history_path,
        lr=np.array(lr_history),
        train_loss=np.array(train_loss_history),
        val_loss=np.array(val_loss_history)
    )
    print(f"Training history saved to '{history_path}'")

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return None


def evaluate_model(model, X_val, y_val, target_cols, device, return_loss=False):
    """
    Evaluate model on validation/test set and compute loss + accuracy per task.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        preds = model(X_val.to(device))
        losses = [criterion(preds[i], y_val[:, i].to(device)).item()
                  for i in range(len(preds))]
        accs = []
        for i, col in enumerate(target_cols):
            pred_labels = preds[i].argmax(dim=1)
            acc = (pred_labels == y_val[:, i].to(device)).float().mean().item()
            accs.append(acc)
            print(f"{col:<20}: Acc={acc:.3f}, Loss={losses[i]:.3f}")
        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(accs) / len(accs)
        print(f"Average Val Loss={avg_loss:.4f}, Average Acc={avg_acc:.4f}")

    if return_loss:
        return avg_loss, accs
    else:
        return accs


def function_neural_network_main(
        processed_data_path,
        test_size,
        random_state,
        config,
        num_epochs,
        lr,
        scheduler_factor,
        scheduler_patience,
        best_model_path,
        history_path
):
    """
    Config templates:
        config = {
        "hidden_layers": [256, 128, 64],
        "activation": nn.LeakyReLU(),
        "dropout": 0.25
    }
    """

    X, y, target_cols = load_processed_data(processed_data_path)

    # ---- Split ----
    X_train, X_val, y_train, y_val = train_test_split(X, y, \
                            test_size=test_size, random_state=random_state)
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)

    # ---- Model config ----
    config = config

    num_classes_per_task = [len(torch.unique(y[:, i])) for i in range(y.shape[1])]
    model = MultiTaxonomyNN(\
        input_dim=X.shape[1], num_classes_per_task=num_classes_per_task, **config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("models", exist_ok=True)
    print(f"Training on {device.upper()} ...")

    # ---- Train + record history ----
    train_model(
        model, train_dl, X_val, y_val, target_cols,
        num_epochs, lr,
        scheduler_factor, scheduler_patience, device,
        best_model_path, history_path
    )
