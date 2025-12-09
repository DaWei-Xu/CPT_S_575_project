import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle   # NEW


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
                 dropout=0.3,use_probs=True):
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
        self.shared_dim = prev_dim
        self.num_tasks = len(num_classes_per_task)
        self.num_classes_per_task = list(num_classes_per_task)
        self.use_probs = use_probs 
        self.heads = nn.ModuleList()
        in_dim = self.shared_dim
        for t, n_classes in enumerate(self.num_classes_per_task):
            self.heads.append(nn.Linear(in_dim, n_classes))
            in_dim = self.shared_dim + n_classes

    def forward(self, x):
        shared_out = self.shared(x)  
        outputs = []

        current_feat = shared_out
        for t, head in enumerate(self.heads):
            logits = head(current_feat)  
            outputs.append(logits)

            if t < self.num_tasks - 1:
                if self.use_probs:
                    extra = torch.softmax(logits, dim=1) 
                else:
                    extra = logits

                current_feat = torch.cat([shared_out, extra], dim=1)  # [B, shared_dim + C_t]

        return outputs 



def train_model(model, train_loader, X_val, y_val, target_cols,
                num_epochs, lr, scheduler_factor, scheduler_patience, device,
                best_model_path, history_path,task_weights=None, unknown_ids=None,skip_if_history_exists: bool = True):
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

    if skip_if_history_exists and os.path.exists(history_path):
        print(f"Training history file already exists at '{history_path}'. Skipping training.")
        return None


    if task_weights is None:
        task_weights = [1.0] * len(target_cols)
    else:
        assert len(task_weights) == len(target_cols), \
            "len(task_weights) must match number of target columns"
        task_weights = [float(w) for w in task_weights]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            #loss = sum(criterion(preds[i], y_batch[:, i]) for i in range(len(preds)))
            per_task_losses = []
            for i in range(len(preds)):
                ce_i = criterion(preds[i], y_batch[:, i])
                w_i = task_weights[i]
                per_task_losses.append(w_i * ce_i)

            loss = sum(per_task_losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ---- Validation after each epoch ----
        val_loss, val_accs = evaluate_model(
            model, X_val, y_val, target_cols, device, return_loss=True,unknown_ids=unknown_ids  
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


def evaluate_model(model, X_val, y_val, target_cols, device, return_loss=False,unknown_ids=None):
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
            y_true_i = y_val[:, i].to(device)
            unk_id = unknown_ids.get(col, None)

            
            if unknown_ids is not None:
                unk_id = unknown_ids.get(col, None)
            else:
                unk_id = None

            if unk_id is not None:
                is_unknown_true = (y_true_i == unk_id)
                is_unknown_pred = (pred_labels == unk_id)

                correct_known = (~is_unknown_true) & (pred_labels == y_true_i)
                correct_unknown = is_unknown_true & (~is_unknown_pred)

                correct = correct_known | correct_unknown
            else:
                correct = (pred_labels == y_true_i)

            acc = correct.float().mean().item()
            accs.append(acc)
            print(f"{col:<20}: Acc={acc:.3f}, Loss={losses[i]:.3f}")

        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(accs) / len(accs)
        print(f"Average Val Loss={avg_loss:.4f}, Average Acc={avg_acc:.4f}")

    if return_loss:
        return avg_loss, accs
    else:
        return accs

def _load_unknown_ids(processed_data_path, target_cols):
 
    encoder_dir = os.path.join(os.path.dirname(processed_data_path),
                               "label_encoders")
    unknown_ids = {}

    for col in target_cols:
        fname = col.replace(" ", "_") + "_encoder.pkl"
        path = os.path.join(encoder_dir, fname)
        if not os.path.exists(path):
            unknown_ids[col] = None
            continue

        with open(path, "rb") as f:
            le = pickle.load(f)

        classes = np.array(le.classes_, dtype=str)
        lower = np.char.lower(classes)
        mask = (lower == "unknown") | (lower == "unassigned")
        if mask.any():
            unknown_ids[col] = int(np.where(mask)[0][0])
        else:
            unknown_ids[col] = None

    return unknown_ids

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
        history_path,
        skip_if_history_exists=True,
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
    unknown_ids = _load_unknown_ids(processed_data_path, target_cols)

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

    task_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0] 

    # ---- Train + record history ----
    train_model(
        model, train_dl, X_val, y_val, target_cols,
        num_epochs, lr,
        scheduler_factor, scheduler_patience, device,
        best_model_path, history_path,
        task_weights=task_weights,          

        unknown_ids=unknown_ids,
        skip_if_history_exists=skip_if_history_exists,
    )


def evaluate_fill_accuracy_by_masking(
    processed_data_path: str,
    best_model_path: str,
    config: dict,
    rank_names=None,
    mask_frac: float = 0.3,
    random_state: int = 42,
):
    
    X, y, target_cols = load_processed_data(processed_data_path)
    df = pd.read_csv(processed_data_path)

    unknown_ids = _load_unknown_ids(processed_data_path, target_cols)

    if rank_names is None:
        rank_names = list(target_cols)

    num_classes_per_task = [
        int(len(torch.unique(y[:, i]))) for i in range(y.shape[1])
    ]
    model = MultiTaxonomyNN(
        input_dim=X.shape[1],
        num_classes_per_task=num_classes_per_task,
        **config,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits_list = model(X.to(device))

    pred_ids_per_rank = [
        logits.argmax(dim=1).cpu().numpy() for logits in logits_list
    ]

    rng = np.random.default_rng(random_state)
    results = []

    for col in rank_names:
        if col not in target_cols:
            print(f"[WARN] Column '{col}' not in target_cols, skip.")
            continue

        t_idx = target_cols.index(col)
        pred_ids = pred_ids_per_rank[t_idx]

        true_ids = df[col].to_numpy()

        unk_id = unknown_ids.get(col, None) if unknown_ids is not None else None
        if unk_id is None:
            is_unknown = np.zeros(len(true_ids), dtype=bool)
        else:
            is_unknown = (true_ids == unk_id)

        known_idx = np.where(~is_unknown)[0]
        n_known = len(known_idx)
        if n_known == 0:
            print(f"[{col}] no known labels, skip.")
            continue

        n_mask = max(1, int(n_known * mask_frac))
        mask_idx = rng.choice(known_idx, size=n_mask, replace=False)

        acc = float(np.mean(pred_ids[mask_idx] == true_ids[mask_idx]))
        print(
            f"[{col}] mask_frac={mask_frac:.2f}, "
            f"n_masked={n_mask}, fill_accuracy≈{acc:.3f}"
        )

        results.append(
            {"rank": col, "n_masked": int(n_mask), "fill_accuracy": acc}
        )

    if not results:
        return pd.DataFrame(columns=["rank", "n_masked", "fill_accuracy"])

    return pd.DataFrame(results)


def knn_baseline_main(
        processed_data_path,
        test_size=0.2,
        random_state=42,
        n_neighbors=5,
        output_path="data/results/knn_accuracy_by_rank.csv",   
):
    X_torch, y_torch, target_cols = load_processed_data(processed_data_path)

    X = X_torch.numpy()
    y = y_torch.numpy()

    unknown_ids = _load_unknown_ids(processed_data_path, target_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    print("\n=== KNN baseline ===")
    for i, col in enumerate(target_cols):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train_scaled, y_train[:, i])

        y_pred = clf.predict(X_test_scaled)
        unk_id = unknown_ids.get(col, None)

        y_true_rank = y_test[:, i]
        correct = (y_true_rank == y_pred)

        if unk_id is not None:
           is_unknown_true = (y_true_rank == unk_id)
           is_unknown_pred = (y_pred == unk_id)

           correct_known = (~is_unknown_true) & (y_true_rank == y_pred)

           correct_unknown = is_unknown_true & (~is_unknown_pred)

           correct = correct_known | correct_unknown
        else:
             correct = (y_true_rank == y_pred)

        acc = correct.mean()
        results.append({"rank": col, "accuracy": acc})
        print(f"{col:<22} accuracy = {acc:.3f}")


    acc_df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    acc_df.to_csv(output_path, index=False)
    print(f"\nKNN accuracy table saved to: {output_path}")

    return acc_df



def _load_label_encoders(processed_data_path, target_cols):
   
    encoder_dir = os.path.join(os.path.dirname(processed_data_path), "label_encoders")
    encoders = {}

    for col in target_cols:
        fname = col.replace(" ", "_") + "_encoder.pkl"
        path = os.path.join(encoder_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Label encoder for '{col}' not found: {path}")
        with open(path, "rb") as f:
            encoders[col] = pickle.load(f)

    return encoders

def fill_unknown_labels_with_model(
    processed_data_path: str,
    best_model_path: str,
    output_path: str,
    config: dict,
):
  
    X, y, target_cols = load_processed_data(processed_data_path)
    unknown_ids = _load_unknown_ids(processed_data_path, target_cols)

    df = pd.read_csv(processed_data_path)

    num_classes_per_task = [
        int(len(torch.unique(y[:, i]))) for i in range(y.shape[1])
    ]
    model = MultiTaxonomyNN(
        input_dim=X.shape[1],
        num_classes_per_task=num_classes_per_task,
        **config,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits_list = model(X.to(device))

    pred_ids_per_rank = [
        logits.argmax(dim=1).cpu().numpy() for logits in logits_list
    ]

    encoders = _load_label_encoders(processed_data_path, target_cols)

    unknown_strs = {"unknown", "unassigned", ""}

    for i, col in enumerate(target_cols):
        encoder = encoders[col]
        pred_ids = pred_ids_per_rank[i]

        pred_labels = encoder.inverse_transform(pred_ids)

        col_true = df[col].astype(str).fillna("")

        mask_un = col_true.str.strip().str.lower().isin(unknown_strs)

        df.loc[mask_un, col] = pred_labels[mask_un.to_numpy()]

    
        n_filled = int(mask_un.sum())
        print(f"[{col}] filled {n_filled} unknown entries.")

    out_dir = os.path.dirname(output_path)
    if out_dir != "" and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved filled taxonomy table to: {output_path}")