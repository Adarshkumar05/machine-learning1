# Neural Network Models for Debutanizer Project

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_autoregressive_matrix(
    X: np.ndarray, y: np.ndarray, lags: Sequence[int] = (1, 2)
) -> Tuple[np.ndarray, np.ndarray]:
    # Add delayed y values as features
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    if not lags:
        raise ValueError("Provide at least one lag.")

    lags = sorted(set(int(abs(lag)) for lag in lags if lag > 0))
    max_lag = max(lags)
    if len(X) <= max_lag:
        raise ValueError("Not enough samples for the requested lags.")

    X_trimmed = X[max_lag:].copy()
    lag_features = []
    for lag in lags:
        lagged = y[max_lag - lag : len(y) - lag].reshape(-1, 1)
        lag_features.append(lagged)

    X_aug = np.hstack([X_trimmed] + lag_features)
    y_trimmed = y[max_lag:].copy()
    return X_aug, y_trimmed


def prepare_lstm_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int = 10, train_ratio: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Create sequences for LSTM
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1.")
    if X.shape[0] <= seq_len:
        raise ValueError("Not enough samples for the requested sequence length.")

    cutoff = max(int(len(X) * train_ratio), seq_len + 1)
    scaler = StandardScaler().fit(X[:cutoff])
    X_scaled = scaler.transform(X)

    sequences: List[np.ndarray] = []
    targets: List[float] = []
    for idx in range(seq_len, len(X_scaled)):
        sequences.append(X_scaled[idx - seq_len : idx])
        targets.append(y[idx])

    sequences = np.asarray(sequences, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    train_limit = max(cutoff - seq_len, 1)
    X_train_seq = sequences[:train_limit]
    y_train_seq = targets[:train_limit]
    X_test_seq = sequences[train_limit:]
    y_test_seq = targets[train_limit:]
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Sequence[int], dropout: float = 0.1, use_batch_norm: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            if use_batch_norm:
                layers.append(nn.LayerNorm(units))  # LayerNorm works better with small batches
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=lstm_dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :]).squeeze(-1)


class NeuralModels:
    def __init__(self, random_state: int = 42, device: str | None = None):
        self.random_state = random_state
        _set_global_seed(random_state)
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _make_loaders(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, val_split: float
    ) -> Tuple[DataLoader, DataLoader]:
        idx = np.arange(len(X))
        if len(idx) < 5:
            split = int(max(len(idx) * (1 - val_split), 1))
            train_idx, val_idx = idx[:split], idx[split:]
        else:
            train_idx, val_idx = train_test_split(
                idx, test_size=val_split, random_state=self.random_state, shuffle=True
            )
        if len(train_idx) == 0:
            train_idx = val_idx[:1]
        if len(val_idx) == 0:
            val_idx = train_idx[-1:]

        X_train_tensor = torch.from_numpy(X[train_idx].astype(np.float32))
        y_train_tensor = torch.from_numpy(y[train_idx].astype(np.float32)).unsqueeze(-1)
        X_val_tensor = torch.from_numpy(X[val_idx].astype(np.float32))
        y_val_tensor = torch.from_numpy(y[val_idx].astype(np.float32)).unsqueeze(-1)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False
        )
        return train_loader, val_loader

    def _fit_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        patience: int,
    ) -> Tuple[nn.Module, List[float]]:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
        )
        best_state = None
        best_val = float("inf")
        patience_counter = 0
        history: List[float] = []

        for epoch in range(max_epochs):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = model(xb)
                    val_losses.append(criterion(preds, yb).item())

            epoch_val = float(np.mean(val_losses))
            history.append(epoch_val)
            scheduler.step(epoch_val)

            if epoch_val + 1e-8 < best_val:
                best_val = epoch_val
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, history

    def _predict(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X.astype(np.float32)).to(self.device)
            preds = model(inputs).cpu().numpy()
        return preds

    def train_feedforward(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        hidden_layer_configs: Iterable[Sequence[int]],
        learning_rates: Iterable[float],
        weight_decays: Iterable[float],
        batch_size: int = 64,
        val_split: float = 0.2,
        max_epochs: int = 400,
        patience: int = 40,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> Dict[str, object]:
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Scale y too - helps with training
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        best_result: Dict[str, object] = {
            "val_loss": float("inf"),
            "model": None,
            "params": None,
            "history": None,
        }

        for layers in hidden_layer_configs:
            for lr in learning_rates:
                for wd in weight_decays:
                    model = FeedForwardNet(X_train.shape[1], layers, dropout=dropout, use_batch_norm=use_batch_norm).to(self.device)
                    train_loader, val_loader = self._make_loaders(
                        X_train_scaled, y_train_scaled, batch_size, val_split
                    )
                    trained_model, history = self._fit_model(
                        model,
                        train_loader,
                        val_loader,
                        learning_rate=lr,
                        weight_decay=wd,
                        max_epochs=max_epochs,
                        patience=patience,
                    )
                    val_loss = history[-1] if history else float("inf")
                    if val_loss < best_result["val_loss"]:
                        best_result = {
                            "val_loss": val_loss,
                            "model": copy.deepcopy(trained_model),
                            "params": {"layers": layers, "lr": lr, "weight_decay": wd},
                            "history": history,
                            "scaler_X": scaler_X,
                            "scaler_y": scaler_y,
                        }

        if best_result["model"] is None:
            raise RuntimeError("Feed-forward training failed.")

        model = best_result["model"]
        scaler_X = best_result["scaler_X"]
        scaler_y = best_result["scaler_y"]
        
        y_pred_train_scaled = self._predict(model, scaler_X.transform(X_train))
        y_pred_test_scaled = self._predict(model, X_test_scaled)
        
        # Convert back to original scale
        y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
        y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
        
        # Handle any NaN/Inf values
        y_pred_train = np.nan_to_num(y_pred_train, nan=np.nanmean(y_train), posinf=np.nanmax(y_train), neginf=np.nanmin(y_train))
        y_pred_test = np.nan_to_num(y_pred_test, nan=np.nanmean(y_test), posinf=np.nanmax(y_test), neginf=np.nanmin(y_test))
        
        best_result.update(
            {
                "train_pred": y_pred_train,
                "test_pred": y_pred_test,
                "n_params": self._count_params(model),
            }
        )
        return best_result

    def train_autoregressive_nn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        hidden_layer_configs: Iterable[Sequence[int]],
        learning_rates: Iterable[float],
        weight_decays: Iterable[float],
        use_batch_norm: bool = True,
        **kwargs,
    ) -> Dict[str, object]:
        return self.train_feedforward(
            X_train,
            y_train,
            X_test,
            y_test,
            hidden_layer_configs=hidden_layer_configs,
            learning_rates=learning_rates,
            weight_decays=weight_decays,
            use_batch_norm=use_batch_norm,
            **kwargs,
        )

    def train_lstm(
        self,
        X_train_seq: np.ndarray,
        y_train_seq: np.ndarray,
        X_test_seq: np.ndarray,
        y_test_seq: np.ndarray,
        hidden_sizes: Iterable[int],
        num_layers_options: Iterable[int],
        learning_rates: Iterable[float],
        weight_decays: Iterable[float],
        batch_size: int = 64,
        val_split: float = 0.2,
        max_epochs: int = 300,
        patience: int = 30,
        dropout: float = 0.1,
    ) -> Dict[str, object]:
        X_train_seq = X_train_seq.astype(np.float32)
        X_test_seq = X_test_seq.astype(np.float32)
        
        scaler_y = StandardScaler()
        y_train_seq_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1)).ravel()
        y_test_seq_scaled = scaler_y.transform(y_test_seq.reshape(-1, 1)).ravel()

        best_result: Dict[str, object] = {
            "val_loss": float("inf"),
            "model": None,
            "params": None,
            "history": None,
            "scaler_y": scaler_y,
        }

        for hidden_size in hidden_sizes:
            for num_layers in num_layers_options:
                for lr in learning_rates:
                    for wd in weight_decays:
                        model = LSTMRegressor(
                            input_dim=X_train_seq.shape[2],
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                        ).to(self.device)
                        train_loader, val_loader = self._make_loaders(
                            X_train_seq, y_train_seq_scaled, batch_size, val_split
                        )
                        trained_model, history = self._fit_model(
                            model,
                            train_loader,
                            val_loader,
                            learning_rate=lr,
                            weight_decay=wd,
                            max_epochs=max_epochs,
                            patience=patience,
                        )
                        val_loss = history[-1] if history else float("inf")
                        if val_loss < best_result["val_loss"]:
                            best_result = {
                                "val_loss": val_loss,
                                "model": copy.deepcopy(trained_model),
                                "params": {
                                    "hidden_size": hidden_size,
                                    "num_layers": num_layers,
                                    "lr": lr,
                                    "weight_decay": wd,
                                },
                                "history": history,
                                "scaler_y": scaler_y,
                            }

        if best_result["model"] is None:
            raise RuntimeError("LSTM training failed.")

        model = best_result["model"]
        scaler_y = best_result["scaler_y"]
        
        y_pred_train_scaled = self._predict(model, X_train_seq)
        y_pred_test_scaled = self._predict(model, X_test_seq)
        
        y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
        y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
        
        # Clean up any bad values
        y_pred_train = np.nan_to_num(y_pred_train, nan=np.nanmean(y_train_seq), posinf=np.nanmax(y_train_seq), neginf=np.nanmin(y_train_seq))
        y_pred_test = np.nan_to_num(y_pred_test, nan=np.nanmean(y_test_seq), posinf=np.nanmax(y_test_seq), neginf=np.nanmin(y_test_seq))
        
        best_result.update(
            {
                "train_pred": y_pred_train,
                "test_pred": y_pred_test,
                "n_params": self._count_params(model),
            }
        )
        return best_result


