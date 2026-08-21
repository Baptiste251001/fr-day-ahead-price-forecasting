"""LSTM point + quantile forecaster, trained with pinball (quantile) loss.

Builds sliding windows of `sequence_length_hours` consecutive feature rows
ending immediately before each target timestamp -- a window containing any
NaN feature value is dropped (NaNs come from insufficient lag/rolling
history at the start of a series; see `fr_power.features.build`). Features
and target are standardized on the training data; quantile predictions are
mapped back to the original scale, which is valid because quantiles
commute with a positive affine rescaling. The median quantile (0.5, or the
closest available) doubles as the point forecast.

A trailing, chronological slice of the training windows is used for
early stopping -- never a random split, to avoid leaking future
information into model selection.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _LSTMNet(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float, n_outputs: int
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        result: torch.Tensor = self.head(last_hidden)
        return result


def _pinball_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: list[float]
) -> torch.Tensor:
    """y_true: (batch,), y_pred: (batch, n_quantiles)."""
    diff = y_true.unsqueeze(1) - y_pred
    q = torch.tensor(quantiles, dtype=y_pred.dtype, device=y_pred.device).unsqueeze(0)
    loss = torch.maximum(q * diff, (q - 1) * diff)
    return loss.mean()


class LSTMQuantileForecaster:
    def __init__(
        self,
        sequence_length_hours: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        batch_size: int,
        learning_rate: float,
        max_epochs: int,
        early_stopping_patience: int,
        quantiles: list[float],
        random_seed: int = 42,
        validation_fraction: float = 0.1,
    ) -> None:
        self.sequence_length_hours = sequence_length_hours
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.quantiles = sorted(quantiles)
        self.random_seed = random_seed
        self.validation_fraction = validation_fraction

        self.feature_columns_: list[str] = []
        self.x_mean_: np.ndarray = np.array([])
        self.x_std_: np.ndarray = np.array([])
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.net_: _LSTMNet | None = None
        self.median_index_ = (
            self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        )

    def _fit_scalers(self, X: pd.DataFrame, y: pd.Series) -> None:
        x_std = X.std(axis=0).to_numpy()
        x_std[x_std == 0] = 1.0
        self.x_mean_ = X.mean(axis=0).to_numpy()
        self.x_std_ = x_std
        y_std = float(y.std())
        self.y_mean_ = float(y.mean())
        self.y_std_ = y_std if y_std > 0 else 1.0

    def _build_windows(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> tuple[np.ndarray, np.ndarray | None, pd.DatetimeIndex]:
        X_scaled = (X.to_numpy(dtype=float) - self.x_mean_) / self.x_std_
        valid_row = ~np.isnan(X_scaled).any(axis=1)
        seq_len = self.sequence_length_hours
        n = len(X_scaled)
        y_arr = y.to_numpy(dtype=float) if y is not None else None

        sequences: list[np.ndarray] = []
        targets: list[float] = []
        end_timestamps: list[pd.Timestamp] = []
        for t in range(seq_len, n):
            window = slice(t - seq_len, t)
            if not valid_row[window].all():
                continue
            if y_arr is not None and np.isnan(y_arr[t]):
                continue
            sequences.append(X_scaled[window])
            end_timestamps.append(X.index[t])
            if y_arr is not None:
                targets.append((y_arr[t] - self.y_mean_) / self.y_std_)

        seq_arr = np.stack(sequences) if sequences else np.empty((0, seq_len, X_scaled.shape[1]))
        target_arr = np.array(targets, dtype=float) if y_arr is not None else None
        return seq_arr, target_arr, pd.DatetimeIndex(end_timestamps)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        torch.manual_seed(self.random_seed)
        self.feature_columns_ = list(X.columns)
        self._fit_scalers(X, y)
        seq_arr, target_arr, _ = self._build_windows(X, y)
        if len(seq_arr) < 2 or target_arr is None:
            raise ValueError("Not enough complete windows to train the LSTM forecaster")

        n_val = max(1, int(len(seq_arr) * self.validation_fraction))
        n_val = min(n_val, len(seq_arr) - 1)
        X_train, y_train = seq_arr[:-n_val], target_arr[:-n_val]
        X_val, y_val = seq_arr[-n_val:], target_arr[-n_val:]

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        self.net_ = _LSTMNet(
            input_size=seq_arr.shape[2],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_outputs=len(self.quantiles),
        )
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.net_.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = self.net_(xb)
                loss = _pinball_loss(yb, preds, self.quantiles)
                loss.backward()
                optimizer.step()

            self.net_.eval()
            with torch.no_grad():
                val_preds = self.net_(X_val_t)
                val_loss = _pinball_loss(y_val_t, val_preds, self.quantiles).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(
                        "LSTM early stopping at epoch %d (best val loss %.4f).",
                        epoch,
                        best_val_loss,
                    )
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.net_ is None:
            raise RuntimeError(
                "LSTMQuantileForecaster.fit() must be called before predict_quantiles()"
            )
        X = X[self.feature_columns_]
        seq_arr, _, end_timestamps = self._build_windows(X, None)
        columns = [f"q{int(q * 100)}" for q in self.quantiles]
        result = pd.DataFrame(np.nan, index=X.index, columns=columns)
        if len(seq_arr) == 0:
            return result

        self.net_.eval()
        with torch.no_grad():
            preds_scaled = self.net_(torch.tensor(seq_arr, dtype=torch.float32)).numpy()
        preds = preds_scaled * self.y_std_ + self.y_mean_
        result.loc[end_timestamps, columns] = preds
        return result

    def predict(self, X: pd.DataFrame) -> pd.Series:
        quantile_df = self.predict_quantiles(X)
        median_col = f"q{int(self.quantiles[self.median_index_] * 100)}"
        return quantile_df[median_col].rename("prediction")
