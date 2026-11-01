"""LSTM over daily sequences with a scikit-learn style interface.

Fixes vs. the original notebook:
* early stopping on an inner validation split that is *grouped by user*;
* normalisation statistics computed on the training fold only, inside ``fit``;
* missing days are represented by explicit mask channels instead of zeros;
* class-weighted loss and full seeding for reproducibility.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

from sleeprisk.data.validation import assert_disjoint_groups


class _Net(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(self.drop(h[-1])).squeeze(-1)


class LSTMClassifier:
    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 60,
        patience: int = 8,
        val_fraction: float = 0.2,
        seed: int = 0,
        device: str | None = None,
    ):
        self.hidden_dim, self.num_layers, self.dropout = hidden_dim, num_layers, dropout
        self.lr, self.weight_decay, self.batch_size = float(lr), float(weight_decay), int(batch_size)
        self.max_epochs, self.patience, self.val_fraction = int(max_epochs), int(patience), float(val_fraction)
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.history_: list[dict] = []

    # ---- preprocessing (fit on training data only) -------------------------------
    def _fit_norm(self, X: np.ndarray) -> None:
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = np.nan_to_num(np.nanmean(flat, axis=0))
        std = np.nanstd(flat, axis=0)
        self.std_ = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)

    def _transform(self, X: np.ndarray) -> torch.Tensor:
        mask = np.isnan(X).astype(np.float32)
        z = np.nan_to_num((X - self.mean_) / self.std_, nan=0.0).astype(np.float32)
        return torch.from_numpy(np.concatenate([z, mask], axis=-1))

    # ---- training -----------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> LSTMClassifier:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        gss = GroupShuffleSplit(n_splits=1, test_size=self.val_fraction, random_state=self.seed)
        tr, va = next(gss.split(X, y, groups))
        assert_disjoint_groups(groups[tr], groups[va], "LSTM inner validation")

        self._fit_norm(X[tr])
        Xtr, Xva = self._transform(X[tr]), self._transform(X[va])
        ytr = torch.from_numpy(y[tr].astype(np.float32))
        yva = torch.from_numpy(y[va].astype(np.float32)).to(self.device)

        pos = float(ytr.sum())
        pos_weight = torch.tensor([(len(ytr) - pos) / max(pos, 1.0)], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.net_ = _Net(Xtr.shape[-1], self.hidden_dim, self.num_layers, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)

        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.net_.train()
            perm = torch.randperm(len(Xtr), generator=gen)
            tr_loss = 0.0
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb, yb = Xtr[idx].to(self.device), ytr[idx].to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net_(xb), yb)
                loss.backward()
                opt.step()
                tr_loss += loss.item() * len(idx)
            self.net_.eval()
            with torch.no_grad():
                va_loss = loss_fn(self.net_(Xva.to(self.device)), yva).item()
            self.history_.append({"epoch": epoch + 1, "train_loss": tr_loss / len(Xtr), "val_loss": va_loss})
            if va_loss < best - 1e-4:
                best, best_state, bad = va_loss, copy.deepcopy(self.net_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmin([h["val_loss"] for h in self.history_]) + 1)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(self._transform(X).to(self.device))).cpu().numpy()
        return np.column_stack([1 - p, p])
