import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _compute_cv_pred(traj: np.ndarray) -> np.ndarray:
    """Constant velocity prediction: last + 2*(last - prev)"""
    return traj[-1] + 2.0 * (traj[-1] - traj[-2])


def _build_features(traj: np.ndarray) -> np.ndarray:
    """
    Returns (T, 9): relative position + velocity + acceleration.
    Positions are centered at the last observed point.
    """
    rel = traj - traj[-1]                        # (T, 3) relative position
    vel = np.diff(traj, axis=0)                  # (T-1, 3)
    vel = np.vstack([vel[:1], vel])              # (T, 3) pad first
    acc = np.diff(vel, axis=0)                   # (T-1, 3)
    acc = np.vstack([acc[:1], acc])              # (T, 3) pad first
    return np.concatenate([rel, vel, acc], axis=1)  # (T, 9)


class MosquitoDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        file_ids: List[str],
        labels: Optional[np.ndarray] = None,
    ):
        trajectories = []
        cv_preds = []
        for file_id in file_ids:
            path = os.path.join(data_dir, f"{file_id}.csv")
            df = pd.read_csv(path)
            traj = df[["x", "y", "z"]].values.astype(np.float32)
            trajectories.append(_build_features(traj))
            cv_preds.append(_compute_cv_pred(traj))

        self.trajectories = np.array(trajectories, dtype=np.float32)  # (N, T, 6)
        self.cv_preds = np.array(cv_preds, dtype=np.float32)           # (N, 3)

        # Store residuals as labels (model learns correction over cv_pred)
        if labels is not None:
            self.labels = (labels - self.cv_preds).astype(np.float32)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int):
        traj = torch.tensor(self.trajectories[idx])
        if self.labels is not None:
            return traj, torch.tensor(self.labels[idx])
        return traj
