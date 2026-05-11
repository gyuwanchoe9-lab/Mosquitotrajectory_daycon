import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

EPS = 1e-8

# (d1, par, perp, d2, jerk, time_scale)
_CANDIDATE_SPECS = [
    (2.00,  0.00,  0.00, 0.0,  0.00, 1.00),  # p0_2d1 (CV baseline)
    (2.00,  0.40,  0.40, 0.0,  0.00, 1.00),
    (2.00,  0.50,  0.50, 0.0,  0.00, 1.00),
    (1.98,  0.56,  0.56, 0.0,  0.00, 1.00),
    (2.00,  0.60,  0.60, 0.0,  0.00, 1.00),
    (1.98,  0.96, -0.08, 0.0,  0.00, 1.00),  # frenet_best
    (1.98,  0.90,  0.00, 0.0,  0.00, 1.00),
    (1.98,  1.00,  0.00, 0.0,  0.00, 1.00),
    (2.00,  1.00, -0.10, 0.0,  0.00, 1.00),
    (1.96,  0.90,  0.20, 0.0,  0.00, 1.00),
    (2.02,  0.80,  0.20, 0.0,  0.00, 1.00),
    (1.94,  1.10, -0.20, 0.0,  0.00, 1.00),
    (2.06,  1.00, -0.08, 0.0,  0.00, 1.00),
    (1.90,  1.00, -0.08, 0.0,  0.00, 1.00),
    (1.98,  0.80, -0.05, 0.0,  0.08, 1.00),  # jerk_small_pos
    (1.98,  0.80, -0.05, 0.0, -0.08, 1.00),  # jerk_small_neg
    (1.98,  0.70, -0.20, 0.0,  0.00, 1.00),
    (1.98,  1.20, -0.20, 0.0,  0.00, 1.00),
    (1.98,  1.20,  0.20, 0.0,  0.00, 1.00),
    (2.08,  1.20, -0.20, 0.0,  0.00, 1.00),
    (1.86,  0.70,  0.20, 0.0,  0.00, 1.00),
    (1.98,  0.96, -0.08, 0.0,  0.00, 0.85),  # latency variants
    (1.98,  0.96, -0.08, 0.0,  0.00, 0.92),
    (1.98,  0.96, -0.08, 0.0,  0.00, 1.08),
    (1.98,  0.96, -0.08, 0.0,  0.00, 1.15),
    (1.98,  1.10, -0.20, 0.0,  0.00, 1.10),
    (1.96,  0.90,  0.20, 0.0,  0.00, 0.90),
]
N_CANDIDATES = len(_CANDIDATE_SPECS)  # 27


def _make_candidates(traj: np.ndarray) -> np.ndarray:
    """Returns (K, 3) physics candidates in Frenet frame."""
    p0 = traj[-1]
    d1 = traj[-1] - traj[-2]
    d2 = traj[-2] - traj[-3]
    acc = d1 - d2
    prev_acc = d2 - (traj[-3] - traj[-4])
    jerk = acc - prev_acc

    tangent = d1 / (np.linalg.norm(d1) + EPS)
    acc_par = np.dot(acc, tangent) * tangent
    acc_perp = acc - acc_par

    preds = []
    for c_d1, c_par, c_perp, c_d2, c_jerk, c_ts in _CANDIDATE_SPECS:
        vs = c_ts          # v_scale = (horizon/2) * time_scale = 1.0 * ts
        as_ = c_ts ** 2   # acc_scale = vs^2
        pred = (p0
                + c_d1 * vs * d1
                + c_d2 * vs * d2
                + c_par * as_ * acc_par
                + c_perp * as_ * acc_perp
                + c_jerk * as_ * jerk)
        preds.append(pred)
    return np.array(preds, dtype=np.float32)  # (K, 3)


def _make_seq_features(traj: np.ndarray) -> np.ndarray:
    """Returns (T, 9) speed/direction physics features per timestep."""
    T = len(traj)
    # pad 3 copies of traj[0] so every timestep has at least 4 lookback points
    padded = np.vstack([np.stack([traj[0]] * 3), traj])  # (T+3, 3)

    features = []
    for t in range(T):
        d1 = padded[t + 3] - padded[t + 2]
        d2 = padded[t + 2] - padded[t + 1]
        prev_d2 = padded[t + 1] - padded[t]
        acc = d1 - d2
        jerk = acc - (d2 - prev_d2)

        speed = np.linalg.norm(d1) + EPS
        prev_speed = np.linalg.norm(d2) + EPS
        tangent = d1 / speed
        acc_par_s = np.dot(acc, tangent)
        acc_perp_v = acc - acc_par_s * tangent

        features.append([
            speed,
            prev_speed / speed,
            np.linalg.norm(acc) / speed,
            abs(acc_par_s) / speed,
            np.linalg.norm(acc_perp_v) / speed,
            np.linalg.norm(jerk) / speed,
            np.dot(d1, d2) / (speed * prev_speed),  # turn_cos
            np.linalg.norm(acc_perp_v) / speed,     # curvature ≈ perp/speed
            1.0,                                     # direction_flag
        ])
    return np.array(features, dtype=np.float32)  # (T, 9)


class MosquitoDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        file_ids: List[str],
        labels: Optional[np.ndarray] = None,
    ):
        seq_feats, candidates = [], []
        for file_id in file_ids:
            df = pd.read_csv(os.path.join(data_dir, f"{file_id}.csv"))
            traj = df[["x", "y", "z"]].values.astype(np.float32)
            seq_feats.append(_make_seq_features(traj))
            candidates.append(_make_candidates(traj))

        self.seq_feats = np.array(seq_feats, dtype=np.float32)   # (N, T, 9)
        self.candidates = np.array(candidates, dtype=np.float32)  # (N, K, 3)
        self.labels = labels.astype(np.float32) if labels is not None else None

    def __len__(self) -> int:
        return len(self.seq_feats)

    def __getitem__(self, idx: int):
        seq = torch.tensor(self.seq_feats[idx])
        cands = torch.tensor(self.candidates[idx])
        if self.labels is not None:
            return seq, cands, torch.tensor(self.labels[idx])
        return seq, cands
