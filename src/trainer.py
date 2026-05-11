from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        r_hit: float = 0.01,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.HuberLoss(delta=0.01)
        self.device = device
        self.r_hit = r_hit

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for seq_feat, candidates, y in loader:
            seq_feat = seq_feat.to(self.device)
            candidates = candidates.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(seq_feat, candidates)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * seq_feat.size(0)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        for seq_feat, candidates, y in loader:
            seq_feat = seq_feat.to(self.device)
            candidates = candidates.to(self.device)
            y = y.to(self.device)
            pred = self.model(seq_feat, candidates)
            total_loss += self.criterion(pred, y).item() * seq_feat.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        dist = np.linalg.norm(preds - targets, axis=1)
        return total_loss / len(loader.dataset), float(np.mean(dist <= self.r_hit))
