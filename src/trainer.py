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
        self.criterion = nn.MSELoss()
        self.device = device
        self.r_hit = r_hit

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader
    ) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)
            loss = self.criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

        val_loss = total_loss / len(loader.dataset)

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        dist = np.linalg.norm(preds - targets, axis=1)
        hit_rate = float(np.mean(dist <= self.r_hit))

        return val_loss, hit_rate
