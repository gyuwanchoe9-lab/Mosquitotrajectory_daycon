import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateSelectorGRU(nn.Module):
    """GRU encodes trajectory sequence → scores K physics candidates → soft weighted sum."""

    def __init__(
        self,
        seq_input_size: int = 9,
        hidden_size: int = 128,
        n_candidates: int = 27,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_candidates),
        )

    def score(self, seq_feat: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq_feat)
        return self.scorer(out[:, -1, :])  # (N, K) raw logits

    def forward(self, seq_feat: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        scores = self.score(seq_feat)                    # (N, K)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (weights * candidates).sum(dim=1)


class LSTMPredictor(nn.Module):
    """Legacy LSTM residual model kept for reference."""

    def __init__(self, input_size=9, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

    def forward(self, seq_feat: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq_feat)
        # predict residual over first candidate (CV baseline)
        residual = self.head(out[:, -1, :])
        return candidates[:, 0, :] + residual


def build_model(cfg) -> nn.Module:
    if cfg.model_type == "lstm":
        return LSTMPredictor(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    return CandidateSelectorGRU(
        seq_input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        n_candidates=cfg.n_candidates,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
