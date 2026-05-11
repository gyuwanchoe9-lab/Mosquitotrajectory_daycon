import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import MosquitoDataset
from src.model import build_model
from src.trainer import Trainer
from src.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.model_type is not None:
        cfg.model_type = args.model_type

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    cfg.output_dir.mkdir(exist_ok=True)

    labels_df = pd.read_csv(cfg.data_dir / cfg.labels_file)
    file_ids = labels_df["id"].tolist()
    labels = labels_df[["x", "y", "z"]].values.astype(np.float32)

    indices = list(range(len(file_ids)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
    )
    train_ids = [file_ids[i] for i in train_idx]
    val_ids = [file_ids[i] for i in val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    train_dir = str(cfg.data_dir / cfg.train_dir)
    train_ds = MosquitoDataset(train_dir, train_ids, train_labels)
    val_ds = MosquitoDataset(train_dir, val_ids, val_labels)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )

    trainer = Trainer(model, optimizer, device, r_hit=cfg.r_hit)

    ckpt_name = f"best_model_{cfg.model_type}_seed{cfg.seed}.pt"
    best_hit_rate = 0.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, hit_rate = trainer.evaluate(val_loader)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Hit Rate: {hit_rate:.4f}"
        )

        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            torch.save(model.state_dict(), cfg.output_dir / ckpt_name)
            print(f"  -> Saved {ckpt_name} (hit_rate={hit_rate:.4f})")

    print(f"\nTraining complete. Best Hit Rate: {best_hit_rate:.4f}")


if __name__ == "__main__":
    main()
