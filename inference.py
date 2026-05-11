import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import MosquitoDataset
from src.model import build_model
from src.utils import get_device, set_seed


def _run_model(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            seq_feat, candidates = batch[0].to(device), batch[1].to(device)
            preds.append(model(seq_feat, candidates).cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    cfg.output_dir.mkdir(exist_ok=True)

    submission_df = pd.read_csv(cfg.data_dir / cfg.submission_file)
    test_ids = submission_df["id"].tolist()

    test_dir = str(cfg.data_dir / cfg.test_dir)
    test_ds = MosquitoDataset(test_dir, test_ids)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    ckpts = sorted(cfg.output_dir.glob("best_model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {cfg.output_dir}")

    all_preds = []
    for ckpt in ckpts:
        model_type = ckpt.stem.split("_")[2]  # best_model_{type}_seed{n}
        cfg.model_type = model_type
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        all_preds.append(_run_model(model, test_loader, device))
        print(f"Loaded {ckpt.name}")

    preds = np.mean(all_preds, axis=0)  # ensemble average
    submission_df["x"] = preds[:, 0]
    submission_df["y"] = preds[:, 1]
    submission_df["z"] = preds[:, 2]

    out_path = cfg.output_dir / "submission.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"\nEnsemble of {len(ckpts)} models → {out_path}")


if __name__ == "__main__":
    main()
