import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import MosquitoDataset
from src.model import build_model
from src.utils import get_device, set_seed


def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    cfg.output_dir.mkdir(exist_ok=True)

    submission_df = pd.read_csv(
        cfg.data_dir / cfg.submission_file
    )
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

    model = build_model(cfg).to(device)
    model.load_state_dict(
        torch.load(
            cfg.output_dir / "best_model.pt",
            map_location=device,
        )
    )
    model.eval()

    residuals = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            residuals.append(model(x).cpu().numpy())

    residuals = np.concatenate(residuals, axis=0)
    # final prediction = constant velocity baseline + learned residual
    preds = test_ds.cv_preds + residuals
    submission_df["x"] = preds[:, 0]
    submission_df["y"] = preds[:, 1]
    submission_df["z"] = preds[:, 2]

    out_path = cfg.output_dir / "submission.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")


if __name__ == "__main__":
    main()
