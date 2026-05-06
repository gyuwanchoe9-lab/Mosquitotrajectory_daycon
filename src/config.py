from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    data_dir: Path = Path(".")
    train_dir: str = "train"
    test_dir: str = "test"
    labels_file: str = "train_labels.csv"
    submission_file: str = "sample_submission.csv"
    output_dir: Path = Path("outputs")

    # Model selection: "lstm" 
    model_type: str = "lstm"
    device: str = "cuda"

    # LSTM hyperparameters
    input_size: int = 9  # rel_xyz + vel_xyz + acc_xyz
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2


    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 4

    # Evaluation
    r_hit: float = 0.01
