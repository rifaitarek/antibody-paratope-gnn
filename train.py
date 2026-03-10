# train.py
"""
Main training script. Uses Hydra for config and W&B for experiment tracking.

Usage:
    python train.py
    python train.py model.num_layers=4 optim.lr=0.0005
    python train.py model.hidden_channels=128 optim.epochs=100
"""

import os
import hydra
import wandb
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from data.dataset import ParatopeDataset
from models.gcn import ParatopeGCN
from utils.metrics import compute_metrics, class_weights_from_dataset
from utils.helpers import set_seed, split_dataset, download_sabdab_sample


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print("Checking/downloading PDB files...")
    download_sabdab_sample(cfg.data.raw_dir, n_structures=20)

    print("Building dataset...")
    dataset = ParatopeDataset(
        pdb_dir=cfg.data.raw_dir,
        processed_dir=cfg.data.processed_dir,
        distance_cutoff=cfg.data.distance_cutoff
    )

    if len(dataset) == 0:
        raise RuntimeError("No valid PDB files processed. Check your data directory.")

    print(f"Total graphs: {len(dataset)}")
    train_data, val_data = split_dataset(dataset, cfg.data.train_split)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=cfg.optim.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.optim.batch_size, shuffle=False)

    # Class weights for imbalanced data
    class_weights = class_weights_from_dataset(train_data).to(device)

    # --- Model ---
    model = ParatopeGCN(
        node_features=cfg.model.node_features,
        hidden_channels=cfg.model.hidden_channels,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.optim.epochs
    )

    # --- W&B Init ---
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"gcn_h{cfg.model.hidden_channels}_l{cfg.model.num_layers}_lr{cfg.optim.lr}"
    )
    wandb.watch(model, log="gradients", log_freq=50)

    # --- Training Loop ---
    best_val_f1 = 0.0

    for epoch in range(1, cfg.optim.epochs + 1):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        # Validate
        if epoch % cfg.wandb.log_interval == 0 or epoch == cfg.optim.epochs:
            model.eval()
            val_logits_all, val_labels_all = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    val_logits_all.append(logits.cpu())
                    val_labels_all.append(batch.y.cpu())

            val_logits = torch.cat(val_logits_all)
            val_labels = torch.cat(val_labels_all)
            val_metrics = compute_metrics(val_logits, val_labels)

            print(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}"
            )

            wandb.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "val/accuracy": val_metrics['accuracy'],
                "val/f1": val_metrics['f1'],
                "val/precision": val_metrics['precision'],
                "val/recall": val_metrics['recall'],
                "val/auc": val_metrics['auc'],
                "lr": scheduler.get_last_lr()[0]
            })

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save(model.state_dict(), "best_model.pt")
                wandb.save("best_model.pt")
                print(f"  ✓ New best model saved (F1={best_val_f1:.4f})")

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()