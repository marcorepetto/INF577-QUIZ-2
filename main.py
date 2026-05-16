import wandb
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.backbone import get_backbone
from models.heads.factory import get_head
from models.optimizer import get_optimizer
from models.scheduler.factory import get_scheduler
from data.dataset import FacialAttributesDataset

def calculate_f1(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return f1

def run_inference(model, device, test_path, batch_size, num_workers, threshold=0.5, tta=True, output_path="outputs/submission.csv", normalization_fun=None):
    import os
    import pandas as pd
    from torch.utils.data import DataLoader
    
    print(f"Starting inference on test set: {test_path} (Threshold: {threshold:.4f}, TTA: {tta})")
    if not os.path.exists(test_path):
        print(f"Test file not found at {test_path}, skipping inference.")
        return

    model.eval()
    test_dataset = FacialAttributesDataset(test_path, transform=None, normalization_fun=normalization_fun)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    all_test_preds = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            
            # Original images
            outputs = model(images)
            
            if tta:
                # Flipped images
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                # Average probabilities (or logits)
                probs = (torch.sigmoid(outputs) + torch.sigmoid(outputs_flipped)) / 2.0
            else:
                probs = torch.sigmoid(outputs)
                
            preds = (probs > threshold).int().cpu().numpy()
            all_test_preds.append(preds)
    
    all_test_preds = np.concatenate(all_test_preds).flatten()
    
    submission = pd.DataFrame({
        "id": np.arange(len(all_test_preds)),
        "Category": all_test_preds
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # 0. Setup Reproducibility
    torch.manual_seed(42)

    # 1. Setup Device
    if cfg.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.training.device)
    print(f"Using device: {device}")

    # 2. Instantiate the Model
    backbone_name = cfg.architecture.backbone
    backbone_model = get_backbone(backbone_name, cfg.backbone)
    head_name = cfg.architecture.head
    head_model = get_head(head_name, cfg.head)
    model = nn.Sequential(backbone_model, head_model).to(device)
    
    # 3. Setup Data
    from torchvision import transforms
    from sklearn.model_selection import train_test_split
    
    aug_cfg = cfg.augmentation
    transform_list = []
    
    if aug_cfg.horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    if "color_jitter" in aug_cfg:
        transform_list.append(transforms.ColorJitter(
            brightness=aug_cfg.color_jitter.brightness,
            contrast=aug_cfg.color_jitter.contrast,
            saturation=aug_cfg.color_jitter.saturation,
            hue=aug_cfg.color_jitter.hue
        ))
    
    if "gaussian_blur" in aug_cfg:
        transform_list.append(transforms.GaussianBlur(
            kernel_size=aug_cfg.gaussian_blur.kernel_size,
            sigma=tuple(aug_cfg.gaussian_blur.sigma)
        ))

    normalization = cfg.get("normalization", "intrgb")
        
    train_transform = transforms.Compose(transform_list)

    data_path = cfg.training.get("data_path", "data/train.npz")
    full_dataset = FacialAttributesDataset(data_path, transform=train_transform, normalization=normalization)
    f_n = full_dataset.get_normalization_function()  # Get the normalization function to apply to val/test datasets
    
    # Stratified Split into train and validation (80/20)
    all_labels = full_dataset.labels
    train_indices, val_indices = train_test_split(
        np.arange(len(full_dataset)),
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    # We need a different dataset object for val to avoid the train_transform
    val_dataset_base = FacialAttributesDataset(data_path, transform=None, normalization=None, normalization_fun=f_n)
    val_dataset = torch.utils.data.Subset(val_dataset_base, val_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    # 4. Optimizer and Criterion
    # Calculate pos_weight for BCEWithLogitsLoss to handle imbalance
    all_labels = full_dataset.labels
    train_labels = all_labels[train_indices]
    pos_count = np.sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = neg_count / (pos_count + 1e-7)
    print(f"Calculated pos_weight: {pos_weight:.4f}")

    optimizer = get_optimizer(model.parameters(), cfg.optimizer)
    scheduler = get_scheduler(optimizer, cfg.scheduler)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    print(f"Starting training: {backbone_name} + {head_name}")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 5. Initialize W&B
    run = wandb.init(
        entity="marco-repetto-universidad-t-cnica-federico-santa-mar-a",
        project="espejito-espejito-INF577-QUIZ-2",
        config=dict(cfg),
    )

    # 6. Training Loop
    best_f1 = 0.0
    best_model_state = None
    smoothing = cfg.get("label_smoothing", 0.0)
    
    # Early Stopping setup
    early_stop_cfg = cfg.get("early_stopping", None)
    patience_counter = 0
    early_stop_patience = early_stop_cfg.patience if early_stop_cfg else None
    early_stop_min_delta = early_stop_cfg.min_delta if early_stop_cfg else 0.0

    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Apply Label Smoothing
            if smoothing > 0:
                with torch.no_grad():
                    labels = labels * (1 - smoothing) + 0.5 * smoothing

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        val_loss /= len(val_loader.dataset)
        val_f1 = calculate_f1(torch.cat(all_preds), torch.cat(all_labels))

        # Track best model
        if val_f1 > best_f1 + early_stop_min_delta:
            best_f1 = val_f1
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{cfg.training.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f} - LR: {current_lr:.6f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "lr": current_lr
        })

        if early_stop_patience and patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Save Best Model at the end
    if best_model_state is not None:
        import os
        os.makedirs("outputs", exist_ok=True)
        save_path = "outputs/best_model.pth"
        torch.save(best_model_state, save_path)
        print(f"Best Val F1: {best_f1:.4f}. Model saved to {save_path}")

        # Run inference on test set
        test_path = cfg.training.get("test_path", "data/test.npz")
        model.load_state_dict(best_model_state)

        # --- Threshold Optimization ---
        print("Finding optimal threshold on validation set...")
        model.eval()
        val_logits = []
        val_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_logits.append(outputs.cpu())
                val_targets.append(labels.cpu())
        
        val_logits = torch.cat(val_logits)
        val_targets = torch.cat(val_targets)
        
        best_threshold = 0.5
        if cfg.get("fit_threshold", True):
            best_val_f1_opt = 0.0
            for threshold in np.arange(0.1, 0.9, 0.05):
                f1 = calculate_f1(val_logits, val_targets, threshold=threshold)
                if f1 > best_val_f1_opt:
                    best_val_f1_opt = f1
                    best_threshold = threshold
        
            print(f"Optimal threshold: {best_threshold:.4f} with optimized Val F1: {best_val_f1_opt:.4f}")

        run_inference(
            model, 
            device, 
            test_path, 
            cfg.training.batch_size, 
            cfg.training.num_workers,
            threshold=best_threshold,
            tta=cfg.get("tta", True),
            normalization_fun=f_n
        )

if __name__ == "__main__":
    main()
