import wandb
import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.backbone import get_backbone
from models.heads.factory import get_head
from models.optimizer import get_optimizer
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
    full_dataset = FacialAttributesDataset("data/train_tiny.npz")
    
    # Split into train and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.training.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.training.num_workers
    )

    # 4. Optimizer and Criterion
    optimizer = get_optimizer(model.parameters(), cfg.optimizer)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Starting training: {backbone_name} + {head_name}")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 5. Initialize W&B
    run = wandb.init(
        entity="marco-repetto-universidad-t-cnica-federico-santa-mar-a",
        project="espejito-espejito-INF577-QUIZ-2",
        config=dict(cfg),
    )

    # 6. Training Loop
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
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

        print(f"Epoch {epoch+1}/{cfg.training.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1
        })

if __name__ == "__main__":
    main()
