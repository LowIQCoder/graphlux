import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

from tqdm import tqdm
from utils.logger import logger
from torchinfo import summary

from src.gml.model import GCN
from src.gml.data import get_dataloaders

def train(
    model: nn.Module,
    optimizer: optim,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: str,
    checkpoint_path: str = None
) -> None:
    """Training and validation loop

    Args:
        model (nn.Module): Model to train
        optimizer (optim): Used optimizer
        loss_fn (nn.Module): Loss function
        train_loader (DataLoader): Train data loader
        val_loader (DataLoader): Validation data loader
        epochs (int): NUmber of epochs
        device (str): Device to train on
        checkpoint_path (str): Path to model checkpoint. Default None
    """
    logger.info(f"Using device: {device}")
    
    model.to(device)
    
    # Loading checkpoint
    best_acc = 0
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_acc = checkpoint["accuracy"]
            train_logger.info(f"Loaded checkpoint from {checkpoint_path} with accuracy {best_acc}")
        except FileNotFoundError:
            train_logger.warning(f"No checkpoint found at {checkpoint_path}")
    else:
        train_logger.info("No checkpoint path provided")
    
    # Training and validation loops
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch in tqdm(train_loader,desc=f"Epoch {epoch:3d} │ Training"):
            pass
            # TODO
    
        # Validating
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds_val = []
        all_labels_val = []
        with torch.inference_mode():
            for batch in tqdm(val_loader,desc=f"Epoch {epoch:3d} │ Validating"):
                pass
                # TODO
        
        # Computing metrics
        avg_val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        avg_train_loss = train_loss / total_train
        train_acc = correct_train / total_train
    
        # Logging metrics
        train_logger.info(
            f"Epoch {epoch+1:3d}/{epochs} │ "
            f"Train Loss: {avg_train_loss:.4f} │ Val Loss: {avg_val_loss:.4f} │ "
            f"Train Acc: {train_acc:.4f} │ Val Acc: {val_acc:.4f}"
        )
        
        # Saving best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": best_acc,
            })
            train_logger.info(f"Saved new best model with accuracy: {best_acc}")


if __name__ == "__main__":
    train_logger = logger.getChild(__name__)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Preparing dataloaders
        train_loader, val_loader, test_loader = get_dataloaders("./data/", 32)
        
        # Model configuration
        ...
        LEARNING_RATE = 0.001
        EPOCHS = 10
        
        model = GCN() # TODO
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()
        
        # Training
        train(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            checkpoint_path="./data/model/"
        )
    except Exception as e:
        train_logger.error(f"Error during training: {e}")
