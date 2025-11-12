import torch
from torch.utils.data import DataLoader
from torch import nn

from tqdm import tqdm

from src.gml.model import GCN
from src.gml.data import get_dataloaders

def test(
    model: nn.Module,
    loss_fn: nn.Module,
    test_loader: DataLoader,
    device: str,
    checkpoint_path: str = None
) -> None:
    """Testing loop

    Args:
        model (nn.Module): Model to train
        loss_fn (nn.Module): Loss function
        test_loader (DataLoader): Test data loader
        device (str): Device to train on
        checkpoint_path (str): Path to model checkpoint. Default None
    """
    model.to(device)
    
    # Loading checkpoint
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path)
            
            model.load_state_dict(checkpoint["model_state_dict"])
        except:
            print("No checjpoint found")
    else:
        print("No checkpint passed")

    # Testing
    model.eval()
    test_loss = 0.0
    total_test = 0
    correct_test = 0
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc=f"Testing"):
            pass
            # TODO
    
    # Computing metrics
    avg_test_loss = test_loss / total_test
    test_acc = correct_test / total_test

    # Logging metrics
    print(
        f"Test Loss: {avg_test_loss:.4f} │ Test Acc: {test_acc:.4f}"
    )
        

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Preparing dataloaders
    train_loader, val_loader, test_loader = get_dataloaders("./data/", 32)
    
    # Model configuration
    ...
    LEARNING_RATE = 0.001
    
    model = GCN() # TODO
    loss_fn = nn.CrossEntropyLoss()
    
    # Testing
    test(
        model=model,
        loss_fn=loss_fn,
        test_loader=test_loader,
        device=device,
        checkpoint_path="./data/model/"
    )
