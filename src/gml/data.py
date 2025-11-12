from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class GraphDataset(Dataset):
    def __init__(
        self,
        path: os.PathLike
    ):
        """Dataset class for graph datasets

        Args:
            path (os.PathLike): Path to dataset
        """
        # Checking file extention
        if path[-3] not in ["tsv"]:
            raise Exception(f".{ path[-3]} file type is not supported")
        
        # Saving dataset
        self.dataset = pd.read_csv(path, sep="\t", names=['from_node', 'edge', 'to_node'])
    
    def __getitem__(self, index):
        return self.dataset.loc[index].tolist()
    
    def __len__(self):
        return len(self.dataset)

def get_dataset(
    dataset_path: os.PathLike = "./data/"
):
    """Downloads dataset if not present yet

    Args:
        dataset_path (os.PathLike): Path to download dataset to

    Returns:
        PathLike: Path to dataset
    """
    # Checking dataset directory
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        
    # TODO
    
    return dataset_path
    
def get_dataloaders(
    dataset_path: os.PathLike,
    batch_size: int
):
    """Generates 3 dataloaders

    Args:
        dataset_path (os.PathLike): Path to dataset data
        batch_size (int): Batch size

    Returns:
        Tuple: Train, validation and test dataloaders
    """
    # Creating datasets
    train_dataset = GraphDataset(os.path.join(dataset_path, "train_split.tsv"))
    val_dataset = GraphDataset(os.path.join(dataset_path, "val_split.tsv"))
    test_dataset = GraphDataset(os.path.join(dataset_path, "test_split.tsv"))
    
    # Creating dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
    )

    return train_loader, val_loader, test_loader


